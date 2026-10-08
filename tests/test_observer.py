# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    Frame,
    InputTransportMessageFrame,
    InterruptionFrame,
    LLMTextFrame,
    MetricsFrame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import STTUsage, STTUsageMetricsData, TTFBMetricsData
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from corral.observer import SPAN_NAME, TurnLatencyObserver

MS = 1_000_000
STT, LLM, TTS = FrameProcessor(), FrameProcessor(), FrameProcessor()
AUDIO = TTSAudioRawFrame(audio=b"\x00\x00", sample_rate=24000, num_channels=1)
USAGE = MetricsFrame(data=[STTUsageMetricsData(processor="x", value=STTUsage(audio_seconds=1))])


def _marker(kind: str, utterance_id: str | None = None) -> InputTransportMessageFrame:
    message = {"type": kind}
    if utterance_id:
        message["utterance_id"] = utterance_id
    return InputTransportMessageFrame(message=message)


def _ttfb(seconds: float) -> MetricsFrame:
    return MetricsFrame(data=[TTFBMetricsData(processor="x", value=seconds)])


class Call:
    def __init__(self) -> None:
        self.exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.now_ns = 0
        self.observer = TurnLatencyObserver(
            attributes={"agent": "x"},
            models={"llm": {"provider": "openai", "model": "gpt-4.1"}},
            stt=STT,
            llm=LLM,
            tts=TTS,
            tracer=provider.get_tracer("test"),
            clock=lambda: self.now_ns,
        )

    async def at(self, ms: int, frame: Frame, source: FrameProcessor | None = None) -> None:
        self.now_ns = ms * MS
        source = source or FrameProcessor()
        pushed = FramePushed(source, source, frame, FrameDirection.DOWNSTREAM, timestamp=0)
        await self.observer.on_push_frame(pushed)

    @property
    def turns(self) -> list[dict[str, Any]]:
        return self.observer.turns

    def stages(self) -> dict[str, ReadableSpan]:
        return {s.name: s for s in self.exporter.get_finished_spans() if s.name != SPAN_NAME}


@pytest.fixture
def call() -> Call:
    return Call()


async def test_a_turn_times_each_provider_at_its_own_request(call: Call) -> None:
    await call.at(1_000, _marker("utterance_end"))
    await call.at(1_200, VADUserStoppedSpeakingFrame())
    await call.at(1_260, _ttfb(0.0), source=STT)
    await call.at(1_300, UserStoppedSpeakingFrame())
    await call.at(1_600, _ttfb(0.25), source=LLM)
    await call.at(1_600, LLMTextFrame("Hi"))
    await call.at(1_800, _ttfb(0.15), source=TTS)
    await call.at(1_800, AUDIO)
    await call.at(1_820, BotStartedSpeakingFrame())
    (turn,) = call.turns
    assert turn["s2s.turn_latency_ms"] == 600.0
    assert turn["s2s.turn_latency.turn_detection_timed_out"] is False
    assert turn["s2s.utterance_end.unix_ms"] == 1_000.0
    assert turn["s2s.turn_latency.stt_ms"] == 60.0
    assert turn["s2s.turn_latency.llm_ms"] == 250.0
    assert turn["s2s.turn_latency.tts_ms"] == 150.0
    assert turn["s2s.turn_latency.transport_ms"] == 20.0
    stages = call.stages()
    assert set(stages) == {"stt", "llm", "tts"}
    assert (stages["llm"].start_time, stages["llm"].end_time) == (1_350 * MS, 1_600 * MS)
    assert dict(stages["llm"].attributes or {}) == {"provider": "openai", "model": "gpt-4.1"}


async def test_the_pause_starts_at_silero_minus_its_window(call: Call) -> None:
    await call.at(1_000, _marker("utterance_end"))
    await call.at(1_200, VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await call.at(1_860, LLMTextFrame("Hi"))
    await call.at(2_121, AUDIO)
    await call.at(2_122, BotStartedSpeakingFrame())
    (turn,) = call.turns
    assert turn["s2s.turn_latency_ms"] == 1_121.0
    assert call.stages() == {}


async def test_the_greeting_and_an_interrupted_answer_are_not_turns(call: Call) -> None:
    await call.at(50, LLMTextFrame("Welcome"))
    await call.at(100, AUDIO)
    await call.at(150, BotStartedSpeakingFrame())
    await call.at(1_000, VADUserStoppedSpeakingFrame())
    await call.at(1_400, AUDIO)
    await call.at(1_450, InterruptionFrame())
    await call.at(2_000, VADUserStoppedSpeakingFrame())
    await call.at(2_500, AUDIO)
    await call.at(2_600, BotStartedSpeakingFrame())
    (turn,) = call.turns
    assert turn["s2s.turn_latency_ms"] == 500.0


async def test_the_previous_answer_still_streaming_is_not_the_next_turns(call: Call) -> None:
    await call.at(1_000, VADUserStoppedSpeakingFrame())
    await call.at(1_400, LLMTextFrame("Sure,"))
    await call.at(1_600, AUDIO)
    await call.at(1_620, BotStartedSpeakingFrame())
    await call.at(1_900, _ttfb(0.1), source=TTS)
    await call.at(1_900, AUDIO)
    await call.at(4_200, VADUserStoppedSpeakingFrame())
    await call.at(4_700, _ttfb(0.4), source=LLM)
    await call.at(4_700, LLMTextFrame("Welcome"))
    await call.at(4_900, AUDIO)
    await call.at(4_920, BotStartedSpeakingFrame())
    _, second = call.turns
    assert second["s2s.turn_latency_ms"] == 700.0
    assert second["s2s.turn_latency.llm_ms"] == 400.0
    assert "s2s.turn_latency.tts_ms" not in second


async def test_the_stt_stage_ends_only_when_the_stt_confirms(call: Call) -> None:
    await call.at(1_000, VADUserStoppedSpeakingFrame())
    await call.at(1_040, _ttfb(0.0), source=STT)
    await call.at(2_000, VADUserStoppedSpeakingFrame())
    await call.at(2_020, _ttfb(0.0))
    await call.at(2_040, USAGE, source=STT)
    await call.at(2_070, _ttfb(0.0), source=STT)
    await call.at(2_600, LLMTextFrame("Hi"))
    await call.at(2_800, AUDIO)
    await call.at(2_820, BotStartedSpeakingFrame())
    (turn,) = call.turns
    assert turn["s2s.turn_latency.stt_confirmed"] is True
    assert turn["s2s.turn_latency.stt_ms"] == 70.0


async def test_smart_turns_silence_fallback_is_flagged(call: Call) -> None:
    await call.at(1_200, VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await call.at(4_000, UserStoppedSpeakingFrame())
    await call.at(4_600, LLMTextFrame("Hi"))
    await call.at(4_800, AUDIO)
    await call.at(4_820, BotStartedSpeakingFrame())
    (turn,) = call.turns
    assert turn["s2s.turn_latency.turn_detection_timed_out"] is True
    assert turn["s2s.turn_latency_ms"] == 3_800.0


async def test_an_stt_that_never_confirms_is_flagged_not_timed(call: Call) -> None:
    await call.at(1_200, VADUserStoppedSpeakingFrame())
    await call.at(1_600, LLMTextFrame("Hi"))
    await call.at(1_800, AUDIO)
    await call.at(1_820, BotStartedSpeakingFrame())
    (turn,) = call.turns
    assert turn["s2s.turn_latency.stt_confirmed"] is False
    assert "s2s.turn_latency.stt_ms" not in turn


async def test_covals_marker_is_recorded_on_the_turn_it_belongs_to(call: Call) -> None:
    await call.at(500, _marker("utterance_started", "u-1"))
    await call.at(520, InterruptionFrame())
    await call.at(1_000, _marker("utterance_end", "u-1"))
    await call.at(1_200, VADUserStoppedSpeakingFrame())
    await call.at(1_700, AUDIO)
    await call.at(1_720, BotStartedSpeakingFrame())
    await call.at(1_800, _marker("utterance_end", "u-1"))
    await call.at(6_100, VADUserStoppedSpeakingFrame())
    await call.at(6_600, AUDIO)
    await call.at(6_650, BotStartedSpeakingFrame())
    first, second = call.turns
    assert first["s2s.utterance_end.unix_ms"] == 1_000.0
    assert "s2s.utterance_end.unix_ms" not in second
