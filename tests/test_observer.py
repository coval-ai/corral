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
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterruptionFrame,
    LLMTextFrame,
    MetricsFrame,
    SpeechOutputAudioRawFrame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import STTUsage, STTUsageMetricsData, TTFBMetricsData
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from corral.observer import SPAN_NAME, TurnLatencyObserver

MS = 1_000_000
OPENAI = {"provider": "openai", "model": "gpt-4.1"}
STT, LLM, TTS = FrameProcessor(), FrameProcessor(), FrameProcessor()
AUDIO = TTSAudioRawFrame(audio=b"\x00\x00", sample_rate=24000, num_channels=1)
SILENCE = SpeechOutputAudioRawFrame(audio=b"\x00\x00" * 240, sample_rate=24000, num_channels=1)
SPEECH = SpeechOutputAudioRawFrame(audio=b"\xff\x3f" * 240, sample_rate=24000, num_channels=1)
USAGE = MetricsFrame(data=[STTUsageMetricsData(processor="x", value=STTUsage(audio_seconds=1))])


def _ttfb(seconds: float) -> MetricsFrame:
    return MetricsFrame(data=[TTFBMetricsData(processor="x", value=seconds)])


REALTIME = {"provider": "openai", "model": "gpt-realtime-2.1"}


class Call:
    def __init__(self, native: bool = False, endpoints: bool = True) -> None:
        self.exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.now_ns = 0
        if native:
            models = {"endpointing": REALTIME, "model": REALTIME, "model_second": REALTIME}
        else:
            models = {"llm": OPENAI, "llm_second": OPENAI}
        self.observer = TurnLatencyObserver(
            attributes={"agent": "x"},
            models=models,
            stt=None if native else STT,
            llm=LLM,
            tts=None if native else TTS,
            tracer=provider.get_tracer("test"),
            clock=lambda: self.now_ns,
            native=native,
            endpoints=endpoints,
        )

    async def at(
        self, ms: int, frame: Frame, source: FrameProcessor | None = None, upstream: bool = False
    ) -> None:
        self.now_ns = ms * MS
        source = source or FrameProcessor()
        direction = FrameDirection.UPSTREAM if upstream else FrameDirection.DOWNSTREAM
        await self.observer.on_push_frame(
            FramePushed(source, source, frame, direction, timestamp=0)
        )

    @property
    def turns(self) -> list[dict[str, Any]]:
        return self.observer.turns

    def stages(self) -> dict[str, ReadableSpan]:
        return {s.name: s for s in self.exporter.get_finished_spans() if s.name != SPAN_NAME}


@pytest.fixture
def call() -> Call:
    return Call()


@pytest.fixture
def native() -> Call:
    return Call(native=True)


@pytest.fixture
def full_duplex() -> Call:
    return Call(native=True, endpoints=False)


async def test_a_turn_times_each_provider_at_its_own_request(call: Call) -> None:
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
    assert turn["s2s.turn_latency.smart_turn_timed_out"] is False
    assert turn["s2s.turn_latency.stt_ms"] == 60.0
    assert turn["s2s.turn_latency.llm_ms"] == 250.0
    assert turn["s2s.turn_latency.tts_ms"] == 150.0
    assert turn["s2s.turn_latency.transport_ms"] == 20.0
    stages = call.stages()
    assert set(stages) == {"stt", "llm", "tts"}
    assert (stages["llm"].start_time, stages["llm"].end_time) == (1_350 * MS, 1_600 * MS)
    assert dict(stages["llm"].attributes or {}) == {"provider": "openai", "model": "gpt-4.1"}


async def test_the_pause_starts_at_silero_minus_its_window(call: Call) -> None:
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
    assert turn["s2s.turn_latency.stt_timed_out"] is False
    assert turn["s2s.turn_latency.stt_ms"] == 70.0


async def test_smart_turns_silence_fallback_is_flagged(call: Call) -> None:
    await call.at(1_200, VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await call.at(4_000, UserStoppedSpeakingFrame())
    await call.at(4_600, LLMTextFrame("Hi"))
    await call.at(4_800, AUDIO)
    await call.at(4_820, BotStartedSpeakingFrame())
    (turn,) = call.turns
    assert turn["s2s.turn_latency.smart_turn_timed_out"] is True
    assert turn["s2s.turn_latency_ms"] == 3_800.0


async def test_an_stt_that_never_answers_is_flagged_not_timed(call: Call) -> None:
    await call.at(1_200, VADUserStoppedSpeakingFrame())
    await call.at(1_600, LLMTextFrame("Hi"))
    await call.at(1_800, AUDIO)
    await call.at(1_820, BotStartedSpeakingFrame())
    (turn,) = call.turns
    assert turn["s2s.turn_latency.stt_timed_out"] is True
    assert turn["s2s.turn_latency.silero_missed"] is False
    assert "s2s.turn_latency.stt_ms" not in turn


async def test_a_turn_silero_never_heard_starts_at_smart_turns_stop(call: Call) -> None:
    await call.at(1_000, UserStoppedSpeakingFrame())
    await call.at(1_400, _ttfb(0.3), source=LLM)
    await call.at(1_400, LLMTextFrame("Hi"))
    await call.at(1_600, AUDIO)
    await call.at(1_620, BotStartedSpeakingFrame())
    (turn,) = call.turns
    assert turn["s2s.turn_latency_ms"] == 600.0
    assert turn["s2s.turn_latency.llm_ms"] == 300.0
    assert turn["s2s.turn_latency.smart_turn_timed_out"] is False
    assert turn["s2s.turn_latency.silero_missed"] is True
    assert turn["s2s.turn_latency.stt_timed_out"] is False
    assert "s2s.vad_stop.unix_ms" not in turn


def _in_progress(call_id: str, name: str, arguments: dict[str, Any]) -> FunctionCallInProgressFrame:
    return FunctionCallInProgressFrame(
        function_name=name, tool_call_id=call_id, arguments=arguments
    )


def _result(
    call_id: str, name: str, result: Any, error: str | None = None
) -> FunctionCallResultFrame:
    return FunctionCallResultFrame(
        function_name=name, tool_call_id=call_id, arguments={}, result=result, error=error
    )


async def test_a_tool_turn_splits_the_llm_around_the_call_and_exports_it(call: Call) -> None:
    await call.at(1_000, VADUserStoppedSpeakingFrame())
    await call.at(1_500, _ttfb(0.4), source=LLM)
    await call.at(1_600, _in_progress("c1", "find_patient", {"name": "Ana"}))
    await call.at(1_600, _in_progress("c1", "find_patient", {"name": "Ana"}), upstream=True)
    await call.at(1_900, _result("c1", "find_patient", {"patient_id": 123}))
    await call.at(1_900, _result("c1", "find_patient", {"patient_id": 123}), upstream=True)
    await call.at(2_400, _ttfb(0.45), source=LLM)
    await call.at(2_400, LLMTextFrame("Found"))
    await call.at(2_600, _ttfb(0.15), source=TTS)
    await call.at(2_600, AUDIO)
    await call.at(2_620, BotStartedSpeakingFrame())
    (turn,) = call.turns
    assert turn["s2s.turn.tool_calls"] == 1
    assert turn["s2s.turn_latency_ms"] == 1_600.0
    assert turn["s2s.turn_latency.llm_ms"] == 500.0
    assert turn["s2s.turn_latency.tool_ms"] == 300.0
    assert turn["s2s.turn_latency.llm_second_ms"] == 450.0
    assert turn["s2s.turn_latency.tts_ms"] == 150.0
    spans = {s.name: s for s in call.exporter.get_finished_spans()}
    assert set(spans) == {SPAN_NAME, "llm", "tool", "llm_second", "tts", "llm_tool_call"}
    tool = spans["llm_tool_call"]
    assert (tool.start_time, tool.end_time) == (1_600 * MS, 1_900 * MS)
    assert tool.parent is None
    assert dict(tool.attributes or {}) == {
        "function.name": "find_patient",
        "function.arguments": '{"name": "Ana"}',
        "function.result": '{"patient_id": 123}',
    }
    assert dict(spans["llm_second"].attributes or {}) == {"provider": "openai", "model": "gpt-4.1"}


async def test_a_tool_that_failed_says_so_on_its_span(call: Call) -> None:
    await call.at(1_000, VADUserStoppedSpeakingFrame())
    await call.at(1_400, _ttfb(0.3), source=LLM)
    await call.at(1_400, LLMTextFrame("One moment."))
    await call.at(1_600, _in_progress("c1", "book", {"date": "x"}))
    await call.at(1_700, _result("c1", "book", "Error", error="LookupError: no slot"))
    await call.at(2_000, _ttfb(0.25), source=LLM)
    await call.at(2_000, LLMTextFrame("Sorry"))
    await call.at(2_200, AUDIO)
    await call.at(2_220, BotStartedSpeakingFrame())
    (turn,) = call.turns
    assert turn["s2s.turn_latency.llm_ms"] == 300.0
    assert turn["s2s.turn_latency.tool_ms"] == 100.0
    assert turn["s2s.turn_latency.llm_second_ms"] == 250.0
    (tool,) = [s for s in call.exporter.get_finished_spans() if s.name == "llm_tool_call"]
    assert dict(tool.attributes or {})["function.error"] == "LookupError: no slot"


async def test_a_turn_without_tools_has_no_tool_stages(call: Call) -> None:
    await call.at(1_000, VADUserStoppedSpeakingFrame())
    await call.at(1_400, _ttfb(0.3), source=LLM)
    await call.at(1_400, LLMTextFrame("Hi"))
    await call.at(1_600, AUDIO)
    await call.at(1_620, BotStartedSpeakingFrame())
    (turn,) = call.turns
    assert turn["s2s.turn.tool_calls"] == 0
    assert "s2s.turn_latency.tool_ms" not in turn and "s2s.turn_latency.llm_second_ms" not in turn


async def test_a_native_turn_splits_at_the_moment_the_vendor_heard_the_caller_stop(
    native: Call,
) -> None:
    await native.at(1_200, VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await native.at(1_350, UserStoppedSpeakingFrame())
    await native.at(1_900, _ttfb(0.55), source=LLM)
    await native.at(1_900, AUDIO)
    await native.at(1_905, LLMTextFrame("Hi"))
    await native.at(1_920, BotStartedSpeakingFrame())
    (turn,) = native.turns
    assert turn["s2s.turn_latency_ms"] == 900.0
    assert turn["s2s.turn_latency.endpointing_ms"] == 350.0
    assert turn["s2s.turn_latency.model_ms"] == 550.0
    assert turn["s2s.turn_latency.transport_ms"] == 20.0
    assert turn["s2s.turn_latency.silero_missed"] is False
    assert "s2s.turn_latency.stt_timed_out" not in turn
    assert "s2s.turn_latency.smart_turn_timed_out" not in turn
    assert not {k for k in turn if k.endswith(("stt_ms", "llm_ms", "tts_ms"))}
    stages = native.stages()
    assert set(stages) == {"endpointing", "model"}
    assert (stages["model"].start_time, stages["model"].end_time) == (1_350 * MS, 1_900 * MS)
    assert dict(stages["endpointing"].attributes or {}) == REALTIME


async def test_a_model_that_jumps_in_before_silero_has_negative_endpointing(native: Call) -> None:
    await native.at(900, UserStoppedSpeakingFrame())
    await native.at(1_200, VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await native.at(1_400, AUDIO)
    await native.at(1_420, BotStartedSpeakingFrame())
    (turn,) = native.turns
    assert turn["s2s.turn_latency_ms"] == 400.0
    assert turn["s2s.turn_latency.endpointing_ms"] == -100.0
    assert turn["s2s.turn_latency.model_ms"] == 500.0


async def test_a_vendor_that_never_says_gets_model_from_the_last_word(native: Call) -> None:
    await native.at(1_200, VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await native.at(1_800, AUDIO)
    await native.at(1_810, BotStartedSpeakingFrame())
    (turn,) = native.turns
    assert turn["s2s.turn_latency_ms"] == 800.0
    assert turn["s2s.turn_latency.model_ms"] == 800.0
    assert "s2s.turn_latency.endpointing_ms" not in turn
    assert set(native.stages()) == {"model"}


async def test_a_native_tool_turn_splits_the_model_around_the_call(native: Call) -> None:
    await native.at(1_000, VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await native.at(1_100, UserStoppedSpeakingFrame())
    await native.at(1_500, _in_progress("c1", "find_patient", {"name": "Ana"}))
    await native.at(1_800, _result("c1", "find_patient", {"patient_id": 123}))
    await native.at(2_300, _ttfb(0.45), source=LLM)
    await native.at(2_300, AUDIO)
    await native.at(2_310, LLMTextFrame("Found"))
    await native.at(2_320, BotStartedSpeakingFrame())
    (turn,) = native.turns
    assert turn["s2s.turn.tool_calls"] == 1
    assert turn["s2s.turn_latency_ms"] == 1_500.0
    assert turn["s2s.turn_latency.endpointing_ms"] == 300.0
    assert turn["s2s.turn_latency.model_ms"] == 400.0
    assert turn["s2s.turn_latency.tool_ms"] == 300.0
    assert turn["s2s.turn_latency.model_second_ms"] == 460.0
    spans = {s.name for s in native.exporter.get_finished_spans()}
    assert spans == {SPAN_NAME, "endpointing", "model", "tool", "model_second", "llm_tool_call"}


async def test_a_tool_called_after_the_agent_starts_answering_is_still_a_span(
    native: Call,
) -> None:
    await native.at(1_000, VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await native.at(1_300, UserStoppedSpeakingFrame())
    await native.at(1_900, AUDIO)
    await native.at(1_920, BotStartedSpeakingFrame())
    await native.at(4_000, _in_progress("c1", "book_appointment", {"date": "x"}))
    await native.at(4_300, _result("c1", "book_appointment", {"ok": True}))
    await native.at(4_900, AUDIO)
    await native.at(4_920, BotStartedSpeakingFrame())
    (turn,) = native.turns
    assert turn["s2s.turn.tool_calls"] == 0
    (tool,) = [s for s in native.exporter.get_finished_spans() if s.name == "llm_tool_call"]
    assert (tool.start_time, tool.end_time) == (4_000 * MS, 4_300 * MS)
    assert dict(tool.attributes or {})["function.name"] == "book_appointment"


async def test_a_full_duplex_turn_runs_from_the_last_word_to_the_first_sound(
    full_duplex: Call,
) -> None:
    await full_duplex.at(1_000, SILENCE)
    await full_duplex.at(1_200, VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await full_duplex.at(1_400, UserStoppedSpeakingFrame())
    await full_duplex.at(1_500, SILENCE)
    await full_duplex.at(1_700, SPEECH)
    await full_duplex.at(1_720, BotStartedSpeakingFrame())
    (turn,) = full_duplex.turns
    assert turn["s2s.turn_latency_ms"] == 700.0
    assert turn["s2s.turn_latency.model_ms"] == 700.0
    assert turn["s2s.turn_latency.transport_ms"] == 20.0
    assert "s2s.turn_latency.endpointing_ms" not in turn
    assert set(full_duplex.stages()) == {"model"}


async def test_a_full_duplex_turn_silero_missed_is_not_counted(full_duplex: Call) -> None:
    await full_duplex.at(1_000, UserStoppedSpeakingFrame())
    await full_duplex.at(1_500, SPEECH)
    await full_duplex.at(1_520, BotStartedSpeakingFrame())
    assert full_duplex.turns == []
