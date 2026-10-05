# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    Frame,
    InputTransportMessageFrame,
    InterruptionFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from agent.observer import SPAN_NAME, TurnLatencyObserver

MS = 1_000_000
Exported = tuple[TurnLatencyObserver, "_Clock", InMemorySpanExporter]


def _audio() -> TTSAudioRawFrame:
    return TTSAudioRawFrame(audio=b"\x00\x00", sample_rate=24000, num_channels=1)


def _words(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="caller", timestamp="")


def _marker() -> InputTransportMessageFrame:
    return InputTransportMessageFrame(message={"type": "utterance_end"})


class _Clock:
    def __init__(self) -> None:
        self.now_ns = 1_000 * MS

    def __call__(self) -> int:
        return self.now_ns


async def _push(observer: TurnLatencyObserver, clock: _Clock, at_ms: int, frame: Frame) -> None:
    clock.now_ns = at_ms * MS
    source = FrameProcessor()
    await observer.on_push_frame(
        FramePushed(source, source, frame, FrameDirection.DOWNSTREAM, timestamp=0)
    )


@pytest.fixture
def exported() -> Iterator[Exported]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    clock = _Clock()
    observer = TurnLatencyObserver(
        attributes={"stack": "x"}, tracer=provider.get_tracer("test"), clock=clock
    )
    yield observer, clock, exporter
    provider.shutdown()


async def test_a_full_turn_breaks_the_pause_into_its_steps(exported: Exported) -> None:
    observer, clock, exporter = exported
    await _push(observer, clock, 1_000, _marker())
    await _push(observer, clock, 1_150, _words("hello"))
    await _push(observer, clock, 1_250, UserStoppedSpeakingFrame())
    await _push(observer, clock, 1_300, _words("hello there"))
    await _push(observer, clock, 1_600, LLMTextFrame("Hi"))
    await _push(observer, clock, 1_650, LLMTextFrame(" there"))
    await _push(observer, clock, 1_800, _audio())
    await _push(observer, clock, 1_810, _audio())
    await _push(observer, clock, 1_820, BotStartedSpeakingFrame())
    (span,) = exporter.get_finished_spans()
    assert span.name == SPAN_NAME
    assert span.attributes is not None
    a = dict(span.attributes)
    assert a["stack"] == "x"
    assert a["s2s.turn.number"] == 1
    assert a["s2s.latency.source"] == "utterance_end"
    assert a["s2s.latency_ms"] == 800.0
    assert a["s2s.endpointing_ms"] == 250.0
    assert a["s2s.stt_ms"] == 300.0  # the second transcript, the one the model read
    assert a["s2s.llm_ms"] == 300.0  # counted from stt_done since it came after vad_done
    assert a["s2s.tts_ms"] == 200.0
    assert a["s2s.transport_ms"] == 20.0
    assert a["s2s.utterance_end.unix_ms"] == 1_000.0
    assert a["s2s.llm_first_token.unix_ms"] == 1_600.0
    assert a["s2s.audio_start.unix_ms"] == 1_820.0
    assert (span.start_time, span.end_time) == (1_000 * MS, 1_820 * MS)


async def test_without_covals_marker_the_pause_starts_at_vad_done(exported: Exported) -> None:
    observer, clock, exporter = exported
    await _push(observer, clock, 1_000, UserStoppedSpeakingFrame())
    await _push(observer, clock, 1_600, _audio())
    await _push(observer, clock, 1_700, BotStartedSpeakingFrame())
    (span,) = exporter.get_finished_spans()
    assert span.attributes is not None
    a = dict(span.attributes)
    assert a["s2s.latency.source"] == "vad_done"
    assert a["s2s.latency_ms"] == 600.0
    assert a["s2s.transport_ms"] == 100.0
    assert "s2s.endpointing_ms" not in a
    assert "s2s.stt_ms" not in a
    assert "s2s.llm_ms" not in a
    assert span.start_time == 1_000 * MS


async def test_the_greeting_and_an_interrupted_answer_produce_no_span(
    exported: Exported,
) -> None:
    observer, clock, exporter = exported
    await _push(observer, clock, 50, LLMTextFrame("Welcome"))
    await _push(observer, clock, 100, _audio())
    await _push(observer, clock, 150, BotStartedSpeakingFrame())
    await _push(observer, clock, 900, _marker())
    await _push(observer, clock, 1_000, UserStoppedSpeakingFrame())
    await _push(observer, clock, 1_400, _audio())
    await _push(observer, clock, 1_450, InterruptionFrame())
    await _push(observer, clock, 2_000, UserStoppedSpeakingFrame())
    await _push(observer, clock, 2_500, _audio())
    await _push(observer, clock, 2_600, BotStartedSpeakingFrame())
    (span,) = exporter.get_finished_spans()
    assert span.attributes is not None
    a = dict(span.attributes)
    assert a["s2s.turn.number"] == 1
    assert a["s2s.latency.source"] == "vad_done"  # the interruption cleared the old marker
    assert a["s2s.latency_ms"] == 500.0
