# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

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
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from corral.observer import SPAN_NAME, TurnLatencyObserver

MS = 1_000_000
Exported = tuple[TurnLatencyObserver, "_Clock", InMemorySpanExporter]


def _audio() -> TTSAudioRawFrame:
    return TTSAudioRawFrame(audio=b"\x00\x00", sample_rate=24000, num_channels=1)


def _words(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="caller", timestamp="")


def _marker(utterance_id: str | None = None) -> InputTransportMessageFrame:
    message: dict[str, str] = {"type": "utterance_end"}
    if utterance_id:
        message["utterance_id"] = utterance_id
    return InputTransportMessageFrame(message=message)


def _started(utterance_id: str) -> InputTransportMessageFrame:
    return InputTransportMessageFrame(
        message={"type": "utterance_started", "utterance_id": utterance_id}
    )


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


def _turn(exporter: InMemorySpanExporter) -> ReadableSpan:
    """The one turn span; the stage spans end first so they come out ahead of it."""
    (turn,) = [s for s in exporter.get_finished_spans() if s.name == SPAN_NAME]
    return turn


def _children(exporter: InMemorySpanExporter) -> dict[str, ReadableSpan]:
    return {s.name: s for s in exporter.get_finished_spans() if s.name != SPAN_NAME}


@pytest.fixture
def exported() -> Iterator[Exported]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    clock = _Clock()
    observer = TurnLatencyObserver(
        attributes={"agent": "x"},
        models={"llm": {"provider": "openai", "model": "gpt-4.1"}},
        tracer=provider.get_tracer("test"),
        clock=clock,
    )
    yield observer, clock, exporter
    provider.shutdown()


async def test_a_full_turn_breaks_the_pause_into_its_steps(exported: Exported) -> None:
    observer, clock, exporter = exported
    await _push(observer, clock, 1_000, _marker())
    await _push(observer, clock, 1_200, VADUserStoppedSpeakingFrame())
    await _push(observer, clock, 1_300, _words("hello there"))
    await _push(observer, clock, 1_350, UserStoppedSpeakingFrame())
    await _push(observer, clock, 1_600, LLMTextFrame("Hi"))
    await _push(observer, clock, 1_650, LLMTextFrame(" there"))
    await _push(observer, clock, 1_800, _audio())
    await _push(observer, clock, 1_810, _audio())
    await _push(observer, clock, 1_820, BotStartedSpeakingFrame())
    span = _turn(exporter)
    assert span.name == SPAN_NAME
    assert span.attributes is not None
    a = dict(span.attributes)
    assert a["agent"] == "x"
    assert a["s2s.turn.number"] == 1
    assert a["s2s.latency.source"] == "utterance_end"
    assert a["s2s.latency_ms"] == 800.0
    assert a["s2s.endpointing_ms"] == 200.0
    assert a["s2s.stt_ms"] == 100.0
    assert a["s2s.llm_ms"] == 250.0
    assert a["s2s.tts_ms"] == 200.0
    assert a["s2s.transport_ms"] == 20.0
    assert a["s2s.utterance_end.unix_ms"] == 1_000.0
    assert a["s2s.llm_first_token.unix_ms"] == 1_600.0
    assert a["s2s.audio_start.unix_ms"] == 1_820.0
    assert (span.start_time, span.end_time) == (1_000 * MS, 1_820 * MS)
    assert observer.turns == [a]


async def test_without_covals_marker_the_pause_starts_at_vad_done(exported: Exported) -> None:
    observer, clock, exporter = exported
    await _push(observer, clock, 1_000, UserStoppedSpeakingFrame())
    await _push(observer, clock, 1_600, _audio())
    await _push(observer, clock, 1_700, BotStartedSpeakingFrame())
    span = _turn(exporter)
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
    span = _turn(exporter)
    assert span.attributes is not None
    a = dict(span.attributes)
    assert a["s2s.turn.number"] == 1
    assert a["s2s.latency.source"] == "vad_done"
    assert a["s2s.latency_ms"] == 500.0


async def test_each_stage_is_a_child_span_with_its_provider_and_model(exported: Exported) -> None:
    observer, clock, exporter = exported
    await _push(observer, clock, 1_000, _marker())
    await _push(observer, clock, 1_200, VADUserStoppedSpeakingFrame())
    await _push(observer, clock, 1_300, _words("hello there"))
    await _push(observer, clock, 1_350, UserStoppedSpeakingFrame())
    await _push(observer, clock, 1_600, LLMTextFrame("Hi"))
    await _push(observer, clock, 1_800, _audio())
    await _push(observer, clock, 1_820, BotStartedSpeakingFrame())
    turn = _turn(exporter)
    stages = _children(exporter)
    assert set(stages) == {"stt", "llm", "tts"}
    for stage in stages.values():
        assert stage.parent is not None and stage.parent.span_id == turn.context.span_id
    assert (stages["stt"].start_time, stages["stt"].end_time) == (1_200 * MS, 1_300 * MS)
    assert (stages["llm"].start_time, stages["llm"].end_time) == (1_350 * MS, 1_600 * MS)
    assert (stages["tts"].start_time, stages["tts"].end_time) == (1_600 * MS, 1_800 * MS)
    assert dict(stages["llm"].attributes or {}) == {"provider": "openai", "model": "gpt-4.1"}
    assert not stages["stt"].attributes


async def test_a_transcript_before_silero_fired_gets_no_stt_span(exported: Exported) -> None:
    observer, clock, exporter = exported
    await _push(observer, clock, 1_000, _marker())
    await _push(observer, clock, 1_150, _words("hello"))
    await _push(observer, clock, 1_200, VADUserStoppedSpeakingFrame())
    await _push(observer, clock, 1_250, UserStoppedSpeakingFrame())
    await _push(observer, clock, 1_500, LLMTextFrame("Hi"))
    await _push(observer, clock, 1_700, _audio())
    await _push(observer, clock, 1_720, BotStartedSpeakingFrame())
    assert set(_children(exporter)) == {"llm", "tts"}
    assert dict(_turn(exporter).attributes or {})["s2s.stt_ms"] == -50.0


async def test_a_marker_that_arrives_after_we_answered_does_not_start_the_next_turn(
    exported: Exported,
) -> None:
    observer, clock, exporter = exported
    await _push(observer, clock, 900, _started("u-1"))
    await _push(observer, clock, 1_000, UserStoppedSpeakingFrame())
    await _push(observer, clock, 1_500, _audio())
    await _push(observer, clock, 1_550, BotStartedSpeakingFrame())
    await _push(observer, clock, 1_600, _marker("u-1"))
    await _push(observer, clock, 5_000, _started("u-2"))
    await _push(observer, clock, 6_000, _marker("u-2"))
    await _push(observer, clock, 6_100, UserStoppedSpeakingFrame())
    await _push(observer, clock, 6_600, _audio())
    await _push(observer, clock, 6_650, BotStartedSpeakingFrame())
    first, second = (s for s in exporter.get_finished_spans() if s.name == SPAN_NAME)
    assert dict(first.attributes or {})["s2s.latency.source"] == "vad_done"
    a = dict(second.attributes or {})
    assert a["s2s.latency.source"] == "utterance_end"
    assert a["s2s.latency_ms"] == 600.0


async def test_the_previous_answer_still_streaming_does_not_eat_the_next_transcript(
    exported: Exported,
) -> None:
    observer, clock, exporter = exported
    await _push(observer, clock, 1_000, UserStoppedSpeakingFrame())
    await _push(observer, clock, 1_400, LLMTextFrame("Sure,"))
    await _push(observer, clock, 1_600, _audio())
    await _push(observer, clock, 1_620, BotStartedSpeakingFrame())
    await _push(observer, clock, 1_700, LLMTextFrame(" here is more."))
    await _push(observer, clock, 1_900, _audio())
    await _push(observer, clock, 4_000, _words("thanks"))
    await _push(observer, clock, 4_200, UserStoppedSpeakingFrame())
    await _push(observer, clock, 4_700, LLMTextFrame("Welcome"))
    await _push(observer, clock, 4_900, _audio())
    await _push(observer, clock, 4_920, BotStartedSpeakingFrame())
    _, second = (s for s in exporter.get_finished_spans() if s.name == SPAN_NAME)
    a = dict(second.attributes or {})
    assert a["s2s.stt_done.unix_ms"] == 4_000.0
    assert a["s2s.llm_ms"] == 500.0
    assert a["s2s.latency_ms"] == 700.0


async def test_locally_the_pause_starts_when_silero_says_you_stopped(exported: Exported) -> None:
    observer, clock, exporter = exported
    await _push(observer, clock, 1_200, VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await _push(observer, clock, 1_324, _words("hello there"))
    await _push(observer, clock, 1_330, UserStoppedSpeakingFrame())
    await _push(observer, clock, 1_860, LLMTextFrame("Hi"))
    await _push(observer, clock, 2_121, _audio())
    await _push(observer, clock, 2_122, BotStartedSpeakingFrame())
    span = _turn(exporter)
    a = dict(span.attributes or {})
    assert a["s2s.latency.source"] == "vad_stop"
    assert a["s2s.latency_ms"] == 1_121.0
    assert a["s2s.stt_ms"] == 124.0
    assert "s2s.endpointing_ms" not in a
    assert span.start_time == 1_000 * MS
