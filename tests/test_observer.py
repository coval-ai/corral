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
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from harness.observer import SPAN_NAME, TurnLatencyObserver

MS = 1_000_000


def _audio() -> TTSAudioRawFrame:
    return TTSAudioRawFrame(audio=b"\x00\x00", sample_rate=24000, num_channels=1)


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
def exported() -> Iterator[tuple[TurnLatencyObserver, _Clock, InMemorySpanExporter]]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    clock = _Clock()
    observer = TurnLatencyObserver(
        attributes={"stack": "x"}, tracer=provider.get_tracer("test"), clock=clock
    )
    yield observer, clock, exporter
    provider.shutdown()


@pytest.mark.asyncio
async def test_one_span_per_answered_turn_with_vad_as_t0(
    exported: tuple[TurnLatencyObserver, _Clock, InMemorySpanExporter],
) -> None:
    observer, clock, exporter = exported
    await _push(observer, clock, 1_000, UserStoppedSpeakingFrame())
    await _push(observer, clock, 1_600, _audio())
    await _push(observer, clock, 1_650, _audio())
    await _push(observer, clock, 1_700, BotStartedSpeakingFrame())
    (span,) = exporter.get_finished_spans()
    assert span.name == SPAN_NAME
    assert span.attributes is not None
    assert span.attributes["stack"] == "x"
    assert span.attributes["s2s.turn.number"] == 1
    assert span.attributes["s2s.t0.source"] == "vad"
    assert span.attributes["s2s.provider_latency_ms"] == 600.0
    assert span.attributes["s2s.harness_overhead_ms"] == 100.0
    assert (span.start_time, span.end_time) == (1_000 * MS, 1_700 * MS)


@pytest.mark.asyncio
async def test_covals_marker_wins_over_vad_when_both_arrive(
    exported: tuple[TurnLatencyObserver, _Clock, InMemorySpanExporter],
) -> None:
    observer, clock, exporter = exported
    await _push(observer, clock, 900, InputTransportMessageFrame(message={"type": "utterance_end"}))
    await _push(observer, clock, 1_100, UserStoppedSpeakingFrame())
    await _push(observer, clock, 1_500, _audio())
    await _push(observer, clock, 1_550, BotStartedSpeakingFrame())
    (span,) = exporter.get_finished_spans()
    assert span.attributes is not None
    assert span.attributes["s2s.t0.source"] == "utterance_end"
    assert span.attributes["s2s.provider_latency_ms"] == 600.0


@pytest.mark.asyncio
async def test_the_greeting_and_an_interrupted_answer_produce_no_span(
    exported: tuple[TurnLatencyObserver, _Clock, InMemorySpanExporter],
) -> None:
    observer, clock, exporter = exported
    await _push(observer, clock, 100, _audio())
    await _push(observer, clock, 150, BotStartedSpeakingFrame())
    await _push(observer, clock, 1_000, UserStoppedSpeakingFrame())
    await _push(observer, clock, 1_400, _audio())
    await _push(observer, clock, 1_450, InterruptionFrame())
    await _push(observer, clock, 2_000, UserStoppedSpeakingFrame())
    await _push(observer, clock, 2_500, _audio())
    await _push(observer, clock, 2_600, BotStartedSpeakingFrame())
    (span,) = exporter.get_finished_spans()
    assert span.attributes is not None
    assert span.attributes["s2s.turn.number"] == 1
    assert span.attributes["s2s.provider_latency_ms"] == 500.0
