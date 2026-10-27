# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSTextFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.text.base_text_aggregator import TextType

from corral.transcript import SPAN_NAME, TranscriptRecorder

MS = 1_000_000
RATE = 24000


def _audio(seconds: float) -> TTSAudioRawFrame:
    return TTSAudioRawFrame(
        audio=b"\x00\x00" * int(RATE * seconds), sample_rate=RATE, num_channels=1
    )


def _text(text: str) -> TTSTextFrame:
    frame = TTSTextFrame(text, text_type=TextType.SENTENCE)
    frame.includes_inter_frame_spaces = True
    return frame


class Call:
    def __init__(self) -> None:
        self.exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.now_ns = 0
        self.recorder = TranscriptRecorder(provider.get_tracer("test"), clock=lambda: self.now_ns)

    async def at(self, ms: int, frame: Frame) -> None:
        self.now_ns = ms * MS
        source = FrameProcessor()
        await self.recorder.on_push_frame(
            FramePushed(source, source, frame, FrameDirection.DOWNSTREAM, timestamp=0)
        )

    async def answer(self, start_ms: int, text: str, seconds: float) -> None:
        await self.at(start_ms, LLMFullResponseStartFrame())
        await self.at(start_ms, _text(text))
        await self.at(start_ms, _audio(seconds))
        await self.at(start_ms, BotStartedSpeakingFrame())

    async def caller(self, start_ms: int, stop_ms: int, text: str | None) -> None:
        await self.at(start_ms + 200, VADUserStartedSpeakingFrame(start_secs=0.2))
        await self.at(stop_ms + 300, VADUserStoppedSpeakingFrame(stop_secs=0.3))
        if text is not None:
            await self.at(stop_ms + 400, TranscriptionFrame(text, "", ""))

    def lines(self) -> list[tuple[int, int, str]]:
        spans = sorted(self.exporter.get_finished_spans(), key=lambda s: s.start_time or 0)
        assert {s.name for s in spans} <= {SPAN_NAME}
        return [
            (
                (s.start_time or 0) // MS,
                (s.end_time or 0) // MS,
                str((s.attributes or {})["transcript"]),
            )
            for s in spans
        ]


@pytest.fixture
def call() -> Call:
    return Call()


async def test_a_backchannel_the_agent_talks_through_keeps_one_agent_line(call: Call) -> None:
    await call.answer(1_000, "We're open Monday through Friday, 9AM to 6PM.", 4.0)
    await call.caller(2_000, 2_600, "Mm-hm.")
    await call.at(5_000, BotStoppedSpeakingFrame())
    await call.answer(8_000, "Anything else?", 1.0)
    assert call.lines() == [
        (1_000, 5_000, "[agent] We're open Monday through Friday, 9AM to 6PM."),
        (2_000, 2_600, "[caller] Mm-hm."),
    ]


async def test_a_cut_off_answer_keeps_only_the_words_that_played(call: Call) -> None:
    await call.answer(1_000, "Our hours are Monday through Friday, nine to six.", 4.0)
    await call.caller(2_800, 3_400, "Mm-hm.")
    await call.at(3_000, InterruptionFrame())
    await call.answer(5_000, "We're open Monday through Friday from 9AM to 6PM.", 3.0)
    await call.at(8_000, BotStoppedSpeakingFrame())
    call.recorder.finish()
    assert call.lines() == [
        (1_000, 3_000, "[agent] Our hours are Monday [cut off]"),
        (2_800, 3_400, "[caller] Mm-hm."),
        (5_000, 8_000, "[agent] We're open Monday through Friday from 9AM to 6PM."),
    ]


async def test_pauses_inside_one_answer_stay_one_line(call: Call) -> None:
    await call.answer(1_000, "Hi. Ask me anything.", 2.0)
    await call.at(1_800, BotStoppedSpeakingFrame())
    await call.at(2_300, BotStartedSpeakingFrame())
    await call.at(3_500, BotStoppedSpeakingFrame())
    call.recorder.finish()
    assert call.lines() == [(1_000, 3_500, "[agent] Hi. Ask me anything.")]


async def test_the_callers_line_is_pushed_back_by_silero_windows(call: Call) -> None:
    await call.caller(4_000, 5_200, "Hi, what are your hours?")
    assert call.lines() == [(4_000, 5_200, "[caller] Hi, what are your hours?")]


async def test_hang_up_sends_what_is_still_open(call: Call) -> None:
    await call.caller(1_000, 1_500, None)
    await call.answer(2_000, "Goodbye.", 1.0)
    await call.at(2_600, BotStoppedSpeakingFrame())
    call.recorder.finish()
    assert call.lines() == [
        (1_000, 1_500, "[caller] (no transcript)"),
        (2_000, 2_600, "[agent] Goodbye."),
    ]


async def test_an_answer_that_never_played_is_not_a_line(call: Call) -> None:
    await call.at(1_000, LLMFullResponseStartFrame())
    await call.at(1_000, _text("Never heard."))
    await call.at(1_100, InterruptionFrame())
    call.recorder.finish()
    assert call.lines() == []


async def test_a_line_the_model_skipped_does_not_take_the_next_lines_text(call: Call) -> None:
    await call.caller(1_000, 1_400, None)
    await call.caller(3_000, 3_800, "Thanks, bye.")
    assert call.lines() == [
        (1_000, 1_400, "[caller] (no transcript)"),
        (3_000, 3_800, "[caller] Thanks, bye."),
    ]


async def test_text_that_arrives_before_silero_hears_the_stop_is_kept(call: Call) -> None:
    await call.at(1_200, VADUserStartedSpeakingFrame(start_secs=0.2))
    await call.at(1_900, TranscriptionFrame("Where are you located?", "", ""))
    await call.at(2_300, VADUserStoppedSpeakingFrame(stop_secs=0.3))
    assert call.lines() == [(1_000, 2_000, "[caller] Where are you located?")]
