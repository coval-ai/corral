# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The call's own transcript, one ``stt`` span per utterance, sent to Coval as it happens.

Coval transcribes the recording after the call, and when the caller talks over the agent it
can merge the agent's cut-off answer and its restart into one line. Corral knows what each
side said and when, so it sends that instead: each utterance is a span from its first sound
to its last, its text in ``transcript`` as ``[agent] …`` or ``[caller] …``. Coval's judges
show ``transcript`` on spans named ``stt``, in time order.

- The agent's utterance is one answer: from the first audio we play to the last, its text the
  model's own transcript of what it said. An answer the caller cuts off keeps only the share
  of its words that played, and says so.
- The caller's utterance runs from Silero's start to its stop, each pushed back by Silero's
  window, its text the model's transcription of what it heard.

Spans go out with Corral's other spans, so they reach Coval while the call runs; Coval holds
its metrics for a while once a call has spans, and the last ones are flushed at hang-up.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable

from opentelemetry.trace import Tracer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSTextFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.transports.base_output import BaseOutputTransport

SPAN_NAME = "stt"
NO_TEXT = "(no transcript)"
CUT_OFF = "[cut off]"


@dataclasses.dataclass
class Answer:
    """The agent's answer so far: what it said, what it generated and what has played."""

    text: str = ""
    start: int | None = None
    end: int | None = None
    speaking_since: int | None = None
    played_ns: int = 0
    generated_ns: int = 0

    def stop(self, now: int) -> None:
        if self.speaking_since is not None:
            self.played_ns += now - self.speaking_since
            self.end = now
            self.speaking_since = None

    def heard(self, interrupted: bool) -> tuple[str, bool]:
        """The words that played; when the caller cut it off, only their share of the text."""
        words = self.text.split()
        if not interrupted or not words or self.played_ns >= self.generated_ns:
            return " ".join(words), False
        kept = max(1, round(len(words) * self.played_ns / self.generated_ns))
        return " ".join(words[:kept]), True


@dataclasses.dataclass
class Utterance:
    role: str
    start: int
    end: int
    text: str | None = None
    cut_off: bool = False


class TranscriptRecorder(BaseObserver):
    def __init__(self, tracer: Tracer, clock: Callable[[], int] = time.time_ns) -> None:
        super().__init__()
        self._tracer = tracer
        self._clock = clock
        self._answer = Answer()
        self._caller_since: int | None = None
        self._unheard: list[Utterance] = []
        self._early_text: str | None = None
        self.utterances: list[Utterance] = []

    async def on_push_frame(self, data: FramePushed) -> None:
        if not data.first_push:
            return
        now = self._clock()
        answer = self._answer
        match data.frame:
            # A second request while the first answer still plays, as around a tool call,
            # is the same utterance to the caller.
            case LLMFullResponseStartFrame() if answer.speaking_since is None:
                self._close_answer(interrupted=False)
            case TTSTextFrame(text=text):
                answer.text += text if data.frame.includes_inter_frame_spaces else f" {text}"
            # The output transport pushes the audio on again as it plays; count it once.
            case TTSAudioRawFrame(audio=audio, sample_rate=rate, num_channels=channels) if (
                not isinstance(data.source, BaseOutputTransport)
            ):
                answer.generated_ns += int(len(audio) / (2 * channels * rate) * 1e9)
            case BotStartedSpeakingFrame():
                if answer.start is None:
                    answer.start = now
                answer.speaking_since = now
            case BotStoppedSpeakingFrame():
                answer.stop(now)
            # Pipecat interrupts at the start of every caller turn; the answer was only cut
            # off if it was still playing.
            case InterruptionFrame():
                playing = answer.speaking_since is not None
                answer.stop(now)
                self._close_answer(interrupted=playing)
            case VADUserStartedSpeakingFrame(start_secs=secs):
                self._caller_since = now - int(secs * 1e9)
            case VADUserStoppedSpeakingFrame(stop_secs=secs) if self._caller_since is not None:
                caller = Utterance("caller", self._caller_since, now - int(secs * 1e9))
                self._caller_since = None
                self._unheard.append(caller)
                if self._early_text is not None:
                    self._hear(self._early_text)
                    self._early_text = None
            case TranscriptionFrame(text=text):
                if self._unheard:
                    self._hear(text)
                elif self._caller_since is not None:
                    self._early_text = f"{self._early_text or ''} {text}".strip()
                else:
                    self._send(Utterance("caller", now, now, text.strip() or None))

    def _hear(self, text: str) -> None:
        """The model transcribes only what it took for a turn, so its text goes to the caller's
        latest line; earlier lines it skipped, like a short "Mm-hm", go out without text."""
        *skipped, caller = self._unheard
        self._unheard = []
        for line in skipped:
            self._send(line)
        caller.text = text.strip() or None
        self._send(caller)

    def finish(self) -> None:
        """At hang-up: the answer still playing, the caller still talking and any caller line
        the model never transcribed."""
        now = self._clock()
        self._answer.stop(now)
        self._close_answer(interrupted=False)
        if self._caller_since is not None:
            self._unheard.append(Utterance("caller", self._caller_since, now))
            self._caller_since = None
        if self._early_text is not None and self._unheard:
            self._hear(self._early_text)
        self._early_text = None
        for caller in self._unheard:
            self._send(caller)
        self._unheard = []

    def _close_answer(self, *, interrupted: bool) -> None:
        answer, self._answer = self._answer, Answer()
        if answer.start is None or answer.end is None:
            return
        text, cut_off = answer.heard(interrupted)
        self._send(Utterance("agent", answer.start, answer.end, text or None, cut_off))

    def _send(self, utterance: Utterance) -> None:
        self.utterances.append(utterance)
        text = utterance.text or NO_TEXT
        if utterance.cut_off:
            text = f"{text} {CUT_OFF}"
        span = self._tracer.start_span(
            SPAN_NAME,
            start_time=utterance.start,
            attributes={"transcript": f"[{utterance.role}] {text}"},
        )
        span.end(end_time=utterance.end)
