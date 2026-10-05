# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Times each turn of the call and sends one ``s2s.turn`` span per answered turn.

We grab the time at six moments:

- ``utterance_end``: Coval tells us the caller finished talking
- ``vad_done``: our VAD and smart-turn agree the caller is done
- ``stt_done``: the last transcript the model actually got to read
- ``llm_first_token``: the model says its first word
- ``tts_first_audio``: we have the first bit of synthesized audio
- ``audio_start``: we send the first audio chunk back to Coval

The span runs from when the caller stopped to ``audio_start``. "When the caller
stopped" is ``utterance_end`` if Coval sent it, otherwise ``vad_done``, and
``s2s.latency.source`` tells you which one you got. Every moment we saw goes on
the span as ``s2s.<moment>.unix_ms``, and the time between them as
``s2s.<step>_ms``: ``latency`` (the headline number, caller stopped to first
synthesized audio), ``endpointing``, ``stt``, ``llm``, ``tts`` and ``transport``.
Spans land in Coval under the call's simulation id.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable
from typing import Any

import structlog
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Tracer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed

logger = structlog.get_logger("agent")

SPAN_NAME = "s2s.turn"
SERVICE_NAME = "coval-bench-s2s-agent"
EXPORT_TIMEOUT_SECS = 30
UTTERANCE_END = "utterance_end"


def _ms(later: int, earlier: int) -> float:
    return (later - earlier) / 1e6


@dataclasses.dataclass
class Moments:
    """When each moment happened this turn, in unix ns. None until it does."""

    utterance_end: int | None = None
    vad_done: int | None = None
    stt_done: int | None = None
    llm_first_token: int | None = None
    tts_first_audio: int | None = None
    audio_start: int | None = None

    @property
    def caller_stopped(self) -> int | None:
        return self.utterance_end if self.utterance_end is not None else self.vad_done

    @property
    def source(self) -> str:
        """Which moment we're calling "caller stopped"."""
        return UTTERANCE_END if self.utterance_end is not None else "vad_done"

    def unix_ms_attributes(self) -> dict[str, float]:
        """``s2s.<moment>.unix_ms`` for every moment we saw."""
        return {
            f"s2s.{field.name}.unix_ms": value / 1e6
            for field in dataclasses.fields(self)
            if (value := getattr(self, field.name)) is not None
        }

    def step_attributes(self) -> dict[str, float]:
        """``s2s.<step>_ms`` for every step we have both ends of."""
        stopped, tts, out = self.caller_stopped, self.tts_first_audio, self.audio_start
        if stopped is None or tts is None or out is None:
            return {}
        steps = {"s2s.latency_ms": _ms(tts, stopped), "s2s.transport_ms": _ms(out, tts)}
        if self.utterance_end is not None and self.vad_done is not None:
            steps["s2s.endpointing_ms"] = _ms(self.vad_done, self.utterance_end)
        if self.stt_done is not None:
            steps["s2s.stt_ms"] = _ms(self.stt_done, stopped)
        if self.llm_first_token is not None:
            # the model can't start until the turn is over and the words are in
            ready = max(t for t in (self.stt_done, self.vad_done) if t is not None)
            steps["s2s.llm_ms"] = _ms(self.llm_first_token, ready)
            steps["s2s.tts_ms"] = _ms(tts, self.llm_first_token)
        return steps


def coval_trace_provider(api_key: str, api_base: str, simulation_id: str) -> TracerProvider:
    """Every span from this provider lands on one simulation.

    Shut it down at the end of the call so the last batch gets flushed.
    """
    exporter = OTLPSpanExporter(
        endpoint=f"{api_base}/traces",
        headers={"X-API-Key": api_key, "X-Simulation-Id": simulation_id},
        timeout=EXPORT_TIMEOUT_SECS,
    )
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


class TurnLatencyObserver(BaseObserver):
    def __init__(
        self,
        *,
        attributes: dict[str, Any],
        tracer: Tracer | None = None,
        clock: Callable[[], int] = time.time_ns,
    ) -> None:
        super().__init__()
        self._attributes = attributes
        self._tracer = tracer
        self._clock = clock
        self._turn = 0
        self._moment = Moments()

    async def on_push_frame(self, data: FramePushed) -> None:
        if not data.first_push:
            return
        now = self._clock()
        moment = self._moment
        match data.frame:
            case InputTransportMessageFrame(message={"type": "utterance_end"}):
                moment.utterance_end = now
            case UserStoppedSpeakingFrame():
                moment.vad_done = now
                moment.llm_first_token = None  # new turn, so forget any answer in flight
                moment.tts_first_audio = None
            case TranscriptionFrame() if moment.llm_first_token is None:
                moment.stt_done = now  # keep the latest one until the model starts
            case LLMTextFrame() if moment.llm_first_token is None:
                moment.llm_first_token = now
            case TTSAudioRawFrame() if moment.tts_first_audio is None:
                moment.tts_first_audio = now
            case InterruptionFrame():
                self._moment = Moments()  # caller cut the agent off, start fresh
            case BotStartedSpeakingFrame():
                moment.audio_start = now
                if moment.caller_stopped is not None and moment.tts_first_audio is not None:
                    self._emit(moment, start=moment.caller_stopped, end=now)
                self._moment = Moments()

    def _emit(self, moment: Moments, *, start: int, end: int) -> None:
        self._turn += 1
        attributes: dict[str, Any] = {
            **self._attributes,
            "s2s.turn.number": self._turn,
            "s2s.latency.source": moment.source,
            **moment.unix_ms_attributes(),
            **moment.step_attributes(),
        }
        logger.info("turn", **attributes)
        if self._tracer is not None:
            span = self._tracer.start_span(SPAN_NAME, start_time=start, attributes=attributes)
            span.end(end_time=end)
