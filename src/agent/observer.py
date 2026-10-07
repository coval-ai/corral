# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""One ``turn`` span per answered turn, with ``stt``, ``llm`` and ``tts`` under it.

We grab the time at seven moments:

- ``utterance_end``: Coval tells us the caller finished talking
- ``vad_stop``: Silero hears enough silence and Pipecat asks the STT to finalize
- ``stt_done``: the STT sends back the transcript the model will read
- ``vad_done``: smart-turn agrees and the transcript is in, so the turn is over
- ``llm_first_token``: the model says its first word
- ``tts_first_audio``: we have the first bit of synthesized audio
- ``audio_start``: we send the first audio chunk back to Coval

The turn runs from when the caller stopped to ``audio_start``. On Coval that's
``utterance_end``. Locally it's ``vad_stop`` minus Silero's silence window, and
``vad_done`` only if Silero never fired. STT latency is ``vad_stop`` to
``stt_done``: how long the STT took to answer the finalize. It carries each moment as
``s2s.<moment>.unix_ms`` and the gaps as ``s2s.<step>_ms``; ``s2s.latency_ms``
is the headline, caller stopped to first synthesized audio. The three child
spans are the stages, as long as the stage took, tagged with provider and model.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Mapping
from typing import Any

import structlog
from opentelemetry import trace
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
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed

from agent.stack import S2SStack

logger = structlog.get_logger("agent")

SPAN_NAME = "turn"
SERVICE_NAME = "coval-bench-s2s-agent"
EXPORT_TIMEOUT_SECS = 30
UTTERANCE_END = "utterance_end"

Roles = Mapping[str, Mapping[str, str]]  # stage -> {"provider": ..., "model": ...}


def _ms(later: int, earlier: int) -> float:
    return (later - earlier) / 1e6


@dataclasses.dataclass
class Moments:
    """When each moment happened this turn, in unix ns."""

    utterance_end: int | None = None
    vad_stop: int | None = None
    stt_done: int | None = None
    vad_done: int | None = None
    llm_first_token: int | None = None
    tts_first_audio: int | None = None
    audio_start: int | None = None
    # Coval pairs utterance_started and utterance_end by id. We only take an end
    # whose start we saw this turn, so a marker that shows up after we've already
    # answered can't start the next turn's clock.
    utterance_id: str | None = None
    vad_stop_secs: float = 0.0

    @property
    def caller_stopped(self) -> int | None:
        if self.utterance_end is not None:
            return self.utterance_end
        if self.vad_stop is not None:
            return self.vad_stop - int(self.vad_stop_secs * 1e9)
        return self.vad_done

    @property
    def source(self) -> str:
        """Which moment counts as the caller stopping."""
        if self.utterance_end is not None:
            return UTTERANCE_END
        return "vad_stop" if self.vad_stop is not None else "vad_done"

    @property
    def model_ready(self) -> int | None:
        """When the model could start: turn over and words in."""
        known = [t for t in (self.stt_done, self.vad_done) if t is not None]
        return max(known) if known else None

    def unix_ms_attributes(self) -> dict[str, float]:
        """``s2s.<moment>.unix_ms`` for each moment we saw."""
        return {
            f"s2s.{field.name}.unix_ms": value / 1e6
            for field in dataclasses.fields(self)
            if isinstance(value := getattr(self, field.name), int)
        }

    def step_attributes(self) -> dict[str, float]:
        """``s2s.<step>_ms`` for each gap we have both ends of."""
        stopped, tts, out = self.caller_stopped, self.tts_first_audio, self.audio_start
        if stopped is None or tts is None or out is None:
            return {}
        steps = {"s2s.latency_ms": _ms(tts, stopped), "s2s.transport_ms": _ms(out, tts)}
        if self.utterance_end is not None and self.vad_stop is not None:
            steps["s2s.endpointing_ms"] = _ms(self.vad_stop, self.utterance_end)
        if self.vad_stop is not None and self.stt_done is not None:
            steps["s2s.stt_ms"] = _ms(self.stt_done, self.vad_stop)
        if self.llm_first_token is not None and self.model_ready is not None:
            steps["s2s.llm_ms"] = _ms(self.llm_first_token, self.model_ready)
            steps["s2s.tts_ms"] = _ms(tts, self.llm_first_token)
        return steps

    def stages(self) -> dict[str, tuple[int, int]]:
        """Start and end of each stage, in unix ns.

        No stt stage when the transcript came before Silero fired; a span can't run
        backwards. ``s2s.stt_ms`` still says so.
        """
        out: dict[str, tuple[int, int]] = {}
        silero, stt = self.vad_stop, self.stt_done
        if silero is not None and stt is not None and stt >= silero:
            out["stt"] = (silero, stt)
        if self.llm_first_token is not None and self.model_ready is not None:
            out["llm"] = (self.model_ready, self.llm_first_token)
            if self.tts_first_audio is not None:
                out["tts"] = (self.llm_first_token, self.tts_first_audio)
        return out


def role_attributes(stack: S2SStack) -> dict[str, dict[str, str]]:
    """Provider and model per stage, for the child spans."""
    return {
        "stt": {"provider": stack.stt.provider, "model": stack.stt.model},
        "llm": {"provider": stack.llm.provider, "model": stack.llm.model},
        "tts": {"provider": stack.tts.provider, "model": stack.tts.model},
    }


def coval_trace_provider(api_key: str, api_base: str, simulation_id: str) -> TracerProvider:
    """Sends every span to one simulation. Shut it down after the call to flush."""
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
        roles: Roles | None = None,
        tracer: Tracer | None = None,
        clock: Callable[[], int] = time.time_ns,
    ) -> None:
        super().__init__()
        self._attributes = attributes
        self._roles = roles or {}
        self._tracer = tracer
        self._clock = clock
        self._turn = 0
        self._moment = Moments()
        self.turns: list[dict[str, Any]] = []  # every emitted turn's attributes, in order

    async def on_push_frame(self, data: FramePushed) -> None:
        if not data.first_push:
            return
        now = self._clock()
        moment = self._moment
        match data.frame:
            case InputTransportMessageFrame(message={"type": "utterance_started", **rest}):
                started = rest.get("utterance_id")
                moment.utterance_id = started if isinstance(started, str) else None
            case InputTransportMessageFrame(message={"type": "utterance_end", **rest}) if rest.get(
                "utterance_id"
            ) in (None, moment.utterance_id):
                moment.utterance_end = now
            case VADUserStoppedSpeakingFrame(stop_secs=secs) if moment.llm_first_token is None:
                moment.vad_stop = now
                moment.vad_stop_secs = secs
            case UserStoppedSpeakingFrame():
                moment.vad_done = now
                moment.llm_first_token = None  # the answer to this turn hasn't started
                moment.tts_first_audio = None
            case TranscriptionFrame() if moment.llm_first_token is None:
                moment.stt_done = now  # keep the latest one until the model starts
            # The model only runs once Pipecat ends the caller's turn, so text or audio
            # before vad_done is the previous answer still streaming. Ignore it.
            case LLMTextFrame() if moment.vad_done is not None and moment.llm_first_token is None:
                moment.llm_first_token = now
            case TTSAudioRawFrame() if (
                moment.vad_done is not None and moment.tts_first_audio is None
            ):
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
        self.turns.append(attributes)
        logger.info("turn", **attributes)
        if self._tracer is not None:
            self._export(self._tracer, moment, attributes, start=start, end=end)

    def _export(
        self, tracer: Tracer, moment: Moments, attributes: dict[str, Any], *, start: int, end: int
    ) -> None:
        """The turn span, then a child per stage."""
        turn = tracer.start_span(SPAN_NAME, start_time=start, attributes=attributes)
        inside = trace.set_span_in_context(turn)
        for stage, (stage_start, stage_end) in moment.stages().items():
            child = tracer.start_span(
                stage, context=inside, start_time=stage_start, attributes=self._roles.get(stage)
            )
            child.end(end_time=stage_end)
        turn.end(end_time=end)
