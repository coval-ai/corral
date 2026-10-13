# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""One ``turn_latency`` span per answered turn, with ``stt``, ``llm`` and ``tts`` under it.

We grab the time at these moments:

- ``vad_stop``: Silero hears enough silence and Pipecat asks the STT to finalize
- ``stt_done``: the STT confirms the transcript is final
- ``vad_done``: smart-turn agrees the caller is done
- ``llm_start``: the LLM request goes out
- ``llm_first_token``: the model says its first word
- ``tts_start``: the first text goes to the TTS
- ``tts_first_audio``: we have the first bit of synthesized audio
- ``audio_start``: we send the first audio chunk back to Coval

Latency is from when the caller stopped to ``tts_first_audio``. The caller stopped at
``vad_stop`` minus Silero's silence window, on our clock, the same locally and on Coval.

Each stage is measured at its own request boundary, so it's the provider's time and
nothing else: ``stt`` is ``vad_stop`` to ``stt_done``, ``llm`` is ``llm_start`` to
``llm_first_token``, ``tts`` is ``tts_start`` to ``tts_first_audio``. Pipecat times the
LLM and TTS requests itself and reports them in a metrics frame; the start moments are
worked back from those. The stages don't add up to the latency. What's left over is
turn-taking and plumbing, which is ours.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Mapping
from typing import Any

from loguru import logger
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Tracer
from pipecat.audio.turn.smart_turn.base_smart_turn import STOP_SECS as SMART_TURN_STOP_SECS
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    InterruptionFrame,
    LLMTextFrame,
    MetricsFrame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameProcessor

from corral.agent import Agent

SPAN_NAME = "turn_latency"
SERVICE_NAME = "coval-bench-s2s-agent"
EXPORT_TIMEOUT_SECS = 30

ModelsByModality = Mapping[str, Mapping[str, str]]


def _ms(later: int, earlier: int) -> float:
    """Milliseconds from one unix-ns moment to another."""
    return (later - earlier) / 1e6


def _ttfb_ns(reports: list[Any]) -> int | None:
    """How long Pipecat says the request took, if this metrics frame carries that."""
    for report in reports:
        if isinstance(report, TTFBMetricsData):
            return int(report.value * 1e9)
    return None


@dataclasses.dataclass
class Moments:
    """When each moment happened this turn, in unix ns."""

    vad_stop: int | None = None
    stt_done: int | None = None
    vad_done: int | None = None
    llm_start: int | None = None
    llm_first_token: int | None = None
    tts_start: int | None = None
    tts_first_audio: int | None = None
    audio_start: int | None = None
    vad_stop_secs: float = 0.0

    @property
    def caller_stopped(self) -> int | None:
        """Silero firing minus its silence window, or smart-turn if Silero never fired."""
        if self.vad_stop is not None:
            return self.vad_stop - int(self.vad_stop_secs * 1e9)
        return self.vad_done

    def unix_ms_attributes(self) -> dict[str, float]:
        """``s2s.<moment>.unix_ms`` for each moment we saw."""
        return {
            f"s2s.{field.name}.unix_ms": value / 1e6
            for field in dataclasses.fields(self)
            if isinstance(value := getattr(self, field.name), int)
        }

    def stages(self) -> dict[str, tuple[int, int]]:
        """Start and end of each stage we have both ends of, in unix ns."""
        pairs = {
            "stt": (self.vad_stop, self.stt_done),
            "llm": (self.llm_start, self.llm_first_token),
            "tts": (self.tts_start, self.tts_first_audio),
        }
        return {
            name: (start, end)
            for name, (start, end) in pairs.items()
            if start is not None and end is not None
        }

    def step_attributes(self) -> dict[str, float]:
        """The headline, transport and each stage as ``s2s.turn_latency.*_ms``."""
        stopped, tts, out = self.caller_stopped, self.tts_first_audio, self.audio_start
        if stopped is None or tts is None or out is None:
            return {}
        steps = {
            "s2s.turn_latency_ms": _ms(tts, stopped),
            "s2s.turn_latency.transport_ms": _ms(out, tts),
        }
        for name, (start, end) in self.stages().items():
            steps[f"s2s.turn_latency.{name}_ms"] = _ms(end, start)
        return steps

    @property
    def turn_detection_timed_out(self) -> bool:
        """Smart-turn never said the caller was done; its silence fallback ended the turn."""
        if self.vad_stop is None or self.vad_done is None:
            return False
        return self.vad_done - self.vad_stop >= (SMART_TURN_STOP_SECS - self.vad_stop_secs) * 1e9


def _describe(number: int, moment: Moments, attributes: dict[str, Any]) -> str:
    """One line per turn in Pipecat's log, e.g. ``Turn 3: 1047 ms (stt 80, llm 539, tts 164)``."""
    parts = [f"{stage} {_ms(end, start):.0f}" for stage, (start, end) in moment.stages().items()]
    if moment.stt_done is None:
        parts.insert(0, "stt unconfirmed")
    if moment.turn_detection_timed_out:
        parts.append("turn detection timed out")
    latency = attributes["s2s.turn_latency_ms"]
    return f"Turn {number}: {latency:.0f} ms ({', '.join(parts)})"


def models_by_modality(agent: Agent) -> dict[str, dict[str, str]]:
    """Provider and model per stage, for the child spans."""
    return {
        "stt": {"provider": agent.stt.provider, "model": agent.stt.name},
        "llm": {"provider": agent.llm.provider, "model": agent.llm.name},
        "tts": {"provider": agent.tts.provider, "model": agent.tts.name},
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
        models: ModelsByModality | None = None,
        stt: FrameProcessor | None = None,
        llm: FrameProcessor | None = None,
        tts: FrameProcessor | None = None,
        tracer: Tracer | None = None,
        clock: Callable[[], int] = time.time_ns,
    ) -> None:
        super().__init__()
        self._attributes = attributes
        self._models = models or {}
        self._stt, self._llm, self._tts = stt, llm, tts
        self._tracer = tracer
        self._clock = clock
        self._turn = 0
        self._moment = Moments()
        self.turns: list[dict[str, Any]] = []

    async def on_push_frame(self, data: FramePushed) -> None:
        """Stamp the clock as each frame first leaves the processor that made it."""
        if not data.first_push:
            return
        now = self._clock()
        moment = self._moment
        match data.frame:
            case VADUserStoppedSpeakingFrame(stop_secs=secs) if moment.llm_first_token is None:
                moment.vad_stop = now
                moment.vad_stop_secs = secs
                moment.stt_done = None
                moment.llm_start = None
                moment.tts_start = None
                moment.tts_first_audio = None
            case UserStoppedSpeakingFrame():
                moment.vad_done = now
            case MetricsFrame(data=reports) if moment.vad_stop is not None:
                self._record_request(data.source, now, _ttfb_ns(reports))
            # The model only runs after Silero fires for the caller's turn, so text or
            # audio before vad_stop is the previous answer still streaming. Ignore it.
            case LLMTextFrame() if moment.vad_stop is not None and moment.llm_first_token is None:
                moment.llm_first_token = now
            case TTSAudioRawFrame() if (
                moment.vad_stop is not None and moment.tts_first_audio is None
            ):
                moment.tts_first_audio = now
            case InterruptionFrame():
                self._moment = Moments()
            case BotStartedSpeakingFrame():
                moment.audio_start = now
                if moment.caller_stopped is not None and moment.tts_first_audio is not None:
                    self._emit(moment, start=moment.caller_stopped, end=now)
                self._moment = Moments()

    def _record_request(self, source: FrameProcessor, now: int, ttfb_ns: int | None) -> None:
        """A service just got its first byte back; Pipecat says how long the request took."""
        moment = self._moment
        if ttfb_ns is None:
            return
        if source is self._stt and moment.stt_done is None:
            moment.stt_done = now
        elif source is self._llm and moment.llm_start is None:
            moment.llm_start = now - ttfb_ns
        elif source is self._tts and moment.tts_start is None:
            moment.tts_start = now - ttfb_ns

    def _emit(self, moment: Moments, *, start: int, end: int) -> None:
        """One answered turn: log it, keep it for the summary, and export it if we're tracing."""
        self._turn += 1
        attributes: dict[str, Any] = {
            **self._attributes,
            "s2s.turn.number": self._turn,
            "s2s.turn_latency.stt_confirmed": moment.stt_done is not None,
            "s2s.turn_latency.turn_detection_timed_out": moment.turn_detection_timed_out,
            **moment.unix_ms_attributes(),
            **moment.step_attributes(),
        }
        self.turns.append(attributes)
        level = "INFO" if moment.stt_done is not None else "WARNING"
        logger.log(level, _describe(self._turn, moment, attributes))
        if self._tracer is not None:
            self._export(self._tracer, moment, attributes, start=start, end=end)

    def _export(
        self, tracer: Tracer, moment: Moments, attributes: dict[str, Any], *, start: int, end: int
    ) -> None:
        """The turn span with a child per stage we have both ends of."""
        turn = tracer.start_span(SPAN_NAME, start_time=start, attributes=attributes)
        inside = trace.set_span_in_context(turn)
        for stage, (stage_start, stage_end) in moment.stages().items():
            child = tracer.start_span(
                stage, context=inside, start_time=stage_start, attributes=self._models.get(stage)
            )
            child.end(end_time=stage_end)
        turn.end(end_time=end)
