# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-turn latency measured where the model is, one ``s2s.turn`` span per turn.

T0 is when the caller stopped: Coval's ``utterance_end`` marker when it sends
one, else the local VAD stop. T2 is the first synthesized audio frame, T3 the
first audio frame the transport starts writing to Coval. Provider latency is
T2 - T0 and harness overhead T3 - T2. Each span goes to Coval's trace ingest
under the call's simulation id, so a custom trace metric can average them.
"""

from __future__ import annotations

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
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed

logger = structlog.get_logger("agent")

SPAN_NAME = "s2s.turn"
SERVICE_NAME = "coval-bench-s2s-agent"
EXPORT_TIMEOUT_SECS = 30
UTTERANCE_END = "utterance_end"


def coval_trace_provider(api_key: str, api_base: str, simulation_id: str) -> TracerProvider:
    """A provider whose every span lands on this one simulation; shut it down to flush."""
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
        self._marker_ns: int | None = None
        self._vad_ns: int | None = None
        self._t2_ns: int | None = None

    async def on_push_frame(self, data: FramePushed) -> None:
        if not data.first_push:
            return
        frame = data.frame
        now = self._clock()
        if isinstance(frame, InputTransportMessageFrame):
            if frame.message.get("type") == UTTERANCE_END:
                self._marker_ns = now
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._vad_ns = now
            self._t2_ns = None
        elif isinstance(frame, InterruptionFrame):
            self._t2_ns = None
        elif isinstance(frame, TTSAudioRawFrame):
            if self._t2_ns is None and self._t0_ns() is not None:
                self._t2_ns = now
        elif isinstance(frame, BotStartedSpeakingFrame):
            t0, t2 = self._t0_ns(), self._t2_ns
            if t0 is not None and t2 is not None:
                self._emit(t0, t2, now)
            self._marker_ns = self._vad_ns = self._t2_ns = None

    def _t0_ns(self) -> int | None:
        return self._marker_ns if self._marker_ns is not None else self._vad_ns

    def _emit(self, t0: int, t2: int, t3: int) -> None:
        self._turn += 1
        attributes: dict[str, Any] = {
            **self._attributes,
            "s2s.turn.number": self._turn,
            "s2s.t0.source": UTTERANCE_END if self._marker_ns is not None else "vad",
            "s2s.t0.unix_ms": t0 / 1e6,
            "s2s.t2.unix_ms": t2 / 1e6,
            "s2s.t3.unix_ms": t3 / 1e6,
            "s2s.provider_latency_ms": (t2 - t0) / 1e6,
            "s2s.harness_overhead_ms": (t3 - t2) / 1e6,
        }
        logger.info("turn", **attributes)
        if self._tracer is not None:
            span = self._tracer.start_span(SPAN_NAME, start_time=t0, attributes=attributes)
            span.end(end_time=t3)
