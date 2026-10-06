# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Talk to a stack on this machine's microphone and speakers. Ctrl-C hangs up and prints
how long each stage took."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Any

from pipecat.frames.frames import Frame, LLMRunFrame
from pipecat.pipeline.worker import PipelineWorker
from pipecat.workers.runner import WorkerRunner

from agent.config import get_settings
from agent.observer import Roles, TurnLatencyObserver, role_attributes
from agent.pipeline import build_worker
from agent.services import resolve
from agent.stack import LoadedStack


async def run(loaded: LoadedStack) -> None:
    # pyaudio comes with the local extra only; nothing else needs it.
    from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

    services = resolve(loaded, get_settings())
    audio = loaded.stack.audio
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=audio.in_sample_rate_hz,
            audio_out_sample_rate=audio.out_sample_rate_hz,
        )
    )
    observer = TurnLatencyObserver(
        attributes={"stack": loaded.slug, "digest": loaded.digest},
        roles=role_attributes(loaded.stack),
    )
    worker = build_worker(loaded, services, transport, observer)

    @worker.event_handler("on_pipeline_started")  # type: ignore[untyped-decorator]
    async def _greet(started: PipelineWorker, _frame: Frame) -> None:
        # Fires once StartFrame has passed every processor, so every service is up.
        await started.queue_frame(LLMRunFrame())

    runner = WorkerRunner()
    await runner.add_workers(worker)
    try:
        await runner.run()
    finally:
        print(summary(observer.turns, role_attributes(loaded.stack), loaded.slug), flush=True)


STEPS = ("latency", "endpointing", "stt", "llm", "tts", "transport")


def summary(turns: Sequence[dict[str, Any]], roles: Roles, slug: str) -> str:
    """Median and worst of each step across the call, one row per step."""
    if not turns:
        return "no answered turns"
    lines = [f"{len(turns)} turns on {slug}", f"{'':34}{'median':>8}{'max':>8}"]
    for step in STEPS:
        values = [t[f"s2s.{step}_ms"] for t in turns if f"s2s.{step}_ms" in t]
        if not values:
            continue
        role = roles.get(step)
        label = f"{step} ({role['provider']}/{role['model']})" if role else step
        lines.append(f"{label:34}{statistics.median(values):>7.0f} {max(values):>7.0f}")
    return "\n".join(lines)
