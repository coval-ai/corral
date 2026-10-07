# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Talk to a stack on this machine's microphone and speakers. Ctrl-C hangs up and prints
how long each stage took."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Any

from pipecat.workers.runner import WorkerRunner

from agent.config import get_settings
from agent.observer import ModelsByModality, models_by_modality
from agent.pipeline import build_call
from agent.stack import Stack


async def run(stack: Stack) -> None:
    from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

    audio = stack.audio
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=audio.in_sample_rate_hz,
            audio_out_sample_rate=audio.out_sample_rate_hz,
        )
    )
    call = build_call(stack, transport, get_settings())
    runner = WorkerRunner()
    await runner.add_workers(call.worker)
    try:
        await runner.run()
    finally:
        print(summary(call.observer.turns, models_by_modality(stack), stack.slug), flush=True)


STEPS = ("latency", "endpointing", "stt", "llm", "tts", "transport")


def summary(turns: Sequence[dict[str, Any]], models: ModelsByModality, slug: str) -> str:
    """Median and worst of each step across the call, one row per step."""
    if not turns:
        return "no answered turns"
    lines = [f"{len(turns)} turns on {slug}", f"{'':34}{'median':>8}{'max':>8}"]
    for step in STEPS:
        values = [t[f"s2s.{step}_ms"] for t in turns if f"s2s.{step}_ms" in t]
        if not values:
            continue
        model = models.get(step)
        label = f"{step} ({model['provider']}/{model['model']})" if model else step
        lines.append(f"{label:34}{statistics.median(values):>7.0f} {max(values):>7.0f}")
    return "\n".join(lines)
