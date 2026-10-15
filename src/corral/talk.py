# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Talk to an agent on this machine's microphone and speakers. Ctrl-C hangs up and prints
how long each stage took and, when the scenario has tools, which ones got called."""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from typing import Any

from pipecat.workers.runner import WorkerRunner

from corral.agent import Agent
from corral.config import IN_SAMPLE_RATE_HZ, OUT_SAMPLE_RATE_HZ, get_settings
from corral.observer import ModelsByModality, models_by_modality
from corral.pipeline import ToolCall, build_call


async def run(agent: Agent, test_case: str | None = None) -> None:
    from pipecat.transports.base_transport import TransportParams

    from corral.local import LocalAudioTransport

    transport = LocalAudioTransport(
        TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=IN_SAMPLE_RATE_HZ,
            audio_out_sample_rate=OUT_SAMPLE_RATE_HZ,
        )
    )
    call = await build_call(agent, transport, get_settings(), test_case=test_case)
    runner = WorkerRunner()
    await runner.add_workers(call.worker)
    try:
        await runner.run()
    finally:
        print(summary(call.observer.turns, models_by_modality(agent), agent.slug), flush=True)
        if call.tool_calls is not None:
            print(tool_log(call.tool_calls), flush=True)


STEPS = ("stt", "llm", "tts")
COLUMNS = {"latency": "s2s.turn_latency_ms"} | {s: f"s2s.turn_latency.{s}_ms" for s in STEPS}


def summary(turns: Sequence[dict[str, Any]], models: ModelsByModality, slug: str) -> str:
    """Median and worst of each step across the call, one row per step."""
    if not turns:
        return "no answered turns"
    lines = [f"{len(turns)} turns on {slug}", f"{'':34}{'median':>8}{'max':>8}"]
    for step, key in COLUMNS.items():
        values = [t[key] for t in turns if key in t]
        if not values:
            continue
        model = models.get(step)
        label = f"{step} ({model['provider']}/{model['model']})" if model else step
        lines.append(f"{label:34}{statistics.median(values):>7.0f} {max(values):>7.0f}")
    stt_timed_out = sum(bool(t.get("s2s.turn_latency.stt_timed_out")) for t in turns)
    if stt_timed_out:
        lines.append(f"{stt_timed_out} turns where the STT timed out")
    smart_turn_timed_out = sum(bool(t.get("s2s.turn_latency.smart_turn_timed_out")) for t in turns)
    if smart_turn_timed_out:
        lines.append(f"{smart_turn_timed_out} turns where smart-turn timed out")
    return "\n".join(lines)


def tool_log(calls: Sequence[ToolCall]) -> str:
    """Each tool call in order, as the LLM made it and what it got back."""
    if not calls:
        return "no tools called"
    return "\n".join(
        f"{call.name}({json.dumps(call.arguments)}) -> {json.dumps(call.result, default=str)}"
        for call in calls
    )
