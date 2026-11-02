# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Talk to an agent on this machine's microphone and speakers. Ctrl-C hangs up and prints
how long each stage took and, when the scenario has tools, which ones got called."""

from __future__ import annotations

import json
import statistics
import sys
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from pipecat.workers.runner import WorkerRunner

from corral.agent import Agent
from corral.config import IN_SAMPLE_RATE_HZ, OUT_SAMPLE_RATE_HZ, get_settings
from corral.observer import ModelsByModality, models_by_modality
from corral.pipeline import ToolCall, build_call
from corral.transcript import CUT_OFF, NO_TEXT, Utterance


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
    call = await build_call(agent, transport, get_settings(), test_case=test_case, heard=say)
    runner = WorkerRunner()
    await runner.add_workers(call.worker)
    print(f"Talking to {agent.slug}. Speak when it greets you; Ctrl-C hangs up.", flush=True)
    try:
        await runner.run()
    finally:
        print(summary(call.observer.turns, models_by_modality(agent), agent.slug), flush=True)
        if call.tool_calls is not None:
            print(tool_log(call.tool_calls), flush=True)
        for argument, rate in call.character_error_rates().items():
            print(f"{argument} character error rate: {rate:.2f}", flush=True)


GREEN, CYAN, RESET = "\x1b[32m", "\x1b[36m", "\x1b[0m"


def line(utterance: Utterance, *, color: bool = False) -> str:
    """One transcript line as the agent said or heard it, stamped with when it started:
    ``14:31:02 agent: …`` or ``14:31:05 user: …``."""
    when = datetime.fromtimestamp(utterance.start / 1e9).strftime("%H:%M:%S")
    role = "user" if utterance.role == "caller" else utterance.role
    text = utterance.text or NO_TEXT
    if utterance.cut_off:
        text = f"{text} {CUT_OFF}"
    if color:
        return f"{GREEN}{when}{RESET} {CYAN}{role}:{RESET} {text}"
    return f"{when} {role}: {text}"


def say(utterance: Utterance) -> None:
    print(line(utterance, color=sys.stdout.isatty()), flush=True)


STEPS = ("stt", "llm", "endpointing", "model", "tool", "llm_second", "model_second", "tts")
COLUMNS = {"latency": "s2s.turn_latency_ms"} | {s: f"s2s.turn_latency.{s}_ms" for s in STEPS}


def summary(turns: Sequence[dict[str, Any]], models: ModelsByModality, slug: str) -> str:
    """Median and worst of each step across the call, one row per step."""
    if not turns:
        return "no answered turns"
    rows: list[tuple[str, list[float]]] = []
    for step, key in COLUMNS.items():
        values = [t[key] for t in turns if key in t]
        model = models.get(step)
        label = f"{step} ({model['provider']}/{model['model']})" if model else step
        if values:
            rows.append((label, values))
    width = max(len(label) for label, _ in rows) + 2
    lines = [f"{len(turns)} turns on {slug}", f"{'':{width}}{'median':>8}{'max':>8}"]
    for label, values in rows:
        lines.append(f"{label:{width}}{statistics.median(values):>7.0f} {max(values):>7.0f}")
    silero_missed = sum(bool(t.get("s2s.turn_latency.silero_missed")) for t in turns)
    if silero_missed:
        lines.append(f"{silero_missed} turns where Silero never heard you; check the mic level")
    stt_timed_out = sum(bool(t.get("s2s.turn_latency.stt_timed_out")) for t in turns)
    if stt_timed_out:
        lines.append(f"{stt_timed_out} turns where the STT timed out")
    smart_turn_timed_out = sum(bool(t.get("s2s.turn_latency.smart_turn_timed_out")) for t in turns)
    if smart_turn_timed_out:
        lines.append(f"{smart_turn_timed_out} turns where smart-turn timed out")
    tool_turns = sum(bool(t.get("s2s.turn.tool_calls")) for t in turns)
    if tool_turns:
        lines.append(f"{tool_turns} turns that called a tool")
    return "\n".join(lines)


def tool_log(calls: Sequence[ToolCall]) -> str:
    """Each tool call in order, as the LLM made it and what it got back."""
    if not calls:
        return "no tools called"
    return "\n".join(
        f"{call.name}({json.dumps(call.arguments)}) -> {json.dumps(call.result, default=str)}"
        for call in calls
    )
