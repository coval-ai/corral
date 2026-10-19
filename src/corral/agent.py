# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""An agent: a stack of models on a scenario's prompt and tools, with a slug and digest.

The stack is a cascade of STT, LLM and TTS, or one native speech-to-speech model.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from corral.config import IN_SAMPLE_RATE_HZ, INTERRUPT_MIN_WORDS, OUT_SAMPLE_RATE_HZ, VAD_STOP_SECS
from corral.models import Cascade, Native, Stack, catalog
from corral.scenarios import behavior_files, load_prompt


@dataclass(frozen=True)
class Agent:
    scenario: str
    stack: Stack
    system_prompt: str
    slug: str
    digest: str


def load_cascade_agent(scenario: str, stt_id: str, llm_id: str, tts_id: str) -> Agent:
    """A cascade, each model named as provider/model, on the scenario's prompt."""
    offered = catalog()
    stack = Cascade(offered.get_stt(stt_id), offered.get_llm(llm_id), offered.get_tts(tts_id))
    return _agent(scenario, stack)


def load_native_agent(scenario: str, s2s_id: str) -> Agent:
    """One native model, named as provider/model, on the scenario's prompt."""
    return _agent(scenario, Native(catalog().get_s2s(s2s_id)))


def _agent(scenario: str, stack: Stack) -> Agent:
    prompt_bytes = load_prompt(scenario)
    system_prompt = prompt_bytes.decode().strip()
    if not system_prompt:
        raise ValueError(f"{scenario}'s prompt is empty")
    hasher = hashlib.sha256(json.dumps(_config(stack), sort_keys=True).encode())
    for name, content in behavior_files(scenario).items():
        hasher.update(name.encode())
        hasher.update(content)
    digest = hasher.hexdigest()
    slug = "--".join(
        re.sub(r"[^a-z0-9.]+", "-", f"{model.provider}-{model.name}".lower())
        for model in stack.rows
    )
    return Agent(scenario, stack, system_prompt, slug, digest)


def _config(stack: Stack) -> dict[str, Any]:
    """Everything about the stack that changes the numbers. A native model does its own
    turn-taking, so the cascade's pinned turn-taking isn't part of its digest."""
    if isinstance(stack, Native):
        return {
            "s2s": stack.s2s.model_dump(),
            "sample_rates_hz": [IN_SAMPLE_RATE_HZ, OUT_SAMPLE_RATE_HZ],
        }
    return {
        "stt": stack.stt.model_dump(),
        "llm": stack.llm.model_dump(),
        "tts": stack.tts.model_dump(),
        "vad_stop_secs": VAD_STOP_SECS,
        "interrupt_min_words": INTERRUPT_MIN_WORDS,
        "sample_rates_hz": [IN_SAMPLE_RATE_HZ, OUT_SAMPLE_RATE_HZ],
    }
