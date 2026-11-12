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
from corral.scenarios import PROMPT_FILE, behavior_files, load_prompt


@dataclass(frozen=True)
class Agent:
    scenario: str
    stack: Stack
    system_prompt: str
    slug: str
    digest: str
    tools_file: bytes | None = None


def load_cascade_agent(scenario: str, stt_id: str, llm_id: str, tts_id: str) -> Agent:
    """A cascade, each model named as provider/model, on the scenario's prompt."""
    offered = catalog()
    stack = Cascade(offered.get_stt(stt_id), offered.get_llm(llm_id), offered.get_tts(tts_id))
    return _agent(scenario, stack)


def load_native_agent(scenario: str, s2s_id: str) -> Agent:
    """One native model, named as provider/model, on the scenario's prompt."""
    return _agent(scenario, Native(catalog().get_s2s(s2s_id)))


def load_asset_agent(
    asset: str,
    system_prompt: str,
    s2s_id: str | None,
    stt_id: str | None = None,
    llm_id: str | None = None,
    tts_id: str | None = None,
    tools_file: bytes | None = None,
) -> Agent:
    """One native model, or the cascade's three, on a prompt and tools taken from the assets.
    ``asset`` names the agent in place of a scenario; without a tools file it has no tools."""
    offered = catalog()
    stack: Stack
    if s2s_id:
        stack = Native(offered.get_s2s(s2s_id))
    elif stt_id and llm_id and tts_id:
        stack = Cascade(offered.get_stt(stt_id), offered.get_llm(llm_id), offered.get_tts(tts_id))
    else:
        raise ValueError("an asset agent needs --s2s, or --stt, --llm and --tts")
    files = {PROMPT_FILE: system_prompt.encode()}
    if tools_file is not None:
        files["tools.json"] = tools_file
    return _build(asset, stack, system_prompt.strip(), files, tools_file)


def _agent(scenario: str, stack: Stack) -> Agent:
    system_prompt = load_prompt(scenario).decode().strip()
    return _build(scenario, stack, system_prompt, behavior_files(scenario))


def _build(
    scenario: str,
    stack: Stack,
    system_prompt: str,
    files: dict[str, bytes],
    tools_file: bytes | None = None,
) -> Agent:
    if not system_prompt:
        raise ValueError(f"{scenario}'s prompt is empty")
    hasher = hashlib.sha256(json.dumps(_config(stack), sort_keys=True).encode())
    for name, content in files.items():
        hasher.update(name.encode())
        hasher.update(content)
    digest = hasher.hexdigest()
    slug = "--".join(
        re.sub(r"[^a-z0-9.]+", "-", f"{model.provider}-{model.name}".lower())
        for model in stack.rows
    )
    return Agent(scenario, stack, system_prompt, slug, digest, tools_file)


def _config(stack: Stack) -> dict[str, Any]:
    """Everything about the stack that changes the numbers. A native model does its own
    turn-taking, so the cascade's pinned turn-taking isn't part of its digest."""
    if isinstance(stack, Native):
        return {
            "s2s": stack.s2s.model_dump(exclude_defaults=True),
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
