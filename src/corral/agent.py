# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""An agent: a model per modality plus a scenario's prompt, with a slug and digest over it."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from corral.config import IN_SAMPLE_RATE_HZ, INTERRUPT_MIN_WORDS, OUT_SAMPLE_RATE_HZ, VAD_STOP_SECS
from corral.models import LlmModel, SttModel, TtsModel, catalog
from corral.scenarios import load_prompt


@dataclass(frozen=True)
class Agent:
    scenario: str
    stt: SttModel
    llm: LlmModel
    tts: TtsModel
    system_prompt: str
    slug: str
    digest: str


def load_agent(scenario: str, stt_id: str, llm_id: str, tts_id: str) -> Agent:
    """A model per modality, each named as provider/model, on the scenario's prompt."""
    offered = catalog()
    stt, llm, tts = offered.get_stt(stt_id), offered.get_llm(llm_id), offered.get_tts(tts_id)
    prompt_bytes = load_prompt(scenario)
    system_prompt = prompt_bytes.decode().strip()
    if not system_prompt:
        raise ValueError(f"{scenario}'s prompt is empty")
    config = {
        "stt": stt.model_dump(),
        "llm": llm.model_dump(),
        "tts": tts.model_dump(),
        "vad_stop_secs": VAD_STOP_SECS,
        "interrupt_min_words": INTERRUPT_MIN_WORDS,
        "sample_rates_hz": [IN_SAMPLE_RATE_HZ, OUT_SAMPLE_RATE_HZ],
    }
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode() + prompt_bytes).hexdigest()
    slug = "--".join(
        re.sub(r"[^a-z0-9.]+", "-", f"{model.provider}-{model.name}".lower())
        for model in (stt, llm, tts)
    )
    return Agent(scenario, stt, llm, tts, system_prompt, slug, digest)
