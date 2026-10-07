# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""A stack: a model per modality plus a scenario's prompt, with a slug and digest over all of it."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from agent.config import IN_SAMPLE_RATE_HZ, INTERRUPT_MIN_WORDS, OUT_SAMPLE_RATE_HZ, VAD_STOP_SECS
from agent.models import LlmPin, SttPin, TtsPin, catalog, find
from agent.scenarios import load_prompt


@dataclass(frozen=True)
class Stack:
    scenario: str
    stt: SttPin
    llm: LlmPin
    tts: TtsPin
    system_prompt: str
    slug: str
    digest: str


def load_stack(
    scenario: str, stt: str | None = None, llm: str | None = None, tts: str | None = None
) -> Stack:
    """A model per modality, by provider/model. Any left out gets the catalog's first."""
    models = catalog()
    stt_pin = find("stt", models.stt, stt)
    llm_pin = find("llm", models.llm, llm)
    tts_pin = find("tts", models.tts, tts)
    prompt_bytes = load_prompt(scenario)
    system_prompt = prompt_bytes.decode().strip()
    if not system_prompt:
        raise ValueError(f"{scenario}'s prompt is empty")
    config = {
        "stt": stt_pin.model_dump(),
        "llm": llm_pin.model_dump(),
        "tts": tts_pin.model_dump(),
        "vad_stop_secs": VAD_STOP_SECS,
        "interrupt_min_words": INTERRUPT_MIN_WORDS,
        "sample_rates_hz": [IN_SAMPLE_RATE_HZ, OUT_SAMPLE_RATE_HZ],
    }
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode() + prompt_bytes).hexdigest()
    slug = "--".join(
        re.sub(r"[^a-z0-9.]+", "-", f"{pin.provider}-{pin.model}".lower())
        for pin in (stt_pin, llm_pin, tts_pin)
    )
    return Stack(scenario, stt_pin, llm_pin, tts_pin, system_prompt, slug, digest)
