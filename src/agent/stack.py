# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""A stack: a model per modality, the harness's turn-taking and audio, and a scenario's prompt."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Literal

from agent.models import JsonModel, LlmPin, SttPin, TtsPin, catalog, find
from agent.scenarios import load_prompt


class TurnTakingPin(JsonModel):
    vad_stop_secs: float
    turn_analyzer: Literal["smart-turn-v3"]
    interrupt_min_words: int


class AudioPin(JsonModel):
    in_sample_rate_hz: int
    out_sample_rate_hz: int


TURN_TAKING = TurnTakingPin(vad_stop_secs=0.2, turn_analyzer="smart-turn-v3", interrupt_min_words=3)
AUDIO = AudioPin(in_sample_rate_hz=16000, out_sample_rate_hz=24000)


@dataclass(frozen=True)
class Stack:
    scenario: str
    stt: SttPin
    llm: LlmPin
    tts: TtsPin
    turn_taking: TurnTakingPin
    audio: AudioPin
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
        "turn_taking": TURN_TAKING.model_dump(),
        "audio": AUDIO.model_dump(),
    }
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode() + prompt_bytes).hexdigest()
    slug = "--".join(
        re.sub(r"[^a-z0-9.]+", "-", f"{pin.provider}-{pin.model}".lower())
        for pin in (stt_pin, llm_pin, tts_pin)
    )
    return Stack(
        scenario, stt_pin, llm_pin, tts_pin, TURN_TAKING, AUDIO, system_prompt, slug, digest
    )
