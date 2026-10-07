# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""A stack: a model per modality, the harness's turn-taking and audio, and a scenario's prompt."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import re
from dataclasses import dataclass
from typing import Literal

from agent.annotated import AnnotatedModel
from agent.catalog import LlmPin, SttPin, TtsPin, catalog, find

SCENARIOS_PACKAGE = "agent.scenarios"
PROMPT_FILE = "system-prompt.txt"
SUMMARY_FILE = "summary.txt"
TALK_SCENARIO = "default"
BENCHMARK_SCENARIO = "baseline"


class TurnTakingPin(AnnotatedModel):
    vad_stop_secs: float
    turn_analyzer: Literal["smart-turn-v3"]
    interrupt_min_words: int


class AudioPin(AnnotatedModel):
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


def scenario_names() -> list[str]:
    """Every scenario directory that ships a prompt."""
    scenarios = importlib.resources.files(SCENARIOS_PACKAGE)
    return sorted(
        entry.name
        for entry in scenarios.iterdir()
        if entry.is_dir() and entry.joinpath(PROMPT_FILE).is_file()
    )


def scenario_summary(scenario: str) -> str:
    """The scenario's one line for a menu, from its summary.txt."""
    path = importlib.resources.files(SCENARIOS_PACKAGE).joinpath(scenario, SUMMARY_FILE)
    return path.read_text().strip() if path.is_file() else ""


def load_stack(
    scenario: str, stt: str | None = None, llm: str | None = None, tts: str | None = None
) -> Stack:
    """A model per modality, by provider/model. Any left out gets the catalog's first."""
    models = catalog()
    stt_pin = find("stt", models.stt, stt)
    llm_pin = find("llm", models.llm, llm)
    tts_pin = find("tts", models.tts, tts)
    prompt_bytes = (
        importlib.resources.files(SCENARIOS_PACKAGE).joinpath(scenario, PROMPT_FILE).read_bytes()
    )
    system_prompt = prompt_bytes.decode().strip()
    if not system_prompt:
        raise ValueError(f"{scenario}/{PROMPT_FILE} is empty")
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
