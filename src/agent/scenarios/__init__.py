# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Scenarios: one directory each with a prompt, a menu line, and Coval ids if it's benchmarked.

The ids in coval.json are public, so they ship here rather than in .env.
"""

from __future__ import annotations

import importlib.resources
import json
from importlib.resources.abc import Traversable

from agent.models import JsonModel

PROMPT_FILE = "system-prompt.txt"
SUMMARY_FILE = "summary.txt"
COVAL_FILE = "coval.json"
TALK_SCENARIO = "default"
BENCHMARK_SCENARIO = "baseline"


class TestSet(JsonModel):
    test_set_id: str
    persona_id: str
    metric_ids: list[str]


def _folder(scenario: str) -> Traversable:
    return importlib.resources.files(__name__).joinpath(scenario)


def scenario_names() -> list[str]:
    """Every scenario directory that ships a prompt."""
    return sorted(
        entry.name
        for entry in importlib.resources.files(__name__).iterdir()
        if entry.is_dir() and entry.joinpath(PROMPT_FILE).is_file()
    )


def scenario_summary(scenario: str) -> str:
    path = _folder(scenario).joinpath(SUMMARY_FILE)
    return path.read_text().strip() if path.is_file() else ""


def load_prompt(scenario: str) -> bytes:
    return _folder(scenario).joinpath(PROMPT_FILE).read_bytes()


def load_test_set(scenario: str) -> TestSet | None:
    path = _folder(scenario).joinpath(COVAL_FILE)
    if not path.is_file():
        return None
    return TestSet.model_validate(json.loads(path.read_bytes()))


def require_test_set(scenario: str) -> TestSet:
    test_set = load_test_set(scenario)
    if test_set is None:
        raise ValueError(f"scenario {scenario!r} has no {COVAL_FILE}; it is for talking to")
    return test_set
