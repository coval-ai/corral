# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Scenarios: one directory each with a prompt, a menu line, and a Coval suite if it's benchmarked.

In coval.json each test set, persona and metric is either the id of one that already exists
in Coval, or its full definition, which Corral creates in your workspace and keeps in sync.
``workspace`` is the id of the workspace that you'd like to run it. If workspace isn't set,
defaults to CORRAL_WORKSPACE_ID, and if that isn't set, to a workspace named Corral.
"""

from __future__ import annotations

import importlib.resources
import json
from importlib.resources.abc import Traversable
from typing import Any

from pydantic import Field

from corral.models import JsonModel

PROMPT_FILE = "system-prompt.txt"
SUMMARY_FILE = "summary.txt"
COVAL_FILE = "coval.json"
TALK_SCENARIO = "default"
BENCHMARK_SCENARIO = "baseline"


Entry = str | dict[str, Any]


class Suite(JsonModel):
    workspace: str | None = None
    test_set: Entry
    personas: list[Entry] = Field(min_length=1)
    metrics: list[Entry] = Field(default_factory=list)


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


def load_suite(scenario: str) -> Suite | None:
    path = _folder(scenario).joinpath(COVAL_FILE)
    if not path.is_file():
        return None
    return Suite.model_validate(json.loads(path.read_bytes()))


def require_suite(scenario: str) -> Suite:
    suite = load_suite(scenario)
    if suite is None:
        raise ValueError(f"scenario {scenario!r} has no {COVAL_FILE}; it is for talking to")
    return suite
