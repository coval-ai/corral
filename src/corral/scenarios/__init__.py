# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Scenarios: one directory each with a prompt, a menu line, and a Coval suite if it's benchmarked.

A model can have its own prompt under ``prompts/``; every other model runs the scenario's.

A scenario can also ship a ``tools.py`` whose ``environment()`` returns a fresh backend for one
call; see ``corral.environment``. Without one the agent has no tools.

In coval.json each test set, persona and metric is either the id of one that already exists
in Coval, or its full definition, which Corral creates in your workspace and keeps in sync.
``workspace`` is the id of the workspace that you'd like to run it. If workspace isn't set,
defaults to CORRAL_WORKSPACE_ID, and if that isn't set, to a workspace named Corral.
Or coval.json names its ``assets``, and the ids come from a manifest; see ``corral.assets``.
"""

from __future__ import annotations

import importlib
import importlib.resources
import json
from importlib.resources.abc import Traversable
from typing import Any

from pydantic import Field, model_validator

from corral.environment import Environment
from corral.models import JsonModel

PROMPT_FILE = "system-prompt.txt"
PROMPTS_DIR = "prompts"
SUMMARY_FILE = "summary.txt"
COVAL_FILE = "coval.json"
TOOLS_FILE = "tools.py"
TALK_SCENARIO = "default"
BENCHMARK_SCENARIO = "id_roulette"


Entry = str | dict[str, Any]


class Suite(JsonModel):
    """``sample`` runs that many of the test cases per run, the same ones all day.

    ``assets`` names a ``<scenario>.<version>`` in the manifest at CORRAL_ASSETS_URI instead of
    ids; ``personas`` and ``metrics`` are then names from that entry, all of them when left out.
    """

    assets: str | None = None
    workspace: str | None = None
    test_set: Entry | None = None
    personas: list[Entry] = Field(default_factory=list)
    metrics: list[Entry] = Field(default_factory=list)
    sample: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _one_source(self) -> Suite:
        if self.assets is None:
            if self.test_set is None:
                raise ValueError("coval.json needs a test_set, or assets naming one")
            if not self.personas:
                raise ValueError("coval.json needs at least one persona")
            return self
        if self.workspace is not None or self.test_set is not None:
            raise ValueError(
                "assets brings its own workspace and test set; drop them from coval.json"
            )
        if not all(isinstance(entry, str) for entry in [*self.personas, *self.metrics]):
            raise ValueError("with assets, personas and metrics are names from the manifest")
        return self

    def keyed_test_cases(self) -> dict[str, str]:
        """Each written-out case's ``test_case``, the seed its backend uses, with its description.
        In file order; empty for a test set id."""
        if not isinstance(self.test_set, dict):
            return {}
        return {
            case["test_case"]: str(case.get("description") or "")
            for case in self.test_set.get("test_cases", [])
            if isinstance(case.get("test_case"), str)
        }


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


def load_prompt(scenario: str, model_id: str) -> bytes:
    """The model's own prompt at ``prompts/<provider>/<model>.txt``, else the scenario's."""
    folder = _folder(scenario)
    own = folder.joinpath(PROMPTS_DIR, *f"{model_id}.txt".split("/"))
    return (own if own.is_file() else folder.joinpath(PROMPT_FILE)).read_bytes()


def behavior_files(scenario: str, model_id: str) -> dict[str, bytes]:
    """Every file that changes what the agent does: the model's prompt, the tools and their data.

    The menu line and the Coval suite don't change the agent, so they're left out.
    """
    folder = _folder(scenario)
    files = {
        entry.name: entry.read_bytes()
        for entry in sorted(folder.iterdir(), key=lambda entry: entry.name)
        if entry.is_file() and entry.name not in (SUMMARY_FILE, COVAL_FILE)
    }
    files[PROMPT_FILE] = load_prompt(scenario, model_id)
    return files


def load_environment(scenario: str) -> Environment | None:
    """A fresh backend for one call from the scenario's tools.py, or None when it has none."""
    if not _folder(scenario).joinpath(TOOLS_FILE).is_file():
        return None
    module = importlib.import_module(f"{__name__}.{scenario}.tools")
    environment: Environment = module.environment()
    return environment


async def confirm_test_cases(scenario: str, keys: list[str]) -> None:
    """Every key seeds the scenario's backend, so a typo fails here and not call by call."""
    if not keys:
        return
    environment = load_environment(scenario)
    if environment is None:
        raise ValueError(f"{scenario} has test case keys but no tools to seed")
    for key in keys:
        await environment.start(key)
        await environment.finish()


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
