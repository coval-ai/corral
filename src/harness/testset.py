# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Named test sets per scenario: the Coval ids one benchmark run names.

The ids are not secrets and the default set is meant to be public, so they
ship beside the scenario prompt rather than in ``.env``. Each entry carries
its own persona and metrics, since a challenge set scores differently from a
baseline. Keys starting with ``_`` are the human names behind the ids.
"""

from __future__ import annotations

import importlib.resources
import json

from harness.annotated import AnnotatedModel
from harness.stack import SCENARIOS_PACKAGE

TEST_SETS_FILE = "test-sets.json"
DEFAULT_TEST_SET = "baseline"  # becomes "default" once the public set exists


class TestSet(AnnotatedModel):
    summary: str
    test_set_id: str
    persona_id: str
    metric_ids: list[str]


def load_test_sets(scenario: str) -> dict[str, TestSet]:
    path = importlib.resources.files(SCENARIOS_PACKAGE).joinpath(scenario, TEST_SETS_FILE)
    raw = json.loads(path.read_bytes())
    return {name: TestSet.model_validate(entry) for name, entry in raw.items()}


def load_test_set(scenario: str, name: str) -> TestSet:
    sets = load_test_sets(scenario)
    if name not in sets:
        raise ValueError(f"no test set {name!r} in {scenario}; have {', '.join(sorted(sets))}")
    return sets[name]
