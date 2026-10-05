# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The Coval side of a scenario: the test set, persona and metrics one benchmark run names.

A scenario is one prompt and one test set. The ids are not secrets, and the
default scenario is meant to be public, so they ship beside the prompt in
``coval.json`` rather than in ``.env``. A scenario without that file is for
talking to only. Keys starting with ``_`` are the human names behind the ids.
"""

from __future__ import annotations

import importlib.resources
import json

from agent.annotated import AnnotatedModel
from agent.stack import SCENARIOS_PACKAGE

COVAL_FILE = "coval.json"


class TestSet(AnnotatedModel):
    test_set_id: str
    persona_id: str
    metric_ids: list[str]


def load_test_set(scenario: str) -> TestSet | None:
    path = importlib.resources.files(SCENARIOS_PACKAGE).joinpath(scenario, COVAL_FILE)
    if not path.is_file():
        return None
    return TestSet.model_validate(json.loads(path.read_bytes()))


def require_test_set(scenario: str) -> TestSet:
    test_set = load_test_set(scenario)
    if test_set is None:
        raise ValueError(f"scenario {scenario!r} has no {COVAL_FILE}; it is for talking to")
    return test_set
