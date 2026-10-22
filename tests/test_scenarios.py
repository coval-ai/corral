# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from corral import scenarios
from corral.environment import JsonEnvironment, ToolsFile
from corral.scenarios import (
    Suite,
    behavior_files,
    confirm_test_cases,
    load_environment,
    load_suite,
    require_suite,
)


def _booking() -> JsonEnvironment:
    """One tool, one answer, one test case that changes it."""
    spec = ToolsFile(
        tools=[{"name": "book", "description": "Book it."}],
        responses={"book": {"ok": True}},
        test_cases={"already-booked": {"book": {"ok": False}}},
    )
    return JsonEnvironment(spec)


def test_a_scenario_carries_its_coval_suite_or_is_for_talking_only() -> None:
    ultrabank = require_suite("ultrabank")
    assert ultrabank.test_set == "QjkAPGkb"
    assert ultrabank.workspace == "01KVGSYSYM8VWWE8M8Y5S14D5N"
    assert len(ultrabank.personas) == 1 and len(ultrabank.metrics) == 3
    assert load_suite("default") is None
    with pytest.raises(ValueError, match="'default' has no coval.json"):
        require_suite("default")


def test_a_written_out_test_case_names_the_seed_its_backend_uses() -> None:
    assert require_suite("ultrabank").keyed_test_cases() == {}
    suite = Suite(
        test_set={"test_cases": [{"test_case": "already-booked", "description": "Taken."}]},
        personas=["p1"],
    )
    assert suite.keyed_test_cases() == {"already-booked": "Taken."}


async def test_every_test_case_key_is_tried_on_the_backend_before_a_run() -> None:
    with patch.object(scenarios, "load_environment", return_value=_booking()):
        await confirm_test_cases("dental", ["already-booked"])
        with pytest.raises(ValueError, match="no test case 'already-booke'"):
            await confirm_test_cases("dental", ["already-booke"])


async def test_a_test_case_key_on_a_scenario_without_tools_is_refused() -> None:
    await confirm_test_cases("default", [])
    with pytest.raises(ValueError, match="default has test case keys but no tools"):
        await confirm_test_cases("default", ["x"])


def test_a_scenario_without_a_tools_py_gives_the_agent_no_backend() -> None:
    assert load_environment("default") is None
    assert load_environment("ultrabank") is None


def test_the_files_that_change_the_agent_are_the_prompt_and_the_tools() -> None:
    assert list(behavior_files("default")) == ["system-prompt.txt"]
    assert list(behavior_files("ultrabank")) == ["system-prompt.txt"]


async def test_every_id_roulette_test_case_has_a_right_answer_to_check() -> None:
    keys = list(require_suite("id_roulette").keyed_test_cases())
    assert len(keys) == 15
    await confirm_test_cases("id_roulette", keys)
    backend = load_environment("id_roulette")
    assert backend is not None
    for key in keys:
        await backend.start(key)
        assert backend.tools[0].character_error_rate["value"]
