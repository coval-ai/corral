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
    persona_name,
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
    id_roulette = require_suite("id_roulette")
    assert id_roulette.workspace is None
    assert len(id_roulette.personas) == 2
    assert load_suite("default") is None
    with pytest.raises(ValueError, match="'default' has no coval.json"):
        require_suite("default")


def test_a_written_out_test_case_names_the_seed_its_backend_uses() -> None:
    assert Suite(test_set="TS", personas=["p1"]).keyed_test_cases() == {}
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


def test_the_files_that_change_the_agent_are_the_prompt_and_the_tools() -> None:
    assert list(behavior_files("default")) == ["system-prompt.txt"]


async def test_every_id_roulette_test_case_has_a_right_answer_to_check() -> None:
    keys = list(require_suite("id_roulette").keyed_test_cases())
    assert len(keys) == 15
    await confirm_test_cases("id_roulette", keys)
    backend = load_environment("id_roulette")
    assert backend is not None
    for key in keys:
        await backend.start(key)
        assert backend.tools[0].character_error_rate["value"]


def test_a_suite_runs_its_first_persona_unless_more_are_named() -> None:
    suite = require_suite("id_roulette")
    names = [persona_name(p) for p in suite.personas]
    assert len(names) == 2 and names[0] == "ID Roulette: quiet room"
    assert [persona_name(p) for p in suite.with_personas([]).personas] == names[:1]
    assert suite.with_personas(["all"]) == suite
    picked = suite.with_personas([names[1], names[0]])
    assert [persona_name(p) for p in picked.personas] == [names[1], names[0]]
    with pytest.raises(ValueError, match="no persona nope in coval.json; it has ID Roulette"):
        suite.with_personas(["nope"])
    by_id = Suite(test_set="TS", personas=["P-1", "P-2"]).with_personas(["P-2"])
    assert by_id.personas == ["P-2"]
    assets = Suite(assets="bank.v1").with_personas(["clean"])
    assert assets.personas == ["clean"]
