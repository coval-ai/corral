# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from corral.scenarios import behavior_files, load_environment, load_suite, require_suite


def test_a_scenario_carries_its_coval_suite_or_is_for_talking_only() -> None:
    ultrabank = require_suite("ultrabank")
    assert ultrabank.test_set == "QjkAPGkb"
    assert ultrabank.workspace == "01KVGSYSYM8VWWE8M8Y5S14D5N"
    assert len(ultrabank.personas) == 1 and len(ultrabank.metrics) == 3
    assert load_suite("default") is None
    demo = require_suite("demo")
    assert isinstance(demo.test_set, dict)
    keys = [case["simulation_metadata_input"]["key"] for case in demo.test_set["test_cases"]]
    assert keys == ["book-cleaning", "already-booked"]
    with pytest.raises(ValueError, match="'default' has no coval.json"):
        require_suite("default")


async def test_a_scenario_with_a_tools_py_gives_the_agent_a_fresh_backend_per_call() -> None:
    assert load_environment("default") is None
    first, second = load_environment("demo"), load_environment("demo")
    assert first is not None and second is not None and first is not second
    assert [tool.schema.name for tool in first.tools] == ["find_patient", "book_appointment"]
    await first.start("already-booked")
    await second.start(None)
    book = {tool.schema.name: tool.handler for tool in second.tools}["book_appointment"]
    assert await book({"patient_id": 123, "date": "2026-10-21"}) == {
        "ok": True,
        "confirmation": "A1B2",
    }


def test_the_files_that_change_the_agent_are_the_prompt_and_the_tools() -> None:
    assert list(behavior_files("default")) == ["system-prompt.txt"]
    assert list(behavior_files("demo")) == ["system-prompt.txt", "tools.json", "tools.py"]
