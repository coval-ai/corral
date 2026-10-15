# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any

import pytest

from corral.environment import JsonEnvironment, ToolsFile

SPEC: dict[str, Any] = {
    "tools": [
        {"name": "find_patient", "description": "Look up.", "parameters": {"name": "string"}},
        {
            "name": "book_appointment",
            "description": "Book.",
            "parameters": {"patient_id": "integer"},
        },
    ],
    "responses": {
        "find_patient": {"patient_id": 123, "next_appointment": None},
        "book_appointment": {"ok": True},
    },
    "test_cases": {
        "book-cleaning": {},
        "already-booked": {"book_appointment": {"ok": False, "error": "already booked"}},
    },
}


async def test_a_json_environment_answers_from_the_defaults_or_the_test_case(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tools.json"
    path.write_text(json.dumps(SPEC))
    environment = JsonEnvironment.load(path)
    find, book = environment.tools
    assert find.schema.name == "find_patient"
    assert find.schema.to_default_dict()["parameters"] == {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    await environment.start(None)
    assert await find.handler({"name": "Ana"}) == {"patient_id": 123, "next_appointment": None}
    assert await book.handler({"patient_id": 123}) == {"ok": True}
    await environment.finish()
    await environment.start("already-booked")
    assert await find.handler({"name": "Ana"}) == {"patient_id": 123, "next_appointment": None}
    assert await book.handler({"patient_id": 123}) == {"ok": False, "error": "already booked"}
    await environment.start("book-cleaning")
    assert await book.handler({"patient_id": 123}) == {"ok": True}
    with pytest.raises(ValueError, match="no test case 'book-cleanin' in tools.json; have already"):
        await environment.start("book-cleanin")


def test_a_tools_file_has_a_response_for_every_tool_and_nothing_else() -> None:
    with pytest.raises(ValueError, match="responses must name every tool"):
        ToolsFile.model_validate({**SPEC, "responses": {"find_patient": {}}})
    with pytest.raises(ValueError, match="responses must name every tool"):
        ToolsFile.model_validate({**SPEC, "responses": {**SPEC["responses"], "cancel": {}}})
    with pytest.raises(
        ValueError, match="test case 'x' answers for tools that don't exist: \\['cancel'\\]"
    ):
        ToolsFile.model_validate({**SPEC, "test_cases": {"x": {"cancel": {}}}})
    with pytest.raises(ValueError, match="two tools share a name"):
        ToolsFile.model_validate({**SPEC, "tools": SPEC["tools"] + SPEC["tools"][:1]})
    with pytest.raises(ValueError, match="descripton"):
        ToolsFile.model_validate({**SPEC, "tools": [{"name": "a", "descripton": "x"}]})
