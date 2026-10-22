# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""A scenario's fake backend: the tools the agent can call and state that lives for one call.

A scenario that ships a ``tools.py`` gives the agent tools. ``tools.py`` doesn't return a list
of tools, it returns an environment: a small backend that ``start`` seeds for the call, the
tools act on, and ``finish`` cleans up. The pipeline runs that lifecycle and doesn't care
what's inside. Handlers take the call's arguments and return a result. They never see Pipecat;
the pipeline wraps them, times them and turns an exception into the result the LLM reads.

``JsonEnvironment`` is the simplest one: canned answers from a ``tools.json``. A tool answers
the same thing however it's called, and a test case swaps in different answers. State that
changes after a write, or answers that depend on the arguments, is where you write your own.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pydantic import Field, model_validator

from corral.models import JsonModel

Handler = Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass(frozen=True)
class Tool:
    """A tool the agent can call, and the right value of each argument its checks cover."""

    schema: FunctionSchema
    handler: Handler
    character_error_rate: dict[str, str] = field(default_factory=dict)


class Environment(Protocol):
    tools: list[Tool]

    async def start(self, test_case: str | None) -> None: ...

    async def finish(self) -> None: ...


class ToolSpec(JsonModel):
    """One tool in ``tools.json``. ``parameters`` maps each argument to its JSON type.

    ``character_error_rate`` maps an argument to its right value, so the agent's first
    call is checked against it. The value is one string for every test case, or a map
    from test case to that case's string. See ``corral.tool_checks``.
    """

    name: str
    description: str
    parameters: dict[str, str] = Field(default_factory=dict)
    character_error_rate: dict[str, str | dict[str, str]] = Field(default_factory=dict)

    def right_values(self, test_case: str | None) -> dict[str, str]:
        """Each checked argument's right value for this test case; per-case ones need a case."""
        values = {}
        for argument, value in self.character_error_rate.items():
            if isinstance(value, str):
                values[argument] = value
            elif test_case is not None and test_case in value:
                values[argument] = value[test_case]
        return values

    def checked_test_cases(self) -> set[str]:
        return {
            case for v in self.character_error_rate.values() if isinstance(v, dict) for case in v
        }

    @model_validator(mode="after")
    def _checks_known_arguments(self) -> ToolSpec:
        if unknown := set(self.character_error_rate) - set(self.parameters):
            raise ValueError(f"{self.name} checks arguments it doesn't take: {sorted(unknown)}")
        return self

    def function_schema(self) -> FunctionSchema:
        return FunctionSchema(
            name=self.name,
            description=self.description,
            properties={arg: {"type": kind} for arg, kind in self.parameters.items()},
            required=list(self.parameters),
        )


class ToolsFile(JsonModel):
    """The schemas once, each tool's default response, and per test case only what's different."""

    tools: list[ToolSpec] = Field(min_length=1)
    responses: dict[str, Any]
    test_cases: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def known_test_cases(self) -> set[str]:
        """A test case is known if it changes an answer or has a right value to check."""
        return set(self.test_cases).union(*(tool.checked_test_cases() for tool in self.tools))

    @model_validator(mode="after")
    def _responses_match_tools(self) -> ToolsFile:
        names = {tool.name for tool in self.tools}
        if len(names) != len(self.tools):
            raise ValueError("two tools share a name")
        if set(self.responses) != names:
            raise ValueError(
                f"responses must name every tool and nothing else; tools are {sorted(names)}"
            )
        for key, overrides in self.test_cases.items():
            if unknown := set(overrides) - names:
                raise ValueError(
                    f"test case {key!r} answers for tools that don't exist: {sorted(unknown)}"
                )
        return self


class JsonEnvironment:
    """Canned answers from a ``tools.json``, swapped per test case."""

    def __init__(self, spec: ToolsFile) -> None:
        self._spec = spec
        self._responses: dict[str, Any] = {}
        self.tools = self._tools(None)

    @classmethod
    def load(cls, path: Path) -> JsonEnvironment:
        return cls(ToolsFile.model_validate(json.loads(path.read_bytes())))

    async def start(self, test_case: str | None) -> None:
        """The default answers with the test case's own laid over them, or just the defaults."""
        if test_case is not None and test_case not in (known := self._spec.known_test_cases()):
            have = ", ".join(sorted(known)) or "none"
            raise ValueError(f"no test case {test_case!r} in tools.json; have {have}")
        overrides = self._spec.test_cases.get(test_case, {}) if test_case else {}
        self._responses = {**self._spec.responses, **overrides}
        self.tools = self._tools(test_case)

    async def finish(self) -> None:
        self._responses = {}

    def _tools(self, test_case: str | None) -> list[Tool]:
        return [
            Tool(tool.function_schema(), self._answer(tool.name), tool.right_values(test_case))
            for tool in self._spec.tools
        ]

    def _answer(self, name: str) -> Handler:
        async def handler(_arguments: dict[str, Any]) -> object:
            return self._responses[name]

        return handler
