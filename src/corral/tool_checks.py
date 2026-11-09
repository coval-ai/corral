# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Checks on what the agent passed to its tools, each named for what it computes.

A tool in ``tools.json`` lists a check by name with the right value for each argument
it covers, one for every test case or one per test case. When the call ends, the
tool's first call is checked, and each check goes to Coval as a span named after it,
one attribute per ``<tool>.<argument>``, plus its ``.expected`` and ``.heard`` values.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Protocol

CHARACTER_ERROR_RATE = "character_error_rate"
NO_CALL = "(no call)"
NOT_SENT = "(not sent)"
MAX_SHOWN_CHARS = 200


class Called(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def arguments(self) -> dict[str, Any]: ...


def character_error_rate(expected: str, heard: str) -> float:
    """Edits to turn ``heard`` into ``expected``, over ``expected``'s length.

    Case and spaces don't count. Punctuation counts only where the right answer has
    it: "4242-4242" reads as 42424242, but in an email the dot, underscore and @ do.
    """
    keep = {char for char in _plain(expected) if not char.isalnum()}
    want, got = _kept(expected, keep), _kept(heard, keep)
    if not want:
        return 0.0 if not got else 1.0
    row = list(range(len(got) + 1))
    for i, char in enumerate(want, 1):
        diagonal, row[0] = row[0], i
        for j, other in enumerate(got, 1):
            diagonal, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, diagonal + (char != other))
    return row[-1] / len(want)


def scored_arguments(
    expected: Mapping[str, Mapping[str, str]], calls: Sequence[Called]
) -> Iterator[tuple[str, str, object, bool]]:
    """Each ``<tool>.<argument>``, its right value, the first call's value, and if it was called."""
    for tool, arguments in expected.items():
        first = next((call for call in calls if call.name == tool), None)
        for argument, value in arguments.items():
            heard = first.arguments.get(argument) if first else None
            yield f"{tool}.{argument}", value, heard, first is not None


def character_error_rates(
    expected: Mapping[str, Mapping[str, str]], calls: Sequence[Called]
) -> dict[str, float]:
    """``<tool>.<argument>`` to its rate, from each tool's first call.

    A tool the agent never called, or an argument it left out, rates 1.
    """
    return {
        key: character_error_rate(value, str(heard)) if heard is not None else 1.0
        for key, value, heard, _ in scored_arguments(expected, calls)
    }


def shown_values(
    expected: Mapping[str, Mapping[str, str]], calls: Sequence[Called]
) -> dict[str, str]:
    """``<tool>.<argument>.expected`` and ``.heard`` for the values each rate scored, as sent."""
    shown = {}
    for key, value, heard, called in scored_arguments(expected, calls):
        shown[f"{key}.expected"] = _clipped(value)
        if heard is None:
            shown[f"{key}.heard"] = NOT_SENT if called else NO_CALL
        else:
            shown[f"{key}.heard"] = _clipped(heard if isinstance(heard, str) else json.dumps(heard))
    return shown


def _clipped(text: str) -> str:
    return text if len(text) <= MAX_SHOWN_CHARS else text[: MAX_SHOWN_CHARS - 1] + "…"


def _plain(text: str) -> str:
    """Upper case, no whitespace, and curly apostrophes as straight ones."""
    return re.sub(r"\s", "", text).replace("\u2019", "'").upper()


def _kept(text: str, punctuation: set[str]) -> str:
    return "".join(char for char in _plain(text) if char.isalnum() or char in punctuation)
