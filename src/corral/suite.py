# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn a scenario's suite into ids in your Coval workspace.

An id is used as is. A definition is found by its name in the workspace, created if it isn't
there, and patched where it drifted from the file, so running again changes nothing and
editing the file updates Coval. Test cases are matched by their input; one removed from the
file is deleted from the scenario's test set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from corral.coval_api import CovalClient, NameTaken, SyncError
from corral.scenarios import Entry, Suite


@dataclass(frozen=True)
class Kind:
    noun: str
    path: str
    one: str
    many: str
    name_field: str


METRIC = Kind("metric", "/metrics", "metric", "metrics", "metric_name")
PERSONA = Kind("persona", "/personas", "persona", "personas", "name")
TEST_SET = Kind("test set", "/test-sets", "test_set", "test_sets", "display_name")
TEST_CASES = "/test-cases"
WORKSPACE_NAME = "Corral"


@dataclass(frozen=True)
class SuiteIds:
    test_set_id: str
    persona_ids: list[str]
    metric_ids: list[str]


@dataclass(frozen=True)
class WorkspaceChoice:
    workspace_id: str
    found_by_name: bool = False


def choose_workspace(client: CovalClient, configured: str | None) -> WorkspaceChoice:
    """The configured workspace, else the one named Corral, made the first time.

    Two named Corral is an error rather than a guess.
    """
    if configured:
        return WorkspaceChoice(configured)
    named = [
        workspace
        for workspace in client.list_workspaces()
        if workspace.get("display_name") == WORKSPACE_NAME and workspace.get("status") == "ACTIVE"
    ]
    if len(named) > 1:
        raise SyncError(
            f"found {len(named)} workspaces named {WORKSPACE_NAME}; "
            "set CORRAL_WORKSPACE_ID to the one you want"
        )
    if named:
        return WorkspaceChoice(str(named[0]["id"]), found_by_name=True)
    created = client.create("/workspaces", "workspace", {"display_name": WORKSPACE_NAME})
    logger.info(f"Created the {WORKSPACE_NAME} workspace")
    return WorkspaceChoice(str(created["id"]), found_by_name=True)


def sync_suite(client: CovalClient, suite: Suite) -> SuiteIds:
    metric_ids = _sync_kind(client, METRIC, suite.metrics)
    persona_ids = _sync_kind(client, PERSONA, suite.personas)
    if isinstance(suite.test_set, str):
        test_set_id = suite.test_set
    else:
        definition = dict(suite.test_set)
        cases = definition.pop("test_cases", [])
        (test_set_id,) = _sync_kind(client, TEST_SET, [definition])
        _sync_test_cases(client, test_set_id, cases)
    return SuiteIds(test_set_id, persona_ids, metric_ids)


def _sync_kind(client: CovalClient, kind: Kind, entries: list[Entry]) -> list[str]:
    """Ids for these entries, in order, syncing the definitions among them."""
    if all(isinstance(entry, str) for entry in entries):
        return [str(entry) for entry in entries]
    live = {item.get(kind.name_field): item for item in client.list_all(kind.path, kind.many, {})}
    ids = []
    for entry in entries:
        if isinstance(entry, str):
            ids.append(entry)
            continue
        name = entry.get(kind.name_field)
        if not isinstance(name, str) or not name:
            raise SyncError(f"every {kind.noun} defined in coval.json needs a {kind.name_field}")
        current = live.get(name)
        if current is None:
            try:
                created = client.create(kind.path, kind.one, entry)
            except NameTaken:
                raise SyncError(
                    f"{kind.noun} {name!r} already exists in another workspace of this "
                    "organization. Coval names are unique across the organization, so rename "
                    "it in coval.json or rename the other one."
                ) from None
            ids.append(str(created["id"]))
            logger.info(f"Created {kind.noun} {name!r}")
            continue
        drift = _drift(current, entry)
        if drift:
            client.update(kind.path, kind.one, str(current["id"]), drift)
            logger.info(f"Updated {kind.noun} {name!r}: {', '.join(drift)}")
        ids.append(str(current["id"]))
    return ids


def _sync_test_cases(client: CovalClient, test_set_id: str, cases: list[dict[str, Any]]) -> None:
    """A case's ``test_case`` goes in Coval's attributes field, which the header template reads."""
    inputs = [case.get("input_str") for case in cases]
    if len(set(inputs)) != len(inputs) or not all(isinstance(i, str) and i for i in inputs):
        raise SyncError("every test case needs an input_str, and no two can share one")
    live = {
        item.get("input_str"): item
        for item in client.list_all(
            TEST_CASES, "test_cases", {"filter": f'test_set_id="{test_set_id}"'}
        )
    }
    for case in cases:
        wanted = {**case, "test_set_id": test_set_id}
        if "test_case" in wanted:
            wanted["expected_output_json"] = {"test_case": wanted.pop("test_case")}
        current = live.pop(case["input_str"], None)
        if current is None:
            client.create(TEST_CASES, "test_case", wanted)
            logger.info(f"Created test case {case['input_str'][:40]!r}")
        elif drift := _drift(current, wanted):
            client.update(TEST_CASES, "test_case", str(current["id"]), drift)
            logger.info(f"Updated test case {case['input_str'][:40]!r}: {', '.join(drift)}")
    for stale in live.values():
        client.delete(TEST_CASES, str(stale["id"]))
        logger.info(f"Deleted test case {str(stale.get('input_str'))[:40]!r}")


def _drift(current: dict[str, Any], wanted: dict[str, Any]) -> dict[str, Any]:
    """The fields the file sets that Coval has differently."""
    return {key: value for key, value in wanted.items() if current.get(key) != value}
