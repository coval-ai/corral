# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import httpx
import pytest

from corral.coval_api import CovalClient, SyncError
from corral.scenarios import Suite
from corral.suite import SuiteIds, WorkspaceChoice, choose_workspace, sync_suite

COLLECTIONS = {
    "metrics": "metric",
    "personas": "persona",
    "test-sets": "test_set",
    "workspaces": "workspace",
}


class Workspace:
    """A Coval workspace in memory: list, create, patch and delete, and every write recorded."""

    def __init__(self) -> None:
        self.items: dict[str, dict[str, dict[str, Any]]] = {
            "metrics": {},
            "personas": {},
            "test-sets": {},
            "test-cases": {},
            "workspaces": {"W0": {"id": "W0", "display_name": "Default", "status": "ACTIVE"}},
        }
        self.writes: list[tuple[str, str, dict[str, Any]]] = []
        self.taken_elsewhere: set[str] = set()

    def client(self) -> CovalClient:
        return CovalClient("coval-key", transport=httpx.MockTransport(self.handle))

    def handle(self, request: httpx.Request) -> httpx.Response:
        collection, _, item_id = request.url.path.removeprefix("/v1/").partition("/")
        store = self.items[collection]
        key = COLLECTIONS.get(collection, "test_case")
        body: dict[str, Any] = json.loads(request.content) if request.content else {}
        if request.method == "GET" and item_id:
            return httpx.Response(200 if item_id in store else 404, json={})
        if request.method == "GET":
            found = list(store.values())
            if wanted := request.url.params.get("filter"):
                found = [item for item in found if item["test_set_id"] == wanted.split('"')[1]]
            return httpx.Response(200, json={key + "s": found})
        self.writes.append((request.method, collection, body))
        if collection == "workspaces":
            body = {**body, "status": "ACTIVE"}
        if request.method == "POST":
            if self.taken_elsewhere & set(map(str, body.values())):
                return httpx.Response(409, json={"error": {"code": "ALREADY_EXISTS"}})
            item_id = f"{collection}-{len(store)}"
            store[item_id] = {**body, "id": item_id}
        elif request.method == "PATCH":
            store[item_id].update(body)
        else:
            del store[item_id]
            return httpx.Response(204)
        return httpx.Response(200, json={key: store[item_id]})


DEFINED: dict[str, Any] = {
    "test_set": {
        "display_name": "Front desk",
        "test_cases": [
            {"input_str": "Ask for opening hours.", "expected_behaviors": ["Gives the hours"]},
            {
                "input_str": "Book a table for two.",
                "expected_behaviors": ["Confirms a time"],
                "test_case": "table-free",
            },
        ],
    },
    "personas": [
        {"name": "Easy caller", "persona_prompt": "Be clear.", "voice_name": "aria"},
        {"name": "Hard caller", "persona_prompt": "Interrupt.", "voice_name": "aria"},
    ],
    "metrics": [
        {"metric_name": "Answered", "metric_type": "METRIC_LLM_BINARY", "prompt": "Answered?"},
        {"metric_name": "Polite", "metric_type": "METRIC_LLM_BINARY", "prompt": "Polite?"},
    ],
}


def _defined() -> Suite:
    return Suite.model_validate(DEFINED)


def test_a_suite_of_ids_is_used_as_is_once_coval_has_each() -> None:
    workspace = Workspace()
    for collection, item_id in [("test-sets", "TS"), ("personas", "P1"), ("metrics", "M1")]:
        workspace.items[collection][item_id] = {"id": item_id}
    suite = Suite(test_set="TS", personas=["P1"], metrics=["M1"])
    assert sync_suite(workspace.client(), suite) == SuiteIds("TS", ["P1"], ["M1"])
    assert workspace.writes == []


def test_an_id_from_another_organization_is_named_before_anything_is_written() -> None:
    workspace = Workspace()
    workspace.items["test-sets"]["TS"] = {"id": "TS"}
    suite = Suite(test_set="TS", personas=["P-old"], metrics=[])
    with pytest.raises(SyncError, match="persona P-old in coval.json isn't in this workspace"):
        sync_suite(workspace.client(), suite)
    assert workspace.writes == []


def test_definitions_are_created_once_then_left_alone() -> None:
    workspace = Workspace()
    first = sync_suite(workspace.client(), _defined())
    assert len(first.persona_ids) == 2 and len(first.metric_ids) == 2
    assert len(workspace.items["test-cases"]) == 2
    booking = next(c for c in workspace.items["test-cases"].values() if "Book" in c["input_str"])
    assert (
        booking["expected_output_json"] == {"test_case": "table-free"}
        and "test_case" not in booking
    )
    created = len(workspace.writes)
    assert sync_suite(workspace.client(), _defined()) == first
    assert len(workspace.writes) == created


def test_editing_the_file_patches_what_changed_and_drops_removed_cases() -> None:
    workspace = Workspace()
    sync_suite(workspace.client(), _defined())
    raw = _defined().model_dump()
    raw["personas"][0]["persona_prompt"] = "Speak slowly."
    raw["test_set"]["test_cases"] = raw["test_set"]["test_cases"][:1]
    workspace.writes.clear()
    sync_suite(workspace.client(), Suite.model_validate(raw))
    assert workspace.writes == [
        ("PATCH", "personas", {"persona_prompt": "Speak slowly."}),
        ("DELETE", "test-cases", {}),
    ]


def test_ids_and_definitions_mix_and_keep_their_order() -> None:
    workspace = Workspace()
    workspace.items["metrics"]["EXISTING"] = {"id": "EXISTING"}
    raw = _defined().model_dump()
    raw["metrics"] = ["EXISTING", raw["metrics"][0]]
    ids = sync_suite(workspace.client(), Suite.model_validate(raw))
    assert ids.metric_ids[0] == "EXISTING" and ids.metric_ids[1].startswith("metrics-")


def test_a_name_taken_in_another_workspace_is_explained() -> None:
    workspace = Workspace()
    workspace.taken_elsewhere.add("Front desk")
    with (
        workspace.client() as client,
        pytest.raises(SyncError, match="test set 'Front desk' already exists in another workspace"),
    ):
        sync_suite(client, Suite.model_validate(DEFINED))


def test_a_definition_without_a_name_is_refused() -> None:
    raw = _defined().model_dump()
    del raw["personas"][0]["name"]
    with pytest.raises(SyncError, match="persona defined in coval.json needs a name"):
        sync_suite(Workspace().client(), Suite.model_validate(raw))


def test_a_configured_workspace_is_used_when_the_key_can_see_it() -> None:
    workspace = Workspace()
    assert choose_workspace(workspace.client(), "W0") == WorkspaceChoice("W0")
    assert workspace.writes == []


def test_a_configured_workspace_from_another_organization_says_where_it_came_from() -> None:
    workspace = Workspace()
    with pytest.raises(SyncError, match=r"workspace W9 from coval.json .* has Default \(W0\)"):
        choose_workspace(workspace.client(), "W9", "coval.json")
    assert workspace.writes == []


def test_without_one_the_corral_workspace_is_made_once_then_found() -> None:
    workspace = Workspace()
    client = workspace.client()
    made = choose_workspace(client, None)
    assert made.found_by_name
    assert workspace.items["workspaces"][made.workspace_id]["display_name"] == "Corral"
    assert choose_workspace(client, None) == made
    assert [write for write in workspace.writes if write[1] == "workspaces"] == [
        ("POST", "workspaces", {"display_name": "Corral"})
    ]


def test_two_workspaces_named_corral_is_an_error_not_a_guess() -> None:
    workspace = Workspace()
    for item_id in ("W1", "W2"):
        workspace.items["workspaces"][item_id] = {
            "id": item_id,
            "display_name": "Corral",
            "status": "ACTIVE",
        }
    with pytest.raises(SyncError, match="2 workspaces named Corral; set CORRAL_WORKSPACE_ID"):
        choose_workspace(workspace.client(), None)
