# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import google.auth
import httpx
import pytest
from pydantic import ValidationError

from corral.assets import (
    AssetRun,
    AssetsError,
    agent_prompt,
    load_asset_run,
    load_manifest,
    resolve,
    resolve_assets,
)
from corral.config import Settings
from corral.scenarios import Suite

MANIFEST = {
    "workflow-bank": {
        "v1": {
            "workspace_id": "W1",
            "test_set_id": "TS1",
            "personas": {"clean": {"id": "P-clean", "version": "v1"}, "low": {"id": "P-low"}},
            "metric_ids": {"judge": "M-judge", "latency": "M-latency"},
            "schedules": {"clean": {"enabled": True}},
        }
    }
}


def test_a_suite_names_its_assets_or_pins_ids_but_not_both() -> None:
    assert Suite(assets="workflow-bank.v1").personas == []
    with pytest.raises(ValidationError, match="drop them from coval.json"):
        Suite(assets="workflow-bank.v1", test_set="TS1")
    with pytest.raises(ValidationError, match="names from the manifest"):
        Suite(assets="workflow-bank.v1", personas=[{"name": "p"}])
    with pytest.raises(ValidationError, match="needs a test_set"):
        Suite(personas=["p1"])
    with pytest.raises(ValidationError, match="at least one persona"):
        Suite(test_set="TS1")


def test_assets_resolve_to_the_first_persona_and_every_metric_unless_told_all() -> None:
    resolved = resolve(Suite(assets="workflow-bank.v1", sample=2), MANIFEST, "map.json")
    assert resolved == Suite(
        workspace="W1",
        test_set="TS1",
        personas=["P-clean"],
        metrics=["M-judge", "M-latency"],
        sample=2,
    )
    every = resolve(Suite(assets="workflow-bank.v1", personas=["all"]), MANIFEST, "map.json")
    assert every.personas == ["P-clean", "P-low"]


def test_assets_pick_the_named_personas_and_metrics_in_the_order_asked() -> None:
    suite = Suite(assets="workflow-bank.v1", personas=["low", "clean"], metrics=["latency"])
    resolved = resolve(suite, MANIFEST, "map.json")
    assert resolved.personas == ["P-low", "P-clean"]
    assert resolved.metrics == ["M-latency"]


@pytest.mark.parametrize(
    ("suite", "message"),
    [
        (Suite(assets="bank.v1"), "no scenario 'bank' in map.json; it has workflow-bank"),
        (Suite(assets="workflow-bank.v9"), "no version 'v9' in map.json; it has v1"),
        (Suite(assets="workflow-bank.v1", personas=["noisy"]), "no persona noisy .* clean, low"),
        (Suite(assets="workflow-bank.v1", metrics=["wer"]), "no metric wer .* judge, latency"),
    ],
)
def test_an_unknown_asset_name_says_what_the_manifest_has(suite: Suite, message: str) -> None:
    with pytest.raises(AssetsError, match=message):
        resolve(suite, MANIFEST, "map.json")


def test_a_malformed_entry_is_named() -> None:
    broken = {"workflow-bank": {"v1": {"workspace_id": "W1"}}}
    with pytest.raises(AssetsError, match="workflow-bank.v1 in map.json is malformed"):
        resolve(Suite(assets="workflow-bank.v1"), broken, "map.json")


def test_a_suite_with_ids_is_left_alone_and_assets_need_a_manifest(tmp_path: Path) -> None:
    pinned = Suite(test_set="TS", personas=["p1"])
    assert resolve_assets(pinned, Settings(corral_assets_uri=None)) is pinned
    with pytest.raises(AssetsError, match="set CORRAL_ASSETS_URI"):
        resolve_assets(Suite(assets="workflow-bank.v1"), Settings(corral_assets_uri=None))
    path = tmp_path / "map.json"
    path.write_text(json.dumps(MANIFEST))
    resolved = resolve_assets(
        Suite(assets="workflow-bank.v1"), Settings(corral_assets_uri=str(path))
    )
    assert resolved.test_set == "TS1"


def test_a_local_manifest_must_exist_and_be_a_json_object(tmp_path: Path) -> None:
    with pytest.raises(AssetsError, match="doesn't exist"):
        load_manifest(str(tmp_path / "missing.json"))
    listed = tmp_path / "list.json"
    listed.write_text("[]")
    with pytest.raises(AssetsError, match="isn't a JSON object"):
        load_manifest(str(listed))


@pytest.fixture
def signed_in(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = SimpleNamespace(token="t0k", refresh=lambda _request: None)  # noqa: S106
    monkeypatch.setattr(google.auth, "default", lambda scopes: (credentials, "project"))


def test_a_gcs_manifest_is_read_with_the_callers_google_credentials(signed_in: None) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=MANIFEST)

    manifest = load_manifest("gs://bucket/org/scenario_map.json", httpx.MockTransport(handler))
    assert manifest == MANIFEST
    (request,) = seen
    assert request.url.raw_path.split(b"?")[0] == b"/storage/v1/b/bucket/o/org%2Fscenario_map.json"
    assert request.url.params["alt"] == "media"
    assert request.headers["Authorization"] == "Bearer t0k"


def test_a_gcs_manifest_the_account_cant_read_says_which_role_it_needs(signed_in: None) -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(403))
    with pytest.raises(AssetsError, match="roles/storage.objectViewer on bucket"):
        load_manifest("gs://bucket/map.json", transport)
    with pytest.raises(AssetsError, match="gs://bucket/object"):
        load_manifest("gs://bucket", transport)


AGENTS = {
    "openai/gpt-realtime-2": {"id": "A-rt2", "registry_model": "gpt-realtime"},
    "google/gemini-3.8-live": {"id": "A-gem"},
}


def _session(instructions: str) -> str:
    return json.dumps({"type": "session.update", "session": {"instructions": instructions}})


@pytest.mark.parametrize(
    ("record", "prompt"),
    [
        (
            {"metadata": {"initialization_json": _session(" Be Ultra Bank Bot. ")}},
            "Be Ultra Bank Bot.",
        ),
        ({"metadata": {"initialization_json": f"[{_session('First frame.')}]"}}, "First frame."),
        ({"metadata": {"gemini_realtime_instructions": "Gemini prompt."}}, "Gemini prompt."),
        ({"metadata": {"initialization_json": "{not json"}}, None),
        ({"metadata": {}}, None),
        ({}, None),
    ],
)
def test_an_agents_prompt_is_read_where_coval_keeps_it(
    record: dict[str, object], prompt: str | None
) -> None:
    assert agent_prompt(record) == prompt


class FakeCoval:
    def __init__(self, records: dict[str, dict[str, object]]) -> None:
        self.records = records
        self.workspace: str | None = None

    def use_workspace(self, workspace_id: str) -> None:
        self.workspace = workspace_id

    def get_agent(self, agent_id: str) -> dict[str, object]:
        return self.records[agent_id]


def _asset_run(
    tmp_path: Path,
    model_id: str,
    prompt_from: str | None = None,
    records: dict[str, dict[str, object]] | None = None,
) -> tuple[AssetRun, FakeCoval]:
    manifest = json.loads(json.dumps(MANIFEST))
    manifest["workflow-bank"]["v1"]["agents"] = AGENTS
    path = tmp_path / "map.json"
    path.write_text(json.dumps(manifest))
    coval = FakeCoval(
        {
            "A-rt2": {"metadata": {"initialization_json": _session("RT2 prompt.")}},
            "A-gem": {"metadata": {"gemini_realtime_instructions": "Gemini prompt."}},
            **(records or {}),
        }
    )
    run = load_asset_run(
        "workflow-bank.v1",
        personas=["clean"],
        metrics=[],
        model_id=model_id,
        prompt_from=prompt_from,
        settings=Settings(corral_assets_uri=str(path)),
        client=coval,
    )
    return run, coval


def test_an_asset_run_takes_the_prompt_of_the_agent_for_its_model(tmp_path: Path) -> None:
    run, coval = _asset_run(tmp_path, "openai/gpt-realtime-2")
    assert run.system_prompt == "RT2 prompt."
    assert run.suite.personas == ["P-clean"] and run.suite.test_set == "TS1"
    assert coval.workspace == "W1"
    assert run.source.endswith("map.json (workflow-bank.v1)")


def test_a_model_the_scenario_has_no_agent_for_borrows_the_first_or_prompt_from(
    tmp_path: Path,
) -> None:
    run, _ = _asset_run(tmp_path, "openai/gpt-realtime-2.1")
    assert run.system_prompt == "RT2 prompt."
    run, _ = _asset_run(tmp_path, "openai/gpt-realtime-2.1", "google/gemini-3.8-live")
    assert run.system_prompt == "Gemini prompt."
    with pytest.raises(AssetsError, match="no agent xai/grok-voice-3.0 in"):
        _asset_run(tmp_path, "openai/gpt-realtime-2.1", "xai/grok-voice-3.0")


def test_an_agent_without_a_saved_prompt_or_a_missing_manifest_stops_the_run(
    tmp_path: Path,
) -> None:
    with pytest.raises(AssetsError, match="has no system prompt saved in Coval"):
        _asset_run(tmp_path, "openai/gpt-realtime-2", records={"A-rt2": {"metadata": {}}})
    with pytest.raises(AssetsError, match="needs CORRAL_ASSETS_URI"):
        load_asset_run(
            "workflow-bank.v1",
            personas=[],
            metrics=[],
            model_id="openai/gpt-realtime-2",
            prompt_from=None,
            settings=Settings(corral_assets_uri=None),
            client=FakeCoval({}),
        )
