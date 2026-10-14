# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import httpx

from corral.config import Settings
from corral.coval_api import (
    CovalClient,
    CovalTarget,
    coval_problem,
    is_terminal,
    launch_body,
    results_lines,
    sync_target,
)

URL = "https://quiet-hill.trycloudflare.com"
DEFINITION = CovalTarget("cascade-x", URL, "s3cret", "TSBANK", "W1")


def _client(state: dict[str, Any]) -> CovalClient:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/v1")
        body: dict[str, Any] = json.loads(request.content) if request.content else {}
        if request.method == "GET" and path == "/agents":
            wanted = request.url.params["filter"].split('"')[1]
            agents = [a for a in state["agents"] if a.get("customer_agent_id") == wanted]
            return httpx.Response(200, json={"agents": agents})
        state["writes"].append((request.method, path, body))
        if path == "/agents":
            created = {**body, "id": "A" * 22}
            state["agents"].append(created)
            return httpx.Response(200, json={"agent": created})
        if path.startswith("/agents/"):
            state["agents"][0].update(body)
            return httpx.Response(200, json={"agent": state["agents"][0]})
        return httpx.Response(400, json={"error": {"message": path}})

    return CovalClient("coval-key", "https://api.coval.dev/v1", httpx.MockTransport(handler))


def test_the_target_speaks_raw_pcm_and_waits_for_session_ready() -> None:
    body = DEFINITION.target_body()
    metadata = body["metadata"]
    assert body["customer_agent_id"] == "cascade-x--W1"
    assert body["test_set_ids"] == ["TSBANK"]
    assert metadata["endpoint"] == "wss://quiet-hill.trycloudflare.com/ws"
    assert metadata["authorization_header"] == "Bearer s3cret"
    assert metadata["handshake_ready_message_type"] == "session_ready"
    assert metadata["speech_activity_enabled"] is False
    assert (metadata["send_sample_rate_hertz"], metadata["receive_sample_rate_hertz"]) == (
        16000,
        24000,
    )


def test_sync_creates_once_then_patches_only_what_changed() -> None:
    state: dict[str, Any] = {"agents": [], "writes": []}
    client = _client(state)
    assert sync_target(client, DEFINITION) == "A" * 22
    assert sync_target(client, DEFINITION) == "A" * 22
    other = "https://other-hill.trycloudflare.com"
    sync_target(client, CovalTarget("cascade-x", other, "s3cret", "TSBANK", "W1"))
    sync_target(client, CovalTarget("cascade-x", other, "s3cret", "TSHARD", "W1"))
    (_, _, created), (_, _, moved), (_, _, attached) = state["writes"]
    assert created["customer_agent_id"] == "cascade-x--W1"
    assert moved == {"metadata": {**created["metadata"], "endpoint": "wss://" + other[8:] + "/ws"}}
    assert attached == {"test_set_ids": ["TSBANK", "TSHARD"]}


def test_an_agent_made_before_workspaces_is_found_under_its_bare_slug() -> None:
    old = {**DEFINITION.target_body(), "customer_agent_id": "cascade-x", "id": "OLD"}
    state: dict[str, Any] = {"agents": [old], "writes": []}
    assert sync_target(_client(state), DEFINITION) == "OLD"
    assert [method for method, _, _ in state["writes"]] == []


def test_launch_runs_the_boards_metrics_one_call_at_a_time() -> None:
    body = launch_body("A" * 22, "P" * 22, "TSBANK", ["M1", "M2"], sample=0, display_name="S2S x")
    assert body["options"] == {"iteration_count": 1, "concurrency": 1}
    assert body["metric_ids"] == ["M1", "M2"]
    sampled = launch_body("A" * 22, "P" * 22, "TSBANK", [], sample=2, display_name="x")
    assert sampled["options"]["sub_sample_size"] == 2
    assert not is_terminal({"status": "RUNNING"}) and is_terminal({"status": "COMPLETED"})


def test_results_print_one_line_per_metric() -> None:
    run = {
        "run_id": "R1",
        "status": "COMPLETED",
        "results": {"metrics": {"m1": {"metric_name": "Turn Count", "values": [{"value": 8.0}]}}},
    }
    assert results_lines(run) == ["run R1 COMPLETED", "  Turn Count: 8.0"]


def test_coval_answers_for_its_key_before_anything_is_served() -> None:
    settings = Settings(_env_file=None, coval_api_key="ck")
    seen: list[httpx.Request] = []

    def ok(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"workspaces": []})

    assert coval_problem(settings, httpx.MockTransport(ok)) is None
    assert seen[0].url == "https://api.coval.dev/v1/workspaces"
    assert seen[0].headers["X-API-Key"] == "ck"
    denied = httpx.MockTransport(lambda _request: httpx.Response(401))
    assert coval_problem(settings, denied) == "coval returned 401"
    assert coval_problem(Settings(_env_file=None), denied) == "COVAL_API_KEY is unset"
