# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import httpx
import pytest

from corral.config import Settings
from corral.coval_api import (
    CovalClient,
    CovalTarget,
    SyncError,
    coval_problem,
    is_terminal,
    launch_body,
    results_lines,
    run_succeeded,
    sync_target,
)

DEFINITION = CovalTarget("cascade-x", "TSBANK", "W1")


def _client(state: dict[str, Any]) -> CovalClient:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/v1")
        body: dict[str, Any] = json.loads(request.content) if request.content else {}
        if request.method == "GET" and path == "/agents":
            wanted = request.url.params["filter"].split('"')[1]
            agents = [a for a in state["agents"] if a.get("customer_agent_id") == wanted]
            return httpx.Response(200, json={"agents": agents})
        if request.method == "GET" and path == "/simulations":
            state["filters"].append(request.url.params["filter"])
            return httpx.Response(200, json={"simulations": [{"simulation_id": "S1"}]})
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


def test_the_target_is_an_sdk_agent_the_harness_dials_in_for() -> None:
    body = DEFINITION.target_body()
    assert body["customer_agent_id"] == "cascade-x--sdk--W1"
    assert body["model_type"] == "MODEL_TYPE_SDK_WEBSOCKET"
    assert body["test_set_ids"] == ["TSBANK"]
    assert "metadata" not in body


def test_the_calling_test_case_is_asked_for_by_simulation_id() -> None:
    attributes = {"sim-booked": "already-booked", "sim-odd": {"seed": 1}}

    def handler(request: httpx.Request) -> httpx.Response:
        sim, _, name = request.url.path.removeprefix("/v1/simulations/").split("/")
        assert (request.method, name) == ("GET", "test_case")
        if sim not in attributes:
            return httpx.Response(404, json={"error": {"message": "no such attribute"}})
        return httpx.Response(200, json=attributes[sim])

    client = CovalClient("coval-key", "https://api.coval.dev/v1", httpx.MockTransport(handler))
    assert client.test_case_key("sim-booked") == "already-booked"
    assert client.test_case_key("sim-plain") is None
    with pytest.raises(SyncError, match="not a name"):
        client.test_case_key("sim-odd")


def test_sync_creates_once_then_attaches_new_test_sets() -> None:
    state: dict[str, Any] = {"agents": [], "writes": []}
    client = _client(state)
    assert sync_target(client, DEFINITION) == "A" * 22
    assert sync_target(client, DEFINITION) == "A" * 22
    sync_target(client, CovalTarget("cascade-x", "TSHARD", "W1"))
    (_, _, created), (_, _, attached) = state["writes"]
    assert created["customer_agent_id"] == "cascade-x--sdk--W1"
    assert attached == {"test_set_ids": ["TSBANK", "TSHARD"]}


def test_a_run_lists_its_simulations() -> None:
    state: dict[str, Any] = {"filters": []}
    client = _client(state)
    assert client.list_simulations("R1") == [{"simulation_id": "S1"}]
    assert state["filters"] == ['run_id="R1"']


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


def test_coval_answers_for_its_key_before_anything_runs() -> None:
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


def test_a_completed_run_with_a_failed_call_is_not_a_success() -> None:
    failed = {
        "run_id": "R2",
        "status": "COMPLETED",
        "progress": {"total_test_cases": 1, "failed_test_cases": 1},
    }
    assert not run_succeeded(failed)
    assert run_succeeded({"status": "COMPLETED", "progress": {"failed_test_cases": 0}})
    assert results_lines(failed) == ["run R2 COMPLETED", "  1 of 1 calls failed"]
