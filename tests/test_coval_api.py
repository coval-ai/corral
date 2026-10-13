# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import httpx

from corral.coval_api import (
    CovalClient,
    CovalTarget,
    is_terminal,
    launch_body,
    results_lines,
    sync_target,
)

DEFINITION = CovalTarget("cascade-x", "https://quiet-hill.trycloudflare.com", "s3cret", "TSBANK")


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
    assert body["customer_agent_id"] == "cascade-x"
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
    sync_target(client, CovalTarget("cascade-x", other, "s3cret", "TSBANK"))
    sync_target(client, CovalTarget("cascade-x", other, "s3cret", "TSHARD"))
    (_, _, created), (_, _, moved), (_, _, attached) = state["writes"]
    assert created["customer_agent_id"] == "cascade-x"
    assert moved == {"metadata": {**created["metadata"], "endpoint": "wss://" + other[8:] + "/ws"}}
    assert attached == {"test_set_ids": ["TSBANK", "TSHARD"]}


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
