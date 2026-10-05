# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import httpx
import pytest

from agent import coval_agent
from agent.coval_agent import (
    CovalS2SAgent,
    is_terminal,
    launch_body,
    results_lines,
    sync_agent,
)
from agent.coval_client import CovalClient
from agent.stack import AudioPin
from agent.testset import load_test_set, load_test_sets

AUDIO = AudioPin(in_sample_rate_hz=16000, out_sample_rate_hz=24000)
DEFINITION = CovalS2SAgent("cascade-x", "https://quiet-hill.trycloudflare.com", "s3cret", "TSBANK")


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


def test_the_agent_speaks_raw_pcm_and_waits_for_session_ready() -> None:
    body = DEFINITION.agent_body(AUDIO)
    metadata = body["metadata"]
    assert body["customer_agent_id"] == "benchmarks-s2s-cascade-x"
    assert body["model_type"] == coval_agent.MODEL_TYPE
    assert metadata["endpoint"] == "wss://quiet-hill.trycloudflare.com/ws"
    assert metadata["send_audio_template"] == "{{audio_data}}"
    assert metadata["handshake_ready_message_type"] == "session_ready"
    assert metadata["handshake_requires_session_id"] is False
    assert metadata["authorization_header"] == "Bearer s3cret"
    assert (metadata["send_sample_rate_hertz"], metadata["receive_sample_rate_hertz"]) == (
        16000,
        24000,
    )
    assert body["test_set_ids"] == ["TSBANK"]


def test_sync_creates_once_then_patches_only_the_new_url() -> None:
    state: dict[str, Any] = {"agents": [], "writes": []}
    client = _client(state)
    assert sync_agent(client, DEFINITION, AUDIO) == "A" * 22
    assert sync_agent(client, DEFINITION, AUDIO) == "A" * 22
    moved = CovalS2SAgent("cascade-x", "https://other-hill.trycloudflare.com", "s3cret", "TSBANK")
    assert sync_agent(client, moved, AUDIO) == "A" * 22
    methods = [(method, path) for method, path, _ in state["writes"]]
    assert methods == [("POST", "/agents"), ("PATCH", "/agents/" + "A" * 22)]
    patch = state["writes"][1][2]
    assert set(patch) == {"metadata"}
    assert patch["metadata"]["endpoint"] == "wss://other-hill.trycloudflare.com/ws"


def test_sync_attaches_a_new_test_set_without_detaching_the_old() -> None:
    state: dict[str, Any] = {"agents": [], "writes": []}
    client = _client(state)
    sync_agent(client, DEFINITION, AUDIO)
    challenge = CovalS2SAgent("cascade-x", DEFINITION.url, "s3cret", "TSHARD")
    sync_agent(client, challenge, AUDIO)
    sync_agent(client, DEFINITION, AUDIO)
    patches = [body for method, _, body in state["writes"] if method == "PATCH"]
    assert patches == [{"test_set_ids": ["TSBANK", "TSHARD"]}]


def test_launch_carries_the_boards_metrics_at_concurrency_one() -> None:
    body = launch_body("A" * 22, "P" * 22, "TSBANK", ["M1", "M2"], sample=0, display_name="S2S x")
    assert body["options"] == {"iteration_count": 1, "concurrency": 1}
    assert body["metric_ids"] == ["M1", "M2"]
    assert body["metadata"] == {"display_name": "S2S x"}
    sampled = launch_body("A" * 22, "P" * 22, "TSBANK", [], sample=2, display_name="x")
    assert sampled["options"]["sub_sample_size"] == 2
    assert not is_terminal({"status": "RUNNING"})
    assert is_terminal({"status": "COMPLETED"})


def test_the_scenario_names_its_test_sets_with_their_coval_ids() -> None:
    sets = load_test_sets("bank")
    assert "baseline" in sets
    baseline = load_test_set("bank", "baseline")
    assert baseline.test_set_id and baseline.persona_id and baseline.metric_ids
    assert baseline.summary
    with pytest.raises(ValueError, match="no test set 'nope' in bank; have baseline"):
        load_test_set("bank", "nope")


def test_results_print_one_line_per_metric() -> None:
    run = {
        "run_id": "R1",
        "status": "COMPLETED",
        "results": {
            "metrics": {
                "m1": {"metric_name": "Turn Count", "values": [{"value": 8.0}]},
                "m2": {"metric_name": "Validate Expected Behaviors", "values": [{"value": 1.0}]},
            }
        },
    }
    assert results_lines(run) == [
        "run R1 COMPLETED",
        "  Turn Count: 8.0",
        "  Validate Expected Behaviors: 1.0",
    ]
