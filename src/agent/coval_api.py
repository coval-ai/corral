# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Coval's HTTP API: the agent record we point at the served stack, and the runs we launch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import httpx

from agent.stack import AudioPin

COVAL_API_BASE = "https://api.coval.dev/v1"
_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


class SyncError(RuntimeError):
    pass


class CovalClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = COVAL_API_BASE,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url, headers={"X-API-Key": api_key}, timeout=_TIMEOUT, transport=transport
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self._client.close()

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self._client.request(method, path, json=payload, params=params)
        if response.status_code >= 400:
            raise SyncError(f"{method} {path} -> {response.status_code}: {response.text[:400]}")
        if not response.content:
            return {}
        parsed: dict[str, Any] = response.json()
        return parsed

    def find_agent(self, customer_agent_id: str) -> dict[str, Any] | None:
        payload = self._request(
            "GET",
            "/agents",
            params={"filter": f'customer_agent_id="{customer_agent_id}"', "page_size": 1},
        )
        for agent in payload.get("agents") or []:
            if isinstance(agent, dict) and agent.get("customer_agent_id") == customer_agent_id:
                return agent
        return None

    def create_agent(self, body: dict[str, Any]) -> dict[str, Any]:
        payload = self._request("POST", "/agents", body)
        agent = payload.get("agent")
        return agent if isinstance(agent, dict) else payload

    def update_agent(self, agent_id: str, body: dict[str, Any]) -> dict[str, Any]:
        payload = self._request("PATCH", f"/agents/{agent_id}", body)
        agent = payload.get("agent")
        return agent if isinstance(agent, dict) else payload

    def launch_run(self, body: dict[str, Any]) -> dict[str, Any]:
        payload = self._request("POST", "/runs", body)
        run = payload.get("run")
        return run if isinstance(run, dict) else payload

    def get_run(self, run_id: str) -> dict[str, Any]:
        payload = self._request("GET", f"/runs/{run_id}")
        run = payload.get("run")
        return run if isinstance(run, dict) else payload


MODEL_TYPE = "MODEL_TYPE_WEBSOCKET"
# PATCH replaces metadata wholesale, so it is reconciled as one value. test_set_ids
# is additive: every set this agent has ever run stays attached.
MANAGED = ("display_name", "metadata", "test_set_ids")
TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "DELETED"})


@dataclass(frozen=True)
class CovalS2SAgent:
    slug: str
    url: str
    token: str | None
    test_set_id: str

    @property
    def customer_agent_id(self) -> str:
        return f"benchmarks-s2s-{self.slug}"

    @property
    def display_name(self) -> str:
        return f"Benchmarks: S2S {self.slug}"

    def agent_body(self, audio: AudioPin) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "connection_mode": "direct",
            "endpoint": "wss" + self.url.removeprefix("https").removeprefix("http") + "/ws",
            "custom_headers": {"X-Coval-Simulation-Id": "{{simulation_id}}"},
            "handshake_ready_message_type": "session_ready",
            "handshake_requires_session_id": False,
            "handshake_timeout_seconds": 30,
            "message_type_path": "type",
            "send_audio_template": "{{audio_data}}",
            "audio_encoding": "pcm",
            "send_sample_rate_hertz": audio.in_sample_rate_hz,
            "receive_sample_rate_hertz": audio.out_sample_rate_hz,
            "receive_audio_channels": 1,
            "pace_inbound_binary_audio": True,
            "speech_activity_enabled": True,
        }
        if self.token:
            metadata["authorization_header"] = f"Bearer {self.token}"
        return {
            "display_name": self.display_name,
            "customer_agent_id": self.customer_agent_id,
            "model_type": MODEL_TYPE,
            "metadata": metadata,
            "test_set_ids": [self.test_set_id],
        }


def sync_agent(client: CovalClient, definition: CovalS2SAgent, audio: AudioPin) -> str:
    """Find-or-create the agent and point it at this serving; returns its id."""
    wanted = definition.agent_body(audio)
    live = client.find_agent(definition.customer_agent_id)
    if live is None:
        return str(client.create_agent(wanted)["id"])
    if live.get("model_type") != MODEL_TYPE:
        raise SyncError(
            f"coval agent {live.get('id')} is {live.get('model_type')!r}, not {MODEL_TYPE}; "
            "model_type cannot be patched, so this record is not ours to reconcile"
        )
    agent_id = str(live["id"])
    attached = [str(item) for item in live.get("test_set_ids") or []]
    if definition.test_set_id not in attached:
        attached.append(definition.test_set_id)
    wanted["test_set_ids"] = attached
    drift = [key for key in MANAGED if live.get(key) != wanted[key]]
    if drift:
        client.update_agent(agent_id, {key: wanted[key] for key in drift})
    return agent_id


def launch_body(
    agent_id: str,
    persona_id: str,
    test_set_id: str,
    metrics: list[str],
    *,
    sample: int,
    display_name: str,
) -> dict[str, Any]:
    options: dict[str, Any] = {"iteration_count": 1, "concurrency": 1}
    if sample:
        options["sub_sample_size"] = sample
    return {
        "agent_id": agent_id,
        "persona_id": persona_id,
        "test_set_id": test_set_id,
        "metric_ids": metrics,
        "options": options,
        "metadata": {"display_name": display_name},
    }


def results_lines(run: dict[str, Any]) -> list[str]:
    """One line per metric: its name and every simulation's value, in output order."""
    results = run.get("results") or {}
    lines = [f"run {run_id_of(run)} {run.get('status')}"]
    for metric in (results.get("metrics") or {}).values():
        values = [entry.get("value") for entry in metric.get("values") or []]
        lines.append(f"  {metric.get('metric_name')}: {', '.join(str(v) for v in values)}")
    return lines


def run_id_of(run: dict[str, Any]) -> str:
    return str(run.get("run_id") or run.get("id"))


def is_terminal(run: dict[str, Any]) -> bool:
    return run.get("status") in TERMINAL_STATUSES
