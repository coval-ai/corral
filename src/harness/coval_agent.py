# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The Coval side of a served stack: its WebSocket agent, pushed idempotently, and one run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.config import Settings
from harness.coval_client import CovalClient, SyncError
from harness.stack import AudioPin

MODEL_TYPE = "MODEL_TYPE_WEBSOCKET"
# PATCH replaces metadata wholesale, so it is reconciled as one value.
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
    drift = [key for key in MANAGED if live.get(key) != wanted[key]]
    if drift:
        client.update_agent(agent_id, {key: wanted[key] for key in drift})
    return agent_id


METRIC_SETTINGS = (
    "coval_s2s_latency_metric_id",
    "coval_s2s_instruction_metric_id",
    "coval_s2s_interruption_metric_id",
    "coval_s2s_call_length_metric_id",
)


def metric_ids(settings: Settings) -> list[str]:
    """The board's metrics, the same ids the fetch reads back; a run scores nothing without them."""
    ids = [(getattr(settings, name) or "").strip() for name in METRIC_SETTINGS]
    missing = [name.upper() for name, value in zip(METRIC_SETTINGS, ids, strict=True) if not value]
    if missing:
        raise RuntimeError(f"{', '.join(missing)} unset")
    return ids


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
