# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Coval's HTTP API: the suite and agent records a benchmark needs, and the runs we launch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import httpx

from corral.config import Settings

COVAL_API_BASE = "https://api.coval.dev/v1"
WORKSPACE_HEADER = "X-Coval-Workspace-Id"
TEST_CASE_ATTRIBUTE = "test_case"
_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


class SyncError(RuntimeError):
    pass


class NameTaken(SyncError):
    """Coval refused to create something because the name is already used in the organization."""


def coval_problem(settings: Settings, transport: httpx.BaseTransport | None = None) -> str | None:
    """One authenticated GET to Coval: None when it answered 200, else why not."""
    if settings.coval_api_key is None or not settings.coval_api_key.get_secret_value():
        return "COVAL_API_KEY is unset"
    headers = {"X-API-Key": settings.coval_api_key.get_secret_value()}
    try:
        with httpx.Client(
            base_url=settings.coval_api_base, headers=headers, timeout=_TIMEOUT, transport=transport
        ) as client:
            response = client.get("/workspaces")
    except httpx.HTTPError as error:
        return f"coval is unreachable: {error}"
    if response.status_code != 200:
        return f"coval returned {response.status_code}"
    return None


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
        if response.status_code == 409:
            raise NameTaken(f"{method} {path} -> 409: {response.text[:400]}")
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

    def use_workspace(self, workspace_id: str) -> None:
        """Every later call acts in this workspace."""
        self._client.headers[WORKSPACE_HEADER] = workspace_id

    def list_workspaces(self) -> list[dict[str, Any]]:
        workspaces = self._request("GET", "/workspaces").get("workspaces") or []
        return [workspace for workspace in workspaces if isinstance(workspace, dict)]

    def list_all(self, path: str, key: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Every page of a list endpoint; ``key`` is where each page keeps its items."""
        items: list[dict[str, Any]] = []
        query = {**params, "page_size": 100}
        while True:
            page = self._request("GET", path, params=query)
            items += [item for item in page.get(key) or [] if isinstance(item, dict)]
            token = page.get("next_page_token")
            if not token:
                return items
            query["page_token"] = token

    def exists(self, path: str, item_id: str) -> bool:
        """Whether this workspace has the item. Coval answers 404 for one in another org."""
        response = self._client.get(f"{path}/{item_id}")
        if response.status_code == 404:
            return False
        if response.status_code >= 400:
            raise SyncError(
                f"GET {path}/{item_id} -> {response.status_code}: {response.text[:400]}"
            )
        return True

    def create(self, path: str, key: str, body: dict[str, Any]) -> dict[str, Any]:
        return _unwrap(self._request("POST", path, body), key)

    def update(self, path: str, key: str, item_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return _unwrap(self._request("PATCH", f"{path}/{item_id}", body), key)

    def delete(self, path: str, item_id: str) -> None:
        self._request("DELETE", f"{path}/{item_id}")

    def create_agent(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.create("/agents", "agent", body)

    def update_agent(self, agent_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return self.update("/agents", "agent", agent_id, body)

    def launch_run(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.create("/runs", "run", body)

    def get_run(self, run_id: str) -> dict[str, Any]:
        return _unwrap(self._request("GET", f"/runs/{run_id}"), "run")

    def list_simulations(self, run_id: str) -> list[dict[str, Any]]:
        return self.list_all("/simulations", "simulations", {"filter": f'run_id="{run_id}"'})

    def test_case_key(self, simulation_id: str) -> str | None:
        """The calling test case's ``test_case`` attribute, or None when it has none."""
        path = f"/simulations/{simulation_id}/attributes/{TEST_CASE_ATTRIBUTE}"
        response = self._client.get(path)
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise SyncError(f"GET {path} -> {response.status_code}: {response.text[:400]}")
        value = response.json()
        if not isinstance(value, str):
            raise SyncError(f"the test case's {TEST_CASE_ATTRIBUTE} is {value!r}, not a name")
        return value


def _unwrap(payload: dict[str, Any], key: str) -> dict[str, Any]:
    """Coval wraps one item under its singular name; fall back to the payload if it doesn't."""
    item = payload.get(key)
    return item if isinstance(item, dict) else payload


MODEL_TYPE = "MODEL_TYPE_SDK_WEBSOCKET"
# test_set_ids is additive: every set this agent has ever run stays attached.
MANAGED = ("display_name", "test_set_ids")
TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "DELETED"})


@dataclass(frozen=True)
class CovalTarget:
    """The agent record a benchmark runs against. Coval reaches it through the SDK's
    relay, so the record carries no address: the harness dials in for each call."""

    slug: str
    test_set_id: str
    workspace_id: str

    @property
    def customer_agent_id(self) -> str:
        """Coval wants it unique across the organization, so it carries the workspace. The
        ``sdk`` keeps it clear of the other records this stack has in Coval."""
        return f"{self.slug}--sdk--{self.workspace_id}"

    @property
    def display_name(self) -> str:
        return self.slug

    def target_body(self) -> dict[str, Any]:
        return {
            "display_name": self.display_name,
            "customer_agent_id": self.customer_agent_id,
            "model_type": MODEL_TYPE,
            "test_set_ids": [self.test_set_id],
        }


def sync_target(client: CovalClient, definition: CovalTarget) -> str:
    """Find-or-create the agent and attach the suite's test set; returns its id."""
    wanted = definition.target_body()
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
    seed: int | None = None,
) -> dict[str, Any]:
    """``sample`` test cases, picked at random or by ``seed``: one seed, one pick."""
    options: dict[str, Any] = {"iteration_count": 1, "concurrency": 1}
    if sample:
        options["sub_sample_size"] = sample
        if seed is not None:
            options["sub_sample_seed"] = seed
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
    progress = run.get("progress") or {}
    lines = [f"run {run_id_of(run)} {run.get('status')}"]
    failed, total = progress.get("failed_test_cases"), progress.get("total_test_cases")
    if failed:
        lines.append(f"  {failed} of {total} calls failed")
    for metric in (results.get("metrics") or {}).values():
        values = [entry.get("value") for entry in metric.get("values") or []]
        lines.append(f"  {metric.get('metric_name')}: {', '.join(str(v) for v in values)}")
    return lines


def run_id_of(run: dict[str, Any]) -> str:
    return str(run.get("run_id") or run.get("id"))


def is_terminal(run: dict[str, Any]) -> bool:
    return run.get("status") in TERMINAL_STATUSES


def run_succeeded(run: dict[str, Any]) -> bool:
    """Completed with every test case through: Coval calls a run with failed calls COMPLETED too."""
    progress = run.get("progress") or {}
    return run.get("status") == "COMPLETED" and not progress.get("failed_test_cases")
