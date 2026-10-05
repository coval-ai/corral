# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The slice of Coval's public API the harness uses: agents and runs."""

from __future__ import annotations

from typing import Any, Self

import httpx

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
