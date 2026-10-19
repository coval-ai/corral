# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import patch

import pytest

from corral import benchmark, pipeline, serve
from corral.agent import load_cascade_agent
from corral.config import Settings
from corral.coval_api import CovalClient
from corral.models import CascadeServices
from corral.scenarios import require_suite
from corral.suite import SuiteIds, WorkspaceChoice
from tests.test_serve import _Echo, _Passthrough

REFERENCE = ("deepgram/nova-3", "openai/gpt-4.1", "elevenlabs/eleven_flash_v2")


@pytest.fixture
async def url() -> AsyncIterator[str]:
    agent = load_cascade_agent("ultrabank", *REFERENCE)
    fakes = CascadeServices(stt=_Echo(), llm=_Passthrough(), tts=_Passthrough())
    settings = Settings(_env_file=None, s2s_agent_token="s3cret")  # noqa: S106
    with (
        patch.object(pipeline, "resolve", return_value=fakes),
        patch.object(serve, "get_settings", return_value=settings),
    ):
        server, task, port = await benchmark.serve_in_process(serve.create_app(agent), 0)
        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            await task


async def test_probe_reads_health_behind_the_bearer(url: str) -> None:
    health = await benchmark.probe(url, "s3cret")
    assert health["agent"] == "deepgram-nova-3--openai-gpt-4.1--elevenlabs-eleven-flash-v2"
    with pytest.raises(Exception, match="403"):
        await benchmark.probe(url, "nope")


def test_the_tunnel_url_is_read_out_of_cloudflared_log_noise() -> None:
    line = "2026-10-02T21:33:43Z INF |  https://quiet-hill-abc.trycloudflare.com    |"
    found = benchmark.TUNNEL_URL.search(line)
    assert found and found.group(0) == "https://quiet-hill-abc.trycloudflare.com"


class _Runs:
    """Launches runs that finish on their second poll."""

    def __init__(self) -> None:
        self.launched: list[dict[str, Any]] = []
        self.polls: dict[str, int] = {}

    def use_workspace(self, workspace_id: str) -> None:
        self.workspace_id = workspace_id

    def launch_run(self, body: dict[str, Any]) -> dict[str, Any]:
        assert all(self.polls.get(f"R{n}", 0) > 1 for n in range(1, len(self.launched) + 1))
        self.launched.append(body)
        return {"run_id": f"R{len(self.launched)}", "status": "PENDING"}

    def get_run(self, run_id: str) -> dict[str, Any]:
        self.polls[run_id] = self.polls.get(run_id, 0) + 1
        return {"run_id": run_id, "status": "COMPLETED" if self.polls[run_id] > 1 else "RUNNING"}


async def test_each_persona_gets_its_own_run_after_the_last_one_finishes() -> None:
    runs = _Runs()
    ids = SuiteIds("TS", ["P1", "P2"], ["M1"])
    request = benchmark.RunRequest(require_suite("ultrabank"), None, sample=1)
    with (
        patch.object(benchmark, "choose_workspace", return_value=WorkspaceChoice("W1", True)),
        patch.object(benchmark, "remember", return_value=True) as remembered,
        patch.object(benchmark, "sync_suite", return_value=ids),
        patch.object(benchmark, "sync_target", return_value="A"),
        patch.object(benchmark, "POLL_SECS", 0),
    ):
        agent = load_cascade_agent("ultrabank", *REFERENCE)
        finals = await benchmark.run_on_coval(
            cast(CovalClient, runs), agent, "https://x.trycloudflare.com", None, request
        )
    assert runs.workspace_id == "W1"
    remembered.assert_called_once_with("CORRAL_WORKSPACE_ID", "W1")
    assert [body["persona_id"] for body in runs.launched] == ["P1", "P2"]
    assert all(
        body["test_set_id"] == "TS" and body["metric_ids"] == ["M1"] for body in runs.launched
    )
    assert [final["status"] for final in finals] == ["COMPLETED", "COMPLETED"]
