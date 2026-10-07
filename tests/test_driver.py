# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest

from agent import driver, serve
from agent.config import Settings
from agent.services import Services
from agent.stack import load_stack
from tests.test_serve import _Echo, _Passthrough


@pytest.fixture
async def url() -> AsyncIterator[str]:
    stack = load_stack("baseline")
    fakes = Services(stt=_Echo(), llm=_Passthrough(), tts=_Passthrough())
    settings = Settings(_env_file=None, s2s_agent_token="s3cret")  # noqa: S106
    with (
        patch.object(serve, "resolve", return_value=fakes),
        patch.object(serve, "get_settings", return_value=settings),
    ):
        server, task, port = await driver.serve_in_process(serve.create_app(stack), 0)
        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            await task


@pytest.mark.asyncio
async def test_probe_reads_health_then_hears_session_ready(url: str) -> None:
    health = await driver.probe(url, "s3cret")
    assert health["stack"] == "deepgram-nova-3--openai-gpt-4.1--elevenlabs-eleven-flash-v2"


@pytest.mark.asyncio
async def test_probe_fails_on_a_wrong_bearer(url: str) -> None:
    with pytest.raises(Exception, match="403"):
        await driver.probe(url, "nope")


def test_the_tunnel_url_is_read_out_of_cloudflared_log_noise() -> None:
    line = "2026-10-02T21:33:43Z INF |  https://quiet-hill-abc.trycloudflare.com    |"
    found = driver.TUNNEL_URL.search(line)
    assert found and found.group(0) == "https://quiet-hill-abc.trycloudflare.com"
