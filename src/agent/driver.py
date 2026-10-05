# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""One benchmark: serve in-process behind a quick tunnel, point Coval at it, run, exit."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

import httpx
import structlog
import uvicorn
from fastapi import FastAPI
from websockets.asyncio.client import connect

from agent.config import Settings, get_settings
from agent.coval import SESSION_READY
from agent.coval_agent import (
    CovalS2SAgent,
    is_terminal,
    launch_body,
    results_lines,
    run_id_of,
    sync_agent,
)
from agent.coval_client import CovalClient
from agent.serve import create_app
from agent.stack import LoadedStack
from agent.testset import TestSet

logger = structlog.get_logger("agent")

PROBE_TIMEOUT_SECS = 30
REACHABLE_TIMEOUT_SECS = 180  # a quick tunnel's hostname takes about 80 s to resolve
POLL_SECS = 30
TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


async def serve_in_process(
    app: FastAPI, port: int
) -> tuple[uvicorn.Server, asyncio.Task[None], int]:
    """Start uvicorn on the loop we are on; returns the server, its task and the bound port."""
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(server.serve(), name="s2s-serve")
    while not server.started:
        if task.done():
            task.result()
            raise RuntimeError("server exited before starting")
        await asyncio.sleep(0.05)
    bound: int = server.servers[0].sockets[0].getsockname()[1]
    return server, task, bound


async def probe(url: str, token: str | None) -> dict[str, Any]:
    """healthz over HTTP, then one authenticated call that must answer session_ready first."""
    deadline = asyncio.get_running_loop().time() + REACHABLE_TIMEOUT_SECS
    async with httpx.AsyncClient(timeout=10) as http:
        while True:
            try:
                response = await http.get(f"{url}/healthz")
                response.raise_for_status()
                break
            except httpx.HTTPError:
                if asyncio.get_running_loop().time() > deadline:
                    raise
                await asyncio.sleep(2)
        health: dict[str, Any] = response.json()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    ws_url = "ws" + url.removeprefix("http") + "/ws"
    async with connect(ws_url, additional_headers=headers, open_timeout=PROBE_TIMEOUT_SECS) as ws:
        first = await asyncio.wait_for(ws.recv(), PROBE_TIMEOUT_SECS)
        if not isinstance(first, str) or json.loads(first) != SESSION_READY:
            raise RuntimeError(f"expected session_ready first, got {first!r}")
    return health


async def open_tunnel(port: int) -> tuple[asyncio.subprocess.Process, str]:
    """A Cloudflare quick tunnel to the port; returns the process and its public URL."""
    process = await asyncio.create_subprocess_exec(
        "cloudflared",
        "tunnel",
        "--url",
        f"http://localhost:{port}",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    stderr = process.stderr
    if stderr is None:
        raise RuntimeError("cloudflared started without a stderr pipe")
    async with asyncio.timeout(PROBE_TIMEOUT_SECS):
        while True:
            line = await stderr.readline()
            if not line:
                raise RuntimeError(f"cloudflared exited with {await process.wait()} before a URL")
            found = TUNNEL_URL.search(line.decode())
            if found:
                asyncio.create_task(_drain(stderr), name="cloudflared-stderr")
                return process, found.group(0)


async def _drain(stream: asyncio.StreamReader) -> None:
    while await stream.readline():
        pass


@dataclass(frozen=True)
class RunRequest:
    test_set: TestSet
    sample: int  # test cases to run; 0 is the whole set


def _secret(settings: Settings, name: str) -> str:
    value = getattr(settings, name)
    if value is None or not value.get_secret_value():
        raise RuntimeError(f"{name.upper()} is unset")
    secret: str = value.get_secret_value()
    return secret


async def run_on_coval(
    coval: CovalClient, loaded: LoadedStack, url: str, token: str | None, request: RunRequest
) -> dict[str, Any]:
    """Upsert the agent at this URL, launch one run, poll it to a terminal status.

    The Coval client is synchronous, so every call goes through a thread: this
    loop is also serving the calls, and a blocked loop is a stalled call.
    """
    test_set = request.test_set
    definition = CovalS2SAgent(loaded.slug, url, token, test_set.test_set_id)
    agent_id = await asyncio.to_thread(sync_agent, coval, definition, loaded.stack.audio)
    body = launch_body(
        agent_id,
        test_set.persona_id,
        test_set.test_set_id,
        test_set.metric_ids,
        sample=request.sample,
        display_name=definition.display_name,
    )
    run = await asyncio.to_thread(coval.launch_run, body)
    run_id = run_id_of(run)
    logger.info("run_launched", run_id=run_id, agent_id=agent_id, status=run.get("status"))
    while not is_terminal(run):
        await asyncio.sleep(POLL_SECS)
        run = await asyncio.to_thread(coval.get_run, run_id)
        logger.info(
            "run_status", run_id=run_id, status=run.get("status"), progress=run.get("progress")
        )
    return run


async def run(loaded: LoadedStack, port: int, tunnel: bool, request: RunRequest) -> int:
    settings = get_settings()
    token = settings.s2s_agent_token.get_secret_value() if settings.s2s_agent_token else None
    api_key = _secret(settings, "coval_api_key")
    server, task, port = await serve_in_process(create_app(loaded), port)
    process = None
    try:
        if tunnel:
            process, url = await open_tunnel(port)
        else:
            url = f"http://127.0.0.1:{port}"
        logger.info("probing", url=url)
        health = await probe(url, token)
        logger.info("agent_up", url=url, **health)
        print(url, flush=True)
        with CovalClient(api_key, settings.coval_api_base) as coval:
            final = await run_on_coval(coval, loaded, url, token, request)
        logger.info("run_finished", run_id=run_id_of(final), status=final.get("status"))
        print("\n".join(results_lines(final)), flush=True)
        return 0 if final.get("status") == "COMPLETED" else 1
    finally:
        if process is not None and process.returncode is None:
            process.terminate()
            await process.wait()
        server.should_exit = True
        await task
