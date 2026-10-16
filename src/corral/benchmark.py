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
import uvicorn
from fastapi import FastAPI
from loguru import logger
from websockets.asyncio.client import connect

from corral.agent import Agent
from corral.config import Settings, get_settings, remember
from corral.coval_api import (
    CovalClient,
    CovalTarget,
    is_terminal,
    launch_body,
    results_lines,
    run_id_of,
    sync_target,
)
from corral.coval_wire import SESSION_READY
from corral.scenarios import Suite, confirm_test_cases
from corral.serve import create_app
from corral.suite import choose_workspace, sync_suite

PROBE_TIMEOUT_SECS = 30
REACHABLE_TIMEOUT_SECS = 180
POLL_SECS = 30
TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


async def serve_in_process(
    app: FastAPI, port: int
) -> tuple[uvicorn.Server, asyncio.Task[None], int]:
    """Start uvicorn on the loop we are on; returns the server, its task and the bound port."""
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", log_config=None)
    )
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
            except httpx.HTTPError as error:
                if asyncio.get_running_loop().time() > deadline:
                    raise
                logger.debug(f"healthz not up yet ({error!r}), retrying")
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
    suite: Suite
    workspace_id: str | None
    sample: int


def _secret(settings: Settings, name: str) -> str:
    value = getattr(settings, name)
    if value is None or not value.get_secret_value():
        raise RuntimeError(f"{name.upper()} is unset")
    secret: str = value.get_secret_value()
    return secret


async def run_on_coval(
    coval: CovalClient, agent: Agent, url: str, token: str | None, request: RunRequest
) -> list[dict[str, Any]]:
    """Pick the workspace, sync the suite and agent, then run each persona in turn.

    The Coval client is synchronous, so every call goes through a thread: this
    loop is also serving the calls, and a blocked loop is a stalled call.
    """
    choice = await asyncio.to_thread(choose_workspace, coval, request.workspace_id)
    coval.use_workspace(choice.workspace_id)
    logger.info(f"Using Coval workspace {choice.workspace_id}")
    if choice.found_by_name:
        if remember("CORRAL_WORKSPACE_ID", choice.workspace_id):
            logger.info("Saved CORRAL_WORKSPACE_ID to .env; later runs use it")
        else:
            logger.info(f"Set CORRAL_WORKSPACE_ID={choice.workspace_id} to pin this workspace")
    keys = request.suite.test_case_keys()
    await confirm_test_cases(agent.scenario, keys)
    ids = await asyncio.to_thread(sync_suite, coval, request.suite)
    definition = CovalTarget(
        agent.slug, url, token, ids.test_set_id, choice.workspace_id, keyed=bool(keys)
    )
    agent_id = await asyncio.to_thread(sync_target, coval, definition)
    # One persona at a time, like one test case at a time, so no two calls share the machine.
    runs = []
    for persona_id in ids.persona_ids:
        body = launch_body(
            agent_id,
            persona_id,
            ids.test_set_id,
            ids.metric_ids,
            sample=request.sample,
            display_name=definition.display_name,
        )
        run = await asyncio.to_thread(coval.launch_run, body)
        logger.info(f"Run {run_id_of(run)} launched on agent {agent_id}: {run.get('status')}")
        while not is_terminal(run):
            await asyncio.sleep(POLL_SECS)
            run = await asyncio.to_thread(coval.get_run, run_id_of(run))
            logger.info(
                f"Run {run_id_of(run)}: {run.get('status')} {run.get('progress') or ''}".rstrip()
            )
        runs.append(run)
    return runs


async def run(agent: Agent, port: int, tunnel: bool, request: RunRequest) -> int:
    settings = get_settings()
    token = settings.s2s_agent_token.get_secret_value() if settings.s2s_agent_token else None
    api_key = _secret(settings, "coval_api_key")
    server, task, port = await serve_in_process(create_app(agent), port)
    process = None
    try:
        if tunnel:
            process, url = await open_tunnel(port)
        else:
            url = f"http://127.0.0.1:{port}"
        logger.debug(f"Probing {url}")
        health = await probe(url, token)
        logger.info(f"Agent {health['agent']} is up at {url}")
        print(url, flush=True)
        with CovalClient(api_key, settings.coval_api_base) as coval:
            finals = await run_on_coval(coval, agent, url, token, request)
        for final in finals:
            level = "INFO" if final.get("status") == "COMPLETED" else "WARNING"
            logger.log(level, f"Run {run_id_of(final)} finished: {final.get('status')}")
            print("\n".join(results_lines(final)), flush=True)
        return 0 if all(final.get("status") == "COMPLETED" for final in finals) else 1
    finally:
        if process is not None and process.returncode is None:
            process.terminate()
            await process.wait()
        server.should_exit = True
        await task
