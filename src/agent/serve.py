# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve the chosen models to Coval, one pipeline per WebSocket call."""

from __future__ import annotations

import asyncio

import structlog
from fastapi import FastAPI, WebSocket
from opentelemetry.sdk.trace import TracerProvider
from pipecat.frames.frames import OutputTransportMessageUrgentFrame
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.workers.runner import WorkerRunner

from agent.config import Settings, get_settings
from agent.coval_wire import (
    SESSION_READY,
    CovalFrameSerializer,
    authorized,
    simulation_id,
)
from agent.observer import coval_trace_provider
from agent.pipeline import build_call
from agent.stack import Stack

logger = structlog.get_logger("agent")

POLICY_VIOLATION = 1008


def create_app(stack: Stack) -> FastAPI:
    app = FastAPI()

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {
            "status": "ok",
            "scenario": stack.scenario,
            "stack": stack.slug,
            "digest": stack.digest,
        }

    @app.websocket("/ws")
    async def call(websocket: WebSocket) -> None:
        settings = get_settings()
        token = settings.s2s_agent_token.get_secret_value() if settings.s2s_agent_token else None
        if not authorized(websocket.headers, token):
            await websocket.close(code=POLICY_VIOLATION)
            return
        await websocket.accept()
        sim = simulation_id(websocket.headers)
        log = logger.bind(simulation_id=sim, stack=stack.slug, digest=stack.digest)
        log.info("call_started")
        await run_call(websocket, stack, sim)
        log.info("call_ended")

    return app


async def run_call(websocket: WebSocket, stack: Stack, sim: str | None) -> None:
    settings = get_settings()
    provider = _trace_provider(settings, sim)
    transport = FastAPIWebsocketTransport(
        websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            serializer=CovalFrameSerializer(),
            allowed_origins=[],
        ),
    )
    call = build_call(
        stack,
        transport,
        settings,
        attributes={"simulation.id": sim or ""},
        tracer=provider.get_tracer(__name__) if provider else None,
        conversation_id=sim,
        before_greeting=[OutputTransportMessageUrgentFrame(message=SESSION_READY)],
    )

    @transport.event_handler("on_client_disconnected")  # type: ignore[untyped-decorator]
    async def _gone(_transport: FastAPIWebsocketTransport, _ws: WebSocket) -> None:
        await call.worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(call.worker)
    try:
        await runner.run()
    finally:
        if provider is not None:
            await asyncio.to_thread(provider.shutdown)


def _trace_provider(settings: Settings, sim: str | None) -> TracerProvider | None:
    """Spans go to Coval only for a real simulation with a key to send them under."""
    if sim is None or settings.coval_api_key is None:
        return None
    key = settings.coval_api_key.get_secret_value()
    return coval_trace_provider(key, settings.coval_api_base, sim) if key else None


def serve(stack: Stack, port: int) -> None:
    """The image's command: the agent on every interface, no tunnel, until stopped."""
    import uvicorn

    uvicorn.run(create_app(stack), host="0.0.0.0", port=port)  # noqa: S104
