# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve the chosen models to Coval, one pipeline per WebSocket call."""

from __future__ import annotations

import asyncio

import structlog
from fastapi import FastAPI, WebSocket
from opentelemetry.sdk.trace import TracerProvider
from pipecat.frames.frames import Frame, LLMRunFrame, OutputTransportMessageUrgentFrame
from pipecat.pipeline.worker import PipelineWorker
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.workers.runner import WorkerRunner

from agent.config import Settings, get_settings
from agent.coval import (
    SESSION_READY,
    CovalFrameSerializer,
    authorized,
    simulation_id,
)
from agent.observer import TurnLatencyObserver, coval_trace_provider, models_by_modality
from agent.pipeline import build_worker
from agent.services import resolve
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
    services = resolve(stack, settings)
    provider = _trace_provider(settings, sim)
    observer = TurnLatencyObserver(
        attributes={"simulation.id": sim or "", "stack": stack.slug, "digest": stack.digest},
        models=models_by_modality(stack),
        tracer=provider.get_tracer(__name__) if provider else None,
    )
    transport = FastAPIWebsocketTransport(
        websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            serializer=CovalFrameSerializer(),
            allowed_origins=[],
        ),
    )
    worker = build_worker(stack, services, transport, observer, conversation_id=sim)

    @worker.event_handler("on_pipeline_started")  # type: ignore[untyped-decorator]
    async def _ready(started: PipelineWorker, _frame: Frame) -> None:
        # Fires once StartFrame has passed every processor, so every service is
        # up. Coval holds the persona until session_ready, then the agent greets.
        await started.queue_frame(OutputTransportMessageUrgentFrame(message=SESSION_READY))
        await started.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")  # type: ignore[untyped-decorator]
    async def _gone(_transport: FastAPIWebsocketTransport, _ws: WebSocket) -> None:
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    try:
        await runner.run()
    finally:
        if provider is not None:
            # shutdown flushes, so every span of this call is out before the call is
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
