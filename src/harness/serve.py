# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve ``S2S_STACK`` running ``S2S_SCENARIO`` to Coval, one pipeline per WebSocket call."""

from __future__ import annotations

import asyncio
import os

import structlog
from fastapi import FastAPI, WebSocket
from opentelemetry.sdk.trace import TracerProvider
from pipecat.frames.frames import Frame, LLMRunFrame, OutputTransportMessageUrgentFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.workers.runner import WorkerRunner

from harness.config import Settings, get_settings
from harness.coval import (
    SESSION_READY,
    CovalFrameSerializer,
    authorized,
    simulation_id,
)
from harness.logging import configure_logging
from harness.observer import TurnLatencyObserver, coval_trace_provider
from harness.pipeline import cascade_processors, context_aggregators
from harness.services import resolve
from harness.stack import LoadedStack, load_stack

logger = structlog.get_logger("harness")

POLICY_VIOLATION = 1008


def create_app(loaded: LoadedStack) -> FastAPI:
    app = FastAPI()

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {
            "status": "ok",
            "scenario": loaded.scenario,
            "stack": loaded.slug,
            "prompt": loaded.prompt_file,
            "digest": loaded.digest,
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
        log = logger.bind(simulation_id=sim, stack=loaded.slug, digest=loaded.digest)
        log.info("call_started")
        await run_call(websocket, loaded, sim)
        log.info("call_ended")

    return app


async def run_call(websocket: WebSocket, loaded: LoadedStack, sim: str | None) -> None:
    settings = get_settings()
    services = resolve(loaded, settings)
    audio = loaded.stack.audio
    provider = _trace_provider(settings, sim)
    observer = TurnLatencyObserver(
        attributes={"simulation.id": sim or "", "stack": loaded.slug, "digest": loaded.digest},
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
    aggregators = context_aggregators(loaded)
    middle = cascade_processors(services.stt, services.llm, services.tts, aggregators)
    worker = PipelineWorker(
        Pipeline([transport.input(), *middle, transport.output(), aggregators.assistant()]),
        params=PipelineParams(
            audio_in_sample_rate=audio.in_sample_rate_hz,
            audio_out_sample_rate=audio.out_sample_rate_hz,
        ),
        conversation_id=sim,
        enable_rtvi=False,
        observers=[observer],
    )

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
            # Shutdown flushes; every span of this call is out before the call is.
            await asyncio.to_thread(provider.shutdown)


def _trace_provider(settings: Settings, sim: str | None) -> TracerProvider | None:
    """Spans go to Coval only for a real simulation with a key to send them under."""
    if sim is None or settings.coval_api_key is None:
        return None
    key = settings.coval_api_key.get_secret_value()
    return coval_trace_provider(key, settings.coval_api_base, sim) if key else None


def main() -> None:
    import uvicorn

    configure_logging()
    loaded = load_stack(
        os.environ.get("S2S_SCENARIO", "bank"),
        os.environ.get("S2S_STACK", "cascade-nova3-gpt41-flash"),
    )
    uvicorn.run(create_app(loaded), host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))  # noqa: S104


if __name__ == "__main__":
    main()
