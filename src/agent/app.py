# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Talk to a stack on this machine's microphone and speakers. Ctrl-C hangs up."""

from __future__ import annotations

from pipecat.frames.frames import Frame, LLMRunFrame
from pipecat.pipeline.worker import PipelineWorker
from pipecat.workers.runner import WorkerRunner

from agent.config import get_settings
from agent.observer import TurnLatencyObserver
from agent.pipeline import build_worker
from agent.services import resolve
from agent.stack import LoadedStack


async def run(loaded: LoadedStack) -> None:
    # pyaudio comes with the local extra only; nothing else needs it.
    from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

    services = resolve(loaded, get_settings())
    audio = loaded.stack.audio
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=audio.in_sample_rate_hz,
            audio_out_sample_rate=audio.out_sample_rate_hz,
        )
    )
    observer = TurnLatencyObserver(attributes={"stack": loaded.slug, "digest": loaded.digest})
    worker = build_worker(loaded, services, transport, observer)

    @worker.event_handler("on_pipeline_started")  # type: ignore[untyped-decorator]
    async def _greet(started: PipelineWorker, _frame: Frame) -> None:
        # Fires once StartFrame has passed every processor, so every service is up.
        await started.queue_frame(LLMRunFrame())

    runner = WorkerRunner()
    await runner.add_workers(worker)
    await runner.run()
