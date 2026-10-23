# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator
from datetime import date
from typing import Any, cast
from unittest.mock import patch

import pytest
from coval_agent_sdk import AudioConnection
from coval_agent_sdk.errors import CovalError
from pipecat.frames.frames import Frame, InputAudioRawFrame, OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from corral import benchmark
from corral.agent import load_cascade_agent
from corral.coval_api import CovalClient
from corral.coval_sdk import SDK_SAMPLE_RATE_HZ, CovalSdkTransport
from corral.scenarios import Suite, require_suite
from corral.suite import SuiteIds, WorkspaceChoice

REFERENCE = ("deepgram/nova-3", "openai/gpt-4.1", "elevenlabs/eleven_flash_v2")
CHUNK = SDK_SAMPLE_RATE_HZ * 2 * 40 // 1000


class _Echo(FrameProcessor):
    """Stands in for the whole cascade: sends the caller's audio straight back out."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            frame = OutputAudioRawFrame(
                audio=frame.audio, sample_rate=frame.sample_rate, num_channels=frame.num_channels
            )
        await self.push_frame(frame, direction)


class _Connection:
    """Stands in for the SDK's connection: two chunks of caller audio, then the call ends."""

    simulation_id = "S1"

    def __init__(self, hangs_up: bool = True) -> None:
        self.closed = False
        self.sent = bytearray()
        self.enough = asyncio.Event()
        self.hangs_up = hangs_up

    async def wait_closed(self) -> None:
        await self.enough.wait()
        if not self.hangs_up:
            await asyncio.Event().wait()
        self.closed = True

    async def receive_audio(self) -> AsyncIterator[bytes]:
        yield bytes(range(256)) * (CHUNK // 256)
        yield bytes(reversed(range(256))) * (CHUNK // 256)

    async def send_audio(self, audio: bytes) -> bool:
        self.sent += audio
        if len(self.sent) >= 2 * CHUNK:
            self.enough.set()
        return True


async def test_the_sdk_transport_carries_audio_both_ways_until_coval_hangs_up() -> None:
    connection = cast(AudioConnection, _Connection())
    transport = CovalSdkTransport()
    transport.attach(connection)
    worker = PipelineWorker(
        Pipeline([transport.input(), _Echo(), transport.output()]),
        params=PipelineParams(
            audio_in_sample_rate=SDK_SAMPLE_RATE_HZ, audio_out_sample_rate=SDK_SAMPLE_RATE_HZ
        ),
    )
    await asyncio.wait_for(benchmark.run_until_hangup(worker, connection), 10)
    expected = bytes(range(256)) * (CHUNK // 256) + bytes(reversed(range(256))) * (CHUNK // 256)
    assert bytes(cast(_Connection, connection).sent[: 2 * CHUNK]) == expected


async def test_cancelling_the_handler_brings_the_pipeline_down_with_it() -> None:
    connection = _Connection(hangs_up=False)
    transport = CovalSdkTransport()
    transport.attach(cast(AudioConnection, connection))
    worker = PipelineWorker(
        Pipeline([transport.input(), _Echo(), transport.output()]),
        params=PipelineParams(
            audio_in_sample_rate=SDK_SAMPLE_RATE_HZ, audio_out_sample_rate=SDK_SAMPLE_RATE_HZ
        ),
    )
    handling = asyncio.create_task(
        benchmark.run_until_hangup(worker, cast(AudioConnection, connection))
    )
    await asyncio.wait_for(connection.enough.wait(), 10)
    handling.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(handling, 10)
    assert worker.has_finished()


def test_a_call_cannot_start_before_the_sdk_connection_is_attached() -> None:
    with pytest.raises(RuntimeError, match="before an SDK connection"):
        print(CovalSdkTransport().connection)


class _Coval:
    """Launches runs whose one simulation starts on the first poll; the run finishes on the
    second poll."""

    def __init__(self) -> None:
        self.launched: list[dict[str, Any]] = []
        self.polls: dict[str, int] = {}
        self.listed: list[str] = []

    def launch_run(self, body: dict[str, Any]) -> dict[str, Any]:
        assert all(self.polls.get(f"R{n}", 0) > 1 for n in range(1, len(self.launched) + 1))
        self.launched.append(body)
        return {"run_id": f"R{len(self.launched)}", "status": "PENDING"}

    def get_run(self, run_id: str) -> dict[str, Any]:
        self.polls[run_id] = self.polls.get(run_id, 0) + 1
        return {"run_id": run_id, "status": "COMPLETED" if self.polls[run_id] > 1 else "RUNNING"}

    def list_simulations(self, run_id: str) -> list[dict[str, Any]]:
        self.listed.append(run_id)
        status = "PENDING" if self.polls.get(run_id, 0) == 0 else "RUNNING"
        return [
            {"simulation_id": f"{run_id}-S1", "status": status},
            {"simulation_id": f"{run_id}-S2", "status": "SKIPPED"},
        ]


async def test_each_persona_gets_its_own_run_and_each_simulation_is_handled_once() -> None:
    coval = _Coval()
    handled: list[str] = []

    async def handler(simulation_id: str) -> None:
        handled.append(simulation_id)

    ids = SuiteIds("TS", ["P1", "P2"], ["M1"])
    request = benchmark.RunRequest(Suite(test_set="TS", personas=["P1"]), None, sample=1)
    with (
        patch.object(benchmark, "sync_suite", return_value=ids),
        patch.object(benchmark, "sync_target", return_value="A"),
        patch.object(benchmark, "POLL_SECS", 0),
    ):
        agent = load_cascade_agent("default", *REFERENCE)
        outcomes = await benchmark.run_suite(
            cast(CovalClient, coval), agent, request, "W1", handler
        )
    assert [body["persona_id"] for body in coval.launched] == ["P1", "P2"]
    assert all(
        body["test_set_id"] == "TS" and body["metric_ids"] == ["M1"] for body in coval.launched
    )
    assert handled == ["R1-S1", "R2-S1"]
    assert [outcome.run["status"] for outcome in outcomes] == ["COMPLETED", "COMPLETED"]
    assert all(outcome.succeeded for outcome in outcomes)


async def test_a_bug_in_a_call_stops_the_run() -> None:
    async def handler(simulation_id: str) -> None:
        raise ValueError(f"{simulation_id} has no backend to seed")

    run = {"run_id": "R1", "status": "PENDING"}
    with patch.object(benchmark, "POLL_SECS", 0), pytest.raises(ValueError, match="R1-S1"):
        await benchmark.await_run(cast(CovalClient, _Coval()), run, handler)


async def test_a_call_the_relay_drops_is_reported_and_the_run_goes_on() -> None:
    async def handler(simulation_id: str) -> None:
        raise CovalError("relay closed without completion (code 1013)")

    run = {"run_id": "R1", "status": "PENDING"}
    with patch.object(benchmark, "POLL_SECS", 0):
        outcome = await benchmark.await_run(cast(CovalClient, _Coval()), run, handler)
    assert outcome.run["status"] == "COMPLETED"
    assert not outcome.succeeded
    assert outcome.lines() == [
        "run R1 COMPLETED",
        "  call R1-S1 failed: relay closed without completion (code 1013)",
    ]


async def test_a_workspace_found_by_name_is_pinned_for_next_time() -> None:
    class _Client:
        def use_workspace(self, workspace_id: str) -> None:
            self.workspace_id = workspace_id

    client = _Client()
    with (
        patch.object(benchmark, "choose_workspace", return_value=WorkspaceChoice("W1", True)),
        patch.object(benchmark, "remember", return_value=True) as remembered,
    ):
        assert await benchmark.pick_workspace(cast(CovalClient, client), None) == "W1"
    assert client.workspace_id == "W1"
    remembered.assert_called_once_with("CORRAL_WORKSPACE_ID", "W1")


def test_a_sampled_suite_runs_the_same_test_cases_all_day_and_a_smoke_runs_one_at_random() -> None:
    sampled = require_suite("id_roulette")
    assert sampled.sample == 1
    day = benchmark.run_request(sampled, "W", smoke=False, today=date(2026, 10, 22))
    assert (day.sample, day.seed) == (1, 20261022)
    smoke = benchmark.run_request(sampled, "W", smoke=True, today=date(2026, 10, 22))
    assert (smoke.sample, smoke.seed) == (1, None)
    every = benchmark.run_request(
        Suite(test_set="TS", personas=["P1"]), None, smoke=False, today=date(2026, 10, 22)
    )
    assert (every.sample, every.seed) == (0, None)
