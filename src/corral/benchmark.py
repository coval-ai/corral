# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""One benchmark: launch the suite on Coval, dial in to each call as it starts, print results."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import date
from functools import partial
from typing import Any

from coval_agent_sdk import AudioConnection, CovalVoiceClient
from coval_agent_sdk.errors import CovalError
from loguru import logger
from pipecat.pipeline.worker import PipelineWorker
from pipecat.workers.runner import WorkerRunner

from corral.agent import Agent
from corral.config import Settings, get_settings, remember
from corral.coval_api import (
    CovalClient,
    CovalTarget,
    is_terminal,
    launch_body,
    results_lines,
    run_id_of,
    run_succeeded,
    sync_target,
)
from corral.coval_sdk import CovalSdkTransport
from corral.observer import tracing
from corral.pipeline import build_call
from corral.scenarios import Suite, confirm_test_cases
from corral.suite import choose_workspace, sync_suite

POLL_SECS = 2
NOT_LIVE = frozenset({"PENDING", "IN QUEUE", "SKIPPED"})

SimulationHandler = Callable[[str], Coroutine[Any, Any, None]]
"""Handles one simulation, by id, until Coval hangs up."""


@dataclass(frozen=True)
class RunRequest:
    """``sample`` test cases per run, all of them at 0, picked by ``seed`` when one is given."""

    suite: Suite
    workspace_id: str | None
    sample: int
    seed: int | None = None


def run_request(suite: Suite, workspace_id: str | None, *, smoke: bool, today: date) -> RunRequest:
    """A smoke runs one test case at random. A suite with ``sample`` runs that many, seeded by the
    UTC date, so every agent and persona benchmarked the same day gets the same ones."""
    if smoke:
        return RunRequest(suite, workspace_id, sample=1)
    if suite.sample:
        return RunRequest(suite, workspace_id, sample=suite.sample, seed=int(f"{today:%Y%m%d}"))
    return RunRequest(suite, workspace_id, sample=0)


@dataclass(frozen=True)
class Outcome:
    """A finished run and the calls the relay dropped, by simulation id, with why."""

    run: dict[str, Any]
    failed_calls: dict[str, str]

    @property
    def succeeded(self) -> bool:
        return run_succeeded(self.run) and not self.failed_calls

    def lines(self) -> list[str]:
        return results_lines(self.run) + [
            f"  call {simulation_id} failed: {why}"
            for simulation_id, why in self.failed_calls.items()
        ]


def _secret(settings: Settings, name: str) -> str:
    value = getattr(settings, name)
    if value is None or not value.get_secret_value():
        raise RuntimeError(f"{name.upper()} is unset")
    secret: str = value.get_secret_value()
    return secret


async def pick_workspace(coval: CovalClient, wanted: str | None) -> str:
    """Choose the workspace, use it, and pin it in .env when it had to be found by name.

    The Coval client is synchronous, so every call goes through a thread: this
    loop is also running the calls, and a blocked loop is a stalled call.
    """
    choice = await asyncio.to_thread(choose_workspace, coval, wanted)
    coval.use_workspace(choice.workspace_id)
    logger.info(f"Using Coval workspace {choice.workspace_id}")
    if choice.found_by_name:
        if remember("CORRAL_WORKSPACE_ID", choice.workspace_id):
            logger.info("Saved CORRAL_WORKSPACE_ID to .env; later runs use it")
        else:
            logger.info(f"Set CORRAL_WORKSPACE_ID={choice.workspace_id} to pin this workspace")
    return choice.workspace_id


async def handle_simulation(
    sdk: CovalVoiceClient,
    coval: CovalClient,
    agent: Agent,
    settings: Settings,
    simulation_id: str,
) -> None:
    """One call: build the pipeline, join the relay, run until Coval hangs up.

    The relay counts unread caller audio from the moment it is ready, so the slow
    part, loading the turn-taking models and seeding the scenario's backend, comes
    before joining.
    """
    test_case = await asyncio.to_thread(coval.test_case_key, simulation_id)
    logger.info(f"Call {simulation_id} started on {agent.slug}, test case {test_case}")
    transport = CovalSdkTransport()
    async with tracing(settings, simulation_id) as tracer:
        call = await build_call(
            agent,
            transport,
            settings,
            test_case=test_case,
            attributes={"simulation.id": simulation_id},
            tracer=tracer,
            conversation_id=simulation_id,
        )
        async with sdk.connect(simulation_id) as connection:
            transport.attach(connection)
            await run_until_hangup(call.worker, connection)
    logger.info(f"Call {simulation_id} ended")


async def run_until_hangup(worker: PipelineWorker, connection: AudioConnection) -> None:
    """Run the pipeline until Coval hangs up, or until the pipeline dies.

    Leaving the relay before Coval ends the call is a failure the SDK raises, so
    a pipeline that dies mid-call fails the run rather than passing as a short
    conversation. A relay failure is raised here too, once the pipeline is down.
    """
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run(), name=f"call-{connection.simulation_id}")
    closing = asyncio.create_task(
        connection.wait_closed(), name=f"relay-{connection.simulation_id}"
    )
    try:
        done, _ = await asyncio.wait({running, closing}, return_when=asyncio.FIRST_COMPLETED)
        if running in done:
            await running
        else:
            await worker.cancel()
        if closing in done:
            await closing
    finally:
        # Reached on cancellation too: the pipeline has to come down with the handler,
        # or its provider connections stay open and shutdown hangs on them.
        if not running.done():
            await worker.cancel()
        closing.cancel()
        await asyncio.gather(running, closing, return_exceptions=True)


async def await_run(coval: CovalClient, run: dict[str, Any], handler: SimulationHandler) -> Outcome:
    """Poll the run, joining each simulation as Coval starts it, until the run is over
    and every call has ended.

    A call the relay drops is logged and reported, and the run goes on: Coval keeps
    going regardless, and the other calls are still worth having. Any other failure
    in a call is a bug here, and stops the run.
    """
    calls: dict[str, asyncio.Task[None]] = {}
    failed: dict[str, str] = {}
    seen: dict[str, Any] = {}
    status = None
    try:
        while True:
            for simulation in await asyncio.to_thread(coval.list_simulations, run_id_of(run)):
                simulation_id = str(simulation.get("simulation_id") or simulation.get("id"))
                if seen.get(simulation_id) != simulation.get("status"):
                    seen[simulation_id] = simulation.get("status")
                    logger.debug(f"Simulation {simulation_id}: {simulation.get('status')}")
                if simulation_id in calls or simulation.get("status") in NOT_LIVE:
                    continue
                if is_terminal(simulation):
                    continue
                calls[simulation_id] = asyncio.create_task(handler(simulation_id))
            for simulation_id, task in calls.items():
                if not task.done() or simulation_id in failed:
                    continue
                error = task.exception()
                if isinstance(error, CovalError):
                    failed[simulation_id] = str(error)
                    logger.error(f"Call {simulation_id} failed: {error}")
                elif error is not None:
                    raise error
            if run.get("status") != status:
                status = run.get("status")
                logger.info(f"Run {run_id_of(run)}: {status} {run.get('progress') or ''}".rstrip())
            if is_terminal(run) and all(task.done() for task in calls.values()):
                return Outcome(run, failed)
            await asyncio.sleep(POLL_SECS)
            run = await asyncio.to_thread(coval.get_run, run_id_of(run))
    finally:
        for task in calls.values():
            task.cancel()
        await asyncio.gather(*calls.values(), return_exceptions=True)


async def run_suite(
    coval: CovalClient,
    agent: Agent,
    request: RunRequest,
    workspace_id: str,
    handler: SimulationHandler,
) -> list[Outcome]:
    """Sync the suite and the agent record, then run each persona in turn."""
    await confirm_test_cases(agent.scenario, list(request.suite.keyed_test_cases()))
    ids = await asyncio.to_thread(sync_suite, coval, request.suite)
    definition = CovalTarget(agent.slug, ids.test_set_id, workspace_id)
    agent_id = await asyncio.to_thread(sync_target, coval, definition)
    # One persona at a time, like one test case at a time, so no two calls share the machine.
    outcomes = []
    for persona_id in ids.persona_ids:
        body = launch_body(
            agent_id,
            persona_id,
            ids.test_set_id,
            ids.metric_ids,
            sample=request.sample,
            display_name=definition.display_name,
            seed=request.seed,
        )
        run = await asyncio.to_thread(coval.launch_run, body)
        logger.info(f"Run {run_id_of(run)} launched on agent {agent_id}: {run.get('status')}")
        outcomes.append(await await_run(coval, run, handler))
    return outcomes


async def run(agent: Agent, request: RunRequest) -> int:
    settings = get_settings()
    api_key = _secret(settings, "coval_api_key")
    with CovalClient(api_key, settings.coval_api_base) as coval:
        workspace_id = await pick_workspace(coval, request.workspace_id)
        sdk = CovalVoiceClient(
            api_key=api_key, api_base_url=settings.coval_api_base, workspace_id=workspace_id
        )
        handler = partial(handle_simulation, sdk, coval, agent, settings)
        outcomes = await run_suite(coval, agent, request, workspace_id, handler)
    for outcome in outcomes:
        level = "INFO" if outcome.succeeded else "WARNING"
        logger.log(level, f"Run {run_id_of(outcome.run)} finished: {outcome.run.get('status')}")
        print("\n".join(outcome.lines()), flush=True)
    return 0 if all(outcome.succeeded for outcome in outcomes) else 1
