# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import time
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.tests.utils import run_test
from pipecat.transports.base_transport import BaseTransport
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies

from corral import pipeline
from corral.agent import load_cascade_agent, load_native_agent
from corral.config import Settings
from corral.environment import JsonEnvironment, Tool, ToolsFile
from corral.models import CascadeServices, NativeService
from corral.pipeline import (
    LastWordWatch,
    ToolCall,
    build_call,
    cascade_processors,
    context_aggregators,
    tool_handler,
)

REFERENCE = ("deepgram/nova-3", "openai/gpt-4.1", "elevenlabs/eleven_flash_v2")
SETTINGS = Settings(_env_file=None)


class _Transport(BaseTransport):
    def input(self) -> FrameProcessor:
        return FrameProcessor()

    def output(self) -> FrameProcessor:
        return FrameProcessor()


def _params(arguments: dict[str, Any], results: list[Any]) -> FunctionCallParams:
    async def deliver(result: Any, **_: Any) -> None:
        results.append(result)

    return cast(FunctionCallParams, SimpleNamespace(arguments=arguments, result_callback=deliver))


def test_cascade_runs_transcribe_accumulate_generate_synthesize() -> None:
    aggregators = context_aggregators()
    stt, llm, tts = FrameProcessor(), FrameProcessor(), FrameProcessor()
    assert cascade_processors(stt, llm, tts, aggregators) == [
        stt,
        aggregators.user(),
        llm,
        tts,
    ]


def test_a_native_aggregator_has_no_turn_taking_of_ours() -> None:
    assert context_aggregators(native=True).user()._vad_controller is None
    assert context_aggregators().user()._vad_controller is not None


def test_a_model_without_turn_ends_proposes_them_and_is_never_interrupted() -> None:
    user = context_aggregators(native=True, reports_turn_end=False).user()
    assert user._vad_controller is None
    strategies = user._params.user_turn_strategies
    assert isinstance(strategies, ExternalUserTurnStrategies)
    assert strategies.enable_interruptions is False


def test_the_context_offers_the_tools_on_every_request() -> None:
    schema = FunctionSchema("find_patient", "Look up.", {}, [])
    assert context_aggregators([schema]).user().context.tools.standard_tools == [schema]
    assert context_aggregators([schema], native=True).user().context.tools.standard_tools == [
        schema
    ]
    assert not context_aggregators().user().context.tools


class _Loud(VADAnalyzer):
    """Hears speech in any chunk that isn't silence, ten milliseconds at a time."""

    def num_frames_required(self) -> int:
        return 160

    def voice_confidence(self, buffer: bytes) -> float:
        return 1.0 if any(buffer) else 0.0


async def test_the_watch_times_the_last_word_upstream_and_passes_the_audio_on() -> None:
    watch = LastWordWatch(_Loud(params=VADParams(start_secs=0.05, stop_secs=0.2, min_volume=0)))
    loud = InputAudioRawFrame(audio=b"\x00\x40" * 160, sample_rate=16000, num_channels=1)
    quiet = InputAudioRawFrame(audio=b"\x00\x00" * 160, sample_rate=16000, num_channels=1)
    chunks = [loud] * 20 + [quiet] * 40
    down, up = await run_test(
        watch,
        frames_to_send=chunks,
        expected_down_frames=[InputAudioRawFrame] * len(chunks),
        expected_up_frames=[VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame],
    )
    assert len(down) == len(chunks) and len(up) == 2
    assert isinstance(up[1], VADUserStoppedSpeakingFrame) and up[1].stop_secs == 0.2


async def test_a_handler_that_raises_answers_the_llm_with_the_error() -> None:
    async def flaky(arguments: dict[str, Any]) -> Any:
        if arguments["name"] == "Ana":
            return {"patient_id": 1}
        raise LookupError("no such patient")

    log: list[ToolCall] = []
    handle = tool_handler(Tool(FunctionSchema("find_patient", "Look up.", {}, []), flaky), log)
    results: list[Any] = []
    await handle(_params({"name": "Ana"}, results))
    await handle(_params({"name": "Bo"}, results))
    assert results == [{"patient_id": 1}, {"error": "LookupError: no such patient"}]
    assert log == [
        ToolCall("find_patient", {"name": "Ana"}, {"patient_id": 1}),
        ToolCall("find_patient", {"name": "Bo"}, {"error": "LookupError: no such patient"}),
    ]


def _booking() -> JsonEnvironment:
    """One tool, one answer, one test case that changes it."""
    spec = ToolsFile(
        tools=[{"name": "book", "description": "Book it."}],
        responses={"book": {"ok": True}},
        test_cases={"already-booked": {"book": {"ok": False}}},
    )
    return JsonEnvironment(spec)


def _cascade(llm: LLMService) -> CascadeServices:
    return CascadeServices(stt=FrameProcessor(), llm=llm, tts=FrameProcessor())


async def test_a_call_seeds_the_backend_and_puts_its_tools_on_the_llm() -> None:
    llm = LLMService()
    with (
        patch.object(pipeline, "resolve", return_value=_cascade(llm)),
        patch.object(pipeline, "load_environment", return_value=_booking()),
    ):
        agent = load_cascade_agent("default", *REFERENCE)
        call = await build_call(agent, _Transport(), SETTINGS, test_case="already-booked")
    assert call.tool_calls == []
    assert llm.has_function("book")
    results: list[Any] = []
    registered = cast(Any, llm._functions["book"]).handler
    await registered(_params({"date": "2026-10-21"}, results))
    assert results == [{"ok": False}]
    assert call.tool_calls == [ToolCall("book", {"date": "2026-10-21"}, {"ok": False})]


async def test_a_test_case_the_backend_cannot_seed_is_refused() -> None:
    with (
        patch.object(pipeline, "resolve", return_value=_cascade(LLMService())),
        patch.object(pipeline, "load_environment", return_value=_booking()),
        pytest.raises(ValueError, match="no test case 'nope'"),
    ):
        await build_call(
            load_cascade_agent("default", *REFERENCE), _Transport(), SETTINGS, test_case="nope"
        )


async def test_a_test_case_on_a_scenario_without_tools_is_refused() -> None:
    with (
        patch.object(pipeline, "resolve", return_value=_cascade(LLMService())),
        pytest.raises(ValueError, match="default has no tools"),
    ):
        await build_call(
            load_cascade_agent("default", *REFERENCE), _Transport(), SETTINGS, test_case="x"
        )


async def test_a_call_without_tools_logs_no_tool_calls() -> None:
    with patch.object(pipeline, "resolve", return_value=_cascade(LLMService())):
        call = await build_call(load_cascade_agent("default", *REFERENCE), _Transport(), SETTINGS)
    assert call.tool_calls is None


async def test_a_native_call_puts_the_tools_on_the_one_model_and_watches_the_last_word() -> None:
    model = LLMService()
    agent = load_native_agent("default", "openai/gpt-realtime-2.1")
    with (
        patch.object(pipeline, "resolve", return_value=NativeService(model)),
        patch.object(pipeline, "load_environment", return_value=_booking()),
    ):
        call = await build_call(agent, _Transport(), SETTINGS, test_case="already-booked")
    assert model.has_function("book")
    assert call.observer._native is True and call.observer._llm is model
    assert call.observer._stt is None and call.observer._tts is None
    inner = next(p for p in call.worker._pipeline._processors if isinstance(p, Pipeline))
    kinds = [type(p).__name__ for p in inner._processors]
    order = [kinds.index(k) for k in ("LastWordWatch", "LLMUserAggregator", "LLMService")]
    assert order == sorted(order)


async def test_a_model_that_waits_for_a_cue_is_told_to_open_the_call() -> None:
    greetings: dict[bool, list[Frame]] = {}
    for waits in (False, True):
        agent = load_native_agent("default", "openai/gpt-realtime-2.1")
        service = NativeService(LLMService(), waits_for_cue=waits)
        with patch.object(pipeline, "resolve", return_value=service):
            call = await build_call(agent, _Transport(), SETTINGS)
        queued: list[Frame] = []

        async def queue(frame: Frame, queued: list[Frame] = queued) -> None:
            queued.append(frame)

        for handler in call.worker._event_handlers["on_pipeline_started"].handlers:
            await handler(SimpleNamespace(queue_frame=queue), None)
        greetings[waits] = queued
    assert [type(frame) for frame in greetings[False]] == [LLMRunFrame]
    (cue,) = greetings[True]
    assert isinstance(cue, LLMMessagesAppendFrame) and cue.run_llm
    assert cue.messages == [{"role": "developer", "content": pipeline.OPENING_CUE}]


async def test_the_backend_is_cleaned_up_when_the_pipeline_finishes() -> None:
    class _Environment:
        def __init__(self) -> None:
            self.tools: list[Tool] = []
            self.events: list[str] = []

        async def start(self, test_case: str | None) -> None:
            self.events.append(f"start {test_case}")

        async def finish(self) -> None:
            self.events.append("finish")

    environment = _Environment()
    fakes = CascadeServices(stt=FrameProcessor(), llm=LLMService(), tts=FrameProcessor())
    with (
        patch.object(pipeline, "resolve", return_value=fakes),
        patch.object(pipeline, "load_environment", return_value=environment),
    ):
        call = await build_call(
            load_cascade_agent("default", *REFERENCE), _Transport(), SETTINGS, test_case="x"
        )
    assert environment.events == ["start x"]
    await call.worker._call_event_handler("on_pipeline_finished", None)
    await call.worker.cleanup()
    assert environment.events == ["start x", "finish"]


async def test_a_finished_call_sends_each_checked_arguments_character_error_rate() -> None:
    spec = ToolsFile(
        tools=[
            {
                "name": "lookup",
                "description": "Look it up.",
                "parameters": {"id": "string"},
                "character_error_rate": {"id": "AB12"},
            }
        ],
        responses={"lookup": {"received": True}},
    )
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    llm = LLMService()
    with (
        patch.object(pipeline, "resolve", return_value=_cascade(llm)),
        patch.object(pipeline, "load_environment", return_value=JsonEnvironment(spec)),
    ):
        call = await build_call(
            load_cascade_agent("default", *REFERENCE),
            _Transport(),
            SETTINGS,
            tracer=provider.get_tracer(__name__),
        )
    await cast(Any, llm._functions["lookup"]).handler(_params({"id": "ab-1"}, []))
    assert call.character_error_rates() == {"lookup.id": 0.25}
    await call.worker._call_event_handler("on_pipeline_finished", None)
    await call.worker.cleanup()
    (rates,) = [s for s in exporter.get_finished_spans() if s.name == "character_error_rate"]
    assert rates.attributes is not None and rates.attributes["lookup.id"] == 0.25
    assert rates.attributes["agent"] == call.observer._attributes["agent"]


async def test_a_finished_call_sends_a_span_from_start_to_finish() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with patch.object(pipeline, "resolve", return_value=_cascade(LLMService())):
        call = await build_call(
            load_cascade_agent("default", *REFERENCE),
            _Transport(),
            SETTINGS,
            tracer=provider.get_tracer(__name__),
        )

    async def queue(_frame: Frame) -> None:
        pass

    with patch.object(time, "time_ns", return_value=1_000):
        for handler in call.worker._event_handlers["on_pipeline_started"].handlers:
            await handler(SimpleNamespace(queue_frame=queue), None)
    await call.worker._call_event_handler("on_pipeline_finished", None)
    await call.worker.cleanup()
    (span,) = [s for s in exporter.get_finished_spans() if s.name == pipeline.CALL_SPAN_NAME]
    assert span.start_time == 1_000
    assert span.attributes is not None
    assert span.attributes["agent"] == call.observer._attributes["agent"]


async def test_a_model_that_can_no_longer_do_its_job_ends_the_call() -> None:
    from pipecat.pipeline.worker import ProcessorUnusablePolicy

    llm = LLMService()
    with (
        patch.object(pipeline, "resolve", return_value=_cascade(llm)),
        patch.object(pipeline, "load_environment", return_value=_booking()),
    ):
        agent = load_cascade_agent("default", *REFERENCE)
        call = await build_call(agent, _Transport(), SETTINGS, test_case="already-booked")
    assert call.worker._processor_unusable_policy is ProcessorUnusablePolicy.CANCEL
