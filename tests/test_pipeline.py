# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.frames.frames import (
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.tests.utils import run_test
from pipecat.transports.base_transport import BaseTransport

from corral import pipeline
from corral.agent import load_cascade_agent, load_native_agent
from corral.config import Settings
from corral.environment import Tool
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


async def test_a_call_seeds_the_backend_and_puts_its_tools_on_the_llm() -> None:
    llm = LLMService()
    fakes = CascadeServices(stt=FrameProcessor(), llm=llm, tts=FrameProcessor())
    with patch.object(pipeline, "resolve", return_value=fakes):
        demo = load_cascade_agent("demo", *REFERENCE)
        call = await build_call(demo, _Transport(), SETTINGS, test_case="already-booked")
        assert call.tool_calls == []
        assert llm.has_function("find_patient") and llm.has_function("book_appointment")
        results: list[Any] = []
        registered = cast(Any, llm._functions["book_appointment"]).handler
        await registered(_params({"patient_id": 123}, results))
        assert results == [{"ok": False, "error": "patient already has an appointment"}]
        assert call.tool_calls == [ToolCall("book_appointment", {"patient_id": 123}, results[0])]
        with pytest.raises(ValueError, match="no test case 'nope'"):
            await build_call(
                load_cascade_agent("demo", *REFERENCE), _Transport(), SETTINGS, test_case="nope"
            )
        with pytest.raises(ValueError, match="default has no tools"):
            await build_call(
                load_cascade_agent("default", *REFERENCE), _Transport(), SETTINGS, test_case="x"
            )
        plain = await build_call(load_cascade_agent("default", *REFERENCE), _Transport(), SETTINGS)
        assert plain.tool_calls is None


async def test_a_native_call_puts_the_tools_on_the_one_model_and_watches_the_last_word() -> None:
    model = LLMService()
    agent = load_native_agent("demo", "openai/gpt-realtime-2.1")
    with patch.object(pipeline, "resolve", return_value=NativeService(model)):
        call = await build_call(agent, _Transport(), SETTINGS, test_case="book-cleaning")
    assert model.has_function("find_patient") and model.has_function("book_appointment")
    assert call.observer._native is True and call.observer._llm is model
    assert call.observer._stt is None and call.observer._tts is None
    inner = next(p for p in call.worker._pipeline._processors if isinstance(p, Pipeline))
    kinds = [type(p).__name__ for p in inner._processors]
    order = [kinds.index(k) for k in ("LastWordWatch", "LLMUserAggregator", "LLMService")]
    assert order == sorted(order)


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
            load_cascade_agent("demo", *REFERENCE), _Transport(), SETTINGS, test_case="x"
        )
    assert environment.events == ["start x"]
    await call.worker._call_event_handler("on_pipeline_finished", None)
    await call.worker.cleanup()
    assert environment.events == ["start x", "finish"]
