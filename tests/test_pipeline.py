# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.transports.base_transport import BaseTransport

from corral import pipeline
from corral.agent import load_agent
from corral.config import Settings
from corral.environment import Tool
from corral.models import Services
from corral.pipeline import (
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


def test_the_context_offers_the_tools_on_every_request() -> None:
    schema = FunctionSchema("find_patient", "Look up.", {}, [])
    assert context_aggregators([schema]).user().context.tools.standard_tools == [schema]
    assert not context_aggregators().user().context.tools


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
    fakes = Services(stt=FrameProcessor(), llm=llm, tts=FrameProcessor())
    with patch.object(pipeline, "resolve", return_value=fakes):
        call = await build_call(
            load_agent("demo", *REFERENCE), _Transport(), SETTINGS, test_case="already-booked"
        )
        assert call.tool_calls == []
        assert llm.has_function("find_patient") and llm.has_function("book_appointment")
        results: list[Any] = []
        registered = cast(Any, llm._functions["book_appointment"]).handler
        await registered(_params({"patient_id": 123}, results))
        assert results == [{"ok": False, "error": "patient already has an appointment"}]
        assert call.tool_calls == [ToolCall("book_appointment", {"patient_id": 123}, results[0])]
        with pytest.raises(ValueError, match="no test case 'nope'"):
            await build_call(
                load_agent("demo", *REFERENCE), _Transport(), SETTINGS, test_case="nope"
            )
        with pytest.raises(ValueError, match="default has no tools"):
            await build_call(
                load_agent("default", *REFERENCE), _Transport(), SETTINGS, test_case="x"
            )
        plain = await build_call(load_agent("default", *REFERENCE), _Transport(), SETTINGS)
        assert plain.tool_calls is None


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
    fakes = Services(stt=FrameProcessor(), llm=LLMService(), tts=FrameProcessor())
    with (
        patch.object(pipeline, "resolve", return_value=fakes),
        patch.object(pipeline, "load_environment", return_value=environment),
    ):
        call = await build_call(
            load_agent("demo", *REFERENCE), _Transport(), SETTINGS, test_case="x"
        )
    assert environment.events == ["start x"]
    await call.worker._call_event_handler("on_pipeline_finished", None)
    await call.worker.cleanup()
    assert environment.events == ["start x", "finish"]
