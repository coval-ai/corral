# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""What sits between the transport's input and output, in order."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from opentelemetry.trace import Tracer
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.audio.vad.vad_controller import VADController
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    LLMRunFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
    FrameProcessorSetup,
)
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.transports.base_transport import BaseTransport
from pipecat.turns.user_start.min_words_user_turn_start_strategy import (
    MinWordsUserTurnStartStrategy,
)
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies, UserTurnStrategies

from corral.agent import Agent
from corral.config import (
    IN_SAMPLE_RATE_HZ,
    INTERRUPT_MIN_WORDS,
    OUT_SAMPLE_RATE_HZ,
    VAD_STOP_SECS,
    Settings,
)
from corral.environment import Tool
from corral.models import CascadeServices, NativeService, resolve
from corral.observer import TurnLatencyObserver, models_by_modality
from corral.scenarios import load_environment
from corral.tool_checks import CHARACTER_ERROR_RATE, character_error_rates


def silero() -> SileroVADAnalyzer:
    return SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS))


def context_aggregators(
    tools: Sequence[FunctionSchema] = (), *, native: bool = False, full_duplex: bool = False
) -> LLMContextAggregatorPair:
    """The conversation context and, for a cascade, the turn-taking the agent pins around it.

    The system prompt is not here; it rides on the LLM service's Settings so
    Pipecat composes it with its own instructions. The tools are, so every
    request offers them. Silero decides when the caller is speaking; smart-turn
    decides when a pause is the end of the turn rather than a breath. Both are
    local models shipped with Pipecat, so the cascade's endpointing is the same
    on every machine. While the agent is speaking, the caller has to say
    ``interrupt_min_words`` before it yields, so a backchannel does not cut it off.

    A native model does its own turn-taking, so its aggregator has none of ours, and a
    full-duplex one's never interrupts.
    """
    context = LLMContext(tools=list(tools)) if tools else LLMContext()
    if full_duplex:
        external = ExternalUserTurnStrategies(enable_interruptions=False)
        return LLMContextAggregatorPair(
            context, user_params=LLMUserAggregatorParams(user_turn_strategies=external)
        )
    if native:
        return LLMContextAggregatorPair(context)
    return LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=silero(),
            user_turn_strategies=UserTurnStrategies(
                start=[MinWordsUserTurnStartStrategy(min_words=INTERRUPT_MIN_WORDS)],
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())],
            ),
        ),
    )


def cascade_processors(
    stt: FrameProcessor,
    llm: FrameProcessor,
    tts: FrameProcessor,
    aggregators: LLMContextAggregatorPair,
) -> list[FrameProcessor]:
    """Between transport input and output: transcribe, accumulate, generate, synthesize.

    The assistant-side aggregator is not here; it belongs after the transport's
    output so it records what was actually said, and the caller places it.
    """
    return [stt, aggregators.user(), llm, tts]


class LastWordWatch(FrameProcessor):
    """Silero on the caller's audio as a stopwatch, not a switch.

    A native model does its own turn-taking, but the observer still wants the
    caller's last word on our clock, the same guess the cascade makes. The VAD
    frames go upstream, where no turn strategy can act on them.
    """

    def __init__(self, analyzer: VADAnalyzer) -> None:
        super().__init__()
        self._analyzer = analyzer
        self._vad = VADController(analyzer, audio_idle_timeout=0)
        self._vad.add_event_handler("on_speech_started", self._started)
        self._vad.add_event_handler("on_speech_stopped", self._stopped)

    async def setup(self, setup: FrameProcessorSetup) -> None:  # type: ignore[override]
        await super().setup(setup)
        await self._vad.setup(setup)

    async def cleanup(self) -> None:
        await self._vad.cleanup()  # type: ignore[no-untyped-call]
        await super().cleanup()  # type: ignore[no-untyped-call]

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            await self._vad.process_frame(frame)

    async def _started(self, _controller: VADController) -> None:
        frame = VADUserStartedSpeakingFrame(start_secs=self._analyzer.params.start_secs)
        await self.push_frame(frame, FrameDirection.UPSTREAM)

    async def _stopped(self, _controller: VADController) -> None:
        frame = VADUserStoppedSpeakingFrame(stop_secs=self._analyzer.params.stop_secs)
        await self.push_frame(frame, FrameDirection.UPSTREAM)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: Any


@dataclass(frozen=True)
class Call:
    worker: PipelineWorker
    observer: TurnLatencyObserver
    tool_calls: list[ToolCall] | None
    expected: dict[str, dict[str, str]] = field(default_factory=dict)

    def character_error_rates(self) -> dict[str, float]:
        """``<tool>.<argument>`` to its character error rate so far; empty with no checks."""
        return character_error_rates(self.expected, self.tool_calls or [])


def tool_handler(
    tool: Tool, log: list[ToolCall]
) -> Callable[[FunctionCallParams], Awaitable[None]]:
    """What Pipecat calls. Runs the scenario's handler and hands the LLM the result.

    A handler that raises doesn't end the call: the LLM gets the error as the
    result, the way a real backend would answer, and carries on.
    """

    async def handle(params: FunctionCallParams) -> None:
        arguments = dict(params.arguments)
        try:
            result = await tool.handler(arguments)
        except Exception as error:  # noqa: BLE001
            logger.warning(f"{tool.schema.name} failed: {error!r}")
            result = {"error": f"{type(error).__name__}: {error}"}
        log.append(ToolCall(tool.schema.name, arguments, result))
        await params.result_callback(result)

    return handle


def _register(processor: FrameProcessor, tools: Sequence[Tool], log: list[ToolCall]) -> None:
    """Put the scenario's tools on the service that answers, which has to be an LLM service."""
    if not tools:
        return
    if not isinstance(processor, LLMService):
        raise TypeError(f"{processor} is not an LLM service, so it can't take tools")
    for tool in tools:
        processor.register_function(tool.schema.name, tool_handler(tool, log))


async def build_call(
    agent: Agent,
    transport: BaseTransport,
    settings: Settings,
    *,
    test_case: str | None = None,
    attributes: Mapping[str, Any] | None = None,
    tracer: Tracer | None = None,
    conversation_id: str | None = None,
) -> Call:
    """One call on any transport: services, backend, pipeline, observer and the greeting.

    Talk and benchmark both build their call here, so a change to how a call works
    reaches both and their latency numbers stay comparable. A scenario with
    tools gets a fresh backend seeded for the test case, and its tools go on the
    LLM, or on the native model, which is one. The backend is cleaned up when the
    pipeline finishes.
    """
    services = resolve(agent.stack, agent.system_prompt, settings)
    environment = load_environment(agent.scenario)
    log: list[ToolCall] = []
    if environment is None:
        if test_case is not None:
            raise ValueError(f"{agent.scenario} has no tools, so there is no test case to seed")
        handlers: list[Tool] = []
    else:
        await environment.start(test_case)
        handlers = list(environment.tools)
    full_duplex = isinstance(services, NativeService) and services.full_duplex
    aggregators = context_aggregators(
        [tool.schema for tool in handlers],
        native=isinstance(services, NativeService),
        full_duplex=full_duplex,
    )
    tags = {**(attributes or {}), "agent": agent.slug, "digest": agent.digest}
    stages = models_by_modality(agent)
    match services:
        case CascadeServices(stt, llm, tts):
            _register(llm, handlers, log)
            middle = cascade_processors(stt, llm, tts, aggregators)
            observer = TurnLatencyObserver(
                attributes=tags, models=stages, stt=stt, llm=llm, tts=tts, tracer=tracer
            )
        case NativeService(model):
            _register(model, handlers, log)
            middle = [LastWordWatch(silero()), aggregators.user(), model]
            observer = TurnLatencyObserver(
                attributes=tags,
                models=stages,
                llm=model,
                tracer=tracer,
                native=True,
                endpoints=not full_duplex,
            )
    worker = PipelineWorker(
        Pipeline([transport.input(), *middle, transport.output(), aggregators.assistant()]),
        params=PipelineParams(
            audio_in_sample_rate=IN_SAMPLE_RATE_HZ,
            audio_out_sample_rate=OUT_SAMPLE_RATE_HZ,
            enable_metrics=True,
        ),
        conversation_id=conversation_id,
        enable_rtvi=False,
        observers=[observer],
    )

    @worker.event_handler("on_pipeline_started")  # type: ignore[untyped-decorator]
    async def _greet(started: PipelineWorker, _frame: Frame) -> None:
        await started.queue_frame(LLMRunFrame())

    expected = {t.schema.name: t.character_error_rate for t in handlers if t.character_error_rate}
    call = Call(worker, observer, log if environment else None, expected)

    @worker.event_handler("on_pipeline_finished")  # type: ignore[untyped-decorator]
    async def _finish(_worker: PipelineWorker, _frame: Frame) -> None:
        if environment is not None:
            await environment.finish()
        if tracer is not None and expected:
            rates = call.character_error_rates()
            tracer.start_span(CHARACTER_ERROR_RATE, attributes={**tags, **rates}).end()

    return call
