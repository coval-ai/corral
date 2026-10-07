# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""What sits between the transport's input and output, in order."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from opentelemetry.trace import Tracer
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.base_transport import BaseTransport
from pipecat.turns.user_start.min_words_user_turn_start_strategy import (
    MinWordsUserTurnStartStrategy,
)
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from corral.agent import Agent
from corral.config import (
    IN_SAMPLE_RATE_HZ,
    INTERRUPT_MIN_WORDS,
    OUT_SAMPLE_RATE_HZ,
    VAD_STOP_SECS,
    Settings,
)
from corral.models import resolve
from corral.observer import TurnLatencyObserver, models_by_modality


def context_aggregators(agent: Agent) -> LLMContextAggregatorPair:
    """The conversation context and the turn-taking the agent pins around it.

    The system prompt is not here; it rides on the LLM service's Settings so
    Pipecat composes it with its own instructions. Silero decides when the
    caller is speaking; smart-turn decides when a pause is the end of the turn
    rather than a breath. Both are local models shipped with Pipecat, so the
    cascade's endpointing is the same on every machine. While the agent is
    speaking, the caller has to say ``interrupt_min_words`` before it yields,
    so a backchannel does not cut it off.
    """
    context = LLMContext()
    return LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS)),
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


@dataclass(frozen=True)
class Call:
    worker: PipelineWorker
    observer: TurnLatencyObserver


def build_call(
    agent: Agent,
    transport: BaseTransport,
    settings: Settings,
    *,
    attributes: Mapping[str, Any] | None = None,
    tracer: Tracer | None = None,
    conversation_id: str | None = None,
    before_greeting: Sequence[Frame] = (),
) -> Call:
    """One call on any transport: the services, the pipeline, the observer and the greeting.

    Talk and serve both build their call here, so a change to how a call works
    reaches both and their latency numbers stay comparable.
    """
    services = resolve(agent, settings)
    observer = TurnLatencyObserver(
        attributes={**(attributes or {}), "agent": agent.slug, "digest": agent.digest},
        models=models_by_modality(agent),
        stt=services.stt,
        tracer=tracer,
    )
    aggregators = context_aggregators(agent)
    middle = cascade_processors(services.stt, services.llm, services.tts, aggregators)
    worker = PipelineWorker(
        Pipeline([transport.input(), *middle, transport.output(), aggregators.assistant()]),
        params=PipelineParams(
            audio_in_sample_rate=IN_SAMPLE_RATE_HZ,
            audio_out_sample_rate=OUT_SAMPLE_RATE_HZ,
        ),
        conversation_id=conversation_id,
        enable_rtvi=False,
        observers=[observer],
    )

    @worker.event_handler("on_pipeline_started")  # type: ignore[untyped-decorator]
    async def _greet(started: PipelineWorker, _frame: Frame) -> None:
        for frame in before_greeting:
            await started.queue_frame(frame)
        await started.queue_frame(LLMRunFrame())

    return Call(worker, observer)
