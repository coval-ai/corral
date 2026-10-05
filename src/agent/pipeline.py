# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""What sits between the transport's input and output, in order."""

from __future__ import annotations

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
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

from agent.observer import TurnLatencyObserver
from agent.services import Services
from agent.stack import LoadedStack


def context_aggregators(loaded: LoadedStack) -> LLMContextAggregatorPair:
    """The conversation context and the turn-taking the stack pins around it.

    The system prompt is not here; it rides on the LLM service's Settings so
    Pipecat composes it with its own instructions. Silero decides when the
    caller is speaking; smart-turn decides when a pause is the end of the turn
    rather than a breath. Both are local models shipped with Pipecat, so the
    cascade's endpointing is the same on every machine. While the agent is
    speaking, the caller has to say ``interrupt_min_words`` before it yields,
    so a backchannel does not cut it off.
    """
    turn_taking = loaded.stack.turn_taking
    context = LLMContext()
    return LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=turn_taking.vad_stop_secs)),
            user_turn_strategies=UserTurnStrategies(
                start=[MinWordsUserTurnStartStrategy(min_words=turn_taking.interrupt_min_words)],
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


def build_worker(
    loaded: LoadedStack,
    services: Services,
    transport: BaseTransport,
    observer: TurnLatencyObserver,
    *,
    conversation_id: str | None = None,
) -> PipelineWorker:
    """The whole pipeline between one transport's input and output, at the stack's rates.

    Both the mic mode and the served mode build their worker here, so a change
    to the pipeline's shape or parameters reaches both and their latency
    numbers stay comparable.
    """
    audio = loaded.stack.audio
    aggregators = context_aggregators(loaded)
    middle = cascade_processors(services.stt, services.llm, services.tts, aggregators)
    return PipelineWorker(
        Pipeline([transport.input(), *middle, transport.output(), aggregators.assistant()]),
        params=PipelineParams(
            audio_in_sample_rate=audio.in_sample_rate_hz,
            audio_out_sample_rate=audio.out_sample_rate_hz,
        ),
        conversation_id=conversation_id,
        enable_rtvi=False,
        observers=[observer],
    )
