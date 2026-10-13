# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0


from pipecat.processors.frame_processor import FrameProcessor

from corral.agent import load_agent
from corral.pipeline import cascade_processors, context_aggregators

REFERENCE = ("deepgram/nova-3", "openai/gpt-4.1", "elevenlabs/eleven_flash_v2")


def test_cascade_runs_transcribe_accumulate_generate_synthesize() -> None:
    aggregators = context_aggregators(load_agent("ultrabank", *REFERENCE))
    stt, llm, tts = FrameProcessor(), FrameProcessor(), FrameProcessor()
    assert cascade_processors(stt, llm, tts, aggregators) == [
        stt,
        aggregators.user(),
        llm,
        tts,
    ]
