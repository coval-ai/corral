# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0


from pipecat.processors.frame_processor import FrameProcessor

from agent.pipeline import cascade_processors, context_aggregators
from agent.stack import load_stack


def test_cascade_runs_transcribe_accumulate_generate_synthesize() -> None:
    aggregators = context_aggregators(load_stack("bank", "cascade-nova3-gpt41-flash"))
    stt, llm, tts = FrameProcessor(), FrameProcessor(), FrameProcessor()
    assert cascade_processors(stt, llm, tts, aggregators) == [
        stt,
        aggregators.user(),
        llm,
        tts,
    ]
