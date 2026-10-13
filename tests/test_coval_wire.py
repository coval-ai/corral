# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
from typing import Any

from pipecat.frames.frames import (
    InputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup

from corral.coval_wire import SESSION_READY, CovalFrameSerializer, authorized, simulation_id

SILENCE_16K_100MS = bytes(2 * 1600)


async def _serializer(pipeline_in_hz: int) -> CovalFrameSerializer:
    serializer = CovalFrameSerializer()
    plumbing: dict[str, Any] = {
        field.name: None
        for field in dataclasses.fields(FrameProcessorSetup)
        if not field.name.startswith("audio_")
    }
    await serializer.setup(
        FrameProcessorSetup(
            **plumbing, audio_in_sample_rate=pipeline_in_hz, audio_out_sample_rate=24000
        )
    )
    return serializer


async def test_inbound_pcm_is_resampled_to_the_pipeline_rate() -> None:
    serializer = await _serializer(24000)
    total = 0
    for _ in range(6):
        frame = await serializer.deserialize(SILENCE_16K_100MS)
        assert isinstance(frame, InputAudioRawFrame) and frame.sample_rate == 24000
        total += len(frame.audio)
    expected = len(SILENCE_16K_100MS) * 1.5 * 6
    assert expected * 0.85 <= total <= expected
    same = await (await _serializer(16000)).deserialize(SILENCE_16K_100MS)
    assert isinstance(same, InputAudioRawFrame) and same.audio == SILENCE_16K_100MS


async def test_inbound_text_is_dropped() -> None:
    serializer = await _serializer(24000)
    marker = {"type": "utterance_end", "utterance_id": "utterance-3"}
    assert await serializer.deserialize(json.dumps(marker)) is None
    assert await serializer.deserialize("not json") is None


async def test_outbound_audio_and_control_go_out_as_coval_expects() -> None:
    serializer = await _serializer(24000)
    audio = bytes(2 * 2400)
    frame = TTSAudioRawFrame(audio=audio, sample_rate=24000, num_channels=1)
    assert await serializer.serialize(frame) == audio
    ready = await serializer.serialize(OutputTransportMessageUrgentFrame(message=SESSION_READY))
    assert isinstance(ready, str) and json.loads(ready) == {"type": "session_ready"}


def test_the_upgrade_headers_carry_the_simulation_id_and_the_bearer() -> None:
    assert simulation_id({"x-coval-simulation-id": "sim-1"}) == "sim-1"
    assert simulation_id({}) is None
    assert authorized({}, None)
    assert authorized({"Authorization": "Bearer s3cret"}, "s3cret")
    assert not authorized({"Authorization": "Bearer other"}, "s3cret")
