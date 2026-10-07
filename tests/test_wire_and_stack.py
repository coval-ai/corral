import dataclasses
import json
from typing import Any

import pytest
from pipecat.frames.frames import (
    InputAudioRawFrame,
    InputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup

from agent import catalog
from agent import stack as stacks
from agent.coval import (
    SESSION_READY,
    CovalFrameSerializer,
    authorized,
    simulation_id,
)

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


@pytest.mark.asyncio
async def test_inbound_pcm_is_resampled_once_to_the_pipeline_rate() -> None:
    serializer = await _serializer(24000)
    total = 0
    for _ in range(6):
        frame = await serializer.deserialize(SILENCE_16K_100MS)
        assert isinstance(frame, InputAudioRawFrame)
        assert frame.sample_rate == 24000 and frame.num_channels == 1
        total += len(frame.audio)
    # The stream resampler emits in blocks and holds a little back, so pin the ratio over six.
    expected = len(SILENCE_16K_100MS) * 1.5 * 6
    assert expected * 0.85 <= total <= expected


@pytest.mark.asyncio
async def test_inbound_pcm_passes_through_at_a_matching_rate() -> None:
    serializer = await _serializer(16000)
    frame = await serializer.deserialize(SILENCE_16K_100MS)
    assert isinstance(frame, InputAudioRawFrame)
    assert frame.audio == SILENCE_16K_100MS and frame.sample_rate == 16000


@pytest.mark.asyncio
async def test_speech_markers_become_transport_messages_and_other_text_is_dropped() -> None:
    serializer = await _serializer(24000)
    marker = {"type": "utterance_end", "utterance_id": "utterance-3"}
    frame = await serializer.deserialize(json.dumps(marker))
    assert isinstance(frame, InputTransportMessageFrame)
    assert frame.message == marker
    assert await serializer.deserialize(json.dumps({"type": "mark"})) is None
    assert await serializer.deserialize("not json") is None


@pytest.mark.asyncio
async def test_outbound_audio_and_control_go_out_as_coval_expects() -> None:
    serializer = await _serializer(24000)
    audio = bytes(2 * 2400)
    frame = TTSAudioRawFrame(audio=audio, sample_rate=24000, num_channels=1)
    out = await serializer.serialize(frame)
    assert out == audio
    ready = await serializer.serialize(OutputTransportMessageUrgentFrame(message=SESSION_READY))
    assert isinstance(ready, str) and json.loads(ready) == {"type": "session_ready"}


def test_simulation_id_comes_from_the_upgrade_header_case_insensitively() -> None:
    assert simulation_id({"X-Coval-Simulation-Id": "sim-1"}) == "sim-1"
    assert simulation_id({"x-coval-simulation-id": ""}) is None
    assert simulation_id({}) is None


def test_bearer_token_is_required_only_when_configured() -> None:
    assert authorized({}, None)
    assert authorized({"Authorization": "Bearer s3cret"}, "s3cret")
    assert not authorized({"Authorization": "Bearer other"}, "s3cret")
    assert not authorized({}, "s3cret")


def test_the_catalog_offers_models_per_role_and_the_first_is_the_default() -> None:
    assert catalog.model_ids("stt") == ["deepgram/nova-3", "soniox/stt-rt-v5"]
    assert catalog.model_ids("llm") == ["openai/gpt-4.1", "google/gemini-2.5-flash"]
    assert catalog.model_ids("tts") == ["elevenlabs/eleven_flash_v2", "soniox/tts-rt-v2"]
    stack = stacks.load_stack("baseline")
    picked = (stack.stt.id, stack.llm.id, stack.tts.id)
    assert picked == ("deepgram/nova-3", "openai/gpt-4.1", "elevenlabs/eleven_flash_v2")
    assert stack.turn_taking.vad_stop_secs == 0.2
    assert stack.slug == "deepgram-nova-3--openai-gpt-4.1--elevenlabs-eleven-flash-v2"
    assert stack.system_prompt.startswith("# Role & Objective")


def test_any_mix_of_models_loads_with_its_own_digest() -> None:
    mixed = stacks.load_stack("baseline", "soniox/stt-rt-v5", "openai/gpt-4.1")
    assert (mixed.stt.provider, mixed.tts.provider) == ("soniox", "elevenlabs")
    assert len(mixed.digest) == 64
    assert mixed.digest != stacks.load_stack("baseline").digest
    talk = stacks.load_stack("default")
    assert talk.system_prompt.startswith("You are a helpful assistant on a voice call.")
    assert talk.digest != stacks.load_stack("baseline").digest


def test_an_unknown_model_names_the_ones_that_exist() -> None:
    with pytest.raises(ValueError, match="no llm model 'openai/nope'; have openai/gpt-4.1, google"):
        stacks.load_stack("baseline", llm="openai/nope")


def test_the_catalog_rejects_unknown_keys_but_keeps_rationale() -> None:
    entry = {"provider": "elevenlabs", "model": "x", "voice": "v"}
    assert catalog.TtsPin.model_validate({**entry, "_voice": "Sarah"})
    with pytest.raises(ValueError, match="unknown key"):
        catalog.TtsPin.model_validate({**entry, "voise": "x"})
