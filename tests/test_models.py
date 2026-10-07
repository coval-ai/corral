# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Any

import pytest
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.soniox.stt import SonioxSTTService
from pipecat.services.soniox.tts import SonioxTTSService

from corral import models
from corral.agent import load_agent
from corral.config import STT_FINALIZE_FALLBACK_SECS, Settings

REFERENCE = ("deepgram/nova-3", "openai/gpt-4.1", "elevenlabs/eleven_flash_v2")
SONIOX_GEMINI = ("soniox/stt-rt-v5", "google/gemini-2.5-flash", "soniox/tts-rt-v2")
KEYS: dict[str, Any] = {
    "deepgram_api_key": "dg",
    "openai_api_key": "oa",
    "elevenlabs_api_key": "el",
    "soniox_api_key": "sx",
    "gemini_api_key": "gm",
}


def test_each_stack_resolves_to_its_vendors_with_our_endpointing() -> None:
    settings = Settings(_env_file=None, **KEYS)
    agent = load_agent("baseline", *REFERENCE)
    reference = models.resolve(agent, settings)
    assert isinstance(reference.stt, DeepgramSTTService)
    assert isinstance(reference.llm, OpenAILLMService)
    assert isinstance(reference.tts, ElevenLabsTTSService)
    assert vars(reference.stt._settings)["endpointing"] is False
    assert reference.stt._stt_ttfb_timeout == STT_FINALIZE_FALLBACK_SECS
    assert vars(reference.llm._settings)["system_instruction"] == agent.system_prompt
    soniox = models.resolve(load_agent("baseline", *SONIOX_GEMINI), settings)
    assert isinstance(soniox.stt, SonioxSTTService)
    assert isinstance(soniox.llm, GoogleLLMService)
    assert isinstance(soniox.tts, SonioxTTSService)
    assert soniox.stt._vad_force_turn_endpoint is True


def test_a_bad_provider_or_a_missing_key_fails_before_anything_connects() -> None:
    agent = load_agent("baseline", *REFERENCE)
    bad = dataclasses.replace(agent, tts=agent.tts.model_copy(update={"provider": "nope"}))
    with pytest.raises(ValueError, match="no tts for 'nope'; have elevenlabs, soniox"):
        models.resolve(bad, Settings(_env_file=None, **KEYS))
    unpaid = Settings(_env_file=None, **{**KEYS, "soniox_api_key": None})
    with pytest.raises(RuntimeError, match="SONIOX_API_KEY is unset"):
        models.resolve(load_agent("baseline", *SONIOX_GEMINI), unpaid)
