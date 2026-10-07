# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Any

import pytest
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.soniox.stt import SonioxSTTService
from pipecat.services.soniox.tts import SonioxTTSService

from corral import models
from corral.agent import load_agent
from corral.config import Settings

REFERENCE = ("deepgram/nova-3", "openai/gpt-4.1", "elevenlabs/eleven_flash_v2")
SONIOX_GEMINI = ("soniox/stt-rt-v5", "google/gemini-2.5-flash", "soniox/tts-rt-v2")
KEYS: dict[str, Any] = {
    "deepgram_api_key": "dg",
    "openai_api_key": "oa",
    "elevenlabs_api_key": "el",
    "soniox_api_key": "sx",
    "gemini_api_key": "gm",
}


def test_each_role_lists_the_vendors_that_fill_it() -> None:
    assert models.supported("stt") == ["deepgram", "soniox"]
    assert models.supported("llm") == ["google", "openai"]
    assert models.supported("tts") == ["elevenlabs", "soniox"]


def test_the_reference_stack_resolves_to_deepgram_openai_elevenlabs() -> None:
    built = models.resolve(load_agent("baseline", *REFERENCE), Settings(_env_file=None, **KEYS))
    assert isinstance(built.stt, DeepgramSTTService)
    assert isinstance(built.llm, OpenAILLMService)
    assert isinstance(built.tts, ElevenLabsTTSService)


def test_the_soniox_gemini_stack_resolves_to_soniox_google_soniox() -> None:
    built = models.resolve(load_agent("baseline", *SONIOX_GEMINI), Settings(_env_file=None, **KEYS))
    assert isinstance(built.stt, SonioxSTTService)
    assert isinstance(built.llm, GoogleLLMService)
    assert isinstance(built.tts, SonioxTTSService)


def test_an_unknown_provider_fails_before_anything_connects() -> None:
    agent = load_agent("baseline", *REFERENCE)
    bad = dataclasses.replace(agent, tts=agent.tts.model_copy(update={"provider": "nope"}))
    with pytest.raises(ValueError, match="no tts for 'nope'; have elevenlabs, soniox"):
        models.resolve(bad, Settings(_env_file=None, **KEYS))


def test_a_missing_key_names_the_env_var() -> None:
    keys = {**KEYS, "soniox_api_key": None}
    with pytest.raises(RuntimeError, match="SONIOX_API_KEY is unset"):
        models.resolve(load_agent("baseline", *SONIOX_GEMINI), Settings(_env_file=None, **keys))


def test_the_system_prompt_rides_on_the_llm_service() -> None:
    agent = load_agent("baseline", *REFERENCE)
    built = models.resolve(agent, Settings(_env_file=None, **KEYS))
    assert isinstance(built.llm, LLMService)
    assert built.llm._settings.system_instruction == agent.system_prompt


def test_every_stack_runs_its_stt_with_endpointing_off() -> None:
    deepgram = models.resolve(load_agent("baseline", *REFERENCE), Settings(_env_file=None, **KEYS))
    assert isinstance(deepgram.stt, DeepgramSTTService)
    assert vars(deepgram.stt._settings)["endpointing"] is False
    soniox = models.resolve(
        load_agent("baseline", *SONIOX_GEMINI), Settings(_env_file=None, **KEYS)
    )
    assert isinstance(soniox.stt, SonioxSTTService)
    assert soniox.stt._vad_force_turn_endpoint is True
