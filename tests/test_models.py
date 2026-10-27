# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from google.genai.types import ThinkingLevel
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.live.llm import OpenAILiveLLMService, ResponsesDelegation
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.services.openai.responses.llm import OpenAIResponsesReasoningConfig
from pipecat.services.soniox.stt import SonioxSTTService
from pipecat.services.soniox.tts import SonioxTTSService
from pipecat.services.xai.realtime.llm import GrokRealtimeLLMService

from corral import models
from corral.agent import load_cascade_agent, load_native_agent
from corral.config import STT_FINALIZE_FALLBACK_SECS, Settings
from corral.models.eleven import ElevenRealtimeLLMService

REFERENCE = ("deepgram/nova-3", "openai/gpt-4.1", "elevenlabs/eleven_flash_v2")
SONIOX_GEMINI = ("soniox/stt-rt-v5", "google/gemini-2.5-flash", "soniox/tts-rt-v2")
KEYS: dict[str, Any] = {
    "deepgram_api_key": "dg",
    "openai_api_key": "oa",
    "elevenlabs_api_key": "el",
    "soniox_api_key": "sx",
    "gemini_api_key": "gm",
    "xai_api_key": "xa",
}


def test_each_stack_resolves_to_its_vendors_with_our_endpointing() -> None:
    settings = Settings(_env_file=None, **KEYS)
    agent = load_cascade_agent("default", *REFERENCE)
    reference = models.resolve(agent.stack, agent.system_prompt, settings)
    assert isinstance(reference, models.CascadeServices)
    assert isinstance(reference.stt, DeepgramSTTService)
    assert isinstance(reference.llm, OpenAILLMService)
    assert isinstance(reference.tts, ElevenLabsTTSService)
    assert vars(reference.stt._settings)["endpointing"] is False
    assert reference.stt._stt_ttfb_timeout == STT_FINALIZE_FALLBACK_SECS
    assert vars(reference.llm._settings)["system_instruction"] == agent.system_prompt
    mixed = load_cascade_agent("default", *SONIOX_GEMINI)
    soniox = models.resolve(mixed.stack, mixed.system_prompt, settings)
    assert isinstance(soniox, models.CascadeServices)
    assert isinstance(soniox.stt, SonioxSTTService)
    assert isinstance(soniox.llm, GoogleLLMService)
    assert isinstance(soniox.tts, SonioxTTSService)
    assert soniox.stt._vad_force_turn_endpoint is True
    assert vars(soniox.llm._settings)["thinking"].thinking_budget == 0


def test_a_native_model_resolves_to_one_service_with_the_prompt_on_it() -> None:
    settings = Settings(_env_file=None, **KEYS)
    agent = load_native_agent("default", "openai/gpt-realtime-2.1")
    native = models.resolve(agent.stack, agent.system_prompt, settings)
    assert isinstance(native, models.NativeService)
    assert isinstance(native.model, OpenAIRealtimeLLMService)
    assert native.reports_turn_end
    session = vars(native.model._settings)["session_properties"]
    assert vars(native.model._settings)["model"] == "gpt-realtime-2.1"
    assert session.instructions == agent.system_prompt
    assert session.audio.output.voice == "marin"
    assert session.audio.input.transcription.model == "gpt-4o-mini-transcribe"
    assert session.audio.input.turn_detection is None
    assert isinstance(agent.stack, models.Native)
    eager = agent.stack.s2s.model_copy(
        update={"options": {"audio": {"input": {"turn_detection": {"type": "semantic_vad"}}}}}
    )
    service = models.resolve(models.Native(eager), agent.system_prompt, settings)
    assert isinstance(service, models.NativeService)
    session = vars(cast(Any, service.model)._settings)["session_properties"]
    assert session.audio.input.turn_detection.type == "semantic_vad"


def test_a_live_model_delegates_to_a_responses_model_on_the_same_prompt() -> None:
    agent = load_native_agent("default", "openai/gpt-live-1")
    native = models.resolve(agent.stack, agent.system_prompt, Settings(_env_file=None, **KEYS))
    assert isinstance(native, models.NativeService) and not native.reports_turn_end
    assert isinstance(native.model, OpenAILiveLLMService)
    live = vars(native.model._settings)
    assert (live["model"], live["voice"]) == ("gpt-live-1", "alloy")
    assert live["system_instruction"] == agent.system_prompt
    delegation: Any = native.model._delegation
    assert isinstance(delegation, ResponsesDelegation)
    assert delegation.service_tier == "priority"
    backend = delegation.settings
    assert backend.model == "gpt-5.6-luna"
    assert backend.system_instruction == agent.system_prompt
    assert backend.max_completion_tokens == 2048
    assert backend.reasoning == OpenAIResponsesReasoningConfig(effort="medium", summary="auto")
    assert backend.extra == {"text": {"verbosity": "low"}}


def test_gemini_3_8_live_runs_on_its_coval_agents_session_settings() -> None:
    agent = load_native_agent("default", "google/gemini-3.8-live")
    gemini = models.resolve(agent.stack, "x", Settings(_env_file=None, **KEYS))
    assert isinstance(gemini, models.NativeService) and not gemini.reports_turn_end
    assert isinstance(gemini.model, GeminiLiveLLMService)
    live = vars(gemini.model._settings)
    assert (live["voice"], live["temperature"]) == ("Charon", 0.8)
    assert gemini.model._resolved_thinking_config() is None


def test_gemini_3_8_live_extended_thinking_thinks_at_medium() -> None:
    agent = load_native_agent("default", "google/gemini-3.8-live-extended-thinking")
    gemini = models.resolve(agent.stack, "x", Settings(_env_file=None, **KEYS))
    assert isinstance(gemini, models.NativeService) and not gemini.reports_turn_end
    assert isinstance(gemini.model, GeminiLiveLLMService)
    live = vars(gemini.model._settings)
    assert (live["voice"], live["temperature"]) == ("Charon", 0.8)
    thinking = gemini.model._resolved_thinking_config()
    assert thinking is not None and thinking.thinking_level == ThinkingLevel.MEDIUM


def test_grok_voice_3_runs_on_its_coval_agents_session_settings() -> None:
    agent = load_native_agent("default", "xai/grok-voice-3.0")
    grok = models.resolve(agent.stack, "x", Settings(_env_file=None, **KEYS))
    assert isinstance(grok, models.NativeService) and grok.reports_turn_end
    assert isinstance(grok.model, GrokRealtimeLLMService)
    session = vars(grok.model._settings)["session_properties"]
    assert session.voice == "ara"
    assert session.turn_detection.silence_duration_ms == 500


def test_eleven_realtime_runs_on_its_coval_agents_session_settings() -> None:
    settings = Settings(_env_file=None, **KEYS)
    for name in ("eleven_realtime_v1_mini", "eleven_realtime_v1_max"):
        agent = load_native_agent("default", f"elevenlabs/{name}")
        eleven = models.resolve(agent.stack, "x", settings)
        assert isinstance(eleven, models.NativeService)
        assert isinstance(eleven.model, ElevenRealtimeLLMService)
        assert eleven.model.base_url == f"wss://api.elevenlabs.io/v1/convai/realtime?model={name}"
        session = vars(eleven.model._settings)["session_properties"]
        assert session.audio.output.voice == "brian"
        assert session.audio.input.format.rate == 24000
        assert session.audio.input.turn_detection.eagerness == "medium"


async def test_eleven_sends_its_key_in_xi_api_key_not_bearer() -> None:
    agent = load_native_agent("default", "elevenlabs/eleven_realtime_v1_mini")
    eleven = models.resolve(agent.stack, "x", Settings(_env_file=None, **KEYS))
    assert isinstance(eleven, models.NativeService)
    service = cast(Any, eleven.model)
    with (
        patch("corral.models.eleven.websocket_connect", AsyncMock()) as connect,
        patch.object(service, "create_task"),
        patch.object(service, "_receive_task_handler", MagicMock()),
    ):
        await service._connect()
    assert connect.call_args.kwargs["additional_headers"] == {"xi-api-key": "el"}


def test_turn_ends_and_delegation_come_from_the_vendors_api_not_its_name() -> None:
    settings = Settings(_env_file=None, **KEYS)
    delegating = models.S2sModel(provider="openai", name="x", delegation={"model": "y"})
    with pytest.raises(ValueError, match="its realtime API doesn't take"):
        models.resolve(models.Native(delegating), "x", settings)
    odd = models.S2sModel(provider="openai", name="x", api="nope")
    with pytest.raises(ValueError, match="no 'nope' s2s for 'openai'; have realtime, live"):
        models.resolve(models.Native(odd), "x", settings)
    plain = models.S2sApi("pipecat.services.openai.live.llm:OpenAILiveLLMService")
    vendor = dataclasses.replace(models.VENDORS["openai"], s2s={"live": plain})
    live = models.S2sModel(provider="openai", name="gpt-live-1", api="live")
    with patch.dict(models.VENDORS, {"openai": vendor}):
        service = models.resolve(models.Native(live), "x", settings)
    assert isinstance(service, models.NativeService) and service.reports_turn_end


def test_a_bad_provider_or_a_missing_key_fails_before_anything_connects() -> None:
    agent = load_cascade_agent("default", *REFERENCE)
    assert isinstance(agent.stack, models.Cascade)
    nope = agent.stack.tts.model_copy(update={"provider": "nope"})
    bad = dataclasses.replace(agent.stack, tts=nope)
    with pytest.raises(ValueError, match="no tts for 'nope'; have elevenlabs, soniox"):
        models.resolve(bad, agent.system_prompt, Settings(_env_file=None, **KEYS))
    unpaid = Settings(_env_file=None, **{**KEYS, "soniox_api_key": None})
    with pytest.raises(RuntimeError, match="SONIOX_API_KEY is unset"):
        models.resolve(load_cascade_agent("default", *SONIOX_GEMINI).stack, "x", unpaid)
    no_openai = Settings(_env_file=None, **{**KEYS, "openai_api_key": None})
    native = load_native_agent("default", "openai/gpt-realtime-2.1")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is unset"):
        models.resolve(native.stack, "x", no_openai)
    assert models.supported("s2s") == ["elevenlabs", "google", "openai", "xai"]


def test_every_vendor_check_sends_its_key_the_way_that_vendor_wants() -> None:
    seen: dict[str, httpx.Request] = {}

    def record(request: httpx.Request) -> httpx.Response:
        seen[request.url.host] = request
        return httpx.Response(200)

    settings = Settings(_env_file=None, **KEYS)
    for provider in models.VENDORS:
        assert models.vendor_problem(provider, settings, httpx.MockTransport(record)) is None
    assert seen["api.deepgram.com"].headers["Authorization"] == "Token dg"
    assert seen["api.openai.com"].headers["Authorization"] == "Bearer oa"
    assert seen["api.elevenlabs.io"].headers["xi-api-key"] == "el"
    assert seen["api.soniox.com"].headers["Authorization"] == "Bearer sx"
    assert seen["generativelanguage.googleapis.com"].headers["x-goog-api-key"] == "gm"
    assert seen["api.x.ai"].headers["Authorization"] == "Bearer xa"
    assert all(request.method == "GET" and not request.url.query for request in seen.values())


def test_a_vendor_problem_is_the_status_the_unset_key_or_the_network() -> None:
    settings = Settings(_env_file=None, **KEYS)
    denied = httpx.MockTransport(lambda _request: httpx.Response(401))
    assert models.vendor_problem("soniox", settings, denied) == "soniox returned 401"

    def down(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    unreachable = models.vendor_problem("openai", settings, httpx.MockTransport(down))
    assert unreachable == "openai is unreachable: no route"
    unpaid = Settings(_env_file=None, **{**KEYS, "gemini_api_key": ""})
    assert models.vendor_problem("google", unpaid, denied) == "GEMINI_API_KEY is unset"
