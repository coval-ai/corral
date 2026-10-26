# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from corral import agent as agents
from corral import models

REFERENCE = ("deepgram/nova-3", "openai/gpt-4.1", "elevenlabs/eleven_flash_v2")


def test_the_catalog_offers_models_per_role_and_any_mix_loads() -> None:
    assert models.model_ids("stt") == ["deepgram/nova-3", "soniox/stt-rt-v5"]
    assert models.model_ids("llm") == ["openai/gpt-4.1", "google/gemini-2.5-flash"]
    assert models.model_ids("tts") == ["elevenlabs/eleven_flash_v2", "soniox/tts-rt-v2"]
    agent = agents.load_cascade_agent("default", *REFERENCE)
    assert agent.slug == "deepgram-nova-3--openai-gpt-4.1--elevenlabs-eleven-flash-v2"
    assert agent.system_prompt.startswith("You are a helpful assistant")
    mixed = agents.load_cascade_agent("default", "soniox/stt-rt-v5", *REFERENCE[1:])
    assert len(mixed.digest) == 64 and mixed.digest != agent.digest
    assert agents.load_cascade_agent("id_roulette", *REFERENCE).digest != agent.digest
    assert isinstance(agent.stack, models.Cascade)


def test_a_native_model_is_a_stack_of_one() -> None:
    assert models.model_ids("s2s") == [
        "openai/gpt-realtime-2.1",
        "openai/gpt-realtime-2.1-mini",
        "openai/gpt-live-1",
        "google/gemini-3.8-live",
        "google/gemini-3.8-live-extended-thinking",
        "xai/grok-voice-3.0",
        "elevenlabs/eleven_realtime_v1_mini",
        "elevenlabs/eleven_realtime_v1_max",
    ]
    agent = agents.load_native_agent("default", "openai/gpt-realtime-2.1")
    assert isinstance(agent.stack, models.Native)
    assert agent.slug == "openai-gpt-realtime-2.1"
    assert agent.stack.rows == (agent.stack.s2s,)
    assert agent.system_prompt == agents.load_cascade_agent("default", *REFERENCE).system_prompt
    mini = agents.load_native_agent("default", "openai/gpt-realtime-2.1-mini")
    assert len(agent.digest) == 64 and mini.digest != agent.digest
    live = agents.load_native_agent("default", "openai/gpt-live-1")
    assert live.slug == "openai-gpt-live-1" and live.digest not in {agent.digest, mini.digest}
    assert set(agents._config(agent.stack)["s2s"]) == {"provider", "name", "options"}
    with pytest.raises(ValueError, match="no s2s model 'openai/nope'; have openai/gpt-realtime"):
        agents.load_native_agent("default", "openai/nope")


def test_an_unknown_model_or_a_misspelled_catalog_key_fails_to_load() -> None:
    with pytest.raises(ValueError, match="no llm model 'openai/nope'; have openai/gpt-4.1, google"):
        agents.load_cascade_agent("default", REFERENCE[0], "openai/nope", REFERENCE[2])
    entry = {"provider": "elevenlabs", "name": "x", "voice": "v", "voise": "x"}
    with pytest.raises(ValueError, match="voise"):
        models.TtsModel.model_validate(entry)


def test_an_asset_agent_runs_the_prompt_it_is_given_and_its_digest_follows_it() -> None:
    asset = agents.load_asset_agent("workflow-bank.v1", " Be brief. ", "openai/gpt-realtime-2.1")
    assert asset.scenario == "workflow-bank.v1"
    assert asset.system_prompt == "Be brief."
    assert asset.slug == agents.load_native_agent("default", "openai/gpt-realtime-2.1").slug
    other = agents.load_asset_agent("workflow-bank.v1", "Be kind.", "openai/gpt-realtime-2.1")
    assert other.digest != asset.digest
    cascade = agents.load_asset_agent("workflow-bank.v1", "Be brief.", None, *REFERENCE)
    assert isinstance(cascade.stack, models.Cascade)
    with pytest.raises(ValueError, match="prompt is empty"):
        agents.load_asset_agent("workflow-bank.v1", "  ", "openai/gpt-realtime-2.1")
