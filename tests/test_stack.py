# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from agent import models
from agent import stack as stacks


def test_the_catalog_offers_models_per_role_and_the_first_is_the_default() -> None:
    assert models.model_ids("stt") == ["deepgram/nova-3", "soniox/stt-rt-v5"]
    assert models.model_ids("llm") == ["openai/gpt-4.1", "google/gemini-2.5-flash"]
    assert models.model_ids("tts") == ["elevenlabs/eleven_flash_v2", "soniox/tts-rt-v2"]
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
    assert models.TtsPin.model_validate({**entry, "_voice": "Sarah"})
    with pytest.raises(ValueError, match="unknown key"):
        models.TtsPin.model_validate({**entry, "voise": "x"})
