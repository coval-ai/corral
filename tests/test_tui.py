# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from agent.cli import choose_mode, choose_stack, choose_test_set


def test_the_menu_lists_each_stacks_components_and_takes_a_number(
    capsys: pytest.CaptureFixture[str],
) -> None:
    answers = iter(["zero", "9", "2"])
    chosen = choose_stack("bank", read=lambda _prompt: next(answers))
    out = capsys.readouterr().out
    assert "1. cascade-nova3-gpt41-flash" in out
    assert "stt deepgram/nova-3  llm openai/gpt-4.1  tts elevenlabs/eleven_flash_v2" in out
    assert "2. cascade-soniox-gemini" in out
    assert "stt soniox/stt-rt-v5  llm google/gemini-2.5-flash  tts soniox/tts-rt-v2" in out
    assert out.count("pick a number from 1 to 2") == 2
    assert chosen.slug == "cascade-soniox-gemini"


def test_the_mode_menu_offers_talk_and_benchmark_but_not_serve(
    capsys: pytest.CaptureFixture[str],
) -> None:
    chosen = choose_mode(read=lambda _prompt: "2")
    out = capsys.readouterr().out
    assert "1. talk" in out
    assert "2. benchmark" in out
    assert "3." not in out
    assert chosen.name == "benchmark"


def test_the_test_set_menu_names_each_set_and_returns_its_ids(
    capsys: pytest.CaptureFixture[str],
) -> None:
    chosen = choose_test_set("bank", read=lambda _prompt: "1")
    out = capsys.readouterr().out
    assert "1. baseline" in out
    assert chosen.test_set_id
