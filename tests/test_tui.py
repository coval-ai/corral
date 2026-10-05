# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import shutil

import pytest

from agent.cli import BENCHMARK, TALK, choose_mode, choose_scenario, choose_stack


def test_the_stack_menu_lists_slugs_and_takes_a_number(capsys: pytest.CaptureFixture[str]) -> None:
    answers = iter(["zero", "9", "2"])
    chosen = choose_stack("baseline", read=lambda _prompt: next(answers))
    out = capsys.readouterr().out
    assert "1. cascade-nova3-gpt41-flash\n2. cascade-soniox-gemini\n" in out
    assert out.count("pick a number from 1 to 2, or Enter for 1") == 2
    assert chosen.slug == "cascade-soniox-gemini"


def test_enter_takes_the_first_option(capsys: pytest.CaptureFixture[str]) -> None:
    prompts: list[str] = []

    def read(prompt: str) -> str:
        prompts.append(prompt)
        return "  "

    assert choose_stack("baseline", read=read).slug == "cascade-nova3-gpt41-flash"
    assert prompts == ["Stack [1-2, default 1]: "]


def test_the_mode_menu_offers_talk_and_benchmark_but_not_serve(
    capsys: pytest.CaptureFixture[str],
) -> None:
    chosen = choose_mode(read=lambda _prompt: "2")
    out = capsys.readouterr().out
    assert "1. talk       to the agent" in out
    assert "2. benchmark  with Coval" in out
    assert "3." not in out
    assert chosen.name == "benchmark"


def test_the_menu_says_what_a_mode_still_needs(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/local/bin/cloudflared")
    choose_mode(read=lambda _prompt: "2")
    talk, benchmark = capsys.readouterr().out.splitlines()[:2]
    assert talk.startswith("1. talk")
    assert "(needs: brew install portaudio && uv sync --extra local)" in talk
    assert "(needs" not in benchmark


def test_talk_lists_every_scenario_default_first(capsys: pytest.CaptureFixture[str]) -> None:
    chosen = choose_scenario(TALK, read=lambda _prompt: "2")
    out = capsys.readouterr().out
    assert "1. default   helpful assistant\n2. baseline  Ultra Bank\n" in out
    assert chosen == "baseline"


def test_benchmark_lists_only_scenarios_with_coval_ids(capsys: pytest.CaptureFixture[str]) -> None:
    chosen = choose_scenario(BENCHMARK, read=lambda _prompt: "1")
    out = capsys.readouterr().out
    assert "1. baseline  Ultra Bank\n" in out
    assert "default" not in out
    assert chosen == "baseline"
