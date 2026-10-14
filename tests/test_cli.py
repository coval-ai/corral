# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import shutil

import pytest

from corral.cli import BENCHMARK, TALK, choose_mode, choose_model, choose_scenario


def test_each_role_lists_its_models_and_enter_takes_the_first(
    capsys: pytest.CaptureFixture[str],
) -> None:
    answers = iter(["zero", "9", "2"])
    assert choose_model("llm", read=lambda _prompt: next(answers)) == "google/gemini-2.5-flash"
    out = capsys.readouterr().out
    assert "1. openai/gpt-4.1\n2. google/gemini-2.5-flash\n" in out
    assert out.count("pick a number from 1 to 2, or Enter for 1") == 2
    assert choose_model("stt", read=lambda _prompt: "  ") == "deepgram/nova-3"


def test_the_mode_menu_says_what_each_mode_still_needs(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/local/bin/cloudflared")
    assert choose_mode(read=lambda _prompt: "2").name == "benchmark"
    talk, benchmark, *rest = capsys.readouterr().out.splitlines()
    assert talk.startswith("1. talk") and "(needs: brew install portaudio" in talk
    assert benchmark.startswith("2. benchmark") and "(needs" not in benchmark
    assert "3." not in "\n".join(rest)


def test_benchmark_offers_only_scenarios_with_coval_ids(capsys: pytest.CaptureFixture[str]) -> None:
    assert choose_scenario(TALK, read=lambda _prompt: "2") == "ultrabank"
    out = capsys.readouterr().out
    assert "1. default    helpful assistant\n" in out
    assert "2. ultrabank  Instruction adherence test set (internal)\n" in out
    assert choose_scenario(BENCHMARK, read=lambda _prompt: "1") == "ultrabank"
    assert "default" not in capsys.readouterr().out


@pytest.mark.parametrize("key", [KeyboardInterrupt, EOFError])
def test_ctrl_c_or_ctrl_d_at_a_menu_exits_without_a_traceback(key: type[BaseException]) -> None:
    def read(_prompt: str) -> str:
        raise key

    with pytest.raises(SystemExit) as exited:
        choose_scenario(TALK, read=read)
    assert exited.value.code == 130
