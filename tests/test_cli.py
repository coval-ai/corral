# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import shutil
from typing import Any

import pytest

from corral.cli import BENCHMARK, TALK, checker, choose_mode, choose_model, choose_scenario
from corral.config import Settings

KEYS: dict[str, Any] = {
    "deepgram_api_key": "dg",
    "openai_api_key": "oa",
    "elevenlabs_api_key": "el",
    "soniox_api_key": "sx",
    "gemini_api_key": "gm",
}
ALL_KEYS = Settings(_env_file=None, **KEYS)


def answers_fine(_provider: str) -> str | None:
    return None


def test_each_role_lists_its_models_and_enter_takes_the_first(
    capsys: pytest.CaptureFixture[str],
) -> None:
    answers = iter(["zero", "9", "2"])
    chosen = choose_model(
        "llm", read=lambda _prompt: next(answers), settings=ALL_KEYS, check=answers_fine
    )
    assert chosen == "google/gemini-2.5-flash"
    out = capsys.readouterr().out
    assert "1. openai/gpt-4.1\n2. google/gemini-2.5-flash\n" in out
    assert out.count("pick a number from 1 to 2, or Enter for 1") == 2
    stt = choose_model("stt", read=lambda _prompt: "  ", settings=ALL_KEYS, check=answers_fine)
    assert stt == "deepgram/nova-3"


def test_models_without_a_key_are_marked_and_enter_skips_them(
    capsys: pytest.CaptureFixture[str],
) -> None:
    no_deepgram = Settings(_env_file=None, **{**KEYS, "deepgram_api_key": None})
    answers = iter(["1", ""])
    prompts: list[str] = []

    def read(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    assert choose_model("stt", read, no_deepgram, answers_fine) == "soniox/stt-rt-v5"
    out = capsys.readouterr().out
    assert "1. deepgram/nova-3  (needs: DEEPGRAM_API_KEY)\n2. soniox/stt-rt-v5\n" in out
    assert prompts == ["STT [1-2, default 2]: "] * 2
    assert "deepgram/nova-3 needs DEEPGRAM_API_KEY in .env" in out


def test_no_key_for_a_modality_says_which_to_set() -> None:
    none = Settings(_env_file=None, **{**KEYS, "openai_api_key": None, "gemini_api_key": None})
    with pytest.raises(SystemExit, match="set one of GEMINI_API_KEY, OPENAI_API_KEY in .env"):
        choose_model("llm", read=lambda _prompt: "", settings=none, check=answers_fine)


def test_a_vendor_that_does_not_answer_is_asked_again(capsys: pytest.CaptureFixture[str]) -> None:
    asked: list[str] = []

    def check(provider: str) -> str | None:
        asked.append(provider)
        return "openai returned 401" if provider == "openai" else None

    answers = iter(["1", "2"])
    chosen = choose_model("llm", read=lambda _prompt: next(answers), settings=ALL_KEYS, check=check)
    assert chosen == "google/gemini-2.5-flash"
    out = capsys.readouterr().out
    assert "openai returned 401\n1. openai/gpt-4.1  (openai returned 401)\n2. google/gemini" in out
    assert asked == ["openai", "google"]


def test_the_checker_asks_each_vendor_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def probe(provider: str, _settings: Settings) -> str | None:
        calls.append(provider)
        return None

    monkeypatch.setattr("corral.cli.vendor_problem", probe)
    check = checker(ALL_KEYS)
    assert check("soniox") is None and check("soniox") is None and check("google") is None
    assert calls == ["soniox", "google"]


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
