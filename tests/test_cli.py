# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import shutil
import socket
from typing import Any

import pytest

from corral.cli import (
    BENCHMARK,
    SERVE,
    TALK,
    build_parser,
    checker,
    choose_mode,
    choose_model,
    choose_port,
    choose_scenario,
    offered_models,
    port_in_use,
)
from corral.config import Settings

KEYS: dict[str, Any] = {
    "coval_api_key": "ck",
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


def test_only_models_with_a_key_are_offered(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    no_deepgram = Settings(_env_file=None, **{**KEYS, "deepgram_api_key": None})
    prompts: list[str] = []

    def read(prompt: str) -> str:
        prompts.append(prompt)
        return ""

    assert choose_model("stt", read, no_deepgram, answers_fine) == "soniox/stt-rt-v5"
    out = capsys.readouterr().out
    assert "1. soniox/stt-rt-v5\n" in out and "deepgram" not in out
    assert prompts == ["STT [1-1, default 1]: "]
    assert [m.id for m in offered_models("stt", no_deepgram)] == ["soniox/stt-rt-v5"]
    monkeypatch.delenv("S2S_STT", raising=False)
    parser = build_parser(no_deepgram)
    assert parser.parse_args(["benchmark"]).stt == "soniox/stt-rt-v5"
    assert "  stt  soniox/stt-rt-v5\n" in parser.format_help()
    with pytest.raises(SystemExit):
        parser.parse_args(["benchmark", "--stt", "deepgram/nova-3"])
    monkeypatch.setenv("S2S_STT", "deepgram/nova-3")
    assert build_parser(no_deepgram).parse_args(["benchmark"]).stt == "deepgram/nova-3"


def test_no_key_for_a_modality_says_which_to_set() -> None:
    none = Settings(_env_file=None, **{**KEYS, "openai_api_key": None, "gemini_api_key": None})
    with pytest.raises(SystemExit, match="set one of GEMINI_API_KEY, OPENAI_API_KEY in .env"):
        choose_model("llm", read=lambda _prompt: "", settings=none, check=answers_fine)
    with pytest.raises(SystemExit, match="no llm model has a key"):
        build_parser(none)


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
    assert choose_mode(read=lambda _prompt: "2", settings=ALL_KEYS).name == "benchmark"
    talk, benchmark, *rest = capsys.readouterr().out.splitlines()
    assert talk.startswith("1. talk") and "(needs: uv sync)" in talk
    assert benchmark.startswith("2. benchmark") and "(needs" not in benchmark
    assert "3." not in "\n".join(rest)
    no_coval = Settings(_env_file=None, **{**KEYS, "coval_api_key": None})
    choose_mode(read=lambda _prompt: "1", settings=no_coval)
    _, benchmark, *_ = capsys.readouterr().out.splitlines()
    assert benchmark.endswith("(needs: COVAL_API_KEY in .env)")


def test_benchmark_offers_only_scenarios_with_coval_ids(capsys: pytest.CaptureFixture[str]) -> None:
    assert choose_scenario(TALK, read=lambda _prompt: "3") == "ultrabank"
    out = capsys.readouterr().out
    assert "1. default    helpful assistant\n" in out
    assert "2. dental     dental office receptionist with a fake booking system\n" in out
    assert "3. ultrabank  Instruction adherence test set (internal)\n" in out
    assert choose_scenario(BENCHMARK, read=lambda _prompt: "1") == "ultrabank"
    assert "default" not in capsys.readouterr().out


@pytest.mark.parametrize("key", [KeyboardInterrupt, EOFError])
def test_ctrl_c_or_ctrl_d_at_a_menu_exits_without_a_traceback(key: type[BaseException]) -> None:
    def read(_prompt: str) -> str:
        raise key

    with pytest.raises(SystemExit) as exited:
        choose_scenario(TALK, read=read)
    assert exited.value.code == 130


def test_a_benchmark_takes_any_free_port_and_serve_takes_the_images() -> None:
    assert choose_port(BENCHMARK, None, None) == 0
    assert choose_port(SERVE, None, None) == 7873
    assert choose_port(BENCHMARK, None, "9000") == 9000
    assert choose_port(SERVE, 8080, "9000") == 8080


def test_a_pinned_port_that_is_taken_is_noticed_before_anything_runs() -> None:
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        assert port_in_use(port)
    assert not port_in_use(port)
    assert not port_in_use(0)
