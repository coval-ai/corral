# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from typing import Any
from unittest.mock import patch

import pytest

from corral import cli
from corral.cli import (
    BENCHMARK,
    TALK,
    build_parser,
    checker,
    choose_agent,
    choose_mode,
    choose_model,
    choose_scenario,
    choose_test_case,
    flagged_agent,
    known_scenario,
    misplaced_flag,
    offered_models,
)
from corral.config import Settings
from corral.scenarios import Suite

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
    assert (
        parser.parse_args(["talk", "--test-case", "already-booked"]).test_case == "already-booked"
    )
    assert parser.parse_args(["talk"]).test_case is None
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
    assert choose_mode(read=lambda _prompt: "2", settings=ALL_KEYS).name == "benchmark"
    talk, benchmark, *rest = capsys.readouterr().out.splitlines()
    assert talk.startswith("1. talk") and "(needs: uv sync)" in talk
    assert benchmark.startswith("2. benchmark") and "(needs" not in benchmark
    assert "3." not in "\n".join(rest)
    no_coval = Settings(_env_file=None, **{**KEYS, "coval_api_key": None})
    choose_mode(read=lambda _prompt: "1", settings=no_coval)
    _, benchmark, *_ = capsys.readouterr().out.splitlines()
    assert benchmark.endswith("(needs: COVAL_API_KEY in .env)")


def test_talk_asks_which_seed_when_the_suite_names_test_cases(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases = [
        {"test_case": "book-cleaning", "description": "The slot is free"},
        {"test_case": "already-booked", "description": "The patient already has one"},
    ]
    suite = Suite(test_set={"test_cases": cases}, personas=["p1"])
    with patch.object(cli, "load_suite", return_value=suite):
        assert choose_test_case("dental", read=lambda _prompt: "2") == "already-booked"
        out = capsys.readouterr().out.splitlines()
        assert out[0].startswith("1. book-cleaning   The slot is free")
        assert out[1].startswith("2. already-booked  The patient already has one")
        assert choose_test_case("dental", read=lambda _prompt: "") == "book-cleaning"


def test_talk_asks_nothing_when_the_suite_has_no_test_case_keys(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert choose_test_case("default", read=lambda _prompt: "9") is None
    assert capsys.readouterr().out == ""


def test_a_scenario_that_does_not_exist_says_which_do() -> None:
    assert known_scenario("id_roulette") == "id_roulette"
    have = "have default, id_roulette"
    with pytest.raises(ValueError, match=f"no scenario 'dental'; {have}"):
        known_scenario("dental")


def test_benchmark_offers_only_scenarios_with_coval_ids(capsys: pytest.CaptureFixture[str]) -> None:
    assert choose_scenario(TALK, read=lambda _prompt: "2") == "id_roulette"
    out = capsys.readouterr().out
    assert "1. default      helpful assistant\n" in out
    assert "2. id_roulette  does the agent hear spelled-out details right the first time\n" in out
    assert choose_scenario(BENCHMARK, read=lambda _prompt: "1") == "id_roulette"
    out = capsys.readouterr().out
    assert "default" not in out and out.startswith("1. id_roulette")


@pytest.mark.parametrize("key", [KeyboardInterrupt, EOFError])
def test_ctrl_c_or_ctrl_d_at_a_menu_exits_without_a_traceback(key: type[BaseException]) -> None:
    def read(_prompt: str) -> str:
        raise key

    with pytest.raises(SystemExit) as exited:
        choose_scenario(TALK, read=read)
    assert exited.value.code == 130


def test_a_flag_for_another_mode_is_named_rather_than_ignored() -> None:
    args = build_parser(ALL_KEYS).parse_args(["benchmark", "--test-case", "x"])
    assert misplaced_flag(BENCHMARK, args) == "--test-case is for talk"
    args = build_parser(ALL_KEYS).parse_args(["talk", "--smoke"])
    assert misplaced_flag(TALK, args) == "--smoke is for benchmark"
    args = build_parser(ALL_KEYS).parse_args(["benchmark", "--smoke"])
    assert misplaced_flag(BENCHMARK, args) is None


def test_a_native_model_is_a_stack_of_its_own_at_the_menu_and_on_the_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    answers = iter(["2", "2"])
    native = choose_agent("default", lambda _prompt: next(answers), ALL_KEYS, answers_fine)
    assert native.slug == "openai-gpt-realtime-2.1-mini"
    out = capsys.readouterr().out
    assert "1. cascade  an STT, an LLM and a TTS\n2. native   one speech-to-speech model\n" in out
    answers = iter(["", "", "2", ""])
    cascade = choose_agent("default", lambda _prompt: next(answers), ALL_KEYS, answers_fine)
    assert cascade.slug == "deepgram-nova-3--google-gemini-2.5-flash--elevenlabs-eleven-flash-v2"
    no_openai = Settings(_env_file=None, **{**KEYS, "openai_api_key": None})
    answers = iter(["2", ""])
    other = choose_agent("default", lambda _prompt: next(answers), no_openai, answers_fine)
    assert other.slug == "google-gemini-3.8-live"
    no_native = {"openai_api_key": None, "gemini_api_key": None}
    assert offered_models("s2s", Settings(_env_file=None, **{**KEYS, **no_native})) == []

    parser = build_parser(ALL_KEYS)
    offered = "  s2s  openai/gpt-realtime-2.1, openai/gpt-realtime-2.1-mini, openai/gpt-live-1, "
    assert offered in parser.format_help()
    args = parser.parse_args(["talk", "--s2s", "openai/gpt-realtime-2.1"])
    assert flagged_agent("default", args).slug == "openai-gpt-realtime-2.1"
    assert flagged_agent("default", parser.parse_args(["talk"])).slug.startswith("deepgram-")
    assert "openai/gpt-live-1" not in build_parser(no_openai).format_help()


def test_the_assets_flags_belong_to_benchmark_and_split_on_commas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("S2S_ASSETS", raising=False)
    flags = ["--assets", "workflow-bank.v1", "--personas", "clean, hard", "--metrics", "judge"]
    args = build_parser(ALL_KEYS).parse_args(["benchmark", *flags])
    assert (args.assets, args.personas, args.metrics) == (
        "workflow-bank.v1",
        ["clean", "hard"],
        ["judge"],
    )
    assert misplaced_flag(BENCHMARK, args) is None
    args = build_parser(ALL_KEYS).parse_args(["talk", "--assets", "workflow-bank.v1"])
    assert misplaced_flag(TALK, args) == "--assets is for benchmark"


def test_the_asset_selectors_need_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("S2S_ASSETS", raising=False)
    flags = ["--scenario", "id_roulette", "--personas", "clean"]
    args = build_parser(ALL_KEYS).parse_args(["benchmark", *flags])
    assert misplaced_flag(BENCHMARK, args) == "--personas needs --assets"
    flags = ["--prompt-from", "openai/gpt-realtime-2"]
    args = build_parser(ALL_KEYS).parse_args(["benchmark", *flags])
    assert misplaced_flag(BENCHMARK, args) == "--prompt-from needs --assets"
