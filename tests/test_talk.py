# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from corral.pipeline import ToolCall
from corral.talk import summary, tool_log


def _turn(**ms: float) -> dict[str, float]:
    keys = {"latency": "s2s.turn_latency_ms"}
    return {keys.get(step, f"s2s.turn_latency.{step}_ms"): value for step, value in ms.items()}


def test_hanging_up_prints_median_and_max_per_stage() -> None:
    turns = [
        _turn(latency=800.0, stt=100.0, llm=500.0, tts=200.0),
        _turn(latency=1000.0, stt=50.0, llm=700.0, tts=250.0),
        {
            **_turn(latency=900.0, llm=600.0, tts=300.0),
            "s2s.turn_latency.stt_timed_out": True,
            "s2s.turn_latency.smart_turn_timed_out": True,
        },
        {
            **_turn(latency=1800.0, llm=500.0, tool=120.0, llm_second=400.0),
            "s2s.turn.tool_calls": 1,
        },
        {**_turn(latency=700.0, llm=400.0, tts=200.0), "s2s.turn_latency.silero_missed": True},
    ]
    models = {"llm": {"provider": "openai", "model": "gpt-4.1"}}
    out = summary(turns, models, "cascade-x").splitlines()
    assert out[0] == "5 turns on cascade-x"
    assert out[2].split() == ["latency", "900", "1800"]
    assert out[3].split() == ["stt", "75", "100"]
    assert out[4].split() == ["llm", "(openai/gpt-4.1)", "500", "700"]
    assert out[5].split() == ["tool", "120", "120"]
    assert out[6].split() == ["llm_second", "400", "400"]
    assert out[-4] == "1 turns where Silero never heard you; check the mic level"
    assert out[-3] == "1 turns where the STT timed out"
    assert out[-2] == "1 turns where smart-turn timed out"
    assert out[-1] == "1 turns that called a tool"
    assert summary([], models, "cascade-x") == "no answered turns"


def test_a_native_call_prints_endpointing_and_model_instead() -> None:
    turns = [
        _turn(latency=700.0, endpointing=150.0, model=550.0),
        {**_turn(latency=900.0, endpointing=-60.0, model=960.0)},
        {
            **_turn(latency=1500.0, model=400.0, tool=100.0, model_second=500.0),
            "s2s.turn.tool_calls": 1,
        },
    ]
    openai = {"provider": "openai", "model": "gpt-realtime-2.1"}
    models = {"endpointing": openai, "model": openai, "model_second": openai}
    out = summary(turns, models, "openai-gpt-realtime-2.1").splitlines()
    assert out[2].split() == ["latency", "900", "1500"]
    assert out[3].split() == ["endpointing", "(openai/gpt-realtime-2.1)", "45", "150"]
    assert out[4].split() == ["model", "(openai/gpt-realtime-2.1)", "550", "960"]
    assert out[5].split() == ["tool", "100", "100"]
    assert out[6].split() == ["model_second", "(openai/gpt-realtime-2.1)", "500", "500"]
    assert out[7] == "1 turns that called a tool"
    assert len(out) == 8


def test_hanging_up_lists_the_tools_called_in_order() -> None:
    calls = [
        ToolCall("find_patient", {"name": "Ana"}, {"patient_id": 123}),
        ToolCall("book_appointment", {"patient_id": 123}, {"ok": False, "error": "taken"}),
    ]
    assert tool_log(calls).splitlines() == [
        'find_patient({"name": "Ana"}) -> {"patient_id": 123}',
        'book_appointment({"patient_id": 123}) -> {"ok": false, "error": "taken"}',
    ]
    assert tool_log([]) == "no tools called"
