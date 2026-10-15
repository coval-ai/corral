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
    ]
    models = {"llm": {"provider": "openai", "model": "gpt-4.1"}}
    out = summary(turns, models, "cascade-x").splitlines()
    assert out[0] == "4 turns on cascade-x"
    assert out[2].split() == ["latency", "950", "1800"]
    assert out[3].split() == ["stt", "75", "100"]
    assert out[4].split() == ["llm", "(openai/gpt-4.1)", "550", "700"]
    assert out[5].split() == ["tool", "120", "120"]
    assert out[6].split() == ["llm_second", "400", "400"]
    assert out[-3] == "1 turns where the STT timed out"
    assert out[-2] == "1 turns where smart-turn timed out"
    assert out[-1] == "1 turns that called a tool"
    assert summary([], models, "cascade-x") == "no answered turns"


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
