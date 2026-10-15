# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from corral.talk import summary


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
    ]
    models = {"llm": {"provider": "openai", "model": "gpt-4.1"}}
    out = summary(turns, models, "cascade-x").splitlines()
    assert out[0] == "3 turns on cascade-x"
    assert out[2].split() == ["latency", "900", "1000"]
    assert out[3].split() == ["stt", "75", "100"]
    assert out[4].split() == ["llm", "(openai/gpt-4.1)", "600", "700"]
    assert out[-2] == "1 turns where the STT timed out"
    assert out[-1] == "1 turns where smart-turn timed out"
    assert summary([], models, "cascade-x") == "no answered turns"
