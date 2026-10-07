# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from agent.talk import summary


def test_hanging_up_prints_median_and_max_per_stage() -> None:
    turns = [
        {"s2s.latency_ms": 800.0, "s2s.stt_ms": 100.0, "s2s.llm_ms": 500.0, "s2s.tts_ms": 200.0},
        {"s2s.latency_ms": 1000.0, "s2s.stt_ms": 50.0, "s2s.llm_ms": 700.0, "s2s.tts_ms": 250.0},
        {"s2s.latency_ms": 900.0, "s2s.llm_ms": 600.0, "s2s.tts_ms": 300.0, "s2s.transport_ms": 1},
    ]
    models = {"llm": {"provider": "openai", "model": "gpt-4.1"}}
    out = summary(turns, models, "cascade-x").splitlines()
    assert out[0] == "3 turns on cascade-x"
    assert out[2].split() == ["latency", "900", "1000"]
    assert "endpointing" not in "\n".join(out)
    assert out[3].split() == ["stt", "75", "100"]
    assert out[4].split() == ["llm", "(openai/gpt-4.1)", "600", "700"]
    assert out[6].split() == ["transport", "1", "1"]
    assert summary([], models, "cascade-x") == "no answered turns"
