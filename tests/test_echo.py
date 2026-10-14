# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import math
import struct

import pytest

pytest.importorskip("livekit")
pytest.importorskip("pyaudio")

from corral.echo import EchoCanceller


def _tone(rate: int, secs: float, hz: float = 440.0) -> bytes:
    count = int(rate * secs)
    return struct.pack(
        f"<{count}h", *(int(8000 * math.sin(2 * math.pi * hz * i / rate)) for i in range(count))
    )


def test_mic_audio_keeps_its_length_including_a_partial_chunk() -> None:
    canceller = EchoCanceller()
    mic = _tone(16000, 0.025)
    assert len(canceller.clean(mic, 16000)) == len(mic)
    assert canceller.clean(mic, 16000)[-160:] == mic[-160:]


def test_what_we_played_is_taken_back_out_of_the_mic() -> None:
    canceller = EchoCanceller()
    speakers, mic = _tone(24000, 3.0), _tone(16000, 3.0)
    chunk_out, chunk_in = 480, 320
    cleaned = b""
    for i in range(len(mic) // chunk_in):
        canceller.played(speakers[i * chunk_out : (i + 1) * chunk_out], 24000)
        cleaned += canceller.clean(mic[i * chunk_in : (i + 1) * chunk_in], 16000)
    tail = slice(len(cleaned) // 2, None)
    assert _rms(cleaned[tail]) < _rms(mic[tail]) / 4


def _rms(audio: bytes) -> float:
    samples = struct.unpack(f"<{len(audio) // 2}h", audio)
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def test_a_closed_canceller_passes_audio_through() -> None:
    canceller = EchoCanceller()
    canceller.close()
    mic = _tone(16000, 0.02)
    canceller.played(_tone(24000, 0.02), 24000)
    assert canceller.clean(mic, 16000) == mic
