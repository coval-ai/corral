# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The speakers taken back out of the mic.

On speakers the mic hears the agent, the STT transcribes it, and the agent
interrupts itself. Everything we play goes to WebRTC's echo canceller as the
reference, and every mic chunk goes through it before the pipeline hears it,
the same thing a browser does on a call. The transport that feeds it is in
``local``.
"""

from __future__ import annotations

import threading

from livekit import rtc

SAMPLE_BYTES = 2
CHUNK_SECS = 0.01


class EchoCanceller:
    """WebRTC's audio processing, fed in the 10 ms chunks it requires."""

    def __init__(self) -> None:
        self._apm: rtc.AudioProcessingModule | None = rtc.AudioProcessingModule(
            echo_cancellation=True
        )
        self._lock = threading.Lock()
        self._played = b""
        self.input_latency_ms = 0
        self.output_latency_ms = 0

    def played(self, audio: bytes, sample_rate: int) -> None:
        """Audio about to go to the speakers, the echo to expect."""
        chunk = _chunk_bytes(sample_rate)
        self._played += audio
        while len(self._played) >= chunk:
            frame, self._played = self._played[:chunk], self._played[chunk:]
            with self._lock:
                if self._apm is not None:
                    self._apm.process_reverse_stream(_frame(frame, sample_rate))

    def clean(self, audio: bytes, sample_rate: int) -> bytes:
        """Mic audio with the speakers taken out. A partial chunk passes through as is."""
        chunk = _chunk_bytes(sample_rate)
        out = bytearray()
        for start in range(0, len(audio) - chunk + 1, chunk):
            frame = _frame(audio[start : start + chunk], sample_rate)
            with self._lock:
                if self._apm is None:
                    return audio
                self._apm.set_stream_delay_ms(self.input_latency_ms + self.output_latency_ms)
                self._apm.process_stream(frame)
            out += bytes(frame.data.cast("B"))
        out += audio[len(out) :]
        return bytes(out)

    def close(self) -> None:
        """Release the canceller while LiveKit's native library is still loaded.

        Left to the garbage collector, it's released after LiveKit has shut down
        at exit, and LiveKit raises.
        """
        with self._lock:
            self._apm = None


def _chunk_bytes(sample_rate: int) -> int:
    return int(sample_rate * CHUNK_SECS) * SAMPLE_BYTES


def _frame(audio: bytes, sample_rate: int) -> rtc.AudioFrame:
    samples = len(audio) // SAMPLE_BYTES
    return rtc.AudioFrame(audio, sample_rate, 1, samples)
