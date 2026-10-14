# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Mic and speakers with the speakers taken back out of the mic.

On speakers the mic hears the agent, the STT transcribes it, and the agent
interrupts itself. Everything we play goes to WebRTC's echo canceller as the
reference, and every mic chunk goes through it before the pipeline hears it,
the same thing a browser does on a call.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping

import pyaudio
from livekit import rtc
from pipecat.frames.frames import OutputAudioRawFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.local.audio import (
    LocalAudioInputTransport,
    LocalAudioOutputTransport,
    LocalAudioTransport,
    LocalAudioTransportParams,
)

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


class _Input(LocalAudioInputTransport):
    def __init__(
        self, canceller: EchoCanceller, py_audio: pyaudio.PyAudio, params: LocalAudioTransportParams
    ) -> None:
        super().__init__(py_audio, params)
        self._canceller = canceller

    def _audio_in_callback(
        self, in_data: bytes, frame_count: int, time_info: Mapping[str, float], status: int
    ) -> tuple[None, int]:
        if self._in_stream is not None:
            self._canceller.input_latency_ms = int(self._in_stream.get_input_latency() * 1000)
        cleaned = self._canceller.clean(in_data, self._sample_rate)
        result: tuple[None, int] = super()._audio_in_callback(  # type: ignore[no-untyped-call]
            cleaned, frame_count, time_info, status
        )
        return result

    async def cleanup(self) -> None:
        await super().cleanup()  # type: ignore[no-untyped-call]
        self._canceller.close()


class _Output(LocalAudioOutputTransport):
    def __init__(
        self, canceller: EchoCanceller, py_audio: pyaudio.PyAudio, params: LocalAudioTransportParams
    ) -> None:
        super().__init__(py_audio, params)
        self._canceller = canceller

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        if self._out_stream is not None:
            self._canceller.output_latency_ms = int(self._out_stream.get_output_latency() * 1000)
        self._canceller.played(frame.audio, frame.sample_rate)
        return await super().write_audio_frame(frame)

    async def cleanup(self) -> None:
        await super().cleanup()  # type: ignore[no-untyped-call]
        self._canceller.close()


class EchoCancelledAudioTransport(LocalAudioTransport):
    """Pipecat's local mic and speakers, sharing one echo canceller."""

    def __init__(self, params: LocalAudioTransportParams) -> None:
        super().__init__(params)
        self._canceller = EchoCanceller()

    def input(self) -> FrameProcessor:
        if not self._input:
            self._input = _Input(self._canceller, self._pyaudio, self._params)
        return self._input

    def output(self) -> FrameProcessor:
        if not self._output:
            self._output = _Output(self._canceller, self._pyaudio, self._params)
        return self._output
