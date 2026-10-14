# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""This machine's mic and speakers, through miniaudio, with the speakers taken out of the mic.

miniaudio ships its own audio library in its wheel on Mac, Windows and Linux, so talk needs
nothing installed beyond ``uv sync``. Its devices run on their own thread and talk to us
through generators: the mic sends us bytes, and the speakers ask us for a number of frames.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Generator

import miniaudio
from loguru import logger
from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameProcessorSetup
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from corral.echo import SAMPLE_BYTES, EchoCanceller

BUFFER_MSECS = 20


class Playout:
    """Audio waiting for the speakers. Writers wait until the speakers have nearly caught up.

    Keeping the queue a buffer deep is what makes an interruption stop the voice at once:
    everything else still to play is in Pipecat's queue, which it clears.
    """

    def __init__(self, limit: int) -> None:
        self._audio = bytearray()
        self._limit = limit
        self._closed = False
        self._changed = threading.Condition()

    def write(self, audio: bytes) -> None:
        with self._changed:
            self._changed.wait_for(lambda: self._closed or len(self._audio) < self._limit)
            if not self._closed:
                self._audio += audio

    def take(self, size: int) -> bytes:
        """The next ``size`` bytes, padded with silence when there's less than that."""
        with self._changed:
            audio, self._audio = bytes(self._audio[:size]), self._audio[size:]
            self._changed.notify_all()
        return audio + bytes(size - len(audio))

    def close(self) -> None:
        """Let any waiting writer go, dropping what it brought."""
        with self._changed:
            self._closed = True
            self._changed.notify_all()


def close_device(device: miniaudio.AbstractDevice) -> None:
    """Close the device, and when it won't stop, unplugged say, don't let miniaudio try again.

    miniaudio only clears its running flag once the native stop succeeds, and tries the
    stop again from ``__del__`` at exit, by then against a freed handle.
    """
    try:
        device.stop()
    except miniaudio.MiniaudioError as error:
        logger.warning(f"The audio device did not stop ({error}); was it unplugged?")
        device.running = False
    device.close()


class _Mic(BaseInputTransport):
    def __init__(self, canceller: EchoCanceller, params: TransportParams) -> None:
        super().__init__(params)
        self._canceller = canceller
        self._device: miniaudio.CaptureDevice | None = None

    async def setup(self, setup: FrameProcessorSetup) -> None:  # type: ignore[override]
        await super().setup(setup)
        self._device = miniaudio.CaptureDevice(
            nchannels=self._params.audio_in_channels,
            sample_rate=self.sample_rate,
            buffersize_msec=BUFFER_MSECS,
        )

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        if self._device is not None:
            listener = self._listen(self.get_event_loop())
            next(listener)
            self._device.start(listener)
            await self.set_transport_ready(frame)

    async def cleanup(self) -> None:
        await super().cleanup()  # type: ignore[no-untyped-call]
        if self._device is not None:
            close_device(self._device)
            self._device = None
        self._canceller.close()

    def _listen(self, loop: asyncio.AbstractEventLoop) -> Generator[None, bytes, None]:
        """Runs on miniaudio's thread: each capture buffer, cleaned, becomes a frame."""
        while True:
            audio = yield
            frame = InputAudioRawFrame(
                audio=self._canceller.clean(bytes(audio), self.sample_rate),
                sample_rate=self.sample_rate,
                num_channels=self._params.audio_in_channels,
            )
            asyncio.run_coroutine_threadsafe(self.push_audio_frame(frame), loop)


class _Speakers(BaseOutputTransport):
    def __init__(self, canceller: EchoCanceller, params: TransportParams) -> None:
        super().__init__(params)
        self._canceller = canceller
        self._device: miniaudio.PlaybackDevice | None = None
        self._playout: Playout | None = None

    async def setup(self, setup: FrameProcessorSetup) -> None:  # type: ignore[override]
        await super().setup(setup)
        channels = self._params.audio_out_channels
        self._device = miniaudio.PlaybackDevice(
            nchannels=channels, sample_rate=self.sample_rate, buffersize_msec=BUFFER_MSECS
        )
        self._playout = Playout(self.sample_rate * BUFFER_MSECS // 1000 * SAMPLE_BYTES * channels)

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        if self._device is not None and self._playout is not None:
            player = self._play(self._playout)
            next(player)
            self._device.start(player)
            await self.set_transport_ready(frame)

    async def cleanup(self) -> None:
        await super().cleanup()  # type: ignore[no-untyped-call]
        if self._playout is not None:
            self._playout.close()
        if self._device is not None:
            close_device(self._device)
            self._device = None
        self._canceller.close()

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        if self._playout is None:
            return False
        self._canceller.played(frame.audio, frame.sample_rate)
        await asyncio.to_thread(self._playout.write, frame.audio)
        return True

    def _play(self, playout: Playout) -> Generator[bytes, int, None]:
        """Runs on miniaudio's thread: hands over as many frames as the speakers ask for."""
        frames = yield b""
        while True:
            frames = yield playout.take(frames * SAMPLE_BYTES * self._params.audio_out_channels)


class LocalAudioTransport(BaseTransport):
    """The mic and the speakers, sharing one echo canceller."""

    def __init__(self, params: TransportParams) -> None:
        super().__init__()
        self._params = params
        self._canceller = EchoCanceller()
        # miniaudio keeps one buffer each way; that's the echo's head start on the mic.
        self._canceller.input_latency_ms = BUFFER_MSECS
        self._canceller.output_latency_ms = BUFFER_MSECS
        self._input: _Mic | None = None
        self._output: _Speakers | None = None

    def input(self) -> FrameProcessor:
        if self._input is None:
            self._input = _Mic(self._canceller, self._params)
        return self._input

    def output(self) -> FrameProcessor:
        if self._output is None:
            self._output = _Speakers(self._canceller, self._params)
        return self._output
