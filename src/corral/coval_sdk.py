# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The agent on Coval's relay: one call through the Agent SDK's audio connection.

The SDK dials out to Coval for each simulation and hands back an ``AudioConnection``.
The caller's audio arrives as an async iterator and the agent's audio leaves through a
paced ``send_audio``. Both ways are PCM16 mono at 16 kHz, so the output side runs at that
rate and Pipecat resamples the pipeline's 24 kHz speech on the way through.
"""

from __future__ import annotations

import asyncio

from coval_agent_sdk import AudioConnection
from coval_agent_sdk.errors import CovalError
from loguru import logger
from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

SDK_SAMPLE_RATE_HZ = 16_000


class _Input(BaseInputTransport):
    """The caller's audio off the relay, a frame per chunk, until Coval hangs up."""

    def __init__(self, transport: CovalSdkTransport, params: TransportParams) -> None:
        super().__init__(params)
        self._transport = transport
        self._reader: asyncio.Task[None] | None = None

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        self._reader = self.create_task(self._read(self._transport.connection))
        await self.set_transport_ready(frame)

    async def cleanup(self) -> None:
        await super().cleanup()  # type: ignore[no-untyped-call]
        if self._reader is not None:
            await self.cancel_task(self._reader)
            self._reader = None

    async def _read(self, connection: AudioConnection) -> None:
        """A relay failure is logged here and raised to whoever waits on the connection."""
        try:
            async for audio in connection.receive_audio():
                frame = InputAudioRawFrame(
                    audio=audio, sample_rate=SDK_SAMPLE_RATE_HZ, num_channels=1
                )
                await self.push_audio_frame(frame)
        except CovalError as error:
            logger.debug(f"The relay stopped delivering audio: {error}")


class _Output(BaseOutputTransport):
    """The agent's audio onto the relay. Each write is paced by the SDK, which is what
    keeps the pipeline from running ahead of the call."""

    def __init__(self, transport: CovalSdkTransport, params: TransportParams) -> None:
        super().__init__(params)
        self._transport = transport

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        connection = self._transport.connection
        if connection.closed:
            return False
        try:
            return await connection.send_audio(frame.audio)
        except CovalError as error:
            logger.debug(f"The relay stopped taking audio: {error}")
            return False


class CovalSdkTransport(BaseTransport):
    """One call on the relay.

    The connection is attached after the pipeline is built: the relay counts unread
    caller audio from the moment it is ready, and building a call is the slow part.
    """

    def __init__(self) -> None:
        super().__init__()
        self._params = TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=SDK_SAMPLE_RATE_HZ,
            audio_out_sample_rate=SDK_SAMPLE_RATE_HZ,
        )
        self._connection: AudioConnection | None = None
        self._input: _Input | None = None
        self._output: _Output | None = None

    def attach(self, connection: AudioConnection) -> None:
        self._connection = connection

    @property
    def connection(self) -> AudioConnection:
        if self._connection is None:
            raise RuntimeError("the call started before an SDK connection was attached")
        return self._connection

    def input(self) -> FrameProcessor:
        if self._input is None:
            self._input = _Input(self, self._params)
        return self._input

    def output(self) -> FrameProcessor:
        if self._output is None:
            self._output = _Output(self, self._params)
        return self._output
