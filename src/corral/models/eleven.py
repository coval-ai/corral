# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Eleven Realtime speaks the OpenAI Realtime protocol, but takes its key in an
``xi-api-key`` header and closes a Bearer connection with 4001. It rejects
``session.model`` once the conversation is under way, which can be before the first
session update lands, so the model only ever travels in the URL. It sends
``response.done`` without a usage block, sends the ``output_audio_buffer`` events
OpenAI keeps for WebRTC over the websocket, and drops the socket after a protocol
error instead of closing it."""

from __future__ import annotations

from typing import Any

from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from websockets.asyncio.client import connect as websocket_connect

WEBRTC_ONLY_EVENTS = (
    "output_audio_buffer.started",
    "output_audio_buffer.stopped",
    "output_audio_buffer.cleared",
)
for _kind in WEBRTC_ONLY_EVENTS:
    events._server_event_types.setdefault(_kind, events.ServerEvent)


class ElevenRealtimeLLMService(OpenAIRealtimeLLMService):
    async def _connect(self) -> None:
        service: Any = self
        if service._websocket:
            return
        try:
            service._websocket = await websocket_connect(
                uri=self.base_url, additional_headers={"xi-api-key": self.api_key}
            )
            service._receive_task = service.create_task(self._receive_task_handler())
        except Exception as error:  # noqa: BLE001
            await self.push_error(error_msg=f"Error connecting: {error}", exception=error)
            service._websocket = None

    async def send_client_event(self, event: events.ClientEvent) -> None:
        if isinstance(event, events.SessionUpdateEvent) and event.session.model is not None:
            session = event.session.model_copy(update={"model": None})
            event = event.model_copy(update={"session": session})
        await super().send_client_event(event)

    async def _handle_evt_response_done(self, evt: events.ResponseDone) -> None:
        if evt.response.usage is None:
            evt.response.usage = events.Usage(
                total_tokens=0,
                input_tokens=0,
                output_tokens=0,
                input_token_details=events.TokenDetails(),
                output_token_details=events.TokenDetails(),
            )
        await super()._handle_evt_response_done(evt)

    async def _receive_task_handler(self) -> None:
        service: Any = self
        parent: Any = super()
        try:
            await parent._receive_task_handler()
        except Exception as error:  # noqa: BLE001
            if service._disconnecting:
                return
            await self.push_error(
                error_msg=f"Eleven connection failed: {error}",
                exception=error,
                force_treat_as_permanent=True,
            )
            return
        if not service._disconnecting:
            await self.push_error(
                error_msg="Eleven closed the connection", force_treat_as_permanent=True
            )
