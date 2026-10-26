# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Eleven Realtime speaks the OpenAI Realtime protocol, but takes its key in an
``xi-api-key`` header and closes a Bearer connection with 4001."""

from __future__ import annotations

from typing import Any

from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from websockets.asyncio.client import connect as websocket_connect


class ElevenRealtimeLLMService(OpenAIRealtimeLLMService):
    async def _connect(self) -> None:
        service: Any = self
        if service._websocket:
            return
        try:
            service._websocket = await websocket_connect(
                uri=self.base_url, additional_headers={"xi-api-key": self.api_key}
            )
            service._receive_task = service.create_task(service._receive_task_handler())
        except Exception as error:  # noqa: BLE001
            await self.push_error(error_msg=f"Error connecting: {error}", exception=error)
            service._websocket = None
