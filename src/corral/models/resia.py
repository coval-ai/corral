# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Resia speaks the OpenAI Live protocol on its own socket. It acknowledges commentary
appended outside a delegation but never speaks it, so the cue that opens the call goes
as an appended instruction, which it answers aloud."""

from __future__ import annotations

from pipecat.services.openai.live import events
from pipecat.services.openai.live.llm import OpenAILiveLLMService


class ResiaLiveLLMService(OpenAILiveLLMService):
    async def _send_context_append(
        self, delegation_id: str | None, text: str, *, spoken: bool
    ) -> None:
        if spoken and delegation_id is None:
            await self.send_client_event(
                events.SessionInstructionsAppendEvent(delegation_id=None, content=text)
            )
            return
        await super()._send_context_append(delegation_id, text, spoken=spoken)
