# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Resia speaks the OpenAI Live protocol on its own socket. It acknowledges commentary
appended outside a delegation but never speaks it, so the cue that opens the call goes
as an appended instruction, which it answers aloud. Nothing else takes that route."""

from __future__ import annotations

from pipecat.services.openai.live import events
from pipecat.services.openai.live.llm import OpenAILiveLLMService


class ResiaLiveLLMService(OpenAILiveLLMService):
    async def _handle_evt_session_started(self, evt: events.SessionStartedEvent) -> None:
        cue, self._opening_instruction = self._opening_instruction, None
        await super()._handle_evt_session_started(evt)
        if cue:
            await self.send_client_event(
                events.SessionInstructionsAppendEvent(delegation_id=None, content=cue)
            )
