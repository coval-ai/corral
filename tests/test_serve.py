# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pipecat.frames.frames import Frame, InputAudioRawFrame, OutputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from starlette.websockets import WebSocketDisconnect

from corral import pipeline, serve
from corral.agent import load_agent
from corral.config import Settings
from corral.models import Services

REFERENCE = ("deepgram/nova-3", "openai/gpt-4.1", "elevenlabs/eleven_flash_v2")


class _Passthrough(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class _Echo(FrameProcessor):
    """Stands in for STT: sends the caller's audio straight back out."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            frame = OutputAudioRawFrame(
                audio=frame.audio, sample_rate=frame.sample_rate, num_channels=frame.num_channels
            )
        await self.push_frame(frame, direction)


@pytest.fixture
def client() -> Iterator[TestClient]:
    agent = load_agent("ultrabank", *REFERENCE)
    fakes = Services(stt=_Echo(), llm=_Passthrough(), tts=_Passthrough())
    settings = Settings(_env_file=None, s2s_agent_token="s3cret")  # noqa: S106
    with (
        patch.object(pipeline, "resolve", return_value=fakes),
        patch.object(serve, "get_settings", return_value=settings),
    ):
        yield TestClient(serve.create_app(agent))


def test_healthz_names_the_agent_and_a_bad_bearer_is_refused(client: TestClient) -> None:
    body = client.get("/healthz").json()
    assert body["agent"] == "deepgram-nova-3--openai-gpt-4.1--elevenlabs-eleven-flash-v2"
    with (
        pytest.raises(WebSocketDisconnect) as refused,
        client.websocket_connect("/ws", headers={"Authorization": "Bearer nope"}),
    ):
        pass
    assert refused.value.code == serve.POLICY_VIOLATION


def test_a_test_case_the_scenario_cannot_seed_is_refused_not_crashed(client: TestClient) -> None:
    headers = {"Authorization": "Bearer s3cret", "X-Corral-Test-Case": "book-cleaning"}
    with (
        client.websocket_connect("/ws", headers=headers) as socket,
        pytest.raises(WebSocketDisconnect) as refused,
    ):
        socket.receive_text()
    assert refused.value.code == serve.POLICY_VIOLATION
