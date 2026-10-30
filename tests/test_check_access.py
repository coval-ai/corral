# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from pipecat.services.openai.live.llm import OpenAILiveLLMService
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.services.xai.realtime.llm import GrokRealtimeLLMService
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.frames import Close
from websockets.http11 import Response

from corral import check_access
from corral.config import Settings
from corral.models import VENDORS, S2sModel, access_probes, catalog
from corral.models.access_probes import Connect, GetProbe, Socket, Target, Transports

KEYS: dict[str, Any] = {
    "coval_api_key": "ck",
    "deepgram_api_key": "dg",
    "openai_api_key": "oa",
    "elevenlabs_api_key": "el",
    "soniox_api_key": "sx",
    "gemini_api_key": "gm",
    "xai_api_key": "xa",
}
ALL_KEYS = Settings(_env_file=None, **KEYS)
OK = httpx.MockTransport(lambda _request: httpx.Response(200))


class FakeSocket:
    """Answers each recv with the next scripted event and remembers what was sent."""

    def __init__(self, *events: Any) -> None:
        self.events = list(events)
        self.sent: list[Any] = []

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    async def recv(self) -> str:
        return json.dumps(self.events.pop(0))


def connecting(socket: FakeSocket, seen: dict[str, Any] | None = None) -> Transports:
    @asynccontextmanager
    async def connect(url: str, headers: Mapping[str, str]) -> AsyncIterator[Socket]:
        if seen is not None:
            seen["url"], seen["headers"] = url, dict(headers)
        yield socket

    return Transports(http=OK, connect=connect)


def refusing(error: Exception) -> Transports:
    @asynccontextmanager
    async def connect(url: str, headers: Mapping[str, str]) -> AsyncIterator[Socket]:
        raise error
        yield  # pragma: no cover

    return Transports(http=OK, connect=connect)


def by_url(connect: Connect) -> Transports:
    return Transports(http=OK, connect=connect)


async def test_a_get_probe_is_the_status_or_the_network() -> None:
    target = Target("soniox", {"Authorization": "Bearer sx"})
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    probe = GetProbe("https://api.soniox.com/v1/models")
    assert await probe.problem(target, Transports(http=httpx.MockTransport(record))) is None
    assert seen[0].method == "GET" and seen[0].headers["Authorization"] == "Bearer sx"
    denied = Transports(http=httpx.MockTransport(lambda _request: httpx.Response(401)))
    assert await probe.problem(target, denied) == "soniox returned 401"

    def down(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    unreachable = Transports(http=httpx.MockTransport(down))
    assert await probe.problem(target, unreachable) == "soniox is unreachable: no route"


async def test_a_realtime_model_is_probed_on_its_own_socket_with_the_vendors_key() -> None:
    seen: dict[str, Any] = {}
    socket = FakeSocket({"type": "session.created", "session": {}})
    model = S2sModel(provider="xai", name="grok-voice-3.0")
    assert await check_access.model_problem(model, ALL_KEYS, connecting(socket, seen)) is None
    assert seen["url"] == "wss://api.x.ai/v1/realtime?model=grok-voice-3.0"
    assert seen["headers"] == {"Authorization": "Bearer xa"}
    assert socket.sent == []
    eleven = S2sModel(provider="elevenlabs", name="eleven_realtime_v1_mini")
    socket = FakeSocket({"type": "session.created"})
    assert await check_access.model_problem(eleven, ALL_KEYS, connecting(socket, seen)) is None
    assert seen["url"].endswith("/convai/realtime?model=eleven_realtime_v1_mini")
    assert seen["headers"] == {"xi-api-key": "el"}


def _default(service: type[Any]) -> str:
    kwdefaults: dict[str, Any] = service.__init__.__kwdefaults__
    url: str = kwdefaults["base_url"]
    return url


def test_each_socket_is_the_one_its_pipecat_class_connects_to() -> None:
    openai = VENDORS["openai"].s2s or {}
    assert openai["realtime"].socket_base_url == _default(OpenAIRealtimeLLMService)
    assert openai["live"].socket_base_url == _default(OpenAILiveLLMService)
    xai = VENDORS["xai"].s2s or {}
    assert xai["realtime"].socket_base_url == _default(GrokRealtimeLLMService)
    eleven = VENDORS["elevenlabs"].s2s or {}
    assert eleven["realtime"].socket_base_url == "wss://api.elevenlabs.io/v1/convai/realtime"
    google = VENDORS["google"].s2s or {}
    assert google["live"].socket_base_url is None
    assert "BidiGenerateContent" in google["live"].protocol.socket_url(None, "gemini-3.8-live")


async def test_a_live_model_starts_a_session_and_gemini_sets_one_up() -> None:
    socket = FakeSocket({"type": "session.started"})
    live = S2sModel(provider="openai", name="gpt-live-1", api="live")
    assert await check_access.model_problem(live, ALL_KEYS, connecting(socket)) is None
    assert socket.sent == [{"type": "session.start", "session": {"model": "gpt-live-1"}}]
    socket = FakeSocket({"setupComplete": {}})
    gemini = S2sModel(provider="google", name="gemini-3.8-live", api="live")
    seen: dict[str, Any] = {}
    assert await check_access.model_problem(gemini, ALL_KEYS, connecting(socket, seen)) is None
    assert socket.sent == [{"setup": {"model": "models/gemini-3.8-live"}}]
    assert seen["headers"] == {"x-goog-api-key": "gm"}
    thinking = S2sModel(
        provider="google",
        name="gemini-3.8-live-extended-thinking",
        api="live",
        options={"thinking": {"thinking_level": "MEDIUM"}},
    )
    socket = FakeSocket({"setupComplete": {}})
    assert await check_access.model_problem(thinking, ALL_KEYS, connecting(socket)) is None
    assert socket.sent[0]["setup"]["generationConfig"] == {
        "thinkingConfig": {"thinkingLevel": "MEDIUM"}
    }


async def test_a_model_the_vendor_wont_serve_is_the_vendors_own_words() -> None:
    model = S2sModel(provider="openai", name="gpt-gone")
    refused = FakeSocket({"type": "error", "error": {"message": "model gpt-gone not found"}})
    problem = await check_access.model_problem(model, ALL_KEYS, connecting(refused))
    assert problem == "model gpt-gone not found"
    odd = FakeSocket({"type": "response.done"})
    problem = await check_access.model_problem(model, ALL_KEYS, connecting(odd))
    assert problem == "unexpected first event: 'response.done'"
    gemini = S2sModel(provider="google", name="gemini-gone", api="live")
    denied = FakeSocket({"error": {"message": "models/gemini-gone is not found"}})
    problem = await check_access.model_problem(gemini, ALL_KEYS, connecting(denied))
    assert problem == "models/gemini-gone is not found"


async def test_a_refused_or_unreachable_socket_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    model = S2sModel(provider="openai", name="gpt-realtime-2.1")
    unauthorized = InvalidStatus(Response(401, "Unauthorized", Headers()))
    assert await check_access.model_problem(model, ALL_KEYS, refusing(unauthorized)) == (
        "openai returned 401"
    )
    down = refusing(OSError("no route"))
    assert await check_access.model_problem(model, ALL_KEYS, down) == (
        "openai is unreachable: no route"
    )
    closed = ConnectionClosed(Close(4001, "Authentication required"), None)
    assert await check_access.model_problem(model, ALL_KEYS, refusing(closed)) == (
        "openai closed the socket: 4001 Authentication required"
    )
    monkeypatch.setattr(access_probes, "PROBE_TIMEOUT_SECS", 0)

    class Silent(FakeSocket):
        async def recv(self) -> str:
            await asyncio.Event().wait()
            return ""

    problem = await check_access.model_problem(model, ALL_KEYS, connecting(Silent()))
    assert problem == "no session event in 0 s"


async def test_the_report_skips_vendors_without_a_key_and_fails_on_any_problem(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(check_access, "coval_problem", lambda _settings: None)

    def keys(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403 if "googleapis" in request.url.host else 200)

    @asynccontextmanager
    async def connect(url: str, headers: Mapping[str, str]) -> AsyncIterator[Socket]:
        if "grok" in url:
            yield FakeSocket({"type": "error", "error": {"message": "retired"}})
        elif "live/sessions" in url:
            yield FakeSocket({"type": "session.started"})
        else:
            yield FakeSocket({"type": "session.created"})

    settings = Settings(_env_file=None, **{**KEYS, "soniox_api_key": None})
    transports = Transports(http=httpx.MockTransport(keys), connect=connect)
    assert await check_access.run(settings, transports) == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["coval", "ok"]
    assert any(
        line.split() == ["soniox", "skipped:", "SONIOX_API_KEY", "is", "unset"] for line in lines
    )
    assert any(line.split() == ["google", "google", "returned", "403"] for line in lines)
    assert any(
        line.split() == ["google/gemini-3.8-live", "skipped:", "its", "vendor", "failed"]
        for line in lines
    )
    assert any(line.split() == ["xai/grok-voice-3.0", "retired"] for line in lines)
    assert any(line.split() == ["openai/gpt-realtime-2.1", "ok"] for line in lines)
    assert lines[-1] == "2 failed"


async def test_a_clean_report_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(check_access, "coval_problem", lambda _settings: None)

    @asynccontextmanager
    async def connect(url: str, headers: Mapping[str, str]) -> AsyncIterator[Socket]:
        if "generativelanguage" in url:
            yield FakeSocket({"setupComplete": {}})
        elif "live/sessions" in url:
            yield FakeSocket({"type": "session.started"})
        else:
            yield FakeSocket({"type": "session.created"})

    assert await check_access.run(ALL_KEYS, by_url(connect)) == 0
    out = capsys.readouterr().out
    assert "failed" not in out
    assert out.count(" ok") == 1 + len(VENDORS) + len(catalog().s2s)
