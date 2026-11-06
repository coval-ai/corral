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
from corral.models.access_probes import GetProbe, ListedProbe, Socket, Target, Transports

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


SONIOX_LISTING = {"models": [{"id": "stt-rt-v5"}, {"id": "stt-rt-v4"}]}
ELEVEN_LISTING = [{"model_id": "eleven_flash_v2"}, {"model_id": "eleven_v3"}]


def vendors_answer(request: httpx.Request) -> httpx.Response:
    """Every key GET and model GET is fine, and the two listings hold corral's models."""
    if request.url.host == "api.soniox.com" and request.url.path == "/v1/models":
        return httpx.Response(200, json=SONIOX_LISTING)
    if request.url.host == "api.elevenlabs.io" and request.url.path == "/v1/models":
        return httpx.Response(200, json=ELEVEN_LISTING)
    return httpx.Response(200)


ANSWERING = httpx.MockTransport(vendors_answer)


async def test_a_get_probe_names_the_model_when_it_has_one() -> None:
    seen: list[str] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(404 if "gone" in str(request.url) else 200)

    probe = GetProbe("https://api.openai.com/v1/models/{model}")
    transports = Transports(http=httpx.MockTransport(record))
    key = Target("openai", {"Authorization": "Bearer oa"})
    assert await probe.problem(Target("openai", key.headers, "gpt-4.1"), transports) is None
    assert seen == ["https://api.openai.com/v1/models/gpt-4.1"]
    gone = Target("openai", key.headers, "gpt-gone")
    assert await probe.problem(gone, transports) == "openai returned 404"


async def test_a_listed_probe_wants_the_model_in_the_vendors_list() -> None:
    probe = ListedProbe("https://api.soniox.com/v1/models", lambda body: {"stt-rt-v5"})
    transports = Transports(http=httpx.MockTransport(lambda _r: httpx.Response(200, json={})))
    headers = {"Authorization": "Bearer sx"}
    assert await probe.problem(Target("soniox", headers, "stt-rt-v5"), transports) is None
    problem = await probe.problem(Target("soniox", headers, "stt-rt-v9"), transports)
    assert problem == "soniox does not list stt-rt-v9"

    def broken(_body: object) -> set[str]:
        raise KeyError("models")

    odd = ListedProbe("https://api.soniox.com/v1/models", broken)
    problem = await odd.problem(Target("soniox", headers, "stt-rt-v5"), transports)
    assert problem == "soniox listing was unreadable: 'models'"


async def test_deepgram_is_asked_at_the_listen_handshake_and_told_no_in_the_status() -> None:
    model = catalog().stt[0]
    assert model.provider == "deepgram"
    service = VENDORS["deepgram"].stt
    assert service is not None and service.model_probe is not None
    seen: dict[str, Any] = {}
    target = Target("deepgram", {"Authorization": "Token dg"}, model.name)
    assert await service.model_probe.problem(target, connecting(FakeSocket(), seen)) is None
    assert seen["url"] == f"wss://api.deepgram.com/v1/listen?model={model.name}"
    forbidden = InvalidStatus(Response(403, "Forbidden", Headers()))
    problem = await service.model_probe.problem(target, refusing(forbidden))
    assert problem == "deepgram returned 403"


async def test_soniox_tts_is_asked_to_say_one_character_and_judged_on_the_reply() -> None:
    service = VENDORS["soniox"].tts
    assert service is not None and service.model_probe is not None
    headers = {"Authorization": "Bearer sx"}
    target = Target("soniox", headers, "tts-rt-v2", {"voice": "Emma"})
    socket = FakeSocket({"stream_id": "check-access", "audio": "AAAA", "audio_end": True})
    seen: dict[str, Any] = {}
    assert await service.model_probe.problem(target, connecting(socket, seen)) is None
    assert seen["url"] == "wss://tts-rt.soniox.com/tts-websocket"
    start, text = socket.sent
    assert start["model"] == "tts-rt-v2" and start["voice"] == "Emma" and start["language"] == "en"
    assert text == {"stream_id": "check-access", "text": ".", "text_end": True}
    gone = FakeSocket({"error_code": 400, "error_message": "The requested model is not available."})
    problem = await service.model_probe.problem(target, connecting(gone))
    assert problem == "The requested model is not available."


async def test_xai_stt_is_asked_to_open_a_transcript_on_the_model() -> None:
    service = VENDORS["xai"].stt
    assert service is not None and service.model_probe is not None
    target = Target("xai", {"Authorization": "Bearer xa"}, "grok-voice-transcribe-2.0")
    seen: dict[str, Any] = {}
    socket = FakeSocket({"type": "transcript.created"})
    assert await service.model_probe.problem(target, connecting(socket, seen)) is None
    assert seen["url"] == (
        "wss://api.x.ai/v1/stt?model=grok-voice-transcribe-2.0&encoding=pcm&sample_rate=16000"
    )
    assert socket.sent == []
    refused = FakeSocket({"type": "error", "error": {"message": "unknown model"}})
    assert await service.model_probe.problem(target, connecting(refused)) == "unknown model"


async def test_xai_tts_is_asked_to_say_one_character_and_judged_on_the_reply() -> None:
    service = VENDORS["xai"].tts
    assert service is not None and service.model_probe is not None
    target = Target("xai", {"Authorization": "Bearer xa"}, "grok-tts", {"voice": "ara"})
    seen: dict[str, Any] = {}
    socket = FakeSocket({"type": "audio.delta", "delta": "AAAA"})
    assert await service.model_probe.problem(target, connecting(socket, seen)) is None
    assert seen["url"] == "wss://api.x.ai/v1/tts?language=en&codec=pcm&sample_rate=24000"
    assert socket.sent == [{"type": "text.delta", "delta": "."}, {"type": "text.done"}]
    refused = FakeSocket({"type": "error", "message": "bad request"})
    assert await service.model_probe.problem(target, connecting(refused)) == "bad request"


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
        if "googleapis" in request.url.host:
            return httpx.Response(403)
        return vendors_answer(request)

    @asynccontextmanager
    async def connect(url: str, headers: Mapping[str, str]) -> AsyncIterator[Socket]:
        if "x.ai/v1/stt" in url:
            yield FakeSocket({"type": "transcript.created"})
        elif "x.ai/v1/tts" in url:
            yield FakeSocket({"type": "audio.delta", "delta": "AAAA"})
        elif "grok" in url:
            yield FakeSocket({"type": "error", "error": {"message": "retired"}})
        elif "live/sessions" in url:
            yield FakeSocket({"type": "session.started"})
        elif "listen" in url:
            yield FakeSocket()
        elif "tts-websocket" in url:
            yield FakeSocket({"audio": "AAAA", "audio_end": True})
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
    assert any(line.split() == ["openai/gpt-4.1", "ok"] for line in lines)
    assert any(line.split() == ["deepgram/nova-3", "ok"] for line in lines)
    assert any(
        line.split() == ["google/gemini-2.5-flash", "skipped:", "its", "vendor", "failed"]
        for line in lines
    )
    assert lines[-1] == "2 failed"


async def test_a_clean_report_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(check_access, "coval_problem", lambda _settings: None)

    @asynccontextmanager
    async def connect(url: str, headers: Mapping[str, str]) -> AsyncIterator[Socket]:
        if "generativelanguage" in url:
            yield FakeSocket({"setupComplete": {}})
        elif "x.ai/v1/stt" in url:
            yield FakeSocket({"type": "transcript.created"})
        elif "x.ai/v1/tts" in url:
            yield FakeSocket({"type": "audio.delta", "delta": "AAAA"})
        elif "live/sessions" in url:
            yield FakeSocket({"type": "session.started"})
        elif "listen" in url:
            yield FakeSocket()
        elif "tts-websocket" in url:
            yield FakeSocket({"audio": "AAAA", "audio_end": True})
        else:
            yield FakeSocket({"type": "session.created"})

    assert await check_access.run(ALL_KEYS, Transports(http=ANSWERING, connect=connect)) == 0
    lines = capsys.readouterr().out.splitlines()
    assert not any("failed" in line for line in lines)
    models = sum(len(getattr(catalog(), m)) for m in ("stt", "llm", "tts", "s2s"))
    assert sum(line.endswith(" ok") for line in lines) == 1 + len(VENDORS) + models
