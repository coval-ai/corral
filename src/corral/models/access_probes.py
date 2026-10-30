# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The cheapest request that proves a key still reaches a vendor, or a model.

A GetProbe is one authenticated GET: on a vendor's root it proves the key is live, on a model
resource it proves the model exists. A ListedProbe is for vendors that only list their models,
and checks the id is in the list. A SocketProbe opens a session on a native model's websocket
and reads the vendor's first event. Same question at every level, so check-access runs them all
the same way.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import quote

import httpx
from websockets.asyncio.client import connect as websocket_connect
from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException

PROBE_TIMEOUT_SECS = 10


@dataclass(frozen=True)
class Target:
    """What a probe is pointed at. ``model`` is None for a key probe; ``options`` only matter
    to a SocketProbe."""

    provider: str
    headers: Mapping[str, str]
    model: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


class Socket(Protocol):
    def send(self, message: str) -> Awaitable[None]: ...
    def recv(self) -> Awaitable[str | bytes]: ...


Connect = Callable[[str, Mapping[str, str]], AbstractAsyncContextManager[Socket]]
"""Opens a websocket; tests swap in a fake."""


@asynccontextmanager
async def _connect(url: str, headers: Mapping[str, str]) -> AsyncIterator[Socket]:
    async with websocket_connect(url, additional_headers=dict(headers)) as socket:
        yield socket


@dataclass(frozen=True)
class Transports:
    """The network, or in tests a stand-in for it."""

    http: httpx.AsyncBaseTransport | None = None
    connect: Connect = _connect


class AccessProbe(Protocol):
    async def problem(self, target: Target, transports: Transports) -> str | None:
        """None when the target answered, else a sentence saying why not."""
        ...


async def _get(url: str, target: Target, transports: Transports) -> httpx.Response | str:
    """The response, or why there wasn't one."""
    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_SECS, transport=transports.http) as c:
            response = await c.get(url, headers=dict(target.headers))
    except httpx.HTTPError as error:
        return f"{target.provider} is unreachable: {error or type(error).__name__}"
    if response.status_code != 200:
        return f"{target.provider} returned {response.status_code}"
    return response


@dataclass(frozen=True)
class GetProbe:
    """One GET that returns 200. ``{model}`` in the URL, if any, is the model's name."""

    url: str

    async def problem(self, target: Target, transports: Transports) -> str | None:
        url = self.url.format(model=quote(target.model, safe="")) if target.model else self.url
        response = await _get(url, target, transports)
        return response if isinstance(response, str) else None


Listed = Callable[[object], set[str]]
"""The model ids in a vendor's listing, as it returns them."""


@dataclass(frozen=True)
class ListedProbe:
    """For vendors that only list models: a GET of the list, and the model must be in it."""

    url: str
    listed: Listed

    async def problem(self, target: Target, transports: Transports) -> str | None:
        if target.model is None:
            raise ValueError(f"a listed probe needs a model; {target.provider} gave none")
        response = await _get(self.url, target, transports)
        if isinstance(response, str):
            return response
        try:
            ids = self.listed(response.json())
        except (ValueError, KeyError, TypeError) as error:
            return f"{target.provider} listing was unreadable: {error}"
        if target.model not in ids:
            return f"{target.provider} does not list {target.model}"
        return None


Opening = Callable[[str, Mapping[str, Any]], list[dict[str, Any]]]
"""What the client says first, in order; empty when the vendor speaks first."""

Started = Callable[[object], str | None]
"""Why the vendor's first event isn't a started session, or None when it is."""

SocketUrl = Callable[[str | None, str], str]
"""The socket to open for a model, given the service's base URL."""


@dataclass(frozen=True)
class SessionProtocol:
    """How a protocol opens a session on a model: where, what to say first, and which first
    event means it worked. ``started`` is None when the handshake itself is the answer."""

    socket_url: SocketUrl
    opening: Opening
    started: Started | None


@dataclass(frozen=True)
class SocketProbe:
    """A session opened on the model's own websocket. ``base_url`` is the service's, the same
    one its Pipecat class connects to; the protocol says where the model goes."""

    base_url: str | None
    protocol: SessionProtocol

    async def problem(self, target: Target, transports: Transports) -> str | None:
        if target.model is None:
            raise ValueError(f"a socket probe needs a model; {target.provider} gave none")
        url = self.protocol.socket_url(self.base_url, target.model)
        try:
            async with (
                asyncio.timeout(PROBE_TIMEOUT_SECS),
                transports.connect(url, target.headers) as socket,
            ):
                for message in self.protocol.opening(target.model, target.options):
                    await socket.send(json.dumps(message))
                if self.protocol.started is None:
                    return None
                event = json.loads(await socket.recv())
        except InvalidStatus as error:
            return f"{target.provider} returned {error.response.status_code}"
        except TimeoutError:
            return f"no session event in {PROBE_TIMEOUT_SECS} s"
        except ConnectionClosed as error:
            close = error.rcvd
            reason = f"{close.code} {close.reason}".strip() if close else "without a close frame"
            return f"{target.provider} closed the socket: {reason}"
        except (OSError, WebSocketException, ValueError) as error:
            return f"{target.provider} is unreachable: {error or type(error).__name__}"
        return self.protocol.started(event)


def _model_in_query(base_url: str | None, model: str) -> str:
    if base_url is None:
        raise ValueError("the realtime protocol needs the service's base URL")
    return f"{base_url}?model={quote(model, safe='')}"


def _base_only(base_url: str | None, _model: str) -> str:
    if base_url is None:
        raise ValueError("the live protocol needs the service's base URL")
    return base_url


def _gemini_socket(_base_url: str | None, _model: str) -> str:
    return (
        "wss://generativelanguage.googleapis.com/ws/"
        "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    )


def _deepgram_listen(_base_url: str | None, model: str) -> str:
    return f"wss://api.deepgram.com/v1/listen?model={quote(model, safe='')}"


def _soniox_tts_socket(_base_url: str | None, _model: str) -> str:
    return "wss://tts-rt.soniox.com/tts-websocket"


def _nothing(_model: str, _options: Mapping[str, Any]) -> list[dict[str, Any]]:
    return []


def _session_start(model: str, _options: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"type": "session.start", "session": {"model": model}}]


def _gemini_setup(model: str, options: Mapping[str, Any]) -> list[dict[str, Any]]:
    setup: dict[str, Any] = {"model": f"models/{model}"}
    level = (options.get("thinking") or {}).get("thinking_level")
    if level:
        setup["generationConfig"] = {"thinkingConfig": {"thinkingLevel": level}}
    return [{"setup": setup}]


def _soniox_one_character(model: str, options: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Soniox only judges the model once asked to speak, so ask for a full stop."""
    start = {
        "stream_id": "check-access",
        "model": model,
        "voice": options["voice"],
        "language": "en",
        "audio_format": "pcm_s16le",
        "sample_rate": 24000,
    }
    return [start, {"stream_id": "check-access", "text": ".", "text_end": True}]


def _soniox_spoke(event: object) -> str | None:
    if not isinstance(event, dict):
        return f"unexpected first event: {event!r}"
    if event.get("error_code") is not None:
        message: str = event.get("error_message") or f"error {event['error_code']}"
        return message
    if "audio" in event or event.get("terminated"):
        return None
    return f"unexpected first event: {sorted(event)}"


def _typed(started: str) -> Started:
    """OpenAI-shaped events: a ``type``, and an ``error`` that explains itself."""

    def check(event: object) -> str | None:
        if not isinstance(event, dict):
            return f"unexpected first event: {event!r}"
        kind = event.get("type")
        if kind == started:
            return None
        if kind == "error":
            error = event.get("error") or {}
            return error.get("message") or json.dumps(error)
        return f"unexpected first event: {kind!r}"

    return check


def _setup_complete(event: object) -> str | None:
    if not isinstance(event, dict):
        return f"unexpected first event: {event!r}"
    if "setupComplete" in event:
        return None
    error = event.get("error") or {}
    return error.get("message") or f"unexpected first event: {sorted(event)}"


OPENAI_REALTIME_PROTOCOL = SessionProtocol(_model_in_query, _nothing, _typed("session.created"))
OPENAI_LIVE_PROTOCOL = SessionProtocol(_base_only, _session_start, _typed("session.started"))
GEMINI_LIVE_PROTOCOL = SessionProtocol(_gemini_socket, _gemini_setup, _setup_complete)
DEEPGRAM_LISTEN_PROTOCOL = SessionProtocol(_deepgram_listen, _nothing, None)
SONIOX_TTS_PROTOCOL = SessionProtocol(_soniox_tts_socket, _soniox_one_character, _soniox_spoke)
