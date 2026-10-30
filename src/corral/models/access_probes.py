# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The cheapest request that proves a key still reaches a vendor, or a model.

A GetProbe is one authenticated GET, which is enough to know a key is live. A SocketProbe
opens a session on a native model's websocket and reads the vendor's first event, which is
enough to know the model is still served to that key. Same question at two levels, so
check-access runs both the same way.
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
    """What a probe is pointed at. ``model`` and ``options`` only matter to a SocketProbe."""

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


@dataclass(frozen=True)
class GetProbe:
    """One GET that returns 200 to a live key."""

    url: str

    async def problem(self, target: Target, transports: Transports) -> str | None:
        try:
            async with httpx.AsyncClient(
                timeout=PROBE_TIMEOUT_SECS, transport=transports.http
            ) as client:
                response = await client.get(self.url, headers=dict(target.headers))
        except httpx.HTTPError as error:
            return f"{target.provider} is unreachable: {error}"
        if response.status_code != 200:
            return f"{target.provider} returned {response.status_code}"
        return None


Opening = Callable[[str, Mapping[str, Any]], dict[str, Any] | None]
"""The first client message, or None when the vendor speaks first."""

Started = Callable[[object], str | None]
"""Why the vendor's first event isn't a started session, or None when it is."""

SocketUrl = Callable[[str | None, str], str]
"""The socket to open for a model, given the service's base URL."""


@dataclass(frozen=True)
class SessionProtocol:
    """How one speech-to-speech protocol opens a session: where, what to say, what to expect."""

    socket_url: SocketUrl
    opening: Opening
    started: Started


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
                opening = self.protocol.opening(target.model, target.options)
                if opening is not None:
                    await socket.send(json.dumps(opening))
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
            return f"{target.provider} is unreachable: {error}"
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


def _nothing(_model: str, _options: Mapping[str, Any]) -> None:
    return None


def _session_start(model: str, _options: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "session.start", "session": {"model": model}}


def _gemini_setup(model: str, options: Mapping[str, Any]) -> dict[str, Any]:
    setup: dict[str, Any] = {"model": f"models/{model}"}
    level = (options.get("thinking") or {}).get("thinking_level")
    if level:
        setup["generationConfig"] = {"thinkingConfig": {"thinkingLevel": level}}
    return {"setup": setup}


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
