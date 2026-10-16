# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import threading
from collections.abc import Callable
from typing import Any

import miniaudio
from pipecat.transports.base_transport import TransportParams

from corral.local import LocalAudioTransport, Playout, _Mic, _Speakers, close_device


def test_the_speakers_get_what_was_written_then_silence() -> None:
    playout = Playout(limit=8)
    playout.write(b"abcdef")
    assert playout.take(4) == b"abcd"
    assert playout.take(4) == b"ef\x00\x00"
    assert playout.take(2) == b"\x00\x00"


class _Answering(threading.Condition):
    """Answers a wait at once and keeps what was asked, so a test can ask it again later."""

    def __init__(self) -> None:
        super().__init__()
        self.asked: list[Callable[[], Any]] = []

    def wait_for(self, predicate: Callable[[], Any], timeout: float | None = None) -> Any:
        self.asked.append(predicate)
        return predicate()


def test_a_writer_waits_until_the_speakers_have_nearly_caught_up() -> None:
    condition = _Answering()
    playout = Playout(limit=4, condition=condition)
    playout.write(b"1234")
    playout.write(b"5678")
    ready = condition.asked[-1]
    assert not ready()
    assert playout.take(2) == b"12"
    assert not ready()
    assert playout.take(4) == b"3456"
    assert ready()


def test_closing_lets_a_waiting_writer_go_without_its_audio() -> None:
    condition = _Answering()
    playout = Playout(limit=2, condition=condition)
    playout.write(b"12")
    playout.write(b"34")
    ready = condition.asked[-1]
    assert not ready()
    playout.close()
    assert ready()
    playout.write(b"56")
    assert playout.take(8) == b"1234" + bytes(4)


def test_the_transport_builds_a_mic_and_speakers_without_opening_a_device() -> None:
    transport = LocalAudioTransport(TransportParams(audio_in_enabled=True, audio_out_enabled=True))
    assert isinstance(transport.input(), _Mic)
    assert isinstance(transport.output(), _Speakers)
    assert transport.input() is transport.input()


def test_a_device_that_will_not_stop_is_closed_once_and_forgotten() -> None:
    calls: list[str] = []

    class Unplugged:
        running = True

        def stop(self) -> None:
            calls.append("stop")
            raise miniaudio.MiniaudioError("failed to stop audio device")

        def close(self) -> None:
            calls.append("close")

    device: Any = Unplugged()
    close_device(device)
    assert calls == ["stop", "close"]
    assert device.running is False
