# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import threading

from pipecat.transports.base_transport import TransportParams

from corral.local import LocalAudioTransport, Playout, _Mic, _Speakers


def test_the_speakers_get_what_was_written_then_silence() -> None:
    playout = Playout(limit=8)
    playout.write(b"abcdef")
    assert playout.take(4) == b"abcd"
    assert playout.take(4) == b"ef\x00\x00"
    assert playout.take(2) == b"\x00\x00"


def test_a_writer_waits_until_the_speakers_have_nearly_caught_up() -> None:
    playout = Playout(limit=4)
    playout.write(b"1234")
    arrived = threading.Event()

    def write_more() -> None:
        playout.write(b"5678")
        arrived.set()

    threading.Thread(target=write_more).start()
    assert not arrived.wait(0.1)
    assert playout.take(2) == b"12"
    assert arrived.wait(1)
    assert playout.take(6) == b"345678"


def test_closing_lets_a_waiting_writer_go_without_its_audio() -> None:
    playout = Playout(limit=2)
    playout.write(b"12")
    released = threading.Event()

    def write_more() -> None:
        playout.write(b"34")
        released.set()

    threading.Thread(target=write_more).start()
    assert not released.wait(0.1)
    playout.close()
    assert released.wait(1)
    assert playout.take(4) == b"12\x00\x00"


def test_the_transport_builds_a_mic_and_speakers_without_opening_a_device() -> None:
    transport = LocalAudioTransport(TransportParams(audio_in_enabled=True, audio_out_enabled=True))
    assert isinstance(transport.input(), _Mic)
    assert isinstance(transport.output(), _Speakers)
    assert transport.input() is transport.input()
