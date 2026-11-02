# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""One log stream: ours, Pipecat's, and anything a library says through stdlib logging."""

from __future__ import annotations

import inspect
import logging
import os
import sys
from types import FrameType
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Record


class _ToLoguru(logging.Handler):
    """Hands a stdlib record to loguru at the same level, credited to the line that logged it."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame: FrameType | None = inspect.currentframe()
        depth = 0
        while frame is not None and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame, depth = frame.f_back, depth + 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


EVERY_RUN = (
    "differs from the recommended default",
    "interruption detected, cancelling",
    "The live model handles being interrupted by itself",
)


def _format(record: Record) -> str:
    message = "{message}" if record["level"].no <= logging.INFO else "<level>{message}</level>"
    return f"<green>{{time:HH:mm:ss}}</green> {message}\n{{exception}}"


def _wanted(record: Record) -> bool:
    return not any(phrase in record["message"] for phrase in EVERY_RUN)


def configure_logging() -> None:
    """Log to stderr at INFO, and route stdlib logging into loguru.

    Pipecat narrates every link, connect and metric at DEBUG; LOGURU_LEVEL=DEBUG brings
    that back. Two of its warnings fire on every run (our deliberate VAD stop_secs, and
    Ctrl-C itself) and are dropped. Libraries like websockets and httpcore log every
    frame and header at DEBUG, so stdlib loggers start at WARNING.
    """
    logger.remove()
    logger.add(
        sys.stderr, level=os.environ.get("LOGURU_LEVEL", "INFO"), format=_format, filter=_wanted
    )
    logging.basicConfig(handlers=[_ToLoguru()], level=logging.WARNING, force=True)
