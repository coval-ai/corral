# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""One log stream: ours, Pipecat's, and anything a library says through stdlib logging."""

from __future__ import annotations

import inspect
import logging
from types import FrameType

from loguru import logger


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


def configure_logging() -> None:
    """Route stdlib logging into loguru.

    Libraries like websockets and httpcore log every frame and header at DEBUG, so
    stdlib loggers start at WARNING. Uvicorn sets its own level when it starts.
    """
    logging.basicConfig(handlers=[_ToLoguru()], level=logging.WARNING, force=True)
