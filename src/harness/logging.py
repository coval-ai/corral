# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Structlog JSON to stdout, with the severity key Cloud Logging filters on."""

from __future__ import annotations

import logging
import sys
from typing import Literal

import structlog
from structlog.typing import EventDict, WrappedLogger

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


def add_gcp_severity(_logger: WrappedLogger, _method_name: str, event_dict: EventDict) -> EventDict:
    level = event_dict.get("level")
    if level is not None:
        event_dict["severity"] = level.upper()
    return event_dict


def configure_logging(level: LogLevel = "INFO") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            add_gcp_severity,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=log_level)
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(max(log_level, logging.WARNING))
