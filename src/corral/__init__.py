# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The S2S agent: a Pipecat pipeline Coval's simulator calls in place of the vendor.

Pipecat announces itself on import through loguru's default sink, before corral has said a
word, so the sink goes here and ``configure_logging`` puts one back when a run starts."""

from loguru import logger

logger.remove()
