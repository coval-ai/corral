# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from corral.environment import JsonEnvironment


def environment() -> JsonEnvironment:
    return JsonEnvironment.load(Path(__file__).with_name("tools.json"))
