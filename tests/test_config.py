# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from corral.config import remember


def test_remember_sets_one_line_in_an_existing_env_and_never_makes_one(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    assert not remember("CORRAL_WORKSPACE_ID", "W1", env)
    assert not env.exists()
    env.write_text("COVAL_API_KEY=k\nCORRAL_WORKSPACE_ID=\n")
    assert remember("CORRAL_WORKSPACE_ID", "W1", env)
    assert remember("CORRAL_WORKSPACE_ID", "W2", env)
    assert env.read_text() == "COVAL_API_KEY=k\nCORRAL_WORKSPACE_ID=W2\n"
