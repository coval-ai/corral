# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Keys from the environment or the repo-root ``.env``, and the harness's fixed numbers."""

from __future__ import annotations

import functools
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]

VAD_STOP_SECS = 0.3
STT_FINALIZE_FALLBACK_SECS = 60 * 60
INTERRUPT_MIN_WORDS = 3
IN_SAMPLE_RATE_HZ = 16000
OUT_SAMPLE_RATE_HZ = 24000


class Settings(BaseSettings):
    # The repo root's .env from any working directory; one in the current
    # directory overrides it. In the image neither exists.
    model_config = SettingsConfigDict(
        env_file=(str(ROOT / ".env"), ".env"),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
    )

    deepgram_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    elevenlabs_api_key: SecretStr | None = None
    soniox_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None

    coval_api_key: SecretStr | None = None
    coval_api_base: str = "https://api.coval.dev/v1"
    corral_workspace_id: str | None = None


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def remember(name: str, value: str, env_file: Path = ROOT / ".env") -> bool:
    """Set ``name=value`` in the repo's .env, replacing its line. False if there's no .env."""
    if not env_file.is_file():
        return False
    lines = [
        line for line in env_file.read_text().splitlines() if line.split("=", 1)[0].strip() != name
    ]
    env_file.write_text("\n".join([*lines, f"{name}={value}"]) + "\n")
    return True
