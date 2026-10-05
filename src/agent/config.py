# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Settings from the environment or the repo-root ``.env``."""

from __future__ import annotations

import functools
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


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

    # Bearer Coval must present to the served agent; unset means open.
    s2s_agent_token: SecretStr | None = None

    coval_api_key: SecretStr | None = None
    coval_api_base: str = "https://api.coval.dev/v1"


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
