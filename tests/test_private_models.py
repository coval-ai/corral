# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from corral import models
from corral.agent import load_native_agent
from corral.assets import AssetsError, load_private_models
from corral.cli import offered_models
from corral.config import Settings
from corral.models.eleven import ElevenRealtimeLLMService

VENDOR: dict[str, Any] = {
    "key_env": "ACME_API_KEY",
    "speaks": "elevenlabs-realtime",
    "base_url": "wss://acme.test/v1/realtime",
}
ACME: dict[str, Any] = {
    "vendors": {"acme": VENDOR},
    "s2s": [
        {"provider": "acme", "name": "voice-1", "options": {"voice": "ash"}},
        {"provider": "openai", "name": "gpt-early-access"},
    ],
}


@pytest.fixture(autouse=True)
def only_public_afterwards() -> Iterator[None]:
    with patch.dict(models.VENDORS), patch.object(models, "_private_s2s", []):
        models.catalog.cache_clear()
        yield
    models.catalog.cache_clear()


def _settings(tmp_path: Path, listing: dict[str, Any]) -> Settings:
    path = tmp_path / "models.json"
    path.write_text(json.dumps(listing))
    return Settings(_env_file=None, corral_models_uri=str(path), openai_api_key="oa")


def test_without_a_private_list_only_the_catalog_is_offered() -> None:
    before = models.model_ids("s2s")
    load_private_models(Settings(_env_file=None))
    assert models.model_ids("s2s") == before and "acme" not in models.VENDORS


def test_a_private_list_adds_its_models_after_the_catalogs(tmp_path: Path) -> None:
    public = models.model_ids("s2s")
    load_private_models(_settings(tmp_path, ACME))
    assert models.model_ids("s2s") == [*public, "acme/voice-1", "openai/gpt-early-access"]


def test_a_private_vendor_runs_on_a_known_service_with_its_own_key_and_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, ACME)
    load_private_models(settings)
    monkeypatch.delenv("ACME_API_KEY", raising=False)
    assert "acme/voice-1" not in [model.id for model in offered_models("s2s", settings)]
    assert models.vendor_problem("acme", settings) == "ACME_API_KEY is unset"
    monkeypatch.setenv("ACME_API_KEY", "ac")
    assert "acme/voice-1" in [model.id for model in offered_models("s2s", settings)]
    assert models.vendor_problem("acme", settings) is None
    agent = load_native_agent("default", "acme/voice-1")
    native = models.resolve(agent.stack, "x", settings)
    assert isinstance(native, models.NativeService)
    assert isinstance(native.model, ElevenRealtimeLLMService)
    assert native.model.base_url == "wss://acme.test/v1/realtime?model=voice-1"
    assert native.model.api_key == "ac"
    api, _ = models.resolve_s2s(agent.stack.rows[0], settings)  # type: ignore[arg-type]
    assert api.model_probe.base_url == "wss://acme.test/v1/realtime"
    assert models.VENDORS["acme"].headers("ac") == {"xi-api-key": "ac"}


def test_a_private_list_that_doesnt_fit_never_prints_what_it_holds(tmp_path: Path) -> None:
    bad = {"s2s": [{"provider": "openai", "name": "hush", "path": "os:system"}]}
    with pytest.raises(AssetsError) as error:
        load_private_models(_settings(tmp_path, bad))
    assert "hush" not in str(error.value)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"vendors": {"openai": VENDOR}}, "already offered"),
        ({"s2s": [{"provider": "nobody", "name": "x"}]}, "needs a vendor"),
        ({"s2s": [{"provider": "openai", "name": "gpt-realtime-2.1"}]}, "already offered"),
        ({"s2s": [{"provider": "acme", "name": "x", "delegation": {}}]}, "has a delegation"),
        ({"vendors": {"acme": {**VENDOR, "key_env": "COVAL_API_KEY"}}}, "can't take its key"),
        ({"vendors": {"acme": {**VENDOR, "key_env": "GITHUB_TOKEN"}}}, "can't take its key"),
        ({"vendors": {"acme": {**VENDOR, "speaks": "some.module:Class"}}}, "have openai-realtime"),
        ({"vendors": {"acme": {**VENDOR, "speaks": "google-live"}}}, "have openai-realtime"),
        ({"vendors": {"acme": {**VENDOR, "base_url": "ws://a"}}}, "needs a wss:// base_url"),
        ({"vendors": {"acme": {**VENDOR, "key_check_url": "http://a"}}}, "needs an https://"),
        ({"s2s": [{"provider": "openai", "name": "hush", "path": "os:system"}]}, "s2s.0.path"),
    ],
)
def test_a_private_list_that_doesnt_fit_stops_before_anything_is_added(
    tmp_path: Path, change: dict[str, Any], message: str
) -> None:
    public = models.model_ids("s2s")
    with pytest.raises(AssetsError, match=message):
        load_private_models(_settings(tmp_path, {**ACME, **change}))
    assert models.model_ids("s2s") == public and "acme" not in models.VENDORS
