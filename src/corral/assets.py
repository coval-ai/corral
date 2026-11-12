# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Coval ids for a suite that names its assets, from a manifest shaped like the scenario map.

``benchmark-infra/assets`` publishes that map for every scenario and version it manages; a
staging manifest is the same shape with the staging org's ids. CORRAL_ASSETS_URI points at one,
as ``gs://bucket/object`` or a local path. The workspace, test set, personas and metric ids make
the suite. A run from ``--assets`` takes its system prompt and tools from the files the entry
lists, or else borrows the prompt of one of its agents as saved in Coval, with no tools; Corral
still makes its own agent record to run it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

import google.auth
import google.auth.exceptions
import google.auth.transport.requests
import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from corral.config import Settings
from corral.scenarios import ALL_PERSONAS, Suite

GCS_READ_SCOPE = "https://www.googleapis.com/auth/devstorage.read_only"
GCS_TIMEOUT_SECS = 10


class AssetsError(ValueError):
    pass


class _Persona(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str


class _Agent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str


class _Entry(BaseModel):
    model_config = ConfigDict(extra="ignore")
    workspace_id: str
    test_set_id: str
    personas: dict[str, _Persona]
    metric_ids: dict[str, str] = {}
    agents: dict[str, _Agent] = {}
    system_prompt_uri: str | None = None
    tools_uri: str | None = None


def load_manifest(uri: str, transport: httpx.BaseTransport | None = None) -> dict[str, Any]:
    raw = _read_gcs(uri, transport) if uri.startswith("gs://") else _read_file(uri)
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as error:
        raise AssetsError(f"manifest {uri} isn't JSON: {error}") from None
    if not isinstance(manifest, dict):
        raise AssetsError(f"manifest {uri} isn't a JSON object")
    return manifest


def read_asset_file(
    uri: str, transport: httpx.BaseTransport | None = None, what: str = "file"
) -> bytes:
    return _read_gcs(uri, transport, what) if uri.startswith("gs://") else _read_file(uri, what)


def _read_file(uri: str, what: str = "manifest") -> bytes:
    path = Path(uri).expanduser()
    if not path.is_file():
        raise AssetsError(f"{what} {uri} doesn't exist")
    return path.read_bytes()


def _read_gcs(uri: str, transport: httpx.BaseTransport | None, what: str = "manifest") -> bytes:
    bucket, _, name = uri.removeprefix("gs://").partition("/")
    if not bucket or not name:
        raise AssetsError(f"{what} {uri} isn't a gs://bucket/object URI")
    try:
        credentials, _ = google.auth.default(scopes=[GCS_READ_SCOPE])
        credentials.refresh(google.auth.transport.requests.Request())  # type: ignore[no-untyped-call]
    except google.auth.exceptions.GoogleAuthError as error:
        raise AssetsError(f"can't sign in to Google to read {uri}: {error}") from None
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{quote(name, safe='')}"
    try:
        with httpx.Client(timeout=GCS_TIMEOUT_SECS, transport=transport) as client:
            response = client.get(
                url,
                params={"alt": "media"},
                headers={"Authorization": f"Bearer {credentials.token}"},
            )
    except httpx.HTTPError as error:
        raise AssetsError(f"can't reach {uri}: {error}") from None
    if response.status_code in (401, 403):
        raise AssetsError(
            f"can't read {uri} (HTTP {response.status_code}); your Google account needs "
            f"roles/storage.objectViewer on {bucket}"
        )
    if response.status_code != 200:
        raise AssetsError(f"can't read {uri} (HTTP {response.status_code})")
    return response.content


def _entry(name: str, manifest: dict[str, Any], source: str) -> _Entry:
    scenario, _, version = name.rpartition(".")
    versions = manifest.get(scenario)
    if not scenario or not isinstance(versions, dict):
        have = ", ".join(sorted(manifest)) or "none"
        raise AssetsError(f"assets {name}: no scenario {scenario!r} in {source}; it has {have}")
    if version not in versions:
        have = ", ".join(sorted(versions)) or "none"
        raise AssetsError(f"assets {name}: no version {version!r} in {source}; it has {have}")
    try:
        return _Entry.model_validate(versions[version])
    except ValidationError as error:
        raise AssetsError(f"assets {name} in {source} is malformed: {error}") from None


def resolve(suite: Suite, manifest: dict[str, Any], source: str) -> Suite:
    """The suite in id form, from the manifest entry its ``assets`` names."""
    entry = _entry(suite.assets or "", manifest, source)
    wanted = [str(p) for p in suite.personas]
    if wanted == [ALL_PERSONAS]:
        personas = list(entry.personas.values())
    else:
        personas = _pick("persona", suite, wanted, entry.personas, source, default_first=True)
    metrics = _pick("metric", suite, [str(m) for m in suite.metrics], entry.metric_ids, source)
    return Suite(
        workspace=entry.workspace_id,
        test_set=entry.test_set_id,
        personas=[persona.id for persona in personas],
        metrics=list(dict.fromkeys(metrics)),
        sample=suite.sample,
    )


def _pick[T](
    noun: str,
    suite: Suite,
    wanted: list[str],
    available: dict[str, T],
    source: str,
    default_first: bool = False,
) -> list[T]:
    """The named entries in the order asked for; when none are named, the first in manifest
    order or every one."""
    if not wanted:
        entries = list(available.values())
        return entries[:1] if default_first else entries
    missing = [name for name in wanted if name not in available]
    if missing:
        have = ", ".join(available) or "none"
        raise AssetsError(
            f"assets {suite.assets}: no {noun} {', '.join(missing)} in {source}; it has {have}"
        )
    return [available[name] for name in wanted]


def resolve_assets(suite: Suite, settings: Settings) -> Suite:
    """The suite unchanged when it pins its own ids, else resolved from CORRAL_ASSETS_URI."""
    if suite.assets is None:
        return suite
    uri = settings.corral_assets_uri
    if not uri:
        raise AssetsError(
            f"coval.json names assets {suite.assets}; set CORRAL_ASSETS_URI to a manifest"
        )
    return resolve(suite, load_manifest(uri), uri)


class AgentReader(Protocol):
    """The part of the Coval client an asset run needs: a workspace and one agent."""

    def use_workspace(self, workspace_id: str) -> None: ...

    def get_agent(self, agent_id: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class AssetRun:
    """A run straight from the assets: the suite in id form, the system prompt and, when the
    entry lists them, the tools."""

    suite: Suite
    system_prompt: str
    source: str
    tools_file: bytes | None = None


def load_asset_run(
    name: str,
    *,
    personas: list[str],
    metrics: list[str],
    model_id: str,
    prompt_from: str | None,
    settings: Settings,
    client: AgentReader,
) -> AssetRun:
    """The suite for ``name``, with the prompt and tools files its entry lists. Without a
    prompt file, the prompt of its agent for ``model_id``: a model with its own agent keeps
    that prompt; ``prompt_from`` names whose to borrow when it hasn't one."""
    uri = settings.corral_assets_uri
    if not uri:
        raise AssetsError(f"--assets {name} needs CORRAL_ASSETS_URI set to a manifest")
    manifest = load_manifest(uri)
    suite = resolve(Suite(assets=name, personas=personas, metrics=metrics), manifest, uri)
    entry = _entry(name, manifest, uri)
    source = f"{uri} ({name})"
    tools = read_asset_file(entry.tools_uri, what="tools") if entry.tools_uri else None
    if entry.system_prompt_uri:
        text = read_asset_file(entry.system_prompt_uri, what="prompt").decode().strip()
        if not text:
            raise AssetsError(f"prompt {entry.system_prompt_uri} is empty")
        return AssetRun(suite, text, source, tools)
    wanted = model_id if model_id in entry.agents else prompt_from or model_id
    agent = entry.agents.get(wanted)
    if agent is None:
        have = ", ".join(sorted(entry.agents)) or "none"
        hint = "" if prompt_from else "; pass --prompt-from with one of them"
        raise AssetsError(f"assets {name} has no agent {wanted} in {uri}{hint}. It has {have}")
    client.use_workspace(entry.workspace_id)
    prompt = agent_prompt(client.get_agent(agent.id))
    if not prompt:
        raise AssetsError(f"agent {wanted} ({agent.id}) has no system prompt saved in Coval")
    return AssetRun(suite, prompt, source, tools)


def agent_prompt(agent: dict[str, Any]) -> str | None:
    """The system prompt saved on a Coval agent: Gemini Live keeps it in its own field, the
    websocket agents in the session.update they send first."""
    metadata = agent.get("metadata") or {}
    gemini = metadata.get("gemini_realtime_instructions")
    if isinstance(gemini, str) and gemini.strip():
        return gemini.strip()
    init = metadata.get("initialization_json")
    if isinstance(init, str):
        try:
            init = json.loads(init)
        except json.JSONDecodeError:
            return None
    if isinstance(init, list):
        init = init[0] if init else None
    session = init.get("session") if isinstance(init, dict) else None
    instructions = session.get("instructions") if isinstance(session, dict) else None
    return instructions.strip() if isinstance(instructions, str) and instructions.strip() else None
