# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""``corral check-access``: does every key still work, and is every model still served?

The key probe is the same GET the menu runs before a pick. It proves the key, not the model,
since a vendor keeps answering it for a model it has retired. So every model in the catalog also
gets its own probe: a GET or a listing lookup for cascade models, a session opened on the
socket for native ones. Vendors with no key are skipped rather than failed, so a machine with
two keys can still check those two. Vendors run concurrently. Meant to run daily.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable
from dataclasses import dataclass

from corral.config import Settings
from corral.coval_api import coval_problem
from corral.models import (
    CASCADE,
    VENDORS,
    Model,
    S2sModel,
    TtsModel,
    Vendor,
    api_key,
    catalog,
    resolve_s2s,
)
from corral.models.access_probes import AccessProbe, Target, Transports


@dataclass(frozen=True)
class Finding:
    """One line of the report."""

    subject: str
    problem: str | None = None
    skipped: bool = False

    @property
    def failed(self) -> bool:
        return self.problem is not None and not self.skipped

    def line(self, indent: int = 0, width: int = 44) -> str:
        label = " " * indent + self.subject
        if self.skipped:
            return f"{label:<{width}} skipped: {self.problem}"
        return f"{label:<{width}} {'ok' if self.problem is None else self.problem}"


def probes() -> dict[str, list[tuple[Model, AccessProbe | None]]]:
    """Every model in the catalog with its probe, by provider, cascade models first. None is a
    model its vendor gives no way to ask about."""
    by_provider: dict[str, list[tuple[Model, AccessProbe | None]]] = {}
    for modality in CASCADE:
        for model in getattr(catalog(), modality):
            service = VENDORS[model.provider].service(modality)
            probe = service.model_probe if service else None
            by_provider.setdefault(model.provider, []).append((model, probe))
    for s2s in catalog().s2s:
        vendor = VENDORS[s2s.provider]
        api = (vendor.s2s or {})[s2s.api]
        by_provider.setdefault(s2s.provider, []).append((s2s, api.model_probe))
    return by_provider


async def model_problem(
    model: S2sModel, settings: Settings, transports: Transports | None = None
) -> str | None:
    """None when a session opened on the model with its vendor's key, else why not."""
    api, key = resolve_s2s(model, settings)
    target = Target(model.provider, VENDORS[model.provider].headers(key), model.name, model.options)
    return await api.model_probe.problem(target, transports or Transports())


async def _vendor_findings(
    provider: str,
    vendor: Vendor,
    models: list[tuple[Model, AccessProbe | None]],
    settings: Settings,
    transports: Transports,
) -> list[Finding]:
    """The vendor's line, then one per model under it."""
    key = api_key(provider, settings)
    if key is None:
        return [Finding(provider, f"{vendor.key_name} is unset", skipped=True)]
    headers = vendor.headers(key)
    problem = await vendor.key_probe.problem(Target(provider, headers), transports)
    report = [Finding(provider, problem)]
    if problem is not None:
        report.extend(Finding(m.id, "its vendor failed", skipped=True) for m, _ in models)
        return report

    async def one(model: Model, probe: AccessProbe | None) -> Finding:
        if probe is None:
            return Finding(model.id, f"{provider} has no way to ask", skipped=True)
        options = {
            **model.options,
            **({"voice": model.voice} if isinstance(model, TtsModel) else {}),
        }
        target = Target(provider, headers, model.name, options)
        return Finding(model.id, await probe.problem(target, transports))

    report.extend(await asyncio.gather(*(one(m, p) for m, p in models)))
    return report


class _Progress:
    """A counter on stderr, rewritten in place, shown only on a terminal."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.done = 0
        self.live = sys.stderr.isatty()
        self._show()

    def _show(self) -> None:
        if self.live:
            print(f"\rchecking {self.done}/{self.total}", end="", file=sys.stderr, flush=True)

    async def track[T](self, task: Awaitable[T]) -> T:
        try:
            return await task
        finally:
            self.done += 1
            self._show()

    def finish(self) -> None:
        if self.live:
            print("\r\033[K", end="", file=sys.stderr, flush=True)


async def findings(settings: Settings, transports: Transports | None = None) -> list[Finding]:
    """Coval first, then each vendor with its models under it, in table order."""
    transports = transports or Transports()
    by_provider = probes()
    progress = _Progress(len(VENDORS) + 1)
    vendors = [
        progress.track(
            _vendor_findings(provider, vendor, by_provider.get(provider, []), settings, transports)
        )
        for provider, vendor in VENDORS.items()
    ]
    coval = Finding("coval", await progress.track(asyncio.to_thread(coval_problem, settings)))
    reports = await asyncio.gather(*vendors)
    progress.finish()
    return [coval, *(f for report in reports for f in report)]


async def run(settings: Settings, transports: Transports | None = None) -> int:
    """Print the report. Exit 0 when everything with a key answered, else 1."""
    report = await findings(settings, transports)
    for finding in report:
        print(finding.line(indent=2 if "/" in finding.subject else 0), flush=True)
    failed = [f for f in report if f.failed]
    if failed:
        print(f"\n{len(failed)} failed", flush=True)
        return 1
    return 0
