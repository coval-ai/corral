# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""``corral check-access``: does every key still work, and is every native model still served?

The key probe is the same GET the menu runs before a pick. It proves the key, not the model,
since a vendor keeps answering it for a model it has retired. So each native model also gets a
session opened on its own socket. Vendors with no key are skipped rather than failed, so a
machine with two keys can still check those two. Meant to run daily.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from corral.config import Settings
from corral.coval_api import coval_problem
from corral.models import VENDORS, S2sModel, api_key, catalog, resolve_s2s
from corral.models.access_probes import Target, Transports


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


async def model_problem(
    model: S2sModel, settings: Settings, transports: Transports | None = None
) -> str | None:
    """None when a session opened on the model with its vendor's key, else why not."""
    api, key = resolve_s2s(model, settings)
    target = Target(model.provider, VENDORS[model.provider].headers(key), model.name, model.options)
    return await api.model_probe.problem(target, transports or Transports())


async def findings(settings: Settings, transports: Transports | None = None) -> list[Finding]:
    """Coval first, then each vendor with its native models under it."""
    transports = transports or Transports()
    report = [Finding("coval", coval_problem(settings))]
    by_provider: dict[str, list[S2sModel]] = {}
    for model in catalog().s2s:
        by_provider.setdefault(model.provider, []).append(model)
    for provider, vendor in VENDORS.items():
        key = api_key(provider, settings)
        if key is None:
            report.append(Finding(provider, f"{vendor.key_name} is unset", skipped=True))
            continue
        problem = await vendor.key_probe.problem(Target(provider, vendor.headers(key)), transports)
        report.append(Finding(provider, problem))
        models = by_provider.get(provider, [])
        if problem is not None:
            report.extend(Finding(m.id, "its vendor failed", skipped=True) for m in models)
            continue
        problems = await asyncio.gather(*(model_problem(m, settings, transports) for m in models))
        report.extend(Finding(m.id, p) for m, p in zip(models, problems, strict=True))
    return report


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
