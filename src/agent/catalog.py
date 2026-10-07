# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The models on offer, one file per modality under models/, the default first."""

from __future__ import annotations

import functools
import importlib.resources
import json
from typing import Any, Literal

from pydantic import Field

from agent.annotated import AnnotatedModel

MODELS_DIR = "models"

Modality = Literal["stt", "llm", "tts"]
MODALITIES: tuple[Modality, ...] = ("stt", "llm", "tts")


class ComponentPin(AnnotatedModel):
    """One modality's vendor and model. ``options`` is for knobs only that vendor understands."""

    provider: str
    model: str
    options: dict[str, Any] = Field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.provider}/{self.model}"


class SttPin(ComponentPin):
    pass


class LlmPin(ComponentPin):
    temperature: float


class TtsPin(ComponentPin):
    voice: str


class Catalog(AnnotatedModel):
    stt: list[SttPin]
    llm: list[LlmPin]
    tts: list[TtsPin]


@functools.cache
def catalog() -> Catalog:
    folder = importlib.resources.files("agent").joinpath(MODELS_DIR)
    return Catalog.model_validate(
        {kind: json.loads(folder.joinpath(f"{kind}.json").read_bytes()) for kind in MODALITIES}
    )


def model_ids(modality: Modality) -> list[str]:
    """Every model offered for a modality, as provider/model. The first is the default."""
    return [pin.id for pin in getattr(catalog(), modality)]


def find[P: ComponentPin](modality: Modality, pins: list[P], model_id: str | None) -> P:
    if model_id is None:
        return pins[0]
    for pin in pins:
        if pin.id == model_id:
            return pin
    raise ValueError(f"no {modality} model {model_id!r}; have {', '.join(model_ids(modality))}")
