# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The models on offer, and the Pipecat services that run them.

stt.json, llm.json and tts.json list what you can pick, the default first. The vendor
table says which Pipecat class runs each provider and which key pays for it. Every
Pipecat service takes the same ``api_key``, ``sample_rate`` and ``Settings``, so
building one works the same for every vendor. Providers and keys are checked before
anything connects, so a bad pick fails at startup with the list of what's supported.
"""

from __future__ import annotations

import functools
import importlib
import importlib.resources
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pipecat.processors.frame_processor import FrameProcessor
from pydantic import BaseModel, ConfigDict, Field

from corral.config import (
    IN_SAMPLE_RATE_HZ,
    OUT_SAMPLE_RATE_HZ,
    STT_FINALIZE_FALLBACK_SECS,
    Settings,
)

if TYPE_CHECKING:
    from corral.agent import Agent

Modality = Literal["stt", "llm", "tts"]
MODALITIES: tuple[Modality, ...] = ("stt", "llm", "tts")


class JsonModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Model(JsonModel):
    """One entry in a models file. ``options`` is for knobs only that vendor understands."""

    provider: str
    name: str
    options: dict[str, Any] = Field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.provider}/{self.name}"


class SttModel(Model):
    pass


class LlmModel(Model):
    temperature: float


class TtsModel(Model):
    voice: str


class Catalog(JsonModel):
    stt: list[SttModel]
    llm: list[LlmModel]
    tts: list[TtsModel]

    def get_stt(self, model_id: str) -> SttModel:
        return _get("stt", self.stt, model_id)

    def get_llm(self, model_id: str) -> LlmModel:
        return _get("llm", self.llm, model_id)

    def get_tts(self, model_id: str) -> TtsModel:
        return _get("tts", self.tts, model_id)


@functools.cache
def catalog() -> Catalog:
    folder = importlib.resources.files(__name__)
    return Catalog.model_validate(
        {kind: json.loads(folder.joinpath(f"{kind}.json").read_bytes()) for kind in MODALITIES}
    )


def model_ids(modality: Modality) -> list[str]:
    """Every model offered for a modality, as provider/model. The first is the default."""
    return [model.id for model in getattr(catalog(), modality)]


def _get[M: Model](modality: Modality, offered: list[M], model_id: str) -> M:
    for model in offered:
        if model.id == model_id:
            return model
    raise ValueError(f"no {modality} model {model_id!r}; have {', '.join(m.id for m in offered)}")


@dataclass(frozen=True)
class Vendor:
    key_attr: str
    stt: str | None = None
    llm: str | None = None
    tts: str | None = None


VENDORS: dict[str, Vendor] = {
    "deepgram": Vendor("deepgram_api_key", stt="pipecat.services.deepgram.stt:DeepgramSTTService"),
    "openai": Vendor("openai_api_key", llm="pipecat.services.openai.llm:OpenAILLMService"),
    "elevenlabs": Vendor(
        "elevenlabs_api_key", tts="pipecat.services.elevenlabs.tts:ElevenLabsTTSService"
    ),
    "soniox": Vendor(
        "soniox_api_key",
        stt="pipecat.services.soniox.stt:SonioxSTTService",
        tts="pipecat.services.soniox.tts:SonioxTTSService",
    ),
    "google": Vendor("gemini_api_key", llm="pipecat.services.google.llm:GoogleLLMService"),
}


@dataclass(frozen=True)
class Services:
    stt: FrameProcessor
    llm: FrameProcessor
    tts: FrameProcessor


def supported(modality: Modality) -> list[str]:
    return sorted(name for name, vendor in VENDORS.items() if getattr(vendor, modality) is not None)


def _resolve_vendor(modality: Modality, provider: str, settings: Settings) -> tuple[str, str]:
    """The class path for this modality and the key that pays for it, or why not."""
    vendor = VENDORS.get(provider)
    path: str | None = getattr(vendor, modality) if vendor else None
    if vendor is None or path is None:
        raise ValueError(f"no {modality} for {provider!r}; have {', '.join(supported(modality))}")
    value = getattr(settings, vendor.key_attr)
    if value is None or not value.get_secret_value():
        raise RuntimeError(f"{vendor.key_attr.upper()} is unset")
    key: str = value.get_secret_value()
    return path, key


def _build(
    path: str, api_key: str, fields: dict[str, Any], sample_rate: int | None, **extra: float
) -> FrameProcessor:
    """Import the Pipecat class and build it with its Settings, the key and anything extra."""
    module_name, class_name = path.split(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "settings": cls.Settings.from_mapping(fields),
        **extra,
    }
    if sample_rate is not None:
        kwargs["sample_rate"] = sample_rate
    service: FrameProcessor = cls(**kwargs)
    return service


def resolve(agent: Agent, settings: Settings) -> Services:
    """Every modality's vendor and key checked first, then the three services built."""
    stt_path, stt_key = _resolve_vendor("stt", agent.stt.provider, settings)
    llm_path, llm_key = _resolve_vendor("llm", agent.llm.provider, settings)
    tts_path, tts_key = _resolve_vendor("tts", agent.tts.provider, settings)
    return Services(
        stt=_build(
            stt_path,
            stt_key,
            {"model": agent.stt.name, **agent.stt.options},
            IN_SAMPLE_RATE_HZ,
            stt_ttfb_timeout=STT_FINALIZE_FALLBACK_SECS,
        ),
        llm=_build(
            llm_path,
            llm_key,
            {
                "model": agent.llm.name,
                "temperature": agent.llm.temperature,
                "system_instruction": agent.system_prompt,
                **agent.llm.options,
            },
            None,
        ),
        tts=_build(
            tts_path,
            tts_key,
            {"model": agent.tts.name, "voice": agent.tts.voice, **agent.tts.options},
            OUT_SAMPLE_RATE_HZ,
        ),
    )
