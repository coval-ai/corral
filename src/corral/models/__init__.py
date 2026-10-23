# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""The models on offer, and the Pipecat services that run them.

stt.json, llm.json, tts.json and s2s.json list what you can pick, the default first. A
stack is a cascade of one STT, LLM and TTS, or a native speech-to-speech model on its own.
The vendor table says which Pipecat class runs each provider and which key pays for it.
Every Pipecat service takes the same ``api_key``, ``sample_rate`` and ``Settings``, so
building one works the same for every vendor. Providers and keys are checked before
anything connects, so a bad pick fails at startup with the list of what's supported, and
each vendor answers one cheap authenticated request before a run is started on it.
"""

from __future__ import annotations

import functools
import importlib
import importlib.resources
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from pipecat.processors.frame_processor import FrameProcessor
from pydantic import BaseModel, ConfigDict, Field

from corral.config import (
    IN_SAMPLE_RATE_HZ,
    OUT_SAMPLE_RATE_HZ,
    STT_FINALIZE_FALLBACK_SECS,
    Settings,
)

Modality = Literal["stt", "llm", "tts", "s2s"]
CASCADE: tuple[Modality, ...] = ("stt", "llm", "tts")
MODALITIES: tuple[Modality, ...] = (*CASCADE, "s2s")


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


class S2sModel(Model):
    """A native speech-to-speech model. Voice and turn detection are vendor knobs, in options.
    ``delegation`` is the Responses model a live model hands its reasoning to."""

    api: str = "realtime"
    delegation: dict[str, Any] | None = None


@dataclass(frozen=True)
class Cascade:
    stt: SttModel
    llm: LlmModel
    tts: TtsModel

    @property
    def rows(self) -> tuple[Model, ...]:
        return (self.stt, self.llm, self.tts)


@dataclass(frozen=True)
class Native:
    s2s: S2sModel

    @property
    def rows(self) -> tuple[Model, ...]:
        return (self.s2s,)


Stack = Cascade | Native


class Catalog(JsonModel):
    stt: list[SttModel]
    llm: list[LlmModel]
    tts: list[TtsModel]
    s2s: list[S2sModel]

    def get_stt(self, model_id: str) -> SttModel:
        return _get("stt", self.stt, model_id)

    def get_llm(self, model_id: str) -> LlmModel:
        return _get("llm", self.llm, model_id)

    def get_tts(self, model_id: str) -> TtsModel:
        return _get("tts", self.tts, model_id)

    def get_s2s(self, model_id: str) -> S2sModel:
        return _get("s2s", self.s2s, model_id)


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


CHECK_TIMEOUT_SECS = 10


@dataclass(frozen=True)
class S2sApi:
    """One speech-to-speech service. A full-duplex one never says when the caller is done."""

    path: str
    full_duplex: bool = False
    delegates: bool = False


@dataclass(frozen=True)
class Vendor:
    """Which key pays for a provider, the GET that proves the key works, and its services."""

    key_attr: str
    check_url: str
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer "
    stt: str | None = None
    llm: str | None = None
    tts: str | None = None
    s2s: Mapping[str, S2sApi] | None = None

    @property
    def key_name(self) -> str:
        return self.key_attr.upper()


VENDORS: dict[str, Vendor] = {
    "deepgram": Vendor(
        "deepgram_api_key",
        "https://api.deepgram.com/v1/projects",
        auth_prefix="Token ",
        stt="pipecat.services.deepgram.stt:DeepgramSTTService",
    ),
    "openai": Vendor(
        "openai_api_key",
        "https://api.openai.com/v1/models",
        llm="pipecat.services.openai.llm:OpenAILLMService",
        s2s={
            "realtime": S2sApi("pipecat.services.openai.realtime.llm:OpenAIRealtimeLLMService"),
            "live": S2sApi(
                "pipecat.services.openai.live.llm:OpenAILiveLLMService",
                full_duplex=True,
                delegates=True,
            ),
        },
    ),
    "elevenlabs": Vendor(
        "elevenlabs_api_key",
        "https://api.elevenlabs.io/v1/user",
        auth_header="xi-api-key",
        auth_prefix="",
        tts="pipecat.services.elevenlabs.tts:ElevenLabsTTSService",
    ),
    "soniox": Vendor(
        "soniox_api_key",
        "https://api.soniox.com/v1/models",
        stt="pipecat.services.soniox.stt:SonioxSTTService",
        tts="pipecat.services.soniox.tts:SonioxTTSService",
    ),
    "google": Vendor(
        "gemini_api_key",
        "https://generativelanguage.googleapis.com/v1beta/models",
        auth_header="x-goog-api-key",
        auth_prefix="",
        llm="pipecat.services.google.llm:GoogleLLMService",
    ),
}


def key_name(provider: str) -> str:
    return VENDORS[provider].key_name


def api_key(provider: str, settings: Settings) -> str | None:
    """The provider's key, or None when it is unset or blank."""
    value = getattr(settings, VENDORS[provider].key_attr)
    if value is None or not value.get_secret_value():
        return None
    key: str = value.get_secret_value()
    return key


def vendor_problem(
    provider: str, settings: Settings, transport: httpx.BaseTransport | None = None
) -> str | None:
    """One cheap authenticated GET to the vendor: None when it answered 200, else why not."""
    vendor = VENDORS[provider]
    key = api_key(provider, settings)
    if key is None:
        return f"{vendor.key_name} is unset"
    headers = {vendor.auth_header: vendor.auth_prefix + key}
    try:
        with httpx.Client(timeout=CHECK_TIMEOUT_SECS, transport=transport) as client:
            response = client.get(vendor.check_url, headers=headers)
    except httpx.HTTPError as error:
        return f"{provider} is unreachable: {error}"
    if response.status_code != 200:
        return f"{provider} returned {response.status_code}"
    return None


@dataclass(frozen=True)
class CascadeServices:
    stt: FrameProcessor
    llm: FrameProcessor
    tts: FrameProcessor


@dataclass(frozen=True)
class NativeService:
    """``full_duplex`` listens while it talks and never says when the caller is done."""

    model: FrameProcessor
    full_duplex: bool = False


Services = CascadeServices | NativeService


def supported(modality: Modality) -> list[str]:
    return sorted(name for name, vendor in VENDORS.items() if getattr(vendor, modality) is not None)


def _resolve_vendor(modality: Modality, provider: str, settings: Settings) -> tuple[str, str]:
    """The class path for this modality and the key that pays for it, or why not."""
    vendor = VENDORS.get(provider)
    path: str | None = getattr(vendor, modality) if vendor else None
    if vendor is None or path is None:
        raise ValueError(f"no {modality} for {provider!r}; have {', '.join(supported(modality))}")
    return path, _key(vendor, provider, settings)


def _resolve_s2s(model: S2sModel, settings: Settings) -> tuple[S2sApi, str]:
    """The model's speech-to-speech service and the key that pays for it, or why not."""
    vendor = VENDORS.get(model.provider)
    if vendor is None or vendor.s2s is None:
        raise ValueError(f"no s2s for {model.provider!r}; have {', '.join(supported('s2s'))}")
    api = vendor.s2s.get(model.api)
    if api is None:
        have = ", ".join(vendor.s2s)
        raise ValueError(f"no {model.api!r} s2s for {model.provider!r}; have {have}")
    if model.delegation is not None and not api.delegates:
        raise ValueError(f"{model.id} has a delegation, which its {model.api} API doesn't take")
    return api, _key(vendor, model.provider, settings)


def _key(vendor: Vendor, provider: str, settings: Settings) -> str:
    key = api_key(provider, settings)
    if key is None:
        raise RuntimeError(f"{vendor.key_name} is unset")
    return key


def _build(
    path: str, api_key: str, fields: dict[str, Any], sample_rate: int | None, **extra: object
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


def resolve(stack: Stack, system_prompt: str, settings: Settings) -> Services:
    """Every vendor and key in the stack checked first, then its services built."""
    if isinstance(stack, Native):
        s2s = stack.s2s
        api, key = _resolve_s2s(s2s, settings)
        fields = {"model": s2s.name, "system_instruction": system_prompt, **s2s.options}
        extra: dict[str, object] = {}
        if s2s.delegation is not None:
            extra["delegation"] = _responses_delegation(s2s.delegation, system_prompt)
        model = _build(api.path, key, fields, None, **extra)
        return NativeService(model, full_duplex=api.full_duplex)
    stt_path, stt_key = _resolve_vendor("stt", stack.stt.provider, settings)
    llm_path, llm_key = _resolve_vendor("llm", stack.llm.provider, settings)
    tts_path, tts_key = _resolve_vendor("tts", stack.tts.provider, settings)
    return CascadeServices(
        stt=_build(
            stt_path,
            stt_key,
            {"model": stack.stt.name, **stack.stt.options},
            IN_SAMPLE_RATE_HZ,
            stt_ttfb_timeout=STT_FINALIZE_FALLBACK_SECS,
        ),
        llm=_build(
            llm_path,
            llm_key,
            {
                "model": stack.llm.name,
                "temperature": stack.llm.temperature,
                "system_instruction": system_prompt,
                **stack.llm.options,
            },
            None,
        ),
        tts=_build(
            tts_path,
            tts_key,
            {"model": stack.tts.name, "voice": stack.tts.voice, **stack.tts.options},
            OUT_SAMPLE_RATE_HZ,
        ),
    )


def _responses_delegation(spec: Mapping[str, Any], system_prompt: str) -> object:
    """The Responses model a live model delegates to, on the same prompt."""
    from pipecat.services.openai.live.llm import ResponsesDelegation
    from pipecat.services.openai.responses.llm import (
        OpenAIResponsesLLMService,
        OpenAIResponsesReasoningConfig,
    )

    fields = {"system_instruction": system_prompt, **spec}
    service_tier = fields.pop("service_tier", None)
    if "reasoning" in fields:
        fields["reasoning"] = OpenAIResponsesReasoningConfig(**fields["reasoning"])
    settings = OpenAIResponsesLLMService.Settings.from_mapping(fields)
    return ResponsesDelegation(settings=settings, service_tier=service_tier)
