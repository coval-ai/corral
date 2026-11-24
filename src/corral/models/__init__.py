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

import asyncio
import functools
import importlib
import importlib.resources
import inspect
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

import httpx
from dotenv import dotenv_values
from pipecat.processors.frame_processor import FrameProcessor
from pydantic import BaseModel, ConfigDict, Field

from corral.config import (
    ENV_FILES,
    IN_SAMPLE_RATE_HZ,
    OUT_SAMPLE_RATE_HZ,
    STT_FINALIZE_FALLBACK_SECS,
    Settings,
)
from corral.models.access_probes import (
    DEEPGRAM_LISTEN_PROTOCOL,
    GEMINI_LIVE_PROTOCOL,
    OPENAI_LIVE_PROTOCOL,
    OPENAI_REALTIME_PROTOCOL,
    SONIOX_TTS_PROTOCOL,
    XAI_STT_PROTOCOL,
    XAI_TTS_PROTOCOL,
    AccessProbe,
    GetProbe,
    KeyTakenProbe,
    ListedProbe,
    SessionProtocol,
    SocketProbe,
    Target,
    Transports,
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
    offered = Catalog.model_validate(
        {kind: json.loads(folder.joinpath(f"{kind}.json").read_bytes()) for kind in MODALITIES}
    )
    return offered.model_copy(update={"s2s": [*offered.s2s, *_private_s2s]})


def model_ids(modality: Modality) -> list[str]:
    """Every model offered for a modality, as provider/model. The first is the default."""
    return [model.id for model in getattr(catalog(), modality)]


def _get[M: Model](modality: Modality, offered: list[M], model_id: str) -> M:
    for model in offered:
        if model.id == model_id:
            return model
    raise ValueError(f"no {modality} model {model_id!r}; have {', '.join(m.id for m in offered)}")


@dataclass(frozen=True)
class S2sApi:
    """One speech-to-speech service: the Pipecat class that runs it and the protocol it speaks.
    ``base_url`` is for a vendor that speaks another vendor's protocol; otherwise the socket is
    the class's own default. ``waits_for_cue`` is a service that only opens the call when told."""

    path: str
    protocol: SessionProtocol
    base_url: str | None = None
    reports_turn_end: bool = True
    waits_for_cue: bool = False
    delegates: bool = False

    @property
    def socket_base_url(self) -> str | None:
        """Where the service connects: ``base_url`` when set, else its class's default."""
        return self.base_url if self.base_url is not None else _default_base_url(self.path)

    @property
    def model_probe(self) -> SocketProbe:
        return SocketProbe(self.socket_base_url, self.protocol)


@functools.lru_cache
def _default_base_url(path: str) -> str | None:
    """The ``base_url`` a Pipecat service connects to unless told otherwise; None if it has none."""
    parameter = inspect.signature(_import(path).__init__).parameters.get("base_url")
    if parameter is None or parameter.default is inspect.Parameter.empty:
        return None
    default: str = parameter.default
    return default


@dataclass(frozen=True)
class Service:
    """A cascade service: the Pipecat class that runs it, and how check-access asks whether a
    model is still served. None when the vendor offers no way to ask short of using it."""

    path: str
    model_probe: AccessProbe | None = None


def _soniox_models(listing: object) -> set[str]:
    return {model["id"] for model in _listing(listing)["models"]}


def _eleven_models(listing: object) -> set[str]:
    if not isinstance(listing, list):
        raise TypeError("expected a list of models")
    return {model["model_id"] for model in listing}


def _listing(listing: object) -> dict[str, Any]:
    if not isinstance(listing, dict):
        raise TypeError("expected an object of models")
    return listing


@dataclass(frozen=True)
class Vendor:
    """Which key pays for a provider, the probe that proves it works, and its services."""

    key_attr: str
    key_probe: GetProbe | KeyTakenProbe | None
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer "
    stt: Service | None = None
    llm: Service | None = None
    tts: Service | None = None
    s2s: Mapping[str, S2sApi] | None = None

    def service(self, modality: Modality) -> Service | None:
        """The cascade service for a modality; None for s2s, which is per API."""
        if modality == "s2s":
            return None
        service: Service | None = getattr(self, modality)
        return service

    @property
    def key_name(self) -> str:
        return self.key_attr.upper()

    def headers(self, key: str) -> dict[str, str]:
        return {self.auth_header: self.auth_prefix + key}


VENDORS: dict[str, Vendor] = {
    "deepgram": Vendor(
        "deepgram_api_key",
        GetProbe("https://api.deepgram.com/v1/projects"),
        auth_prefix="Token ",
        stt=Service(
            "pipecat.services.deepgram.stt:DeepgramSTTService",
            SocketProbe(None, DEEPGRAM_LISTEN_PROTOCOL),
        ),
    ),
    "openai": Vendor(
        "openai_api_key",
        GetProbe("https://api.openai.com/v1/models"),
        llm=Service(
            "pipecat.services.openai.llm:OpenAILLMService",
            GetProbe("https://api.openai.com/v1/models/{model}"),
        ),
        s2s={
            "realtime": S2sApi(
                "pipecat.services.openai.realtime.llm:OpenAIRealtimeLLMService",
                OPENAI_REALTIME_PROTOCOL,
            ),
            "live": S2sApi(
                "pipecat.services.openai.live.llm:OpenAILiveLLMService",
                OPENAI_LIVE_PROTOCOL,
                reports_turn_end=False,
                waits_for_cue=True,
                delegates=True,
            ),
        },
    ),
    "resia": Vendor(
        "resia_api_key",
        KeyTakenProbe("https://api.resia.ai/v1/live/sessions", 400),
        s2s={
            "live": S2sApi(
                "corral.models.resia:ResiaLiveLLMService",
                OPENAI_LIVE_PROTOCOL,
                base_url="wss://api.resia.ai/v1/live/sessions",
                reports_turn_end=False,
                waits_for_cue=True,
                delegates=True,
            )
        },
    ),
    "elevenlabs": Vendor(
        "elevenlabs_api_key",
        GetProbe("https://api.elevenlabs.io/v1/voices"),
        auth_header="xi-api-key",
        auth_prefix="",
        tts=Service(
            "pipecat.services.elevenlabs.tts:ElevenLabsTTSService",
            ListedProbe("https://api.elevenlabs.io/v1/models", _eleven_models),
        ),
        s2s={
            "realtime": S2sApi(
                "corral.models.eleven:ElevenRealtimeLLMService",
                OPENAI_REALTIME_PROTOCOL,
                base_url="wss://api.elevenlabs.io/v1/convai/realtime",
            )
        },
    ),
    "soniox": Vendor(
        "soniox_api_key",
        GetProbe("https://api.soniox.com/v1/models"),
        stt=Service(
            "pipecat.services.soniox.stt:SonioxSTTService",
            ListedProbe("https://api.soniox.com/v1/models", _soniox_models),
        ),
        tts=Service(
            "pipecat.services.soniox.tts:SonioxTTSService",
            SocketProbe(None, SONIOX_TTS_PROTOCOL),
        ),
    ),
    "google": Vendor(
        "gemini_api_key",
        GetProbe("https://generativelanguage.googleapis.com/v1beta/models"),
        auth_header="x-goog-api-key",
        auth_prefix="",
        llm=Service(
            "pipecat.services.google.llm:GoogleLLMService",
            GetProbe("https://generativelanguage.googleapis.com/v1beta/models/{model}"),
        ),
        s2s={
            "live": S2sApi(
                "pipecat.services.google.gemini_live.llm:GeminiLiveLLMService",
                GEMINI_LIVE_PROTOCOL,
                reports_turn_end=False,
                waits_for_cue=True,
            )
        },
    ),
    "xai": Vendor(
        "xai_api_key",
        GetProbe("https://api.x.ai/v1/models"),
        stt=Service("pipecat.services.xai.stt:XAISTTService", SocketProbe(None, XAI_STT_PROTOCOL)),
        llm=Service(
            "pipecat.services.xai.llm:GrokLLMService",
            GetProbe("https://api.x.ai/v1/models/{model}"),
        ),
        tts=Service("pipecat.services.xai.tts:XAITTSService", SocketProbe(None, XAI_TTS_PROTOCOL)),
        s2s={
            "realtime": S2sApi(
                "pipecat.services.xai.realtime.llm:GrokRealtimeLLMService",
                OPENAI_REALTIME_PROTOCOL,
            )
        },
    ),
}


PUBLIC_VENDORS: Mapping[str, Vendor] = dict(VENDORS)
KEY_ENV = re.compile(r"[A-Z][A-Z0-9_]*_API_KEY")

_private_s2s: list[S2sModel] = []


def s2s_services() -> dict[str, tuple[Vendor, S2sApi]]:
    """Each speech-to-speech service a private list can pick, by name, with the vendor whose
    auth it sends. Only the ones that connect wherever they are told to."""
    return {
        f"{provider}-{name}": (vendor, api)
        for provider, vendor in PUBLIC_VENDORS.items()
        for name, api in (vendor.s2s or {}).items()
        if api.socket_base_url is not None
    }


class PrivateVendor(JsonModel):
    """A vendor only a private list names: the known service it speaks like, where, and the
    variable its key is read from. It sends its key the way that service does."""

    key_env: str
    key_check_url: str | None = None
    speaks: str
    base_url: str


class PrivateModels(JsonModel):
    """The private list at CORRAL_MODELS_URI: models, and vendors, kept out of this repo."""

    vendors: dict[str, PrivateVendor] = Field(default_factory=dict)
    s2s: list[S2sModel] = Field(default_factory=list)


def add_private_models(listing: PrivateModels) -> None:
    """Offer a private list's models next to the catalog's, for this process only."""
    vendors = {
        provider: _private_vendor(f"vendors.{index}", provider, spec)
        for index, (provider, spec) in enumerate(listing.vendors.items())
    }
    offered = {model.id for model in catalog().s2s}
    for index, model in enumerate(listing.s2s):
        where = f"s2s.{index}"
        vendor = vendors.get(model.provider) or VENDORS.get(model.provider)
        api = (vendor.s2s or {}).get(model.api) if vendor else None
        if api is None:
            raise ValueError(f"{where}: its provider has no vendor with that s2s API")
        if model.delegation is not None and not api.delegates:
            raise ValueError(f"{where}: has a delegation, which its API doesn't take")
        if model.id in offered:
            raise ValueError(f"{where}: is already offered")
        offered.add(model.id)
    VENDORS.update(vendors)
    _private_s2s.extend(listing.s2s)
    catalog.cache_clear()


def _private_vendor(where: str, provider: str, spec: PrivateVendor) -> Vendor:
    """The vendor a private entry describes. What is wrong names the entry, never its values."""
    if provider in VENDORS:
        raise ValueError(f"{where}: that vendor is already offered")
    key_attr = spec.key_env.lower()
    if not KEY_ENV.fullmatch(spec.key_env) or key_attr in Settings.model_fields:
        raise ValueError(
            f"{where}.key_env: must be upper-case, end in _API_KEY and not be Corral's own"
        )
    services = s2s_services()
    if spec.speaks not in services:
        raise ValueError(f"{where}.speaks: must be one of {', '.join(services)}")
    if not spec.base_url.startswith("wss://"):
        raise ValueError(f"{where}.base_url: must start with wss://")
    if spec.key_check_url and not spec.key_check_url.startswith("https://"):
        raise ValueError(f"{where}.key_check_url: must start with https://")
    like, api = services[spec.speaks]
    return Vendor(
        key_attr,
        GetProbe(spec.key_check_url) if spec.key_check_url else None,
        auth_header=like.auth_header,
        auth_prefix=like.auth_prefix,
        s2s={spec.speaks.split("-", 1)[1]: replace(api, base_url=spec.base_url)},
    )


def key_name(provider: str) -> str:
    return VENDORS[provider].key_name


def api_key(provider: str, settings: Settings) -> str | None:
    """The provider's key, or None when it is unset or blank."""
    vendor = VENDORS[provider]
    if vendor.key_attr not in Settings.model_fields:
        return os.environ.get(vendor.key_name) or _dotenv().get(vendor.key_name) or None
    value = getattr(settings, vendor.key_attr)
    if value is None or not value.get_secret_value():
        return None
    key: str = value.get_secret_value()
    return key


@functools.cache
def _dotenv() -> dict[str, str | None]:
    """The .env files Settings reads, for a key Settings has no field for."""
    values: dict[str, str | None] = {}
    for path in ENV_FILES:
        values.update(dotenv_values(path))
    return values


def vendor_problem(
    provider: str, settings: Settings, transport: httpx.AsyncBaseTransport | None = None
) -> str | None:
    """The vendor's key probe, run from the menu: None when the key is live, else why not."""
    vendor = VENDORS[provider]
    key = api_key(provider, settings)
    if key is None:
        return f"{vendor.key_name} is unset"
    if vendor.key_probe is None:
        return None
    target = Target(provider, vendor.headers(key))
    return asyncio.run(vendor.key_probe.problem(target, Transports(http=transport)))


@dataclass(frozen=True)
class CascadeServices:
    stt: FrameProcessor
    llm: FrameProcessor
    tts: FrameProcessor


@dataclass(frozen=True)
class NativeService:
    model: FrameProcessor
    reports_turn_end: bool = True
    waits_for_cue: bool = False


Services = CascadeServices | NativeService


def supported(modality: Modality) -> list[str]:
    return sorted(name for name, vendor in VENDORS.items() if getattr(vendor, modality) is not None)


def _resolve_vendor(modality: Modality, provider: str, settings: Settings) -> tuple[str, str]:
    """The class path for this modality and the key that pays for it, or why not."""
    vendor = VENDORS.get(provider)
    service = vendor.service(modality) if vendor else None
    if vendor is None or service is None:
        raise ValueError(f"no {modality} for {provider!r}; have {', '.join(supported(modality))}")
    return service.path, _key(vendor, provider, settings)


def resolve_s2s(model: S2sModel, settings: Settings) -> tuple[S2sApi, str]:
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


def _import(path: str) -> type[Any]:
    """The class a ``module:Class`` path names."""
    module_name, class_name = path.split(":")
    cls: type[Any] = getattr(importlib.import_module(module_name), class_name)
    return cls


def _build(
    path: str, api_key: str, fields: dict[str, Any], sample_rate: int | None, **extra: object
) -> FrameProcessor:
    """Import the Pipecat class and build it with its Settings, the key and anything extra."""
    cls = _import(path)
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
        api, key = resolve_s2s(s2s, settings)
        fields = {"model": s2s.name, "system_instruction": system_prompt, **s2s.options}
        extra: dict[str, object] = {} if api.base_url is None else {"base_url": api.base_url}
        if s2s.delegation is not None:
            extra["delegation"] = _responses_delegation(s2s.delegation, system_prompt)
        model = _build(api.path, key, fields, None, **extra)
        return NativeService(
            model, reports_turn_end=api.reports_turn_end, waits_for_cue=api.waits_for_cue
        )
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
