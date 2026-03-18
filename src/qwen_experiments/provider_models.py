from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ProviderModelInfo(BaseModel):
    id: str
    label: str
    supports_batch: bool = True
    supports_streaming: bool = False
    default_voice: str


class ProviderVoiceInfo(BaseModel):
    id: str
    label: str
    language: str = "English"
    tags: list[str] = Field(default_factory=list)


class ProviderSpeechRequest(BaseModel):
    model: str
    input: str
    voice: str
    response_format: Literal["wav"] = "wav"
    language: str = "English"
    instructions: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderSpeechResponse(BaseModel):
    model: str
    format: Literal["wav"] = "wav"
    sample_rate: int
    audio_b64: str
    timings: dict[str, int] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)


class ProviderWarmupResponse(BaseModel):
    status: str
    model: str
    ready: bool
    elapsed_ms: float
