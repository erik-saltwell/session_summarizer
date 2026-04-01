from __future__ import annotations

from pydantic import BaseModel, Field


class VadSettings(BaseModel, frozen=True):
    """Hyperparameters for NeMo VAD post-processing."""

    model_name: str = Field(description="Pretrained NeMo VAD model name")
    onset: float = Field(description="Speech onset probability threshold (0.0–1.0)")
    offset: float = Field(description="Speech offset probability threshold (0.0–1.0)")
    min_duration_on: float = Field(description="Minimum speech segment duration in seconds")
    min_duration_off: float = Field(description="Minimum silence segment duration in seconds")
    pad_onset: float = Field(description="Padding added before speech onset in seconds")
    pad_offset: float = Field(description="Padding added after speech offset in seconds")
