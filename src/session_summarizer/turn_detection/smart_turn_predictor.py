from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np


class SmartTurnPredictor(Protocol):
    def predict_endpoint(self, audio_f32_16k: np.ndarray) -> dict[str, float]:
        """Run Smart Turn inference on an 8-second, 16 kHz mono audio window.

        Returns a dict with at least:
            "probability": float  — end-of-turn probability in [0, 1]
            "prediction":  int    — binary prediction (0 or 1)
        """
        ...


class LocalSmartTurnPredictor:
    """Wraps the local Smart Turn model for end-of-turn prediction."""

    _predict_fn: Callable[..., Any]

    def __init__(self, device: str = "cpu") -> None:
        try:
            from smart_turn.inference import predict_endpoint
        except ImportError as exc:
            raise ImportError(
                "The 'smart_turn' package is required for turn-end scoring. "
                "Install it before running the update-turn-end command."
            ) from exc
        self._predict_fn = predict_endpoint
        self._device = device

    def predict_endpoint(self, audio_f32_16k: np.ndarray) -> dict[str, float]:
        result: dict[str, float] = self._predict_fn(audio_f32_16k)
        return result
