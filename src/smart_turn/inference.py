"""Smart Turn ONNX inference for end-of-turn prediction.

Vendored from https://github.com/pipecat-ai/smart-turn (BSD 2-Clause).

Adaptations for this project:
  - Resolves the ONNX model from ``models/smart-turn/`` inside the repo root.
  - Uses a package-relative import for ``audio_utils``.
  - Lazily initialises the ONNX session on first call so import alone is free.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor

from .audio_utils import truncate_audio_to_last_n_seconds

# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "smart-turn"
# Resolved: <repo_root>/models/smart-turn/

_CPU_MODEL_NAME = "smart-turn-v3.2-cpu.onnx"
_GPU_MODEL_NAME = "smart-turn-v3.2-gpu.onnx"


def _resolve_model_path() -> Path:
    """Return the path to the ONNX model, preferring GPU, falling back to CPU."""
    model_dir = _DEFAULT_MODEL_DIR.resolve()
    gpu_path = model_dir / _GPU_MODEL_NAME
    cpu_path = model_dir / _CPU_MODEL_NAME
    if gpu_path.exists():
        return gpu_path
    if cpu_path.exists():
        return cpu_path
    raise FileNotFoundError(
        f"Smart Turn ONNX model not found. Expected one of:\n"
        f"  {gpu_path}\n"
        f"  {cpu_path}\n"
        f"Download from: https://huggingface.co/pipecat-ai/smart-turn-v3\n"
        f"  huggingface-cli download pipecat-ai/smart-turn-v3 {_CPU_MODEL_NAME} --local-dir {model_dir}"
    )


# ---------------------------------------------------------------------------
# Lazy singleton session
# ---------------------------------------------------------------------------

_feature_extractor: WhisperFeatureExtractor | None = None
_session: ort.InferenceSession | None = None


def _ensure_loaded() -> tuple[WhisperFeatureExtractor, ort.InferenceSession]:
    global _feature_extractor, _session  # noqa: PLW0603
    if _session is None:
        model_path = _resolve_model_path()
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = ort.InferenceSession(str(model_path), sess_options=so)
        _feature_extractor = WhisperFeatureExtractor(chunk_length=8)
    assert _feature_extractor is not None
    return _feature_extractor, _session


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_endpoint(audio_array: np.ndarray) -> dict[str, float]:
    """Predict whether an audio segment is complete (turn ended) or incomplete.

    Args:
        audio_array: Numpy array containing audio samples at 16 kHz.

    Returns:
        Dictionary with:
          - ``prediction``: 1 for complete, 0 for incomplete
          - ``probability``: end-of-turn probability (sigmoid output)
    """
    feature_extractor, session = _ensure_loaded()

    # Truncate to 8 seconds (keeping the end) or pad to 8 seconds
    audio_array = truncate_audio_to_last_n_seconds(audio_array, n_seconds=8)

    # Process audio using Whisper's feature extractor
    inputs = feature_extractor(
        audio_array,
        sampling_rate=16000,
        return_tensors="np",
        padding="max_length",
        max_length=8 * 16000,
        truncation=True,
        do_normalize=True,
    )

    # Extract features and ensure correct shape for ONNX
    input_features = inputs.input_features.squeeze(0).astype(np.float32)
    input_features = np.expand_dims(input_features, axis=0)  # Add batch dimension

    # Run ONNX inference
    outputs = session.run(None, {"input_features": input_features})

    # Extract probability (ONNX model returns sigmoid probabilities)
    probability = outputs[0][0].item()

    # Make prediction (1 for Complete, 0 for Incomplete)
    prediction = 1 if probability > 0.5 else 0

    return {
        "prediction": prediction,
        "probability": probability,
    }
