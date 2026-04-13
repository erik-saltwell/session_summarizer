from __future__ import annotations

from typing import Any, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "google/DiarizationLM-8b-Fisher-v2"
_UNREASONABLE_TOKENIZER_CONTEXT = 1_000_000


class DiarizationLMModel:
    """Wraps the DiarizationLM-8b-Fisher-v2 model for inference."""

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        self._device = device
        self._model_name = model_name
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype=torch.float16,
            device_map=self._device,
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def context_window(self) -> int | None:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        config = getattr(self._model, "config", None)
        for attr in ("max_position_embeddings", "n_positions", "seq_length"):
            value = getattr(config, attr, None)
            if isinstance(value, int) and 0 < value < _UNREASONABLE_TOKENIZER_CONTEXT:
                return value

        tokenizer_limit = getattr(self._tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < _UNREASONABLE_TOKENIZER_CONTEXT:
            return tokenizer_limit

        return None

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        inputs = self._tokenizer([text], return_tensors="pt")
        return int(inputs.input_ids.shape[1])

    def infer(self, prompt: str, max_new_tokens: int | None = None) -> str:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self._tokenizer([prompt], return_tensors="pt").to(self._device)
        prompt_length = inputs.input_ids.shape[1]

        # The completion reproduces the input text with corrected speaker tags,
        # so it needs at least as many tokens as the prompt.  Use 2x headroom;
        # the model stops at EOS anyway, so a generous limit only costs memory.
        if max_new_tokens is None:
            max_new_tokens = int(prompt_length * 2.0)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )

        completion_tokens = outputs[:, prompt_length:]
        decoded = cast(list[str], self._tokenizer.batch_decode(completion_tokens, skip_special_tokens=True))
        return decoded[0]

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        torch.cuda.empty_cache()
