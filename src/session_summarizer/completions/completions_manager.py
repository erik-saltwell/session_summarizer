from __future__ import annotations

from time import perf_counter
from typing import Any, cast

from litellm import completion
from litellm.types.utils import ModelResponse

from ..utils import Tracer
from .model_settings import ModelEffort, ModelString
from .prompt_data import PromptData


def trace(name: str, value: Any, tracer: Tracer | None) -> None:
    if tracer is None:
        return
    tracer.add_context(name, value)


def get_completion(prompt_data: PromptData, model: ModelString, effort: ModelEffort, tracer: Tracer | None) -> str:
    trace("sys_prompt_len", len(prompt_data.system_prompt), tracer)
    trace("user_prompt_len", len(prompt_data.user_prompt), tracer)
    start = perf_counter()
    response: ModelResponse = cast(
        ModelResponse,
        completion(
            model=model.value,
            messages=[
                {"role": "system", "content": prompt_data.system_prompt},
                {"role": "user", "content": prompt_data.user_prompt},
            ],
            reasoning_effort=effort.to_litellm_reasoning_effort(),
        ),
    )
    end = perf_counter()
    trace("llm_call_duration", f"{(end - start):.4f}", tracer)

    content: str | None = response.choices[0].message.content
    if content is None:
        raise ValueError("Completion response did not include message content.")
    trace("completion_len", len(content), tracer)
    return content
