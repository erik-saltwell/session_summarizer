from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from litellm import completion
from litellm.types.utils import ModelResponse

from .model_settings import ModelEffort, ModelString


@dataclass
class PromptData:
    system_prompt: str
    user_prompt: str

    def dump(self, system_prompt_path: Path, user_prompt_path: Path) -> None:
        system_prompt_path.write_text(self.system_prompt)
        user_prompt_path.write_text(self.user_prompt)

    def get_completion(self, model: ModelString, effort: ModelEffort) -> str:
        response = cast(
            ModelResponse,
            completion(
                model=model.name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.user_prompt},
                ],
                reasoning_effort=effort.to_litellm_reasoning_effort(),
            ),
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Completion response did not include message content.")

        return content
