from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PromptData:
    system_prompt: str
    user_prompt: str

    def dump(self, system_prompt_path: Path, user_prompt_path: Path) -> None:
        system_prompt_path.write_text(self.system_prompt)
        user_prompt_path.write_text(self.user_prompt)
