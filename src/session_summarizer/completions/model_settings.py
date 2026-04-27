from enum import StrEnum
from typing import Literal, cast

type LiteLLMReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh"]


class ModelString(StrEnum):
    GPT_5_4 = "gpt-5.4"
    GPT_5_5 = "gpt-5.5"
    SONNET_4_6 = "claude-sonnet-4-6"
    OPUS_4_6 = "claude-opus-4-6"
    OPUS_4_7 = "claude-opus-4-7"
    GEMINI_3_1_PRO = "gemini/gemini-3.1-pro-preview"


class ModelEffort(StrEnum):
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"

    def to_litellm_reasoning_effort(self) -> LiteLLMReasoningEffort:
        return cast(LiteLLMReasoningEffort, self.value)  # type:ignore
