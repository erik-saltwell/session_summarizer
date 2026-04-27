from __future__ import annotations

from pathlib import Path

from ..completions import PromptData, get_completion
from ..processing_results import SpeechClipFlags, SpeechClipSet
from ..protocols import LoggingProtocol
from ..settings.session_settings import SessionSettings
from ..utils import FragmentID, Tracer, get_fragment


def _construct_input(settings: SessionSettings, clips: SpeechClipSet) -> str:
    result: str = ""
    for clip in clips:
        if clip.identity is not None and not clip.has_flag(SpeechClipFlags.IS_BACKCHANNEL):
            result += f"**{clip.identity}**:\n"
            result += f"{clip.text}\n\n"

    return result


def simplify_transcript(
    settings: SessionSettings,
    session_dir: Path,
    clips: SpeechClipSet,
    logger: LoggingProtocol,
    tracer: Tracer,
) -> str:
    system_prompt: str = get_fragment(FragmentID.TRANSCRIPT_CLEANER_PROMPT)
    input: str = _construct_input(settings, clips)
    prompt = PromptData(system_prompt, input)
    llm_settings = settings.llm.session_logs

    output: str = get_completion(prompt, llm_settings.model, llm_settings.effort, tracer)

    return output
