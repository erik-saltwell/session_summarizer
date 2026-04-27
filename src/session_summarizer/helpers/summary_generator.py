from __future__ import annotations

from pathlib import Path

from ..completions import PromptData, get_completion
from ..processing_results import SpeechClipFlags, SpeechClipSet
from ..protocols import LoggingProtocol
from ..settings.session_settings import CampaignInfo, SessionInfo, SessionSettings
from ..utils import FragmentID, Tracer, get_fragment


def _construct_input(settings: SessionSettings, clips: SpeechClipSet, logger: LoggingProtocol) -> str:
    session_info: SessionInfo = settings.session_info
    campaign_context: CampaignInfo = settings.campaign_info

    result: str = ""
    result += "<session>\n"
    result += f"session date:{session_info.session_date}\n"
    result += f"adventure name:{session_info.adventure_name}\n"
    result += f"campaign name:{session_info.campaign_name}\n"
    result += "</session>\n\n"

    result += "<campaign_context>\n"
    result += "  <players>\n"
    for player in campaign_context.players.keys():
        result += f"    - {player}:{campaign_context.players[player]}\n"
    result += "  </players>\n"

    result += "  <glossary>\n"
    for entry in campaign_context.glossary:
        result += f"    - {entry.term}"
        if entry.description is not None:
            result += f": {entry.description}"
        result += "\n"
    result += "  </glossary>\n"
    result += "</campaign_context>\n\n"

    result += "<transcript>\n"
    for clip in clips:
        if clip.identity is not None and (not clip.has_flag(SpeechClipFlags.IS_BACKCHANNEL)):
            result += f"**{clip.identity}**:\n"
            result += f"{clip.text}\n"
            result += "\n"
    result += "</transcript>\n"
    return result


def generate_summary(
    settings: SessionSettings, session_dir: Path, clips: SpeechClipSet, logger: LoggingProtocol, tracer: Tracer
) -> str:
    system_prompt: str = get_fragment(FragmentID.SUMMARIZE_SESSION_SYSTEM_PROMPT)
    input: str = _construct_input(settings, clips, logger)
    prompt = PromptData(system_prompt, input)
    llm_settings = settings.llm.session_summary

    output: str = get_completion(
        prompt,
        llm_settings.model,
        llm_settings.effort,
        tracer,
    )

    return output
