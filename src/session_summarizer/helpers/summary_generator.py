from __future__ import annotations

from pathlib import Path

import anthropic

from ..processing_results import SpeechClipFlags, SpeechClipSet
from ..protocols import LoggingProtocol
from ..settings.session_settings import CampaignInfo, SessionInfo, SessionSettings
from ..utils import FragmentID, get_fragment


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


def _get_output(
    settings: SessionSettings, clips: SpeechClipSet, system_prompt: str, input: str, logger: LoggingProtocol
) -> str:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=system_prompt,
        messages=[{"role": "user", "content": input}],
    ) as stream:
        final = stream.get_final_message()

    return next((block.text for block in final.content if block.type == "text"), "")


def _log(input: str, title: str, logger: LoggingProtocol) -> None:
    logger.report_message(f"{title.upper()}: \n")
    logger.report_message(input)
    logger.add_break()
    return


def generate_summary(
    settings: SessionSettings, session_dir: Path, clips: SpeechClipSet, logger: LoggingProtocol
) -> str:
    system_prompt: str = get_fragment(FragmentID.SUMMARIZE_SESSION_SYSTEM_PROMPT)
    input: str = _construct_input(settings, clips, logger)
    _log(input, "Session Input", logger)
    output: str = _get_output(settings, clips, system_prompt, input, logger)
    _log(output, "Response Summary", logger)

    return output
