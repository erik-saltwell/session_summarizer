from __future__ import annotations


def test_clean_audio_eleven_labs_command_imports() -> None:
    from session_summarizer.commands.clean_audio_eleven_labs import CleanAudioElevenLabsCommand

    assert CleanAudioElevenLabsCommand.__name__ == "CleanAudioElevenLabsCommand"
