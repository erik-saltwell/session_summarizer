from __future__ import annotations

from pathlib import Path

import pytest

from session_summarizer.commands.create_speaker_clips import (
    _inferred_export_speaker_name,
    should_export_known_speaker_clip,
)
from session_summarizer.helpers.infer_speakers import UNKNOWN_SPEAKER_IDENTITY
from session_summarizer.helpers.speaker_clip_exporter import (
    SpeakerClipExport,
    export_speaker_audio_clips,
    validate_speaker_sample_name,
)
from session_summarizer.processing_results import SpeechClip, SpeechClipFlags
from session_summarizer.protocols import NullLogger


def _clip(
    speakers: set[str],
    identity: str | None = "Morgan",
    residual: float | None = 0.8,
    flags: SpeechClipFlags = SpeechClipFlags.NONE,
) -> SpeechClip:
    return SpeechClip(
        start_time=1.0,
        end_time=4.0,
        speakers=speakers,
        text="hello",
        identity=identity,
        similarity_residual=residual,
        flags=flags,
    )


def test_known_speaker_filter_keeps_residual_gate() -> None:
    assert should_export_known_speaker_clip(_clip({"1"}, residual=0.8), 0.2, False)
    assert not should_export_known_speaker_clip(_clip({"1"}, residual=0.1), 0.2, False)


def test_known_speaker_filter_skips_multispeaker_unless_enabled() -> None:
    clip = _clip({"1", "2"}, residual=0.8)

    assert not should_export_known_speaker_clip(clip, 0.2, False)
    assert should_export_known_speaker_clip(clip, 0.2, True)


def test_inferred_export_uses_real_participant_name_for_single_label() -> None:
    assert _inferred_export_speaker_name(_clip({"1"}, identity="Game Master"), {"1": "Morgan"}) == "Morgan"


def test_inferred_export_allows_multilabel_clip_for_same_participant() -> None:
    assert (
        _inferred_export_speaker_name(_clip({"1", "3"}, identity="Game Master"), {"1": "Morgan", "3": "Morgan"})
        == "Morgan"
    )


def test_inferred_export_skips_multilabel_clip_for_different_participants() -> None:
    assert (
        _inferred_export_speaker_name(_clip({"1", "2"}, identity="Game Master"), {"1": "Morgan", "2": "Avery"}) is None
    )


def test_inferred_export_skips_unknown_and_backchannels() -> None:
    assert _inferred_export_speaker_name(_clip({"1"}, identity=UNKNOWN_SPEAKER_IDENTITY), {"1": "Morgan"}) is None
    assert (
        _inferred_export_speaker_name(
            _clip({"1"}, identity="Game Master", flags=SpeechClipFlags.IS_BACKCHANNEL), {"1": "Morgan"}
        )
        is None
    )


def test_export_speaker_audio_clips_appends_without_clearing() -> None:
    calls: list[tuple[Path, str, Path | None]] = []

    def fake_save(
        cleaned_audio_path: Path,
        clip: SpeechClip,
        speaker_name: str,
        lead_in_seconds: float,
        lead_out_seconds: float,
        temp_folder: Path | None,
    ) -> None:
        calls.append((cleaned_audio_path, speaker_name, temp_folder))

    summary = export_speaker_audio_clips(
        Path("clean.wav"),
        [SpeakerClipExport(clip=_clip({"1"}), speaker_name="Morgan")],
        skipped_count=2,
        lead_in_seconds=0.25,
        lead_out_seconds=0.5,
        temp_folder=None,
        logger=NullLogger(),
        save_clip_fn=fake_save,
    )

    assert calls == [(Path("clean.wav"), "Morgan", None)]
    assert summary.saved_count == 1
    assert summary.skipped_count == 2
    assert summary.speaker_clip_counts == {"Morgan": 1}


def test_validate_speaker_sample_name_rejects_path_components() -> None:
    assert validate_speaker_sample_name("Morgan") == "Morgan"
    with pytest.raises(ValueError, match="path separators"):
        validate_speaker_sample_name("../Morgan")
