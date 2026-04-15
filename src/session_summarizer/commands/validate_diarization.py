from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..evaluation.evaluate_diatization import DiarizationValidationResult, evaluate_diarization_result
from ..processing_results.speech_clip_set import SpeechClipSet
from ..settings.session_settings import SessionSettings
from .add_embeddings import AddEmbeddingsCommand
from .diarizationlm_command import DiarizationLMCommand
from .diarize_audio import DiarizeAudioCommand
from .identify_speakers import IdentifySpeakersCommand
from .indeterminate_speaker_assignment import IndeterminantSpeakerAssignmentCommand
from .mark_backchannels import MarkBackchannelsCommand
from .session_processing_command import SessionProcessingCommand
from .stitch_identities import StitichIdentitiesCommand

# ---------------------------------------------------------------------------
# Hypothesis registry
# ---------------------------------------------------------------------------
# Each entry is (display_name, path_to_SpeechClipSet_json).
# To add a new hypothesis file, append one tuple here — nothing else to change.


def get_diarization_registry(settings: SessionSettings, session_dir: Path) -> list[tuple[str, Path]]:
    results: list[tuple[str, Path]] = []
    results.append(("Speaker Identified", session_dir / settings.paths.base_diarization))
    results.append(("BaseDiarization", session_dir / settings.paths.diarizationlm_processed))
    results.append(("Embedded", session_dir / settings.paths.clips_with_embeddings))
    results.append(("Identified Speakers", session_dir / settings.paths.identified_speakers))
    results.append(("Identity Stitched", settings.paths.identity_stitched))
    results.append(("Unassign Speakers", settings.paths.indeterminate_speakers))
    results.append(("Backchannel Marked", settings.paths.backchannel_marked))

    return results


_METRIC_LABELS: list[str] = [
    "DER",
    "JER",
    "WDER Match Rate",
    "WDER",
    "old_WDER",
]


def _format_metric(value: float) -> str:
    return "N/A" if math.isnan(value) else f"{value:.4f}"


def _result_column(result: DiarizationValidationResult) -> list[str]:
    return [
        _format_metric(result.DER),
        _format_metric(result.JER),
        _format_metric(result.wder_match_rate),
        _format_metric(result.WDER),
        _format_metric(result.old_WDER),
    ]


@dataclass
class ValidateDiarizationCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Validate Diarization"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.extend(item[1] for item in get_diarization_registry(settings, session_dir))
        self.dependencies.extend(
            [
                DiarizeAudioCommand(self.session_id, self.tracer),
                DiarizationLMCommand(self.session_id, self.tracer),
                AddEmbeddingsCommand(self.session_id, self.tracer),
                IdentifySpeakersCommand(self.session_id, self.tracer),
                StitichIdentitiesCommand(self.session_id, self.tracer),
                IndeterminantSpeakerAssignmentCommand(self.session_id, self.tracer),
                MarkBackchannelsCommand(self.session_id, self.tracer),
            ]
        )

    def process_session(self, settings: SessionSettings, session_dir: Path) -> None:
        # DiarizationLMCommand(self.session_id).execute(self.logger)

        results: dict[str, DiarizationValidationResult] = {}
        failed: list[str] = []

        with tempfile.TemporaryDirectory() as tmp:
            ref_path = Path(tmp) / "reference.json"
            reference_clips: SpeechClipSet = SpeechClipSet.load_from_test_meeting()
            reference_clips.sort_clips()
            reference_clips.save_to_json(ref_path)

            for name, hyp_path in get_diarization_registry(settings, session_dir):
                self.logger.report_message(f"[blue]Evaluating {name}...[/blue]")
                try:
                    result = evaluate_diarization_result(hyp_path, ref_path, settings.epsilon)
                    results[name] = result
                except Exception as exc:
                    self.logger.report_error(f"[red]{name} failed: {exc}[/red]")
                    failed.append(name)

        # Build table: rows = metrics, columns = hypothesis names
        names = list(results.keys())
        headers: list[str] = ["Metric"] + names
        columns = [_result_column(results[name]) for name in names]
        rows: list[list[str]] = []
        for i, label in enumerate(_METRIC_LABELS):
            rows.append([label] + [col[i] for col in columns])

        self.logger.add_break()
        if results:
            self.logger.report_message("[bold]Diarization Validation Results[/bold]")
            self.logger.report_multicolumn_table(headers, rows)
        if failed:
            self.logger.report_message(f"[red bold]Failed evaluations: {', '.join(failed)}[/red bold]")
