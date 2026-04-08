from __future__ import annotations

from pathlib import Path

import torch

from session_summarizer.protocols.logging_protocol import GpuLogger

from ..processing_results.speech_clip_set import SpeechClipSet
from ..protocols import LoggingProtocol, SessionSettings
from ..speaker_embeddings import RegisteredSpeakers
from ..utils import common_paths

_REGISTERED_SPEAKERS_FILE = "registered_speakers.yaml"


def _resolve_speakers_file(session_dir: Path) -> Path:
    session_file = session_dir / _REGISTERED_SPEAKERS_FILE
    if session_file.exists():
        return session_file
    return common_paths.build_speakers_file_path()


def identify_speakers(
    settings: SessionSettings,
    session_dir: Path,
    clips_with_embeddings: SpeechClipSet,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> SpeechClipSet:
    speakers = RegisteredSpeakers.load(_resolve_speakers_file(session_dir))
    attendee_names: set[str] = set(settings.attendees)
    filtered: dict[str, list[float]] = {name: emb for name, emb in speakers.items() if name in attendee_names}

    if not filtered:
        logger.report_message("[yellow]No registered speakers match the attendee list.[/yellow]")
        return clips_with_embeddings

    logger.report_message(f"Matching against {len(filtered)} attendee(s): {sorted(filtered.keys())}")

    # Build a tensor of shape (num_attendees, embedding_dim) for batch cosine similarity
    names: list[str] = list(filtered.keys())
    attendee_tensor = torch.tensor([filtered[n] for n in names], dtype=torch.float32)

    assigned = 0
    skipped = 0
    with logger.progress("Identifying speakers", total=len(clips_with_embeddings)) as progress:
        for clip in clips_with_embeddings:
            if clip.embedding is None:
                skipped += 1
                progress.advance()
                continue

            clip_tensor = torch.tensor(clip.embedding, dtype=torch.float32).unsqueeze(0)
            similarities = torch.nn.functional.cosine_similarity(clip_tensor, attendee_tensor)
            best_idx = int(similarities.argmax())
            clip.identity = names[best_idx]
            assigned += 1
            progress.advance()

    logger.report_message(f"Assigned {assigned} clip(s), skipped {skipped} (no embedding)")
    return clips_with_embeddings
