from __future__ import annotations

import shutil
from pathlib import Path

import torch

from ..protocols import LoggingProtocol
from ..settings.session_settings import SessionSettings
from ..speaker_embeddings import get_embeddings_factory


def create_clips_without_outliers(
    settings: SessionSettings, input_dir: Path, output_dir: Path, logger: LoggingProtocol
) -> list[float] | None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(input_dir, output_dir)

    factory = get_embeddings_factory(settings.device)
    wav_files = sorted(output_dir.glob("*.wav"))
    if not wav_files:
        logger.report_message("[yellow]No .wav files found — nothing to do.[/yellow]")
        return None

    logger.report_message(f"[blue]Removing clips with similarity < {settings.speaker_clips.min_centroid_similarity}.[/blue]")

    embeddings: dict[Path, torch.Tensor] = {}
    for wav in wav_files:
        emb = factory.extract(wav, logger)
        embeddings[wav] = torch.tensor(emb, dtype=torch.float32)

    total = len(embeddings)
    deleted = 0

    while len(embeddings) > 1:
        paths = list(embeddings.keys())
        stack = torch.stack([embeddings[p] for p in paths])
        centroid = stack.mean(dim=0)

        similarities = torch.nn.functional.cosine_similarity(stack, centroid.unsqueeze(0))
        worst_idx = int(similarities.argmin())
        worst_sim = float(similarities[worst_idx])
        worst_path = paths[worst_idx]

        logger.report_message(
            f"[dim]n={len(embeddings)} | min similarity: {worst_sim:.6f} |"
            f"threshold: {settings.speaker_clips.min_centroid_similarity}[/dim]"
        )
        if worst_sim >= settings.speaker_clips.min_centroid_similarity:
            break

        worst_path.unlink()
        del embeddings[worst_path]
        deleted += 1

    kept = len(embeddings)
    logger.report_message(f"[green]Done — kept {kept} clip(s), deleted {deleted} of {total}.[/green]")

    paths = list(embeddings.keys())
    stack = torch.stack([embeddings[p] for p in paths])
    centroid = stack.mean(dim=0)
    return [float(x) for x in centroid]
