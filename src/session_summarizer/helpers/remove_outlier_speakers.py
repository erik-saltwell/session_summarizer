from __future__ import annotations

import shutil
from pathlib import Path

import torch

from ..protocols import LoggingProtocol
from ..settings.session_settings import SessionSettings
from ..speaker_embeddings import get_embeddings_factory


def create_clips_without_outliers(
    settings: SessionSettings, input_dir: Path, output_dir: Path, logger: LoggingProtocol
) -> None:
    shutil.copytree(input_dir, output_dir)

    factory = get_embeddings_factory(settings.device)
    wav_files = sorted(output_dir.glob("*.wav"))
    if not wav_files:
        logger.report_message("[yellow]No .wav files found — nothing to do.[/yellow]")
        return

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
        worst_path = paths[worst_idx]

        # Recompute centroid without the least-similar clip
        remaining = torch.stack([embeddings[p] for p in paths if p != worst_path])
        new_centroid = remaining.mean(dim=0)

        delta = 1.0 - float(torch.nn.functional.cosine_similarity(centroid.unsqueeze(0), new_centroid.unsqueeze(0)))
        if delta <= settings.stable_centroid_epsilon:
            break

        worst_path.unlink()
        del embeddings[worst_path]
        deleted += 1

    kept = len(embeddings)
    logger.report_message(f"[green]Done — kept {kept} clip(s), deleted {deleted} of {total}.[/green]")
