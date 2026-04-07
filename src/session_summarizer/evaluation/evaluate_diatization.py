from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

from ..processing_results.speech_clip_set import SpeechClipSet
from .evaluate_word_diarization import compute_matched_words_ratio, compute_wder, compute_wder_old


@dataclass
class DiarizationValidationResult:
    name: str
    DER: float
    JER: float
    wder_match_rate: float
    WDER: float
    old_WDER: float


def evaluate_diarization_result(hypothesis_path: Path, reference_path: Path) -> DiarizationValidationResult:
    FILE_ID = "session"

    hyp = SpeechClipSet.load_from_json(hypothesis_path)
    ref = SpeechClipSet.load_from_json(reference_path)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        hyp_rttm = tmp_path / "hyp.rttm"
        ref_rttm = tmp_path / "ref.rttm"
        hyp_seglst = tmp_path / "hyp.seglst.json"
        ref_seglst = tmp_path / "ref.seglst.json"

        hyp.save_to_rttm(hyp_rttm, file_id=FILE_ID)
        ref.save_to_rttm(ref_rttm, file_id=FILE_ID)
        hyp.save_to_seglst(hyp_seglst, session_id=FILE_ID)
        ref.save_to_seglst(ref_seglst, session_id=FILE_ID)

        # DER and JER via pyannote.metrics
        ref_ann = load_rttm(str(ref_rttm))[FILE_ID]
        hyp_ann = load_rttm(str(hyp_rttm))[FILE_ID]

        der_metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
        jer_metric = JaccardErrorRate(collar=0.0, skip_overlap=False)
        der = float(cast(dict, der_metric(ref_ann, hyp_ann, detailed=True))["diarization error rate"])
        jer = float(cast(dict, jer_metric(ref_ann, hyp_ann, detailed=True))["jaccard error rate"])

        # tcpWER via meeteval (collar=5.0s per research guidance).
        # Pre-identity stages may have >20 diarization clusters; meeteval refuses above that limit.
        # try:
        #     tcp_results = tcpwer(reference=str(ref_seglst), hypothesis=str(hyp_seglst), collar=Decimal("5.0"))
        #     tcp_wer = float(next(iter(tcp_results.values())).error_rate)
        # except RuntimeError:
        #     tcp_wer = math.nan

    wder = compute_wder(hyp, ref)
    old_wder = compute_wder_old(hyp, ref)
    match_rate = compute_matched_words_ratio(hyp)

    return DiarizationValidationResult(
        name=hypothesis_path.stem, DER=der, JER=jer, wder_match_rate=match_rate, WDER=wder, old_WDER=old_wder
    )
