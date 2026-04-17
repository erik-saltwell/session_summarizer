from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths
from session_summarizer.processing_results.speech_clip_set import SpeechClipSet

from ..settings.session_settings import SessionSettings
from .session_processing_command import SessionProcessingCommand


@dataclass
class TestCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Test"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        return

    def output_confidence_success_map(self, settings: SessionSettings, session_dir: Path) -> None:
        csv_path = session_dir / "conf_wder.csv"

        with csv_path.open("w") as writer:
            writer.write("confidence,success_rate\n")
            clips: SpeechClipSet = SpeechClipSet.load_from_json(session_dir / settings.identity_stitched_path)
            for clip in clips:
                assert clip.identity is not None
                identity: str = clip.identity.lower()
                if clip.words is None:
                    continue
                for word in clip.words:
                    if word.ground_truth is not None:
                        writer.write(f"{word.confidence},{1.0 if word.ground_truth.lower() == identity else 0.0}\n")

    def output_success_data(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        csv_path = session_dir / "similarity_succes.csv"

        with csv_path.open("w") as writer:
            writer.write(
                "speaker_count,word_count,duration,similarity,similarity_residual,avg_confidence,success_rate,text\n"
            )
            clips: SpeechClipSet = SpeechClipSet.load_from_json(session_dir / settings.identity_stitched_path)
            for clip in clips:
                if clip.cosine_similarity and clip.words:
                    writer.write(
                        f"{len(clip.speakers)},"
                        f"{clip.word_count},"
                        f"{clip.duration},"
                        f"{clip.cosine_similarity},"
                        f"{clip.similarity_residual},"
                        f"{clip.confidence_avg},"
                        f"{1.0 - clip.wder},"
                        f"{clip.text}"
                        "\n"
                    )

    def output_words_before_period_distribution(self, settings: SessionSettings, session_dir: Path) -> None:
        clips: SpeechClipSet = SpeechClipSet.load_from_json(session_dir / settings.identity_stitched_path)
        counts: Counter[tuple[int, int]] = Counter()
        single_word_counts: Counter[str] = Counter()
        two_word_counts: Counter[str] = Counter()

        for clip in clips:
            period_idx = clip.text.find(".")
            if period_idx == -1:
                continue
            before_text = clip.text[:period_idx]
            before_words = before_text.split()
            before = len(before_words)
            after = len(clip.text[period_idx + 1 :].split())
            counts[(before, after)] += 1
            if before == 1:
                single_word_counts[before_words[0]] += 1
            elif before == 2:
                two_word_counts[" ".join(before_words)] += 1

        if not counts:
            self.logger.report_message("No clips with periods found.")
            return

        self.logger.report_message("Words before/after first period — clip distribution")
        rows = [[str(b), str(a), str(n)] for (b, a), n in sorted(counts.items())]
        self.logger.report_multicolumn_table(["Words before period", "Words after period", "Clip count"], rows)

        if single_word_counts:
            self.logger.report_message("Single-word-before-period clips — word counts")
            single_rows = [[word, str(n)] for word, n in sorted(single_word_counts.items(), key=lambda x: -x[1])]
            self.logger.report_multicolumn_table(["Word", "Count"], single_rows)

        if two_word_counts:
            self.logger.report_message("Two-words-before-period clips — phrase counts")
            two_rows = [[phrase, str(n)] for phrase, n in sorted(two_word_counts.items(), key=lambda x: -x[1])]
            self.logger.report_multicolumn_table(["Phrase", "Count"], two_rows)

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        # self.output_success_data(settings, session_dir)
        self.output_words_before_period_distribution(settings, session_dir)
