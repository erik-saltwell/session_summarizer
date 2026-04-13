from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..protocols import TextPhraseBuilder, TextPhraseSet
from .alignment_result import WordAlignment
from .process_result_protocol import ProcessResultProtocol
from .speech_clip import SpeechClip, SpeechClipFlags


class SpeechClipSet(list["SpeechClip"], ProcessResultProtocol, TextPhraseSet):
    def name(self) -> str:
        return "SpeechClipSet"

    def plain_text(self) -> str:
        return " ".join(clip.text for clip in self)

    def phrase_builders_in_order(self) -> Iterator[TextPhraseBuilder]:
        return self.all_words_in_order()

    def all_words_in_order(self) -> Iterator[WordAlignment]:
        self.sort_clips()
        for clip in self:
            if clip.words is None:
                continue
            clip.sort_words()
            yield from clip.words

    def phrase_separator_length(self) -> int:
        return 1

    def save_to_human_format(self, path: Path, include_details: bool = False) -> None:
        with path.open("w", encoding="utf-8") as f:
            for clip in self:
                if clip.has_flag(SpeechClipFlags.IS_BACKCHANNEL):
                    continue

                speakers: str
                if clip.identity is None:
                    speakers = ", ".join(sorted(clip.speakers))
                else:
                    speakers = clip.identity

                f.write(f"{speakers}\n")
                if include_details:
                    flags = " ".join(
                        flag.name for flag in SpeechClipFlags if flag and clip.has_flag(flag) and flag.name is not None
                    )
                    flag_str = f"[{flags if flags else 'NO_FLAGS'}]"
                    start_str = f"{clip.start_time: 0.5f}".strip()
                    end_str = f"{clip.end_time: 0.5f}".strip()
                    f.write(f"({start_str},{end_str}): {flag_str}\n")
                f.write(f"{clip.text}\n")
                f.write("\n")

    def save_to_error_formatted_text(self, path: Path, include_details: bool = False) -> None:
        with path.open("w", encoding="utf-8") as f:
            for clip in self:
                speakers: str
                if clip.identity is None:
                    speakers = ", ".join(sorted(clip.speakers))
                else:
                    speakers = clip.identity

                f.write(f"{speakers}\n")
                if include_details:
                    flags = " ".join(
                        flag.name for flag in SpeechClipFlags if flag and clip.has_flag(flag) and flag.name is not None
                    )
                    flag_str = f"[{flags if flags else 'NO_FLAGS'}]"
                    start_str = f"{clip.start_time: 0.5f}".strip()
                    end_str = f"{clip.end_time: 0.5f}".strip()
                    f.write(f"({start_str},{end_str}): {flag_str}\n")
                f.write(f"{clip.error_formatted_text}\n")
                f.write("\n")

    def _speaker_label(self, clip: SpeechClip) -> str:
        if clip.identity:
            return clip.identity
        return "+".join(sorted(clip.speakers))

    def save_to_markdown(self, path: Path, include_timestamps: bool = False) -> None:
        self.sort_clips()
        with path.open("w", encoding="utf-8") as f:
            for clip in self:
                if clip.has_flag(SpeechClipFlags.IS_BACKCHANNEL):
                    continue
                speaker = clip.identity if clip.identity else "+".join(sorted(clip.speakers))

                if include_timestamps:
                    total_seconds = int(clip.start_time)
                    hours = total_seconds // 3600
                    minutes = (total_seconds % 3600) // 60
                    seconds = total_seconds % 60
                    f.write(f"**{speaker}** [{hours:02d}:{minutes:02d}:{seconds:02d}]\n\n")
                else:
                    f.write(f"**{speaker}**\n\n")

                f.write(f"{clip.text}\n\n")

    def save_to_rttm(self, path: Path, file_id: str | None = None) -> None:
        fid = file_id if file_id is not None else path.stem
        with path.open("w", encoding="utf-8") as f:
            for clip in self:
                dur = clip.end_time - clip.start_time
                if dur <= 0:
                    continue
                speaker = self._speaker_label(clip)
                f.write(f"SPEAKER {fid} 1 {clip.start_time:.6f} {dur:.6f} <NA> <NA> {speaker} <NA> <NA>\n")

    def save_to_seglst(self, path: Path, session_id: str | None = None) -> None:
        sid = session_id if session_id is not None else path.stem
        data = [
            {
                "session_id": sid,
                "start_time": clip.start_time,
                "end_time": clip.end_time,
                "speaker": self._speaker_label(clip),
                "words": clip.text,
            }
            for clip in self
        ]
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save_to_json(self, path: Path) -> None:
        data = [
            {
                "start_time": clip.start_time,
                "end_time": clip.end_time,
                "speakers": sorted(clip.speakers),
                "text": clip.text,
                "identity": clip.identity,
                "embedding": clip.embedding,
                "flags": int(clip.flags),
                "cosine_similarity": clip.cosine_similarity,
                "similarity_residual": clip.similarity_residual,
                "end_of_turn_probability": clip.end_of_turn_probability,
                "words": [
                    {
                        "word": w.word,
                        "start_time": w.start_time,
                        "end_time": w.end_time,
                        "confidence": w.confidence,
                        "ground_truth": w.ground_truth,
                    }
                    for w in clip.words
                ]
                if clip.words is not None
                else None,
            }
            for clip in self
        ]
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add_clip(self, clip: SpeechClip) -> None:
        self.append(clip)

    def extend_clips(self, clips: list[SpeechClip]) -> None:
        self.extend(clips)

    def sort_clips(self) -> None:
        self.sort(key=lambda c: (c.start_time, c.end_time))

    @classmethod
    def load_from_test_meeting(cls) -> SpeechClipSet:
        import session_summarizer.utils.common_paths as common_paths

        data = json.loads(common_paths.test_transcript_path().read_text(encoding="utf-8"))
        instance = cls()
        for phrase in data["phrases"]:
            speaker: str = phrase["speaker"]
            instance.append(
                SpeechClip(
                    start_time=float(phrase["start"]),
                    end_time=float(phrase["end"]),
                    speakers={speaker},
                    text=phrase["text"],
                    identity=speaker,
                )
            )
        return instance

    @classmethod
    def load_from_json(cls, path: Path) -> SpeechClipSet:
        with path.open("r", encoding="utf-8") as f:
            data: list[dict] = json.load(f)
        instance = cls()
        for item in data:
            raw_words = item.get("words")
            words = (
                [
                    WordAlignment(
                        word=w["word"],
                        start_time=w["start_time"],
                        end_time=w["end_time"],
                        confidence=w.get("confidence", 0.0),
                        ground_truth=w.get("ground_truth", None),
                    )
                    for w in raw_words
                ]
                if raw_words is not None
                else None
            )
            clip = SpeechClip(
                start_time=item["start_time"],
                end_time=item["end_time"],
                speakers=set(item["speakers"]),
                text=item["text"],
                identity=item.get("identity"),
                embedding=item.get("embedding"),
                cosine_similarity=item.get("cosine_similarity"),
                similarity_residual=item.get("similarity_residual"),
                flags=SpeechClipFlags(item.get("flags", 0)),
                end_of_turn_probability=item.get("end_of_turn_probability"),
                words=words,
            )
            instance.append(clip)
        return instance
