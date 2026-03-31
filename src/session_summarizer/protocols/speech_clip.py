from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class SpeechClipSet(list["SpeechClip"]):
    def save(self, path: Path) -> None:
        data = [
            {
                "clip_id": clip.clip_id,
                "start_time": clip.start_time,
                "end_time": clip.end_time,
                "speakers": clip.speakers,
                "confidence_avg": clip.confidence_avg,
                "text": clip.text,
                "identity": clip.identity,
                "embedding": clip.embedding,
            }
            for clip in self
        ]
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> SpeechClipSet:
        with path.open("r", encoding="utf-8") as f:
            data: list[dict] = json.load(f)
        instance = cls()
        for item in data:
            clip = SpeechClip(
                clip_id=item["clip_id"],
                parent=instance,
                start_time=item["start_time"],
                end_time=item["end_time"],
                speakers=item["speakers"],
                confidence_avg=item["confidence_avg"],
                text=item["text"],
                identity=item.get("identity"),
                embedding=item.get("embedding"),
            )
            instance.append(clip)
        return instance


@dataclass
class SpeechClip:
    clip_id: int
    parent: SpeechClipSet
    start_time: float
    end_time: float
    speakers: list[str]
    confidence_avg: float
    text: str
    identity: str | None = None
    embedding: list[float] | None = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def previous_clip(self) -> SpeechClip | None:
        if self.clip_id == 0:
            raise ValueError("Clip ID is 0, cannot determine previous clip")
        return self.parent[self.clip_id - 1]

    @property
    def next_clip(self) -> SpeechClip | None:
        if self.clip_id >= len(self.parent) - 1:
            raise ValueError("Clip ID is at the end of the list, cannot determine next clip")
        return self.parent[self.clip_id + 1]

    @property
    def is_multispeaker(self) -> bool:
        return len(self.speakers) > 1
