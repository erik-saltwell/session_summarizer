from __future__ import annotations

from session_summarizer.diarizationlm.diarizationlm_processor import DiarizationLMProcessor
from session_summarizer.processing_results import SpeechClip, SpeechClipSet, WordAlignment
from session_summarizer.utils.tracer import Tracer


class FakeDiarizationLMModel:
    def __init__(self, context_window: int):
        self.context_window = context_window
        self.prompts: list[str] = []

    @property
    def is_loaded(self) -> bool:
        return True

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def infer(self, prompt: str, max_new_tokens: int | None = None) -> str:
        self.prompts.append(prompt)
        del max_new_tokens
        words = [token for token in prompt.removesuffix(" --> ").split() if not token.startswith("<speaker:")]
        return "<speaker:2> " + " ".join(words)


class LocallyFailingFakeDiarizationLMModel(FakeDiarizationLMModel):
    def infer(self, prompt: str, max_new_tokens: int | None = None) -> str:
        self.prompts.append(prompt)
        del max_new_tokens
        words = [token for token in prompt.removesuffix(" --> ").split() if not token.startswith("<speaker:")]
        if "w0" in words:
            return ""
        return "<speaker:1> " + " ".join(words)


def _make_clip(speaker: str, words: list[str], start_index: int) -> SpeechClip:
    alignments = [
        WordAlignment(
            word=word,
            start_time=float(start_index + idx),
            end_time=float(start_index + idx) + 0.5,
        )
        for idx, word in enumerate(words)
    ]
    return SpeechClip(
        start_time=alignments[0].start_time,
        end_time=alignments[-1].end_time,
        speakers={speaker},
        text=" ".join(words),
        identity=speaker,
        words=alignments,
    )


def test_processor_uses_single_prompt_when_it_fits_context() -> None:
    clips = SpeechClipSet()
    clips.extend_clips(
        [
            _make_clip("A", ["hello", "there"], 0),
            _make_clip("B", ["general", "kenobi"], 2),
        ]
    )

    model = FakeDiarizationLMModel(context_window=1000)
    result = DiarizationLMProcessor(model).process(clips, epsilon=0.000001, tracer=Tracer())

    assert len(model.prompts) == 1
    assert len(result) == 1
    assert result[0].speakers == {"B"}
    assert result[0].text == "hello there general kenobi"


def test_processor_falls_back_to_original_when_inference_fails() -> None:
    clips = SpeechClipSet()
    clips.extend_clips(
        [
            _make_clip("A", ["w0", "w1", "w2", "w3"], 0),
            _make_clip("B", ["w4", "w5", "w6", "w7"], 4),
        ]
    )

    model = LocallyFailingFakeDiarizationLMModel(context_window=264)
    result = DiarizationLMProcessor(model).process(clips, epsilon=0.000001, tracer=Tracer())

    assert len(model.prompts) == 1
    assert len(result) == 2
    assert result[0].speakers == {"A"}
    assert result[1].speakers == {"B"}
