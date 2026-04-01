from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ProcessResultProtocol(Protocol):
    """Structural protocol for all pipeline processing results.

    Every stage of the processing pipeline (transcription, alignment, diarization, etc.)
    produces a result object that must satisfy this protocol. This enables pipeline
    orchestration code to handle results generically — logging them, persisting them,
    and passing them to the next stage — without knowing their concrete types.

    Implementers must also provide a classmethod with the following signature:

        @classmethod
        def load_from_json(cls, path: Path) -> Self: ...

    This method is the counterpart to ``save_to_json`` and must round-trip correctly:
    loading a previously saved result should reproduce an equivalent object.
    It cannot be expressed in a structural Protocol because Protocol does not support
    classmethods that return ``Self``, so it is enforced by convention.

    See ``TranscriptionResult``, ``AlignmentResult``, and ``SpeechClipSet`` for
    reference implementations.
    """

    def name(self) -> str:
        """Human-readable name identifying the result type (e.g. "TranscriptionResult").

        Used in log messages and progress reporting so operators can tell which
        stage produced the result without inspecting its contents.
        """
        ...

    def plain_text(self) -> str:
        """Return the full textual content of the result as a single string.

        Stitches together all text contained in the result (words, segments, clips, etc.)
        joined by spaces. Intended for display, diffing, and ground-truth comparison —
        not for downstream structured processing.
        """
        ...

    def save_to_json(self, path: Path) -> None:
        """Serialize the result to a JSON file at ``path``.

        The file must be loadable via the corresponding ``load_from_json`` classmethod.
        Overwrites any existing file at that path.
        """
        ...
