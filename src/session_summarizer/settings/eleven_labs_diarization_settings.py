from pydantic import BaseModel, field_validator


class ElevenLabsDiarizationSettings(BaseModel, frozen=True):
    """Configuration for the ElevenLabs Scribe v2 diarization path.

    Unlike the DiariZen pipeline (which runs separate ASR, alignment, and
    diarization stages and then stitches the results together), the
    ElevenLabs path obtains word-level transcription and per-word speaker
    assignments from a single API call.  Building a ``SpeechClipSet`` from
    that response is mostly mechanical, with one tunable: how aggressively
    to split a single speaker's contiguous words into separate clips when
    natural pauses occur.

    See helpers/audio_diarizaer_eleven_labs.py for the consumer.
    """

    # Maximum gap (seconds) between two consecutive same-speaker words that
    # will still be kept inside a single ``SpeechClip``.  When the gap
    # between adjacent words exceeds this threshold, a new clip is started
    # even though the ``speaker_id`` has not changed.  Without this split,
    # a single uninterrupted speaker (e.g. a Game Master narrating for
    # several minutes between player questions) would collapse into one
    # enormous clip and frustrate downstream steps that operate on a
    # per-utterance basis (backchannel detection, identity stitching, etc.).
    # A speaker change always splits, regardless of this value.
    # Used by: helpers/audio_diarizaer_eleven_labs.py to decide intra-speaker
    #          clip boundaries when grouping ElevenLabs words into clips.
    clip_gap_seconds: float

    @field_validator("clip_gap_seconds")
    @classmethod
    def _clip_gap_seconds_must_be_non_negative(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"clip_gap_seconds must be >= 0.0, got {v!r}")
        return v
