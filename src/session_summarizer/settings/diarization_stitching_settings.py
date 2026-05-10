from pydantic import BaseModel, field_validator


class DiarizationStitchingSettings(BaseModel, frozen=True):
    # Maximum gap (seconds) between two clips with the same identified
    # speaker that can still be merged into a single clip during identity
    # stitching.  Larger values allow merging across longer pauses; smaller
    # values keep clips separate when the same speaker resumes after silence.
    # Used by: identity_stitch.py when merging same-speaker clips.
    identity_merge_max_gap_seconds: float

    @field_validator("identity_merge_max_gap_seconds")
    @classmethod
    def non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("must be non-negative")
        return v
