from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StitchResults:
    pre_stitching_segments: int = 0
    post_stitching_segments: int = 0
