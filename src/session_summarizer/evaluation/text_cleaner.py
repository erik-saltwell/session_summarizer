from __future__ import annotations

from typing import cast

import jiwer
from mathspell import analyze_text


def clean_text_for_evaluation(text: str) -> str:
    retVal: str = text
    retVal = analyze_text(retVal)

    transform: jiwer.Compose

    transform = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ]
    )
    retVal = cast(str, transform(retVal))
    return retVal
