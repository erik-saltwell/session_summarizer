from __future__ import annotations

from typing import cast

import jiwer
from mathspell import analyze_text


def clean_text(text: str, reduce_for_wer: bool = False) -> str:
    retVal: str = text
    retVal = analyze_text(retVal)

    transform: jiwer.Compose

    if reduce_for_wer:
        transform = jiwer.Compose(
            [
                jiwer.ToLowerCase(),
                jiwer.RemovePunctuation(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )
    else:
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
