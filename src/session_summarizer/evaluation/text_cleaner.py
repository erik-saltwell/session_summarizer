from __future__ import annotations

from typing import cast

import jiwer
from mathspell import analyze_text

_AT_TOKEN = "{[at]}"


def wrap_for_mathspell_issues(text: str) -> str:
    return text.replace("@", _AT_TOKEN)


def unwrap_for_mathspell_issues(text: str) -> str:
    return text.replace(_AT_TOKEN, "@")


def clean_text_for_evaluation(text: str, do_mathspell: bool) -> str:
    retVal: str = text
    if do_mathspell:
        retVal = wrap_for_mathspell_issues(retVal)
        retVal = analyze_text(retVal)
        retVal = unwrap_for_mathspell_issues(retVal)

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
