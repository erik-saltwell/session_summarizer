from __future__ import annotations

import logging
import os
import sys
import warnings
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import TextIO


@contextmanager
def silence_python_noise() -> Iterator[TextIO]:
    previous_disable_level = logging.root.manager.disable
    real_stdout = sys.stdout

    with open(os.devnull, "w") as devnull:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logging.disable(logging.CRITICAL)
            with redirect_stdout(devnull), redirect_stderr(devnull):
                try:
                    yield real_stdout
                finally:
                    logging.disable(previous_disable_level)
