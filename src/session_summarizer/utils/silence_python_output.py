from __future__ import annotations

import logging
import os
import sys
import warnings
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import TextIO


@contextmanager
def silence_os_noise() -> Iterator[TextIO]:
    """Suppress output at both the Python and OS file-descriptor levels.

    Catches C-library writes (e.g. TensorFlow) that bypass Python's sys.stdout/stderr,
    and Python-level writes (e.g. tqdm) that bypass OS-level fd redirection.
    """
    real_stdout = sys.stdout

    # Capture fd numbers before any redirection — sys.stdout.fileno() changes
    # inside redirect_stdout(), so we need the originals for the restore.
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)

    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stdout_fd)
            os.dup2(devnull.fileno(), stderr_fd)
            with redirect_stdout(devnull), redirect_stderr(devnull):
                yield real_stdout
    finally:
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


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
