from __future__ import annotations

import random
import shutil
import string
from pathlib import Path

from session_summarizer.utils.common_paths import delete_session, ensure_session, session_dir


class TempSession:
    def __init__(self) -> None:
        self.session_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        ensure_session(self.session_id)

    def copy_in(self, filepath: Path, new_filename: str | None = None) -> None:
        if not filepath.exists():
            raise FileNotFoundError(f"Source file does not exist: {filepath}")

        if not filepath.is_file():
            raise IsADirectoryError(f"Source path is not a file: {filepath}")

        destination_filename = new_filename if new_filename is not None else filepath.name
        destination_dir = session_dir(self.session_id)

        destination_path = destination_dir / destination_filename
        shutil.copy2(filepath, destination_path)

    def __enter__(self) -> TempSession:
        return self

    def __exit__(self, *args: object) -> None:
        delete_session(self.session_id)
