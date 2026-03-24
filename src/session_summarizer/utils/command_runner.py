from __future__ import annotations

import subprocess


def run_command(cmd: list[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    """Run a shell command and raise if it fails."""
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=capture_output,
    )
