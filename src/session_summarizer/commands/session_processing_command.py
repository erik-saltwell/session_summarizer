from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch
import typer

from ..protocols import CommmandProtocol, LoggingProtocol, NullLogger, SessionSettings
from ..utils import common_paths


@dataclass
class SessionProcessingCommand(ABC, CommmandProtocol):
    session_id: str
    logger: LoggingProtocol = NullLogger()
    gpu_logging_enabled: bool = False

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def process_session(self, settings: SessionSettings, session_dir: Path) -> None: ...

    def report_message(self, message: str) -> None:
        self.logger.report_message(f"[blue]{message}[/blue]")

    def report_gpu_usage(self, label: str) -> None:
        if not self.gpu_logging_enabled:
            return
        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        self.logger.report_message(
            f"[dim]vRAM ({label}): {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total[/dim]"
        )

    def execute(self, logger: LoggingProtocol) -> None:
        self.logger = logger
        session_dir: Path = common_paths.session_dir(self.session_id)
        if not session_dir.exists():
            raise FileNotFoundError(f"Could not find directory: {session_dir}")
        settings: SessionSettings = SessionSettings.load_cascading(self.session_id)
        self.report_message(f"Processing commang: {self.name()}")
        start = time.perf_counter()
        try:
            self.process_session(settings, session_dir)
            end = time.perf_counter()
            logger.report_message(f"[green]Command completed in {(end - start):.6f} seconds.[/green]")
        except Exception as exc:
            logger.report_exception(f"Error processing {self.name()}", exc)
            raise typer.Exit(code=1) from exc
