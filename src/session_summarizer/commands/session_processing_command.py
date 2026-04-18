from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch
import typer

from ..evaluation import clean_text_for_evaluation
from ..processing_results import AlignmentResult, SpeechClipSet
from ..protocols import CommmandProtocol, LoggingProtocol, NullLogger, SessionSettings
from ..utils import Tracer, common_paths, flush_gpu_memory, silence_python_noise


class PlainTextContainer(Protocol):
    def plain_text(self) -> str: ...


@dataclass
class SessionProcessingCommand(ABC, CommmandProtocol):
    session_id: str
    tracer: Tracer
    force: bool = False
    logger: LoggingProtocol = NullLogger()
    gpu_logging_enabled: bool = False
    inputs: list[Path] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    dependencies: list[CommmandProtocol] = field(default_factory=list)
    test_clips: SpeechClipSet | None = None
    detailed_logging: bool = False

    def should_log_gpu_load(self) -> bool:
        return False

    def enable_clip_test(self, clips: SpeechClipSet) -> None:
        self.test_clips = clips

    def set_detailed_logging(self, should_log: bool) -> None:
        self.detailed_logging = should_log

    def initialize_for_processing(self, settings: SessionSettings, session_dir: Path) -> None: ...

    def validate_clips(self) -> None:
        if self.test_clips is None:
            return
        longest_clip_duration = max([clip.duration for clip in self.test_clips])
        if longest_clip_duration > 600:  # 10 minutes
            # raise RuntimeError(f"Longest clip was {longest_clip_duration}, over 600 second limit.")
            pass

    @property
    def should_process(self) -> bool:
        if self.force or len(self.outputs) == 0:
            return True

        for output in self.outputs:
            if not output.exists():
                return True
        newest_input_mtime = max(path.stat().st_mtime for path in self.inputs)
        newest_output_mtime = max(path.stat().st_mtime for path in self.outputs)
        return newest_input_mtime > newest_output_mtime

    @abstractmethod
    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None: ...

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def process_session(self, settings: SessionSettings, session_dir: Path) -> None: ...

    @property
    def safe_name(self) -> str:
        return self.name().replace(" ", "_")

    def report_detailed_message(self, message: str) -> None:
        if self.detailed_logging:
            self.logger.report_message(message)

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
        self.gpu_logging_enabled = self.should_log_gpu_load()

        if not session_dir.exists():
            raise FileNotFoundError(f"Could not find directory: {session_dir}")
        settings: SessionSettings = SessionSettings.load_cascading(self.session_id)
        self.add_dependencies(settings, session_dir)

        for dependency in self.dependencies:
            dependency.execute(logger)

        if not self.should_process:
            return

        with silence_python_noise():
            self.initialize_for_processing(settings, session_dir)

        self.report_gpu_usage(f"Before Processing {self.name()}")

        start = time.perf_counter()
        try:
            with silence_python_noise():
                with logger.status(f"[green]{self.name()}...[/green]", spinner="toggle6", spinner_style="green"):
                    self.process_session(settings, session_dir)
            self.validate_clips()
            end = time.perf_counter()
            logger.report_message(f"[green]{self.name()} completed in {(end - start):.6f} seconds.[/green]")
            self.tracer.add_context("duration", (end - start))
            self.tracer.log(self.safe_name)
        except Exception as exc:
            logger.report_exception(f"Error processing {self.name()}", exc)
            self.tracer.log_exception(exc, self.safe_name)
            raise typer.Exit(code=1) from exc
        finally:
            flush_gpu_memory()
            self.report_gpu_usage(f"After Processing {self.name()}")

    def postpend_text(self, input: Path, tag: str, suffix: str) -> Path:
        return input.with_name(f"{input.stem}{tag}{suffix}")

    def save_cleaned_text(self, text_container: PlainTextContainer, session_dir: Path, json_filename: Path) -> None:
        text: str = text_container.plain_text()
        cleaned_text = clean_text_for_evaluation(text)
        saved_text = cleaned_text.replace(" ", "\n")
        full_text_path = session_dir / Path(json_filename.stem + "_fulltext.txt")
        with open(full_text_path, "w") as f:
            f.write(saved_text)

    def save_speech_clip(self, clips: SpeechClipSet, session_dir: Path, json_filename: Path) -> None:
        self.enable_clip_test(clips)
        json_path = session_dir / json_filename
        human_path = session_dir / Path(json_filename.stem + "_human.txt")
        error_formated_path = session_dir / Path(json_filename.stem + "_formatted.md")

        markdown_path = session_dir / Path(json_filename.stem + "_markdown.md")

        clips.save_to_json(json_path)
        clips.save_to_human_format(human_path)
        clips.save_to_error_formatted_text(error_formated_path)
        clips.save_to_markdown(markdown_path)
        self.save_cleaned_text(clips, session_dir, json_filename)

    def save_alignment_result(self, alignment_result: AlignmentResult, session_dir: Path, json_filename: Path) -> None:
        alignment_result.save_to_json(session_dir / json_filename)
        self.save_cleaned_text(alignment_result, session_dir, json_filename)
