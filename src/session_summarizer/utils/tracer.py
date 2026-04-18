from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import structlog
from structlog.typing import Processor

from session_summarizer.utils import common_paths

log_path: Path = common_paths.generate_logfile_path()
common_paths.ensure_directory(log_path.parent)
log_file = open(common_paths.generate_tracefile_path(), "a", encoding="utf-8")


def initialize_tracing(indent: int | None = None) -> None:
    json_renderer: Processor
    if indent is not None:
        json_renderer = structlog.processors.JSONRenderer(indent=indent, sort_keys=True)
    else:
        json_renderer = structlog.processors.JSONRenderer(sort_keys=True)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.dict_tracebacks,
            json_renderer,
        ],
        logger_factory=structlog.WriteLoggerFactory(file=log_file),
    )


def initialize_request() -> str:
    req_id: str = f"req_{uuid.uuid4().hex}"
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=req_id,  # your custom ID
        start_time=datetime.now(),
    )
    return req_id


class StructLoggerProtocol(structlog.typing.BindableLogger, Protocol):
    def bind(self, **new_values: Any) -> StructLoggerProtocol: ...
    def info(self, event: str | None = None, **kw: Any) -> Any: ...
    def error(self, event: str | None = None, **kw: Any) -> Any: ...


@dataclass
class Tracer:
    logger: StructLoggerProtocol = field(default_factory=structlog.get_logger)

    def add_context(self, name: str, value: Any) -> None:
        self.logger = self.logger.bind(**{name: value})

    def add_multi_context(self, name: str, value: Any) -> None:
        current_context = dict(structlog.get_context(self.logger))
        existing_value = current_context.get(name)

        if existing_value is None:
            combined_value = [value]
        elif isinstance(existing_value, list):
            combined_value = [*existing_value, value]
        else:
            combined_value = [existing_value, value]

        self.logger = self.logger.bind(**{name: combined_value})

    def log(self, event_name: str) -> None:
        self.logger.info(event_name)

    def log_exception(
        self,
        exception: BaseException,
        event_name: str | None = None,
    ) -> None:
        self.logger.error(event_name, exc_info=exception)
