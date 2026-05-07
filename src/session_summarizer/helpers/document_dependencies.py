from __future__ import annotations

import dataclasses
import importlib
import pkgutil
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import session_summarizer.utils.common_paths as common_paths

from ..commands.session_processing_command import SessionProcessingCommand
from ..protocols import LoggingProtocol, SessionSettings
from ..utils import Tracer

_TEST_SESSION = "test"


@dataclass
class _CommandEdges:
    command_name: str
    input_filenames: list[str]
    output_filenames: list[str]


def _import_all_command_modules() -> None:
    commands_package = importlib.import_module("session_summarizer.commands")
    for module_info in pkgutil.iter_modules(commands_package.__path__):
        if module_info.ispkg:
            continue
        importlib.import_module(f"session_summarizer.commands.{module_info.name}")


def _all_subclasses(cls: type) -> set[type]:
    subclasses: set[type] = set()
    for sub in cls.__subclasses__():
        subclasses.add(sub)
        subclasses.update(_all_subclasses(sub))
    return subclasses


def _dummy_value_for(field_type: Any) -> Any:
    type_str = str(field_type)
    if field_type is bool or "bool" in type_str:
        return False
    if field_type is int or "int" in type_str:
        return 0
    if field_type is float or "float" in type_str:
        return 0.0
    if field_type is str or "str" in type_str:
        return ""
    if field_type is Path or "Path" in type_str:
        return Path("/tmp")
    if "list" in type_str.lower():
        return []
    if "dict" in type_str.lower():
        return {}
    return None


def _try_instantiate(cls: type, session_id: str, tracer: Tracer) -> SessionProcessingCommand | None:
    kwargs: dict[str, Any] = {"session_id": session_id, "tracer": tracer}
    for field in dataclasses.fields(cls):
        if field.name in kwargs:
            continue
        has_default = field.default is not dataclasses.MISSING or field.default_factory is not dataclasses.MISSING
        if has_default:
            continue
        kwargs[field.name] = _dummy_value_for(field.type)
    try:
        return cls(**kwargs)
    except Exception:
        return None


def _collect_command_edges(
    settings: SessionSettings, session_dir: Path, logger: LoggingProtocol
) -> list[_CommandEdges]:
    _import_all_command_modules()

    results: list[_CommandEdges] = []
    tracer = Tracer()

    for cls in sorted(_all_subclasses(SessionProcessingCommand), key=lambda c: c.__name__):
        instance = _try_instantiate(cls, _TEST_SESSION, tracer)
        if instance is None:
            logger.report_message(f"[yellow]Skipping {cls.__name__}: could not instantiate.[/yellow]")
            continue

        try:
            instance.add_dependencies(settings, session_dir)
        except Exception as exc:
            logger.report_message(f"[yellow]Skipping {cls.__name__}: add_dependencies failed ({exc}).[/yellow]")
            continue

        input_names = [p.name for p in instance.inputs]
        output_names = [p.name for p in instance.outputs]

        if not input_names and not output_names:
            continue

        results.append(
            _CommandEdges(
                command_name=instance.name(),
                input_filenames=input_names,
                output_filenames=output_names,
            )
        )

    return results


def _sanitize_node_id(filename: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", filename)


def _build_mermaid(edges: list[_CommandEdges]) -> str:
    node_ids: dict[str, str] = {}
    for edge in edges:
        for name in edge.input_filenames + edge.output_filenames:
            if name not in node_ids:
                node_ids[name] = _sanitize_node_id(name)

    lines: list[str] = ["flowchart LR"]

    for filename, node_id in sorted(node_ids.items()):
        lines.append(f'    {node_id}["{filename}"]')

    lines.append("")

    for edge in edges:
        for input_name in edge.input_filenames:
            for output_name in edge.output_filenames:
                src = node_ids[input_name]
                dst = node_ids[output_name]
                lines.append(f'    {src} -->|"{edge.command_name}"| {dst}')

    return "\n".join(lines) + "\n"


def document_dependencies(logger: LoggingProtocol) -> str:
    session_dir: Path = common_paths.session_dir(_TEST_SESSION)
    if not session_dir.exists():
        raise FileNotFoundError(
            f"Test session directory not found at {session_dir}. "
            "The document-dependencies command requires the 'test' session to resolve settings."
        )
    settings: SessionSettings = SessionSettings.load_cascading(_TEST_SESSION)
    edges = _collect_command_edges(settings, session_dir, logger)
    return _build_mermaid(edges)
