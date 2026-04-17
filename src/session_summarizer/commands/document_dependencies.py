from __future__ import annotations

from dataclasses import dataclass

import session_summarizer.utils.common_paths as common_paths

from ..helpers.document_dependencies import document_dependencies
from ..protocols import LoggingProtocol, NullLogger


@dataclass
class DocumentDependenciesCommand:
    logger: LoggingProtocol = NullLogger()

    def name(self) -> str:
        return "Document Dependencies"

    def execute(self, logger: LoggingProtocol) -> None:
        self.logger = logger
        logger.report_message(f"[blue]{self.name()}: Processing...[/blue]")

        mermaid_text: str = document_dependencies(logger)

        output_path = common_paths.command_dependencies_mermaid_file()
        common_paths.ensure_directory(output_path.parent)
        output_path.write_text(mermaid_text, encoding="utf-8")

        logger.report_message(f"[green]Command dependency graph written to {output_path}[/green]")
