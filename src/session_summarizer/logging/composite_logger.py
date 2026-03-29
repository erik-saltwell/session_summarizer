from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any

from ..protocols import LoggingProtocol, ProgressTask, StatusHandle


@dataclass
class CompositeStatusHandle(StatusHandle):
    members: list[StatusHandle]

    def update(self, message: str) -> None:
        for member in self.members:
            member.update(message)

    def close(self) -> None:
        for member in self.members:
            member.close()


@dataclass
class CompositeProgressTask(ProgressTask):
    members: list[ProgressTask]

    def advance(self, n: int = 1) -> None:
        for member in self.members:
            member.advance(n)

    def set_total(self, total: int | None) -> None:
        for member in self.members:
            member.set_total(total)

    def set_completed(self, completed: int) -> None:
        for member in self.members:
            member.set_completed(completed)

    def set_description(self, description: str) -> None:
        for member in self.members:
            member.set_description(description)

    def close(self) -> None:
        for member in self.members:
            member.close()


@dataclass
class CompositeLogger(LoggingProtocol):
    members: list[LoggingProtocol]

    def report_message(self, message: str) -> None:
        for member in self.members:
            member.report_message(message)

    def report_warning(self, message: str) -> None:
        for member in self.members:
            member.report_warning(message)

    def report_error(self, message: str) -> None:
        for member in self.members:
            member.report_error(message)

    def report_exception(self, context: str, exc: BaseException) -> None:
        for member in self.members:
            member.report_exception(context, exc)

    def report_table_message(self, row_data: dict[str, Any]) -> None:
        for member in self.members:
            member.report_table_message(row_data)

    def report_multicolumn_table(self, headers: list[str], rows: list[list[str]]) -> None:
        for member in self.members:
            member.report_multicolumn_table(headers, rows)

    def add_break(self, break_count: int = 1) -> None:
        for member in self.members:
            member.add_break(break_count)

    @contextmanager
    def status(self, message: str) -> Iterator[StatusHandle]:
        with ExitStack() as stack:
            items: list[StatusHandle] = []
            for member in self.members:
                items.append(stack.enter_context(member.status(message)))
            yield CompositeStatusHandle(items)

    @contextmanager
    def progress(self, description: str, total: int | None = None) -> Iterator[ProgressTask]:
        with ExitStack() as stack:
            items: list[ProgressTask] = []
            for member in self.members:
                items.append(stack.enter_context(member.progress(description, total)))
            yield CompositeProgressTask(items)
