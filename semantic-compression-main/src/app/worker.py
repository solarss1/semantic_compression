"""Фонові потоки для важких операцій (DL, стиснення)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal


class TaskWorker(QThread):
    """Універсальний worker для блокуючих задач."""

    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(
        self,
        task: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._task = task
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            result = self._task(*self._args, **self._kwargs)
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))
