from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.indexer.run_indexer import IndexStats, run


@dataclass
class IndexJobState:
    status: str = "idle"  # idle | running | success | error
    stage: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    message: str | None = None
    logs: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None


class _ThreadLogHandler(logging.Handler):
    def __init__(self, thread_ident: int, sink: list[str]) -> None:
        super().__init__()
        self._thread_ident = thread_ident
        self._sink = sink
        self.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread != self._thread_ident:
            return
        try:
            self._sink.append(self.format(record))
        except Exception:
            # Logging must never break the indexer thread.
            pass


class IndexJobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = IndexJobState()
        self._thread: threading.Thread | None = None

    def get_state(self) -> IndexJobState:
        with self._lock:
            return IndexJobState(**self._state.__dict__)

    def start(
        self,
        repo_path: str,
        *,
        workspace: bool = False,
        module_name: str | None = None,
        reset_logs: bool = True,
    ) -> IndexJobState:
        with self._lock:
            if self._state.status == "running":
                return IndexJobState(**self._state.__dict__)
            if reset_logs:
                self._state.logs = []
            self._state.status = "running"
            self._state.stage = "starting"
            self._state.started_at = datetime.utcnow().isoformat() + "Z"
            self._state.finished_at = None
            self._state.message = None
            self._state.result = None

        thread = threading.Thread(
            target=self._run,
            args=(repo_path, workspace, module_name),
            daemon=True,
        )
        self._thread = thread
        thread.start()
        return self.get_state()

    def _set_stage(self, stage: str, message: str | None = None) -> None:
        with self._lock:
            self._state.stage = stage
            if message:
                self._state.message = message

    def _run(self, repo_path: str, workspace: bool, module_name: str | None) -> None:
        thread_ident = threading.get_ident()
        handler = _ThreadLogHandler(thread_ident=thread_ident, sink=self._state.logs)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

        def on_stage(stage: str, message: str | None = None) -> None:
            self._set_stage(stage, message)

        try:
            stats: IndexStats = run(
                repo_path,
                workspace=workspace,
                module_name=module_name,
                on_stage=on_stage,
            )
            with self._lock:
                self._state.status = "success"
                self._state.stage = "done"
                self._state.finished_at = datetime.utcnow().isoformat() + "Z"
                self._state.result = stats.model_dump()
        except Exception as exc:
            with self._lock:
                self._state.status = "error"
                self._state.stage = "error"
                self._state.finished_at = datetime.utcnow().isoformat() + "Z"
                self._state.message = str(exc)
        finally:
            try:
                root_logger.removeHandler(handler)
            except Exception:
                pass


index_job_manager = IndexJobManager()

