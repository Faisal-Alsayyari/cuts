"""Runs backend commands one at a time and reports what they are doing.

One at a time on purpose. Pass 1 already saturates the machine (OCR's ONNX
session uses every core; see the backend's performance notes), so two
analyses side by side would each run at half speed and both finish later
than running them in turn.

Commands are opaque here — `cuts_gui.backend` builds them and interprets
their output. This module only owns the process: start, stream, cancel, and
decide success by exit code and whether the index was actually written.
"""

from __future__ import annotations

import os
import shlex
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from . import backend
from .backend import Backend, Signal as LogSignal

_LOG_CAP = 5000


@dataclass
class Job:
    session_id: str
    program: str
    args: List[str]
    index_path: str
    # What the run is, for display: "Analyzing", "Re-chaptering".
    label: str = "Analyzing"

    @property
    def command_line(self) -> str:
        return " ".join(shlex.quote(a) for a in [self.program, *self.args])


@dataclass
class RunState:
    """Live picture of one session's most recent run, for the Run panel."""

    label: str = ""
    command: str = ""
    stage: str = ""
    detail: str = ""
    done: Dict[str, float] = field(default_factory=dict)
    seen: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: str = ""
    started: float = 0.0
    finished: float = 0.0
    active: bool = False
    queued: bool = False
    # The most recent _LOG_CAP lines. `total_lines` counts every line ever
    # added, so a viewer can tell what is new even after the front is trimmed.
    log: List[str] = field(default_factory=list)
    total_lines: int = 0

    def add_line(self, line: str) -> None:
        self.log.append(line)
        self.total_lines += 1
        if len(self.log) > _LOG_CAP:
            del self.log[: len(self.log) - _LOG_CAP]

    def lines_since(self, shown: int) -> List[str]:
        """Lines added after the first `shown` (as many as are still kept)."""
        new = self.total_lines - shown
        if new <= 0:
            return []
        return self.log[-new:] if new < len(self.log) else list(self.log)

    @property
    def elapsed(self) -> float:
        if not self.started:
            return 0.0
        return (self.finished or time.time()) - self.started


class JobRunner(QObject):
    # session_id
    started = Signal(str)
    # session_id — the RunState changed (stage, warning, log line)
    updated = Signal(str)
    # session_id, ok, message
    finished = Signal(str, bool, str)

    def __init__(self, backend_: Backend, parent=None) -> None:
        super().__init__(parent)
        self.backend = backend_
        self.states: Dict[str, RunState] = {}
        self._queue: Deque[Job] = deque()
        self._current: Optional[Job] = None
        self._proc: Optional[QProcess] = None
        self._buffer = ""
        self._cancelling = False
        self._index_mtime = 0.0

    # -- queries ----------------------------------------------------------------

    def state(self, session_id: str) -> Optional[RunState]:
        return self.states.get(session_id)

    def is_active(self, session_id: str) -> bool:
        return (self._current is not None
                and self._current.session_id == session_id)

    def is_queued(self, session_id: str) -> bool:
        return any(j.session_id == session_id for j in self._queue)

    @property
    def current(self) -> Optional[Job]:
        return self._current

    @property
    def queue_length(self) -> int:
        return len(self._queue)

    # -- control ----------------------------------------------------------------

    def submit(self, job: Job) -> None:
        if self.is_active(job.session_id) or self.is_queued(job.session_id):
            return
        self.states[job.session_id] = RunState(
            label=job.label, command=job.command_line, queued=True)
        self._queue.append(job)
        self.updated.emit(job.session_id)
        self._next()

    def cancel(self, session_id: str) -> None:
        for job in list(self._queue):
            if job.session_id == session_id:
                self._queue.remove(job)
                st = self.states.get(session_id)
                if st:
                    st.queued = False
                self.finished.emit(session_id, False, "cancelled")
                return
        if self.is_active(session_id) and self._proc is not None:
            self._cancelling = True
            self._proc.terminate()
            # Terminate is a request; a child stuck in a native call may
            # ignore it. Escalate rather than leave the queue wedged.
            proc = self._proc

            def escalate() -> None:
                try:
                    if proc.state() != QProcess.NotRunning:
                        proc.kill()
                except RuntimeError:  # already finished and deleted
                    pass

            QTimer.singleShot(4000, escalate)

    def shutdown(self) -> None:
        self._queue.clear()
        if self._proc is not None and self._proc.state() != QProcess.NotRunning:
            self._cancelling = True
            self._proc.kill()
            self._proc.waitForFinished(2000)

    # -- internals --------------------------------------------------------------

    def _next(self) -> None:
        if self._current is not None or not self._queue:
            return
        job = self._queue.popleft()
        self._current = job
        self._cancelling = False
        self._buffer = ""
        try:
            self._index_mtime = os.path.getmtime(job.index_path)
        except OSError:
            self._index_mtime = 0.0

        st = self.states.setdefault(job.session_id, RunState())
        st.queued, st.active, st.started = False, True, time.time()
        st.add_line(f"$ {job.command_line}")

        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.setWorkingDirectory(self.backend.root)
        env = QProcessEnvironment()
        for k, v in self.backend.environment().items():
            env.insert(k, v)
        proc.setProcessEnvironment(env)
        proc.readyReadStandardOutput.connect(self._read)
        proc.finished.connect(self._done)
        proc.errorOccurred.connect(self._failed_to_start)
        self._proc = proc
        proc.start(job.program, job.args)
        self.started.emit(job.session_id)
        self.updated.emit(job.session_id)

    def _read(self) -> None:
        if self._proc is None or self._current is None:
            return
        chunk = bytes(self._proc.readAllStandardOutput()).decode(
            "utf-8", errors="replace")
        self._buffer += chunk.replace("\r\n", "\n").replace("\r", "\n")
        *lines, self._buffer = self._buffer.split("\n")
        if lines:
            self._consume(lines)

    def _consume(self, lines: List[str]) -> None:
        sid = self._current.session_id
        st = self.states[sid]
        for line in lines:
            st.add_line(line)
            sig: Optional[LogSignal] = backend.interpret(line)
            if sig is None:
                continue
            if sig.kind == "stage":
                st.stage, st.detail = sig.stage, sig.text
                if sig.stage not in st.seen:
                    st.seen.append(sig.stage)
            elif sig.kind == "done" and sig.stage:
                st.done[sig.stage] = st.done.get(sig.stage, 0.0) + (sig.value or 0)
                if sig.stage not in st.seen:
                    st.seen.append(sig.stage)
            elif sig.kind == "warning":
                text = ("Credentials found: " + sig.text
                        if sig.stage == "redact" else sig.text)
                if text not in st.warnings:
                    st.warnings.append(text)
            elif sig.kind == "error" and not st.error:
                st.error = sig.text
        self.updated.emit(sid)

    def _failed_to_start(self, err) -> None:
        if err == QProcess.FailedToStart and self._current is not None:
            st = self.states[self._current.session_id]
            st.error = (f"Could not start {self._current.program!r}. Check the "
                        f"Python interpreter in Settings.")
            # `finished` is not emitted for a process that never started.
            self._done(-1, QProcess.CrashExit)

    def _done(self, exit_code: int, exit_status) -> None:
        job, proc = self._current, self._proc
        if job is None:
            return
        if self._buffer:
            self._consume([self._buffer])
            self._buffer = ""
        st = self.states[job.session_id]
        st.active, st.finished = False, time.time()

        try:
            written = os.path.getmtime(job.index_path) > self._index_mtime
        except OSError:
            written = False
        ok = (not self._cancelling and exit_status == QProcess.NormalExit
              and exit_code == 0 and written)
        if self._cancelling:
            message = "cancelled"
        elif ok:
            message = ""
        else:
            if not st.error:
                tail = [ln for ln in st.log if ln.strip()][-1:] or [""]
                st.error = (tail[0].strip() if exit_code
                            else "the backend exited without writing an index")
            message = st.error

        self._current, self._proc = None, None
        if proc is not None:
            proc.deleteLater()
        self.updated.emit(job.session_id)
        self.finished.emit(job.session_id, ok, message)
        self._next()
