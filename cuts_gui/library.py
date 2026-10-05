"""The persistent list of sessions the user has added.

GUI state only. The backend's outputs stay where the backend writes them
(beside the videos); the library just remembers which files belong together,
how they should be analyzed, and where their index is.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import List, Optional

from PySide6.QtCore import QObject, QStandardPaths, Signal

from .model import Session, Status


def default_path() -> str:
    base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    return os.path.join(base or os.path.expanduser("~/.cuts_gui"), "library.json")


class Library(QObject):
    """Ordered sessions, saved on every change."""

    # Emitted with a session id when that one entry changed, or "" when the
    # list itself changed (added, removed, reordered).
    changed = Signal(str)

    def __init__(self, path: Optional[str] = None, parent=None) -> None:
        super().__init__(parent)
        self.path = path or default_path()
        self.sessions: List[Session] = []
        self._load()

    # -- queries ----------------------------------------------------------------

    def get(self, session_id: str) -> Optional[Session]:
        return next((s for s in self.sessions if s.id == session_id), None)

    def find_by_index(self, index_path: str) -> Optional[Session]:
        target = os.path.abspath(index_path)
        return next((s for s in self.sessions
                     if os.path.abspath(s.index_path) == target), None)

    # -- mutations --------------------------------------------------------------

    def add(self, session: Session) -> Session:
        existing = self.find_by_index(session.index_path)
        if existing is not None:
            return existing
        self.sessions.insert(0, session)
        self._save()
        self.changed.emit("")
        return session

    def remove(self, session_id: str) -> None:
        self.sessions = [s for s in self.sessions if s.id != session_id]
        self._save()
        self.changed.emit("")

    def touch(self, session: Session) -> None:
        """Persist and announce an in-place change to `session`."""
        self._save()
        self.changed.emit(session.id)

    # -- storage ----------------------------------------------------------------

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        for d in data.get("sessions", []):
            try:
                s = Session.from_dict(d)
            except TypeError:
                continue
            # Nothing survives a restart mid-run: the subprocess died with the
            # previous GUI. Whatever is on disk now is the truth.
            if s.status in (Status.RUNNING, Status.QUEUED):
                s.status = Status.CANCELLED
                s.message = "interrupted"
            if os.path.exists(s.index_path):
                if s.status in (Status.NEW, Status.CANCELLED):
                    s.status = Status.READY
            elif s.status == Status.READY:
                s.status = Status.NEW
                s.message = "index file is missing"
            self.sessions.append(s)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        payload = {"version": 1,
                   "sessions": [s.to_dict() for s in self.sessions]}
        fd, tmp = tempfile.mkstemp(prefix=".library_", suffix=".json",
                                   dir=os.path.dirname(self.path))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
