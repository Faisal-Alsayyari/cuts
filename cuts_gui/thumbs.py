"""Asynchronous thumbnails for the filmstrip and the note inspector.

Requests are served newest-first. Scrolling the timeline issues a burst of
requests for whatever is on screen now, and the ones from a moment ago are
the least useful — so a stack, not a queue, with old requests dropped once
it grows past what one screen could possibly show.

Two worker threads, not more: thumbnails are a nicety, and an analysis may be
running in the background with every core already spoken for.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import List, Optional, Tuple

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
from PySide6.QtGui import QImage

from . import media

_MAX_PENDING = 48
_CACHE_SIZE = 800


def _key(path: str, t: float, width: int) -> str:
    return f"{path}|{t:.2f}|{width}"


class _Worker(QRunnable):
    def __init__(self, service: "ThumbnailService") -> None:
        super().__init__()
        self.service = service

    def run(self) -> None:
        while True:
            job = self.service._pop()
            if job is None:
                return
            key, path, t, width = job
            img = QImage()
            try:
                arr = media.grab_frame(path, t, width)
                if arr is not None:
                    h, w = arr.shape[:2]
                    img = QImage(arr.data, w, h, arr.strides[0],
                                 QImage.Format_RGB888).copy()
            except Exception:  # noqa: BLE001 - a bad frame must not kill the worker
                img = QImage()
            self.service._ready.emit(key, img)


class ThumbnailService(QObject):
    """`get()` returns a cached image or None, and schedules the miss."""

    # key — a thumbnail is now available via get()
    ready = Signal(str)
    _ready = Signal(str, QImage)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(2)
        self._cache: "OrderedDict[str, QImage]" = OrderedDict()
        self._failed: set = set()
        self._pending: List[Tuple[str, str, float, int]] = []
        self._pending_keys: set = set()
        self._lock = threading.Lock()
        self._workers = 0
        self._ready.connect(self._store)

    @property
    def enabled(self) -> bool:
        return media.available()

    def get(self, path: str, t: float, width: int) -> Optional[QImage]:
        if not self.enabled or not path:
            return None
        key = _key(path, t, width)
        img = self._cache.get(key)
        if img is not None:
            self._cache.move_to_end(key)
            return img
        if key in self._failed:
            return None
        with self._lock:
            if key in self._pending_keys:
                # Re-requested: move it to the top of the stack.
                self._pending = [j for j in self._pending if j[0] != key]
            self._pending.append((key, path, t, width))
            self._pending_keys.add(key)
            while len(self._pending) > _MAX_PENDING:
                dropped = self._pending.pop(0)
                self._pending_keys.discard(dropped[0])
            spawn = self._workers < self._pool.maxThreadCount()
            if spawn:
                self._workers += 1
        if spawn:
            self._pool.start(_Worker(self))
        return None

    def _pop(self):
        with self._lock:
            if not self._pending:
                self._workers -= 1
                return None
            job = self._pending.pop()
            self._pending_keys.discard(job[0])
            return job

    def _store(self, key: str, img: QImage) -> None:
        if img.isNull():
            self._failed.add(key)
            return
        self._cache[key] = img
        while len(self._cache) > _CACHE_SIZE:
            self._cache.popitem(last=False)
        self.ready.emit(key)

    def shutdown(self) -> None:
        with self._lock:
            self._pending.clear()
            self._pending_keys.clear()
        self._pool.waitForDone(3000)
