"""Application entry point: `python -m cuts_gui [video-or-index ...]`."""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from . import backend, theme


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = QApplication(argv)
    app.setApplicationName("Cuts")
    app.setOrganizationName("cuts")
    app.setApplicationDisplayName("Cuts")
    theme.apply(app)

    from .window import MainWindow

    win = MainWindow()
    win.show()

    # Paths on the command line open as if dropped on the window.
    paths = [os.path.abspath(p) for p in argv[1:] if os.path.exists(p)]
    indexes = [p for p in paths if p.endswith(backend.INDEX_SUFFIX)]
    videos = [p for p in paths if p.lower().endswith(backend.VIDEO_EXTENSIONS)]
    for p in indexes:
        QTimer.singleShot(0, lambda p=p: win.open_index(p))
    if videos:
        QTimer.singleShot(0, lambda: win.add_videos(videos))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
