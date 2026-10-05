"""The main window: library on the left, player and timeline in the middle,
chapters / notes / run on the right.

This is wiring. It owns the selection (which session, which chapter, which
note, where the playhead is) and keeps every view in agreement about it; the
views themselves only report what the user did.
"""

from __future__ import annotations

import html
import os
import subprocess
import sys
from typing import List, Optional

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import backend, media, theme
from .dialogs import (
    AddVideosDialog,
    AnalyzeDialog,
    Preferences,
    RechapterDialog,
    SettingsDialog,
    VIDEO_FILTER,
)
from .jobs import Job, JobRunner
from .library import Library
from .model import (
    Index,
    Session,
    SourceFile,
    Status,
    format_duration,
    format_time,
)
from .panels import ChaptersPanel, LibraryPanel, NotesPanel, RunPanel
from .player import TimelinePlayer
from .thumbs import ThumbnailService
from .timeline import TimelineView


def bare_index(session: Session) -> Index:
    """A chapterless index for a session that has not been analyzed yet.

    Lets an unanalyzed recording still be watched and scrubbed on its
    timeline — useful for checking file order before paying for a run.
    """
    sources: List[SourceFile] = []
    cursor = 0.0
    for i, f in enumerate(session.files):
        dur = media.probe(f).duration if os.path.exists(f) else 0.0
        sources.append(SourceFile(path=f, index=i, duration=dur, offset=cursor))
        cursor += dur
    return Index(title=session.name, duration=cursor, sources=sources,
                 chapters=[], notes_unavailable="")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Cuts")
        self.resize(1480, 920)
        self.setAcceptDrops(True)

        self.prefs = Preferences()
        self.library = Library(parent=self)
        self.runner = JobRunner(self.prefs.backend(), self)
        self.thumbs = ThumbnailService(self)
        self.session: Optional[Session] = None
        self.index: Optional[Index] = None
        self._load_error = ""

        self._build()
        self._build_menus()
        self._connect()

        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._on_tick)
        self._tick.start()

        self.library_panel.set_sessions(self.library.sessions)
        self._update_statusbar()
        if self.library.sessions:
            self.library_panel.select(self.library.sessions[0].id)
        else:
            self._show_session(None)

    # =====================================================================
    # Layout
    # =====================================================================

    def _build(self) -> None:
        self.library_panel = LibraryPanel()
        self.library_panel.setMinimumWidth(240)

        # -- centre --------------------------------------------------------
        self.center = QStackedWidget()
        self.center.addWidget(self._build_welcome())
        session_view = QWidget()
        cv = QVBoxLayout(session_view)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        cv.addWidget(self._build_header())
        cv.addWidget(self._build_banner())

        self.video = QVideoWidget()
        self.video.setMinimumHeight(220)
        self.video_stack = QStackedWidget()
        self.video_stack.setObjectName("VideoArea")
        self.video_stack.setAttribute(Qt.WA_StyledBackground, True)
        self.video_stack.addWidget(self.video)
        self.video_message = QLabel("")
        self.video_message.setAlignment(Qt.AlignCenter)
        self.video_message.setObjectName("Subtle")
        self.video_message.setWordWrap(True)
        self.video_stack.addWidget(self.video_message)
        cv.addWidget(self.video_stack, 1)
        cv.addWidget(self._build_caption())
        cv.addWidget(self._build_transport())
        self.timeline = TimelineView(self.thumbs)
        cv.addWidget(self.timeline)
        self.center.addWidget(session_view)

        # -- right ---------------------------------------------------------
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.chapters_panel = ChaptersPanel()
        self.notes_panel = NotesPanel(self.thumbs)
        self.run_panel = RunPanel()
        self.tabs.addTab(self.chapters_panel, "Chapters")
        self.tabs.addTab(self.notes_panel, "Notes")
        self.tabs.addTab(self.run_panel, "Run")
        right = QWidget()
        right.setObjectName("Panel")
        right.setAttribute(Qt.WA_StyledBackground, True)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.tabs)
        right.setMinimumWidth(340)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.library_panel)
        split.addWidget(self.center)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([270, 820, 400])
        split.setChildrenCollapsible(False)
        self.setCentralWidget(split)

        self.player = TimelinePlayer(self.video, self)
        self.status_left = QLabel("")
        self.status_right = QLabel("")
        self.statusBar().addWidget(self.status_left, 1)
        self.statusBar().addPermanentWidget(self.status_right)

    def _build_welcome(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addStretch(2)
        title = QLabel("Turn long recordings into chapters")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        lay.addWidget(title)
        sub = QLabel("Drop one or more screen recordings here, or add them "
                     "from disk.\nSeveral files from one session can be "
                     "analyzed as a single timeline.")
        sub.setAlignment(Qt.AlignCenter)
        sub.setObjectName("Subtle")
        lay.addWidget(sub)
        lay.addSpacing(16)
        row = QHBoxLayout()
        row.addStretch(1)
        add = QPushButton("Add videos…")
        add.setObjectName("Primary")
        add.clicked.connect(lambda: self.add_videos())
        row.addWidget(add)
        op = QPushButton("Open existing index…")
        op.clicked.connect(self.open_index)
        row.addWidget(op)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addStretch(3)
        return w

    def _build_header(self) -> QWidget:
        w = QWidget()
        w.setObjectName("Header")
        w.setAttribute(Qt.WA_StyledBackground, True)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(16, 10, 12, 10)
        col = QVBoxLayout()
        col.setSpacing(2)
        self.h_title = QLabel("")
        self.h_title.setObjectName("Title")
        self.h_sub = QLabel("")
        self.h_sub.setObjectName("Subtle")
        col.addWidget(self.h_title)
        col.addWidget(self.h_sub)
        lay.addLayout(col, 1)
        self.h_rechapter = QPushButton("Re-chapter…")
        self.h_rechapter.setToolTip("Try a different intent against the "
                                    "frame notes already taken — seconds, "
                                    "not a re-analysis")
        self.h_rechapter.clicked.connect(lambda: self.rechapter())
        lay.addWidget(self.h_rechapter)
        self.h_analyze = QPushButton("Analyze…")
        self.h_analyze.clicked.connect(lambda: self.analyze())
        lay.addWidget(self.h_analyze)
        return w

    def _build_banner(self) -> QWidget:
        self.banner = QWidget()
        self.banner.setObjectName("Banner")
        self.banner.setAttribute(Qt.WA_StyledBackground, True)
        lay = QHBoxLayout(self.banner)
        lay.setContentsMargins(16, 7, 12, 7)
        self.banner_text = QLabel("")
        self.banner_text.setWordWrap(True)
        self.banner_text.setTextFormat(Qt.RichText)
        lay.addWidget(self.banner_text, 1)
        self.banner_btn = QPushButton("")
        self.banner_btn.clicked.connect(self._banner_action)
        lay.addWidget(self.banner_btn)
        self.banner_btn2 = QPushButton("")
        self.banner_btn2.clicked.connect(self._banner_action2)
        lay.addWidget(self.banner_btn2)
        self._banner_actions = ("", "")
        return self.banner

    def _build_caption(self) -> QWidget:
        w = QWidget()
        w.setObjectName("Caption")
        w.setAttribute(Qt.WA_StyledBackground, True)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(16, 7, 16, 7)
        lay.setSpacing(2)
        self.cap_chapter = QLabel("")
        self.cap_chapter.setTextFormat(Qt.RichText)
        self.cap_note = QLabel("")
        self.cap_note.setObjectName("Subtle")
        self.cap_note.setWordWrap(True)
        self.cap_note.setMinimumHeight(18)
        lay.addWidget(self.cap_chapter)
        lay.addWidget(self.cap_note)
        return w

    def _build_transport(self) -> QWidget:
        w = QWidget()
        w.setObjectName("Transport")
        w.setAttribute(Qt.WA_StyledBackground, True)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(6)

        def tb(text: str, tip: str, slot) -> QToolButton:
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setObjectName("Flat")
            b.clicked.connect(slot)
            lay.addWidget(b)
            return b

        tb("⏮", "Previous chapter  [", lambda: self.jump_chapter(-1))
        self.play_btn = tb("▶", "Play / pause  Space", lambda: self.player.toggle())
        self.play_btn.setFixedWidth(34)
        tb("⏭", "Next chapter  ]", lambda: self.jump_chapter(1))
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setFont(theme.mono_font(11))
        lay.addSpacing(6)
        lay.addWidget(self.time_label)
        self.file_label = QLabel("")
        self.file_label.setObjectName("Faint")
        lay.addSpacing(8)
        lay.addWidget(self.file_label)
        lay.addStretch(1)

        self.speed = QComboBox()
        for r in (0.5, 1.0, 1.5, 2.0, 4.0):
            self.speed.addItem(f"{r:g}×", r)
        self.speed.setCurrentIndex(1)
        self.speed.setToolTip("Playback speed")
        self.speed.currentIndexChanged.connect(
            lambda _i: self.player.set_rate(self.speed.currentData()))
        lay.addWidget(self.speed)
        self.mute = tb("🔊", "Mute", lambda: None)
        self.mute.setCheckable(True)
        self.mute.toggled.connect(self._mute)
        self.volume = QSlider(Qt.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(80)
        self.volume.setFixedWidth(80)
        self.volume.valueChanged.connect(lambda v: self.player.set_volume(v / 100))
        lay.addWidget(self.volume)
        lay.addSpacing(12)
        tb("−", "Zoom out  Ctrl+-", lambda: self.timeline.zoom_by(1.6))
        tb("Fit", "Fit whole timeline  Ctrl+0", lambda: self.timeline.zoom_to_fit())
        tb("+", "Zoom in  Ctrl+=", lambda: self.timeline.zoom_by(1 / 1.6))
        return w

    def _build_menus(self) -> None:
        mb = self.menuBar()

        def act(menu, text, slot, shortcut=None, checkable=False):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.setCheckable(checkable)
            a.triggered.connect(slot)
            menu.addAction(a)
            return a

        f = mb.addMenu("&File")
        act(f, "Add Videos…", lambda: self.add_videos(), QKeySequence.Open)
        act(f, "Open Existing Index…", self.open_index, "Ctrl+Shift+O")
        f.addSeparator()
        act(f, "Settings…", self.show_settings, QKeySequence.Preferences)
        f.addSeparator()
        act(f, "Quit", self.close, QKeySequence.Quit)

        s = mb.addMenu("&Session")
        self.a_analyze = act(s, "Analyze…", lambda: self.analyze(), "Ctrl+R")
        self.a_rechapter = act(s, "Re-chapter with New Intent…",
                               lambda: self.rechapter(), "Ctrl+Shift+R")
        self.a_cancel = act(s, "Cancel Run", lambda: self.cancel(), "Ctrl+.")
        s.addSeparator()
        act(s, "Copy Chapter List", self._copy_chapters, "Ctrl+Shift+C")
        act(s, "Show in Finder", lambda: self.reveal())
        act(s, "Rename…", lambda: self.rename())
        act(s, "Remove from Library", lambda: self.remove())

        p = mb.addMenu("&Playback")
        act(p, "Play / Pause", lambda: self.player.toggle(), "Space")
        act(p, "Back 5 Seconds", lambda: self.player.step(-5), "Left")
        act(p, "Forward 5 Seconds", lambda: self.player.step(5), "Right")
        act(p, "Back 30 Seconds", lambda: self.player.step(-30), "Shift+Left")
        act(p, "Forward 30 Seconds", lambda: self.player.step(30), "Shift+Right")
        p.addSeparator()
        act(p, "Previous Chapter", lambda: self.jump_chapter(-1), "[")
        act(p, "Next Chapter", lambda: self.jump_chapter(1), "]")
        act(p, "Previous Note", lambda: self.jump_note(-1), ",")
        act(p, "Next Note", lambda: self.jump_note(1), ".")

        v = mb.addMenu("&View")
        act(v, "Zoom In", lambda: self.timeline.zoom_by(1 / 1.6), "Ctrl+=")
        act(v, "Zoom Out", lambda: self.timeline.zoom_by(1.6), "Ctrl+-")
        act(v, "Fit Timeline", self.timeline.zoom_to_fit, "Ctrl+0")
        fs = act(v, "Show Filmstrip", self.timeline.set_frames_visible,
                 checkable=True)
        fs.setChecked(True)
        fs.setEnabled(self.thumbs.enabled)
        v.addSeparator()
        act(v, "Chapters", lambda: self.tabs.setCurrentIndex(0), "Ctrl+1")
        act(v, "Notes", lambda: self.tabs.setCurrentIndex(1), "Ctrl+2")
        act(v, "Run Log", lambda: self.tabs.setCurrentIndex(2), "Ctrl+3")
        act(v, "Search Notes", self._search_notes, QKeySequence.Find)

    def _connect(self) -> None:
        self.library.changed.connect(self._on_library_changed)
        lp = self.library_panel
        lp.session_selected.connect(self._select)
        lp.add_requested.connect(lambda: self.add_videos())
        lp.open_index_requested.connect(self.open_index)
        lp.action_requested.connect(self._library_action)

        self.runner.started.connect(self._on_job_started)
        self.runner.updated.connect(self._on_job_updated)
        self.runner.finished.connect(self._on_job_finished)
        self.run_panel.cancel_requested.connect(lambda: self.cancel())

        self.player.position_changed.connect(self._on_position)
        self.player.playing_changed.connect(
            lambda on: self.play_btn.setText("⏸" if on else "▶"))
        self.player.source_changed.connect(self._on_source)
        self.player.error.connect(self._on_player_error)

        self.timeline.seek_requested.connect(self.player.seek)
        self.timeline.chapter_picked.connect(self._chapter_picked)
        self.timeline.note_picked.connect(self._note_picked)
        self.chapters_panel.chapter_activated.connect(self._chapter_activated)
        self.notes_panel.note_activated.connect(self._note_activated)

    # =====================================================================
    # Selection & loading
    # =====================================================================

    def _select(self, session_id: str) -> None:
        self._show_session(self.library.get(session_id))

    def _show_session(self, session: Optional[Session],
                      keep_position: bool = False) -> None:
        pos = self.player.position if keep_position else 0.0
        self.session = session
        self.player.pause()
        if session is None:
            self.index = None
            self.center.setCurrentIndex(0)
            self.player.set_sources([])
            for v in (self.timeline, self.chapters_panel, self.notes_panel):
                v.set_index(None)
            self.run_panel.show_state("", None)
            self._update_actions()
            return

        self.center.setCurrentIndex(1)
        self._load_error = ""
        index: Optional[Index] = None
        if os.path.exists(session.index_path):
            try:
                index = backend.load_index(session)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self._load_error = f"The index could not be read: {exc}"
        if index is None:
            index = bare_index(session)
        self.index = index
        if index.duration and abs(index.duration - session.duration) > 1:
            session.duration = index.duration
            self.library.touch(session)

        self.timeline.set_index(index)
        self.chapters_panel.set_index(index if index.chapters else None)
        self.notes_panel.set_index(index if index.chapters else None)
        self.player.set_sources(index.sources)
        self.video_stack.setCurrentIndex(0)
        if keep_position and pos:
            self.player.seek(pos)
        self.run_panel.show_state(session.id, self.runner.state(session.id))
        self._update_header()
        self._update_banner()
        self._update_caption(self.player.position)
        self._update_actions()
        self._update_time(self.player.position)

    def _update_header(self) -> None:
        s, ix = self.session, self.index
        if s is None or ix is None:
            return
        self.h_title.setText(s.name)
        bits = []
        if s.is_multi:
            bits.append(f"{len(s.files)} files")
        bits.append(format_duration(ix.duration) if ix.duration else "length unknown")
        if ix.chapters:
            bits.append(f"{len(ix.chapters)} chapters")
        if ix.notes:
            bits.append(f"{len(ix.notes)} frame notes")
        intent = ix.facts.get("intent") if ix.chapters else ""
        if intent:
            bits.append(f"intent: “{intent}”")
        self.h_sub.setText("  ·  ".join(bits))
        self.h_sub.setToolTip("\n".join(s.files))

    def _update_actions(self) -> None:
        s = self.session
        busy = bool(s) and (self.runner.is_active(s.id) or self.runner.is_queued(s.id))
        ready = bool(s) and os.path.exists(s.index_path)
        has_notes = bool(self.index and self.index.notes)
        self.h_analyze.setText("Re-analyze…" if ready else "Analyze…")
        self.h_analyze.setObjectName("" if ready else "Primary")
        self.h_analyze.style().polish(self.h_analyze)
        self.h_analyze.setEnabled(bool(s) and not busy)
        self.h_rechapter.setVisible(ready)
        self.h_rechapter.setEnabled(has_notes and not busy)
        if ready and not has_notes:
            self.h_rechapter.setToolTip("Needs the per-frame notes from a full "
                                        "(llm) analysis")
        self.a_analyze.setEnabled(bool(s) and not busy)
        self.a_rechapter.setEnabled(has_notes and not busy)
        self.a_cancel.setEnabled(busy)

    def _update_banner(self) -> None:
        s = self.session
        if s is None:
            self.banner.hide()
            return
        st = self.runner.state(s.id)
        e = html.escape
        text, b1, b2 = "", ("", ""), ("", "")
        missing = s.missing_files
        if self.runner.is_active(s.id) and st:
            stage = dict(backend.STAGES).get(st.stage, "Starting")
            detail = f" — {e(st.detail)}" if st.detail else ""
            text = (f"<b style='color:{theme.ACCENT}'>◐ {e(st.label)}</b> · "
                    f"{stage}{detail} · {format_duration(st.elapsed)}")
            b1, b2 = ("View log", "log"), ("Cancel", "cancel")
        elif self.runner.is_queued(s.id):
            cur = self.runner.current
            other = self.library.get(cur.session_id) if cur else None
            text = (f"Queued{' behind ' + e(other.name) if other else ''}. "
                    f"Analyses run one at a time.")
            b1 = ("Cancel", "cancel")
        elif missing:
            text = (f"<span style='color:{theme.DANGER}'>⚠ "
                    f"{'File' if len(missing) == 1 else f'{len(missing)} files'} "
                    f"not found:</span> {e(os.path.basename(missing[0]))}"
                    f"{' …' if len(missing) > 1 else ''}")
            b1 = ("Locate…", "relink")
        elif self._load_error:
            text = f"<span style='color:{theme.DANGER}'>{e(self._load_error)}</span>"
            b1 = ("Analyze…", "analyze")
        elif s.status == Status.FAILED:
            text = (f"<span style='color:{theme.DANGER}'>✕ Analysis failed:</span> "
                    f"{e(s.message or 'see the log')}")
            b1, b2 = ("View log", "log"), ("Try again…", "analyze")
        elif not os.path.exists(s.index_path):
            text = ("Not analyzed yet. You can watch and scrub the footage; "
                    "chapters and frame notes appear after analysis.")
            b1 = ("Analyze…", "analyze")
        elif s.redaction_warning:
            text = (f"<span style='color:{theme.WARN}'>⚠ Credentials were on "
                    f"screen</span> ({e(s.redaction_warning)}). They were "
                    f"blacked out before upload, but go rotate them — "
                    f"redaction reduces exposure, it doesn't guarantee it.")
            b1 = ("Show flagged", "flagged")
            b2 = ("Dismiss", "dismiss")
        self.banner.setVisible(bool(text))
        self.banner_text.setText(text)
        self._banner_actions = (b1[1], b2[1])
        for btn, (label, _a) in ((self.banner_btn, b1), (self.banner_btn2, b2)):
            btn.setText(label)
            btn.setVisible(bool(label))

    def _banner_action(self) -> None:
        self._do_banner(self._banner_actions[0])

    def _banner_action2(self) -> None:
        self._do_banner(self._banner_actions[1])

    def _do_banner(self, action: str) -> None:
        s = self.session
        if s is None:
            return
        if action == "log":
            self.tabs.setCurrentWidget(self.run_panel)
        elif action == "flagged":
            self.tabs.setCurrentWidget(self.notes_panel)
            self.notes_panel.flagged.setChecked(True)
        elif action == "dismiss":
            s.redaction_warning = ""
            self.library.touch(s)
            self._update_banner()
        else:
            self._library_action(action, s.id)

    # =====================================================================
    # Playback sync
    # =====================================================================

    def _on_position(self, t: float) -> None:
        if self.index is None:
            return
        playing = self.player.is_playing
        self.timeline.set_playhead(t, follow=playing)
        self.chapters_panel.set_playing(self.index.chapter_at(t))
        self.notes_panel.set_playing(self.index.note_at(t), follow=playing)
        self._update_caption(t)
        self._update_time(t)

    def _update_time(self, t: float) -> None:
        dur = self.index.duration if self.index else 0.0
        self.time_label.setText(f"{format_time(t)} / {format_time(dur)}")
        if self.index and self.index.is_multi:
            src, local = self.index.source_at(t)
            self.file_label.setText(f"{src.name} @ {format_time(local)}")
        else:
            self.file_label.setText("")

    def _update_caption(self, t: float) -> None:
        ix = self.index
        if ix is None or not ix.chapters:
            self.cap_chapter.setText(
                f"<span style='color:{theme.FAINT}'>No chapters yet</span>")
            self.cap_note.setText("")
            return
        ci = ix.chapter_at(t)
        if ci >= 0:
            c = ix.chapters[ci]
            col = theme.kind_color(c.kind, ci).name()
            kind = (f"  <span style='color:{theme.MUTED}'>{html.escape(c.kind)}"
                    f"</span>" if c.kind else "")
            self.cap_chapter.setText(
                f"<span style='color:{col}'>●</span>  <b>"
                f"{html.escape(c.display_title)}</b>"
                f"  <span style='color:{theme.FAINT}'>{ci + 1}/{len(ix.chapters)}"
                f"</span>{kind}")
        ni = ix.note_at(t)
        if ni >= 0:
            n = ix.notes[ni]
            body = n.description or ("withheld — not described" if n.withheld
                                     else "")
            self.cap_note.setText(f"{format_time(n.t)}  {body}")
        else:
            self.cap_note.setText("")

    def _on_source(self, idx: int) -> None:
        if self.index and 0 <= idx < len(self.index.sources):
            src = self.index.sources[idx]
            if src.exists:
                self.video_stack.setCurrentIndex(0)

    def _on_player_error(self, msg: str) -> None:
        self.video_message.setText(msg)
        self.video_stack.setCurrentIndex(1)

    def _mute(self, on: bool) -> None:
        self.player.set_muted(on)
        self.mute.setText("🔇" if on else "🔊")

    def jump_chapter(self, d: int) -> None:
        ix = self.index
        if not ix or not ix.chapters:
            return
        t = self.player.position
        ci = ix.chapter_at(t)
        if d < 0 and ci >= 0 and t - ix.chapters[ci].start > 2.0:
            target = ci
        else:
            target = max(0, min(len(ix.chapters) - 1, ci + d))
        self._chapter_activated(target)
        self.chapters_panel.select(target)

    def jump_note(self, d: int) -> None:
        ix = self.index
        if not ix or not ix.notes:
            return
        ni = ix.note_at(self.player.position)
        target = max(0, min(len(ix.notes) - 1, ni + d if ni >= 0 else 0))
        self._note_activated(target)
        self.notes_panel.select(target)

    def _chapter_activated(self, i: int) -> None:
        if self.index and 0 <= i < len(self.index.chapters):
            c = self.index.chapters[i]
            self.timeline.select_chapter(i)
            self.timeline.reveal(c.start)
            self.player.seek(c.start)

    def _note_activated(self, i: int) -> None:
        if self.index and 0 <= i < len(self.index.notes):
            self.timeline.select_note(i, reveal=True)
            self.player.seek(self.index.notes[i].t)

    def _chapter_picked(self, i: int) -> None:
        self.chapters_panel.select(i)

    def _note_picked(self, i: int) -> None:
        self.tabs.setCurrentWidget(self.notes_panel)
        self.notes_panel.select(i)

    # =====================================================================
    # Library actions
    # =====================================================================

    def add_videos(self, files: Optional[List[str]] = None) -> None:
        if files is None:
            files, _ = QFileDialog.getOpenFileNames(
                self, "Add videos", "", VIDEO_FILTER)
            if not files:
                return
        dlg = AddVideosDialog(files, self.prefs.root, self)
        if dlg.exec() != AddVideosDialog.Accepted:
            return
        first: Optional[Session] = None
        for s in dlg.result_sessions():
            if os.path.exists(s.index_path):
                s.status = Status.READY
            existing = self.library.find_by_index(s.index_path)
            s = self.library.add(s)
            first = first or s
            if dlg.start and s.status != Status.READY and existing is None:
                self._submit_analysis(s)
        if first:
            self.library_panel.select(first.id)

    def open_index(self, path: str = "") -> None:
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open index", "", "Cuts index (*.events.json);;JSON (*.json)")
            if not path:
                return
        try:
            s = backend.session_from_index(path)
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Cannot open index",
                                f"{os.path.basename(path)} could not be read:\n{exc}")
            return
        s = self.library.add(s)
        self.library_panel.select(s.id)

    def analyze(self, session: Optional[Session] = None) -> None:
        s = session or self.session
        if s is None or self.runner.is_active(s.id) or self.runner.is_queued(s.id):
            return
        if s.missing_files:
            QMessageBox.warning(self, "Missing files",
                                "Locate the missing video files first.")
            return
        dlg = AnalyzeDialog(s, self.prefs.root, self)
        if dlg.exec() != AnalyzeDialog.Accepted:
            return
        s.settings = dlg.settings()
        self._submit_analysis(s)

    def _submit_analysis(self, s: Session) -> None:
        program, args = self.runner.backend.analyze_command(s)
        self._submit(s, Job(s.id, program, args, s.index_path, "Analyzing"))

    def rechapter(self, session: Optional[Session] = None) -> None:
        s = session or self.session
        if s is None or self.runner.is_active(s.id) or self.runner.is_queued(s.id):
            return
        n = len(self.index.notes) if (self.index and s is self.session) else 0
        if not n:
            try:
                n = len(backend.load_index(s).notes)
            except (OSError, ValueError):
                n = 0
        cmd = self.runner.backend.rechapter_command(s, s.settings) if n else None
        if cmd is None:
            QMessageBox.information(
                self, "Re-chapter",
                "There are no cached frame notes that match this session's "
                "files, so chapters can only be redone by a full analysis.")
            return
        dlg = RechapterDialog(s, n, self)
        if dlg.exec() != RechapterDialog.Accepted:
            return
        s.settings = dlg.settings()
        program, args = self.runner.backend.rechapter_command(s, s.settings)
        self._submit(s, Job(s.id, program, args, s.index_path, "Re-chaptering"))

    def _submit(self, s: Session, job: Job) -> None:
        s.status, s.message = Status.QUEUED, ""
        self.library.touch(s)
        self.runner.submit(job)
        if s is self.session:
            self.tabs.setCurrentWidget(self.run_panel)

    def cancel(self, session: Optional[Session] = None) -> None:
        s = session or self.session
        if s is not None:
            self.runner.cancel(s.id)

    def reveal(self, session: Optional[Session] = None) -> None:
        s = session or self.session
        if s is None:
            return
        target = s.index_path if os.path.exists(s.index_path) else s.files[0]
        if sys.platform == "darwin" and os.path.exists(target):
            subprocess.Popen(["open", "-R", target])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(target)))

    def rename(self, session: Optional[Session] = None) -> None:
        s = session or self.session
        if s is None:
            return
        name, ok = QInputDialog.getText(self, "Rename", "Name", text=s.name)
        if ok and name.strip():
            s.name = name.strip()
            self.library.touch(s)
            if s is self.session:
                self._update_header()

    def remove(self, session: Optional[Session] = None) -> None:
        s = session or self.session
        if s is None:
            return
        ok = QMessageBox.question(
            self, "Remove from library",
            f"Remove “{s.name}” from the library?\n\nNothing is deleted from "
            f"disk: the videos, index and frame notes stay where they are, and "
            f"adding the videos again picks the index back up.")
        if ok != QMessageBox.Yes:
            return
        self.runner.cancel(s.id)
        was_current = s is self.session
        self.library.remove(s.id)
        if was_current:
            nxt = self.library.sessions[0] if self.library.sessions else None
            if nxt:
                self.library_panel.select(nxt.id)
            else:
                self._show_session(None)

    def relink(self, session: Optional[Session] = None) -> None:
        s = session or self.session
        if s is None:
            return
        for i, f in enumerate(list(s.files)):
            if os.path.exists(f):
                continue
            path, _ = QFileDialog.getOpenFileName(
                self, f"Locate {os.path.basename(f)}",
                os.path.dirname(s.index_path), VIDEO_FILTER)
            if not path:
                return
            s.files[i] = os.path.abspath(path)
        self.library.touch(s)
        if s is self.session:
            self._show_session(s)

    def _library_action(self, action: str, session_id: str) -> None:
        s = self.library.get(session_id)
        if s is None:
            return
        {
            "analyze": self.analyze,
            "rechapter": self.rechapter,
            "cancel": self.cancel,
            "reveal": self.reveal,
            "rename": self.rename,
            "remove": self.remove,
            "relink": self.relink,
        }.get(action, lambda _s: None)(s)

    def _on_library_changed(self, session_id: str) -> None:
        if session_id:
            self.library_panel.refresh()
        else:
            self.library_panel.set_sessions(
                self.library.sessions,
                self.session.id if self.session else "")

    # =====================================================================
    # Jobs
    # =====================================================================

    def _on_job_started(self, sid: str) -> None:
        s = self.library.get(sid)
        if s:
            s.status = Status.RUNNING
            self.library.touch(s)
        self._refresh_job_views(sid)

    def _on_job_updated(self, sid: str) -> None:
        self._refresh_job_views(sid)

    def _refresh_job_views(self, sid: str) -> None:
        if self.session and self.session.id == sid:
            self.run_panel.show_state(sid, self.runner.state(sid))
            self._update_banner()
            self._update_actions()
        self.library_panel.refresh()
        self._update_statusbar()

    def _on_job_finished(self, sid: str, ok: bool, message: str) -> None:
        s = self.library.get(sid)
        st = self.runner.state(sid)
        if s is None:
            return
        if ok:
            s.status = Status.READY
            try:
                ix = backend.load_index(s)
                s.message = f"{len(ix.chapters)} chapters"
                s.duration = ix.duration or s.duration
            except (OSError, ValueError):
                s.message = ""
        elif message == "cancelled":
            s.status = Status.READY if os.path.exists(s.index_path) \
                else Status.CANCELLED
            s.message = "run cancelled"
        else:
            s.status, s.message = Status.FAILED, message
        if st:
            redact = [w for w in st.warnings if w.startswith("Credentials found")]
            if redact:
                s.redaction_warning = redact[-1].split(": ", 1)[-1]
        self.library.touch(s)
        if self.session is s:
            self._show_session(s, keep_position=True)
            if ok:
                self.tabs.setCurrentWidget(self.chapters_panel)
        self._update_statusbar()
        QApplication.alert(self)

    def _on_tick(self) -> None:
        cur = self.runner.current
        if cur is None:
            return
        if self.session and self.session.id == cur.session_id:
            self.run_panel.show_state(cur.session_id,
                                      self.runner.state(cur.session_id))
            self._update_banner()
        self._update_statusbar()

    def _update_statusbar(self) -> None:
        cur = self.runner.current
        if cur:
            s = self.library.get(cur.session_id)
            st = self.runner.state(cur.session_id)
            stage = dict(backend.STAGES).get(st.stage, "starting") if st else ""
            q = self.runner.queue_length
            self.status_left.setText(
                f"{cur.label} {s.name if s else ''} — {stage} · "
                f"{format_duration(st.elapsed) if st else ''}"
                + (f" · {q} queued" if q else ""))
        else:
            self.status_left.setText("Idle")
        src = backend.credentials_source(self.prefs.root)
        self.status_right.setText("API key: " + ("environment" if src == "environment"
                                                 else ".env" if src else "not found"))

    # =====================================================================
    # Misc
    # =====================================================================

    def show_settings(self) -> None:
        dlg = SettingsDialog(self.prefs, self)
        if dlg.exec() == SettingsDialog.Accepted:
            # Only affects runs submitted from now on.
            self.runner.backend = self.prefs.backend()
            self._update_statusbar()

    def _copy_chapters(self) -> None:
        if self.index and self.index.chapters:
            QApplication.clipboard().setText(
                backend.chapter_list(self.index, show_source=self.index.is_multi))
            self.statusBar().showMessage("Chapter list copied", 2500)

    def _search_notes(self) -> None:
        self.tabs.setCurrentWidget(self.notes_panel)
        self.notes_panel.focus_search()

    # -- drag & drop -------------------------------------------------------

    def _dropped(self, ev) -> tuple:
        paths = [u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()]
        videos = [p for p in paths if p.lower().endswith(backend.VIDEO_EXTENSIONS)]
        indexes = [p for p in paths if p.endswith(backend.INDEX_SUFFIX)]
        return videos, indexes

    def dragEnterEvent(self, ev) -> None:
        videos, indexes = self._dropped(ev)
        if videos or indexes:
            ev.acceptProposedAction()

    def dropEvent(self, ev) -> None:
        videos, indexes = self._dropped(ev)
        ev.acceptProposedAction()
        for p in indexes:
            self.open_index(p)
        if videos:
            # Deferred so the drag finishes (and the Finder stops waiting on
            # us) before a modal dialog opens.
            QTimer.singleShot(0, lambda: self.add_videos(videos))

    def closeEvent(self, ev) -> None:
        if self.runner.current is not None:
            ok = QMessageBox.question(
                self, "Analysis running",
                "An analysis is still running. Quit and stop it?\n\n"
                "Frame notes are only cached once the reading pass completes.")
            if ok != QMessageBox.Yes:
                ev.ignore()
                return
        self.player.pause()
        self.runner.shutdown()
        self.thumbs.shutdown()
        super().closeEvent(ev)
