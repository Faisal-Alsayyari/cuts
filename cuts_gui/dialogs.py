"""Dialogs: adding videos, choosing how to analyze them, and preferences."""

from __future__ import annotations

import os
import re
from dataclasses import replace
from typing import List, Optional

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import backend, media
from .model import RunSettings, Session, format_duration

VIDEO_FILTER = ("Videos (" + " ".join(f"*{e}" for e in backend.VIDEO_EXTENSIONS)
                + ");;All files (*)")


def natural_key(path: str):
    """Sort "(2) b.mp4" before "(10) a.mp4" — how people number recordings."""
    name = os.path.basename(path).lower()
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def _note(text: str, kind: str = "Subtle") -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName(kind)
    lbl.setWordWrap(True)
    return lbl


# ---------------------------------------------------------------------------
# Analysis settings (shared by "add" and "re-analyze")
# ---------------------------------------------------------------------------

class RunSettingsForm(QWidget):
    """The knobs a user would plausibly touch, with the rest behind Advanced.

    Every numeric field has a "Default" state that passes nothing, so the
    backend's own defaults apply and the GUI never goes stale against them.
    """

    def __init__(self, settings: RunSettings, total_duration: float = 0.0,
                 parent=None) -> None:
        super().__init__(parent)
        self.total_duration = total_duration
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.intent = QLineEdit(settings.intent)
        self.intent.setPlaceholderText(
            "What is this index for? e.g. “a tutorial walkthrough”, "
            "“find where I broke the build”")
        form.addRow("Intent", self.intent)
        form.addRow("", _note(
            "Optional, but it matters: what counts as worth its own chapter "
            "depends on what the chapters are for."))

        row = QHBoxLayout()
        self.interval = QSpinBox()
        self.interval.setRange(0, 600)
        self.interval.setSpecialValueText("Default")
        self.interval.setSuffix(" s")
        self.interval.setValue(int(settings.interval or 0))
        self.interval.valueChanged.connect(self._estimate)
        row.addWidget(self.interval)
        self.estimate = QLabel("")
        self.estimate.setObjectName("Subtle")
        row.addWidget(self.estimate, 1)
        form.addRow("Sample every", row)
        lay.addLayout(form)

        self.adv_toggle = QToolButton()
        self.adv_toggle.setObjectName("Flat")
        self.adv_toggle.setCheckable(True)
        self.adv_toggle.setText("▸ Advanced")
        lay.addWidget(self.adv_toggle, 0, Qt.AlignLeft)

        self.adv = QWidget()
        af = QFormLayout(self.adv)
        af.setLabelAlignment(Qt.AlignRight)
        af.setContentsMargins(0, 0, 0, 0)
        self.segmenter = QComboBox()
        self.segmenter.addItem("Default", None)
        self.segmenter.addItem("LLM — judges where activity changes", "llm")
        self.segmenter.addItem("Curve — deterministic similarity baseline",
                               "curve")
        self.segmenter.setCurrentIndex(
            max(0, self.segmenter.findData(settings.segmenter)))
        af.addRow("Segmenter", self.segmenter)
        self.min_chapter = QSpinBox()
        self.min_chapter.setRange(0, 3600)
        self.min_chapter.setSpecialValueText("Default")
        self.min_chapter.setSuffix(" s")
        self.min_chapter.setValue(int(settings.min_chapter or 0))
        af.addRow("Shortest chapter", self.min_chapter)
        self.transcribe = QCheckBox("Transcribe speech when there is audio")
        self.transcribe.setChecked(settings.transcribe)
        af.addRow("", self.transcribe)
        self.refine = QCheckBox("Refine chapter boundaries (one extra look "
                                "at each)")
        self.refine.setChecked(settings.refine)
        af.addRow("", self.refine)
        self.screen_text = QCheckBox("Give the chaptering model on-screen text "
                                     "(~8x larger prompt)")
        self.screen_text.setChecked(settings.include_screen_text)
        af.addRow("", self.screen_text)
        self.cache = QCheckBox("Reuse cached frame notes when inputs match")
        self.cache.setChecked(settings.use_cache)
        af.addRow("", self.cache)
        self.redact = QCheckBox("Black out credentials before frames are "
                                "uploaded")
        self.redact.setChecked(settings.redact)
        self.redact.toggled.connect(self._redact_changed)
        af.addRow("", self.redact)
        self.redact_warn = _note(
            "Frames will be sent to the API exactly as recorded, including any "
            "keys, tokens or .env files on screen. Only for footage you know "
            "is clean.", "Warning")
        af.addRow("", self.redact_warn)
        self.describe_model = QLineEdit(settings.describe_model)
        self.describe_model.setPlaceholderText("Default")
        af.addRow("Frame model", self.describe_model)
        self.chapter_model = QLineEdit(settings.chapter_model)
        self.chapter_model.setPlaceholderText("Default")
        af.addRow("Chapter model", self.chapter_model)
        lay.addWidget(self.adv)

        self.adv_toggle.toggled.connect(self._toggle_adv)
        self._toggle_adv(False)
        self._redact_changed(self.redact.isChecked())
        self._estimate()

    def _toggle_adv(self, on: bool) -> None:
        self.adv.setVisible(on)
        self.adv_toggle.setText(("▾ " if on else "▸ ") + "Advanced")
        self.window().adjustSize() if self.window() is not self else None

    def _redact_changed(self, on: bool) -> None:
        self.redact_warn.setVisible(not on)

    def set_total_duration(self, seconds: float) -> None:
        self.total_duration = seconds
        self._estimate()

    def _estimate(self) -> None:
        iv = self.interval.value()
        if self.total_duration and iv:
            self.estimate.setText(
                f"≈ {int(self.total_duration / iv) + 1} frames to describe")
        else:
            self.estimate.setText("Lower catches faster changes; each frame "
                                  "is one description call")

    def settings(self, base: Optional[RunSettings] = None) -> RunSettings:
        base = base or RunSettings()
        return replace(
            base,
            intent=self.intent.text().strip(),
            interval=float(self.interval.value()) or None,
            segmenter=self.segmenter.currentData(),
            min_chapter=float(self.min_chapter.value()) or None,
            transcribe=self.transcribe.isChecked(),
            redact=self.redact.isChecked(),
            refine=self.refine.isChecked(),
            include_screen_text=self.screen_text.isChecked(),
            describe_model=self.describe_model.text().strip(),
            chapter_model=self.chapter_model.text().strip(),
            use_cache=self.cache.isChecked(),
        )


def _credentials_note(root: str) -> Optional[QLabel]:
    if backend.credentials_source(root):
        return None
    return _note(
        "No ANTHROPIC_API_KEY found in the environment or a .env file. "
        "Analysis will still run, but on the visual channel only, with rough "
        "keyword titles. See Settings.", "Warning")


# ---------------------------------------------------------------------------
# Adding videos
# ---------------------------------------------------------------------------

class AddVideosDialog(QDialog):
    """Pick files, order them, decide whether they are one recording or many.

    The question "separate or together" is asked explicitly because the
    backend means something specific by "together": one continuous timeline,
    read in the order given, with a forced chapter break at every file join.
    """

    def __init__(self, files: List[str], root: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add videos")
        self.setMinimumWidth(640)
        self.root = root
        self._info = {}
        self.start = True

        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        head = QHBoxLayout()
        title = QLabel("Recordings")
        title.setObjectName("Title")
        head.addWidget(title)
        head.addStretch(1)
        for text, slot in [("Add…", self._browse), ("Sort by name", self._sort),
                           ("↑", lambda: self._move(-1)),
                           ("↓", lambda: self._move(1)),
                           ("Remove", self._remove)]:
            b = QToolButton()
            b.setText(text)
            b.clicked.connect(slot)
            head.addWidget(b)
        lay.addLayout(head)

        self.list = QListWidget()
        self.list.setDragDropMode(QAbstractItemView.InternalMove)
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.setMinimumHeight(110)
        self.list.model().rowsMoved.connect(self._changed)
        lay.addWidget(self.list)

        self.mode_box = QWidget()
        ml = QVBoxLayout(self.mode_box)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(4)
        self.separate = QRadioButton("Separate recordings — one library entry "
                                     "each")
        self.together = QRadioButton("One recording split across files — "
                                     "analyzed as a single timeline, in the "
                                     "order above")
        self.separate.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.separate)
        grp.addButton(self.together)
        ml.addWidget(self.separate)
        ml.addWidget(self.together)
        self.together_note = _note(
            "Each file join always starts a new chapter: a new file means the "
            "recording stopped and restarted.")
        ml.addWidget(self.together_note)
        nrow = QHBoxLayout()
        nrow.addWidget(QLabel("Name"))
        self.name = QLineEdit()
        nrow.addWidget(self.name, 1)
        self.name_row = QWidget()
        self.name_row.setLayout(nrow)
        ml.addWidget(self.name_row)
        self.together.toggled.connect(self._changed)
        lay.addWidget(self.mode_box)

        self.existing = _note("", "Subtle")
        lay.addWidget(self.existing)

        lay.addSpacing(4)
        self.form = RunSettingsForm(RunSettings())
        lay.addWidget(self.form)
        cred = _credentials_note(root)
        if cred:
            lay.addWidget(cred)

        btns = QDialogButtonBox()
        self.add_only = btns.addButton("Add", QDialogButtonBox.AcceptRole)
        self.add_run = btns.addButton("Add && analyze",
                                      QDialogButtonBox.AcceptRole)
        self.add_run.setObjectName("Primary")
        self.add_run.setDefault(True)
        btns.addButton(QDialogButtonBox.Cancel)
        self.add_only.clicked.connect(lambda: self._finish(False))
        self.add_run.clicked.connect(lambda: self._finish(True))
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

        self._add_files(sorted(files, key=natural_key))

    # -- file list -----------------------------------------------------------

    def _add_files(self, files: List[str]) -> None:
        present = set(self.files())
        for f in files:
            f = os.path.abspath(f)
            if f in present:
                continue
            info = self._info.get(f) or media.probe(f)
            self._info[f] = info
            bits = [format_duration(info.duration)] if info.duration else []
            if info.resolution:
                bits.append(info.resolution)
            bits.append("audio" if info.has_audio else "no audio")
            label = f"{info.name}    —    {' · '.join(bits)}"
            if info.error:
                label = f"{info.name}    —    ⚠ {info.error}"
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, f)
            it.setToolTip(f)
            self.list.addItem(it)
        self._changed()

    def files(self) -> List[str]:
        return [self.list.item(i).data(Qt.UserRole)
                for i in range(self.list.count())]

    def _browse(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add videos", "",
                                                VIDEO_FILTER)
        self._add_files(paths)

    def _sort(self) -> None:
        files = sorted(self.files(), key=natural_key)
        self.list.clear()
        self._add_files(files)

    def _move(self, d: int) -> None:
        row = self.list.currentRow()
        if row < 0 or not (0 <= row + d < self.list.count()):
            return
        it = self.list.takeItem(row)
        self.list.insertItem(row + d, it)
        self.list.setCurrentRow(row + d)
        self._changed()

    def _remove(self) -> None:
        for it in self.list.selectedItems():
            self.list.takeItem(self.list.row(it))
        self._changed()

    def _changed(self, *_a) -> None:
        files = self.files()
        multi = len(files) > 1
        self.mode_box.setVisible(multi)
        together = multi and self.together.isChecked()
        self.together_note.setVisible(together)
        self.name_row.setVisible(together)
        if files and not self.name.isModified():
            self.name.setText(os.path.splitext(os.path.basename(files[0]))[0]
                              + (f" + {len(files) - 1} more" if multi else ""))
        total = sum(self._info[f].duration for f in files if f in self._info)
        groups = self.groups()
        self.form.set_total_duration(
            total if together or not multi else
            max((self._info[f].duration for f in files if f in self._info),
                default=0.0))
        done = [g for g in groups if os.path.exists(backend.default_index_path(g))]
        if done:
            self.existing.setText(
                f"{len(done)} of these already "
                f"{'has an index' if len(done) == 1 else 'have indexes'} on "
                f"disk; {'it opens' if len(done) == 1 else 'they open'} as "
                f"Ready and won't be re-analyzed unless you ask.")
        self.existing.setVisible(bool(done))
        self.add_only.setEnabled(bool(files))
        self.add_run.setEnabled(bool(files))

    def groups(self) -> List[List[str]]:
        files = self.files()
        if len(files) > 1 and self.together.isChecked():
            return [files]
        return [[f] for f in files]

    def _finish(self, start: bool) -> None:
        self.start = start
        self.accept()

    def result_sessions(self) -> List[Session]:
        settings = self.form.settings()
        out: List[Session] = []
        for g in self.groups():
            name = (self.name.text().strip() if len(g) > 1 else
                    os.path.splitext(os.path.basename(g[0]))[0])
            out.append(Session(
                name=name or os.path.basename(g[0]),
                files=g,
                index_path=backend.default_index_path(g),
                settings=replace(settings),
                duration=sum(self._info[f].duration for f in g
                             if f in self._info),
            ))
        return out


# ---------------------------------------------------------------------------
# Re-analyze / re-chapter
# ---------------------------------------------------------------------------

class AnalyzeDialog(QDialog):
    """Confirm (and adjust) settings before a full analysis."""

    def __init__(self, session: Session, root: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Analyze")
        self.setMinimumWidth(560)
        self.session = session
        lay = QVBoxLayout(self)
        title = QLabel(session.name)
        title.setObjectName("Title")
        lay.addWidget(title)
        lay.addWidget(_note(
            "Reads frames from the video, describes them, then decides "
            "chapters. Frame notes are cached, so re-running with the same "
            "sampling only repeats the cheap second half."))
        self.form = RunSettingsForm(session.settings, session.duration)
        lay.addWidget(self.form)
        cred = _credentials_note(root)
        if cred:
            lay.addWidget(cred)
        btns = QDialogButtonBox(QDialogButtonBox.Cancel)
        go = btns.addButton("Analyze", QDialogButtonBox.AcceptRole)
        go.setObjectName("Primary")
        go.setDefault(True)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def settings(self) -> RunSettings:
        return self.form.settings(self.session.settings)


class RechapterDialog(QDialog):
    """Re-decide chapters from cached notes: one model call, no video read."""

    def __init__(self, session: Session, n_notes: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Re-chapter")
        self.setMinimumWidth(520)
        self.session = session
        s = session.settings
        lay = QVBoxLayout(self)
        title = QLabel("Re-chapter with a new intent")
        title.setObjectName("Title")
        lay.addWidget(title)
        lay.addWidget(_note(
            f"Re-decides chapters from the {n_notes} frame notes already "
            f"taken. No video is re-read: this is one model call over text, "
            f"usually seconds. The current chapters are replaced."))
        form = QFormLayout()
        self.intent = QLineEdit(s.intent)
        self.intent.setPlaceholderText("e.g. “find where I broke the build”")
        form.addRow("Intent", self.intent)
        self.min_chapter = QSpinBox()
        self.min_chapter.setRange(0, 3600)
        self.min_chapter.setSpecialValueText("Default")
        self.min_chapter.setSuffix(" s")
        self.min_chapter.setValue(int(s.min_chapter or 0))
        form.addRow("Shortest chapter", self.min_chapter)
        self.screen_text = QCheckBox("Include on-screen text (~8x larger "
                                     "prompt)")
        self.screen_text.setChecked(s.include_screen_text)
        form.addRow("", self.screen_text)
        self.model = QLineEdit(s.chapter_model)
        self.model.setPlaceholderText("Default")
        form.addRow("Chapter model", self.model)
        lay.addLayout(form)
        lay.addWidget(_note(
            "Chapters from this quick pass skip boundary refinement and the "
            "speech transcript; run a full analysis for those."))
        btns = QDialogButtonBox(QDialogButtonBox.Cancel)
        go = btns.addButton("Re-chapter", QDialogButtonBox.AcceptRole)
        go.setObjectName("Primary")
        go.setDefault(True)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)
        self.intent.setFocus()

    def settings(self) -> RunSettings:
        return replace(
            self.session.settings,
            intent=self.intent.text().strip(),
            min_chapter=float(self.min_chapter.value()) or None,
            include_screen_text=self.screen_text.isChecked(),
            chapter_model=self.model.text().strip(),
        )


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------

class Preferences:
    """Where the backend lives. Stored with QSettings."""

    def __init__(self) -> None:
        self._s = QSettings()

    @property
    def python(self) -> str:
        return str(self._s.value("backend/python", "") or backend.default_python())

    @property
    def root(self) -> str:
        return str(self._s.value("backend/root", "") or backend.default_root())

    def set(self, python: str, root: str) -> None:
        self._s.setValue("backend/python",
                         "" if python == backend.default_python() else python)
        self._s.setValue("backend/root",
                         "" if root == backend.default_root() else root)

    def backend(self) -> backend.Backend:
        return backend.Backend(python=self.python, root=self.root)


class SettingsDialog(QDialog):
    def __init__(self, prefs: Preferences, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(620)
        self.prefs = prefs
        lay = QVBoxLayout(self)
        title = QLabel("Backend")
        title.setObjectName("Title")
        lay.addWidget(title)
        lay.addWidget(_note(
            "The GUI runs analyses as `python -m cuts …` with this interpreter, "
            "from this folder. Point them elsewhere to use a different checkout "
            "or environment of the backend."))
        form = QFormLayout()
        self.python = QLineEdit(prefs.python)
        form.addRow("Python", self._with_browse(self.python, file=True))
        self.root = QLineEdit(prefs.root)
        self.root.textChanged.connect(self._update_creds)
        form.addRow("Backend folder", self._with_browse(self.root, file=False))
        self.creds = QLabel("")
        self.creds.setWordWrap(True)
        form.addRow("API key", self.creds)
        lay.addLayout(form)
        btns = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel
                                | QDialogButtonBox.RestoreDefaults)
        btns.accepted.connect(self._save)
        btns.rejected.connect(self.reject)
        btns.button(QDialogButtonBox.RestoreDefaults).clicked.connect(
            self._defaults)
        lay.addWidget(btns)
        self._update_creds()

    def _with_browse(self, edit: QLineEdit, file: bool) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, 1)
        b = QPushButton("Browse…")

        def pick() -> None:
            p = (QFileDialog.getOpenFileName(self, "Python interpreter",
                                             edit.text())[0] if file else
                 QFileDialog.getExistingDirectory(self, "Backend folder",
                                                  edit.text()))
            if p:
                edit.setText(p)
        b.clicked.connect(pick)
        h.addWidget(b)
        return w

    def _update_creds(self) -> None:
        src = backend.credentials_source(self.root.text())
        if src == "environment":
            self.creds.setText("Found in the environment")
            self.creds.setObjectName("Subtle")
        elif src:
            self.creds.setText(f"Found in {src}")
            self.creds.setObjectName("Subtle")
        else:
            self.creds.setText("Not found. Add ANTHROPIC_API_KEY=… to a .env "
                               "file in the backend folder, or set it in the "
                               "environment before launching.")
            self.creds.setObjectName("Warning")
        self.creds.style().polish(self.creds)

    def _defaults(self) -> None:
        self.python.setText(backend.default_python())
        self.root.setText(backend.default_root())

    def _save(self) -> None:
        self.prefs.set(self.python.text().strip(), self.root.text().strip())
        self.accept()
