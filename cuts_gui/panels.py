"""The side panels: library, chapters, notes, and run.

Each panel is written against `cuts_gui.model` only and talks to the window
through signals. None of them knows how an index was produced or where it
came from.
"""

from __future__ import annotations

import html
from typing import List, Optional

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QRect,
    QSize,
    QSortFilterProxyModel,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QFont, QGuiApplication, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableView,
    QTextBrowser,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import backend, theme
from .jobs import RunState
from .model import Index, Note, Session, Status, format_duration, format_time
from .thumbs import ThumbnailService, pixelate


def dot_icon(color: str, size: int = 10, hollow: bool = False) -> QIcon:
    pm = QPixmap(size * 2, size * 2)
    pm.setDevicePixelRatio(2.0)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    if hollow:
        p.setPen(QColor(color))
        p.setBrush(Qt.NoBrush)
    else:
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(color))
    p.drawEllipse(1, 1, size - 2, size - 2)
    p.end()
    return QIcon(pm)


def section_label(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setObjectName("SectionLabel")
    return lbl


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

_SID = Qt.UserRole + 1


class _SessionDelegate(QStyledItemDelegate):
    """Two-line library row: name, then status pill and summary."""

    def sizeHint(self, option, index) -> QSize:
        return QSize(option.rect.width(), 52)

    def paint(self, p: QPainter, option: QStyleOptionViewItem, index) -> None:
        s: Session = index.data(Qt.UserRole)
        if s is None:
            return
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        r = option.rect.adjusted(6, 2, -6, -2)
        if option.state & QStyle.State_Selected:
            p.setBrush(QColor("#2c3a52"))
        elif option.state & QStyle.State_MouseOver:
            p.setBrush(QColor(theme.HOVER))
        else:
            p.setBrush(Qt.NoBrush)
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(r, 6, 6)

        inner = r.adjusted(10, 6, -10, -6)
        bold = QFont(option.font)
        bold.setWeight(QFont.DemiBold)
        p.setFont(bold)
        p.setPen(QColor(theme.TEXT))
        fm = p.fontMetrics()
        p.drawText(QRect(inner.x(), inner.y(), inner.width(), fm.height()),
                   Qt.AlignLeft | Qt.AlignVCenter,
                   fm.elidedText(s.name, Qt.ElideRight, inner.width()))

        small = QFont(option.font)
        small.setPointSizeF(max(8.0, option.font.pointSizeF() - 1.5))
        p.setFont(small)
        fm = p.fontMetrics()
        y = inner.bottom() - fm.height()
        status = s.status.value
        color = QColor(theme.STATUS_COLORS.get(status, theme.MUTED))
        if s.redaction_warning and s.status == Status.READY:
            color = QColor(theme.WARN)
        p.setBrush(color)
        p.setPen(Qt.NoPen)
        p.drawEllipse(inner.x(), y + fm.height() // 2 - 3, 7, 7)
        bits = [theme.STATUS_LABELS.get(status, status)]
        if s.message and s.status != Status.RUNNING:
            bits.append(s.message)
        if s.is_multi:
            bits.append(f"{len(s.files)} files")
        if s.duration:
            bits.append(format_duration(s.duration))
        if s.missing_files:
            bits = ["Missing file"] + bits[1:]
            p.setBrush(QColor(theme.DANGER))
        p.setPen(QColor(theme.MUTED))
        text = "  ·  ".join(bits)
        p.drawText(QRect(inner.x() + 13, y, inner.width() - 13, fm.height()),
                   Qt.AlignLeft | Qt.AlignVCenter,
                   fm.elidedText(text, Qt.ElideRight, inner.width() - 13))
        p.restore()


class LibraryPanel(QWidget):
    session_selected = Signal(str)
    add_requested = Signal()
    open_index_requested = Signal()
    # (action, session_id): "analyze", "rechapter", "cancel", "reveal",
    # "rename", "remove", "relink"
    action_requested = Signal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setAttribute(Qt.WA_StyledBackground, True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 10, 0, 6)
        lay.setSpacing(8)

        head = QHBoxLayout()
        head.setContentsMargins(14, 0, 10, 0)
        title = QLabel("Library")
        title.setObjectName("Title")
        head.addWidget(title)
        head.addStretch(1)
        open_btn = QToolButton()
        open_btn.setText("Open index…")
        open_btn.setToolTip("Open an existing .events.json index "
                            "(e.g. one made from the command line)")
        open_btn.clicked.connect(self.open_index_requested)
        head.addWidget(open_btn)
        lay.addLayout(head)

        add = QPushButton("＋  Add videos…")
        add.setObjectName("Primary")
        add.clicked.connect(self.add_requested)
        wrap = QHBoxLayout()
        wrap.setContentsMargins(12, 0, 12, 0)
        wrap.addWidget(add)
        lay.addLayout(wrap)

        self.list = QListWidget()
        self.list.setItemDelegate(_SessionDelegate(self.list))
        self.list.setMouseTracking(True)
        self.list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._menu)
        self.list.currentItemChanged.connect(
            lambda cur, _prev: cur and self.session_selected.emit(
                cur.data(_SID)))
        lay.addWidget(self.list, 1)

        self.empty = QLabel("Drop recordings anywhere\nin this window to start.")
        self.empty.setObjectName("Faint")
        self.empty.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.empty)

    def set_sessions(self, sessions: List[Session], current: str = "") -> None:
        self.list.blockSignals(True)
        self.list.clear()
        for s in sessions:
            it = QListWidgetItem()
            it.setData(Qt.UserRole, s)
            it.setData(_SID, s.id)
            it.setToolTip("\n".join(s.files))
            self.list.addItem(it)
            if s.id == current:
                self.list.setCurrentItem(it)
        self.list.blockSignals(False)
        self.empty.setVisible(not sessions)

    def refresh(self) -> None:
        self.list.viewport().update()

    def select(self, session_id: str) -> None:
        for i in range(self.list.count()):
            if self.list.item(i).data(_SID) == session_id:
                self.list.setCurrentRow(i)
                return

    def _menu(self, pos) -> None:
        it = self.list.itemAt(pos)
        if it is None:
            return
        s: Session = it.data(Qt.UserRole)
        m = QMenu(self)
        busy = s.status in (Status.RUNNING, Status.QUEUED)
        entries = [
            ("Cancel run" if busy else
             ("Re-analyze…" if s.status == Status.READY else "Analyze…"),
             "cancel" if busy else "analyze"),
            ("Re-chapter with a new intent…", "rechapter"),
            None,
            ("Rename…", "rename"),
            ("Locate missing file…", "relink") if s.missing_files else None,
            ("Show in Finder", "reveal"),
            None,
            ("Remove from library", "remove"),
        ]
        for e in entries:
            if e is None:
                m.addSeparator()
                continue
            label, action = e
            act = m.addAction(label)
            act.triggered.connect(
                lambda _=False, a=action: self.action_requested.emit(a, s.id))
            if action == "rechapter":
                act.setEnabled(s.status == Status.READY and not busy)
        m.exec(self.list.viewport().mapToGlobal(pos))


# ---------------------------------------------------------------------------
# Chapters
# ---------------------------------------------------------------------------

class ChaptersPanel(QWidget):
    chapter_activated = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.index: Optional[Index] = None
        self._playing = -1
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        bar = QHBoxLayout()
        bar.setContentsMargins(12, 8, 8, 8)
        self.summary = QLabel("")
        self.summary.setObjectName("Subtle")
        bar.addWidget(self.summary, 1)
        self.copy_btn = QToolButton()
        self.copy_btn.setText("Copy list")
        self.copy_btn.setPopupMode(QToolButton.MenuButtonPopup)
        self.copy_btn.setToolTip("Copy the chapter list — the format that "
                                 "pastes into a video description")
        self.copy_btn.clicked.connect(lambda: self._copy(False))
        menu = QMenu(self.copy_btn)
        menu.addAction("Copy list", lambda: self._copy(False))
        menu.addAction("Copy list with file offsets", lambda: self._copy(True))
        menu.addSeparator()
        menu.addAction("Save list as text…", self._save)
        self.copy_btn.setMenu(menu)
        bar.addWidget(self.copy_btn)
        lay.addLayout(bar)

        split = QSplitter(Qt.Vertical)
        split.setChildrenCollapsible(False)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Start", "Chapter", "Length"])
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(True)
        hdr = self.tree.header()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tree.itemSelectionChanged.connect(self._selected)
        self.tree.itemActivated.connect(
            lambda it, _c: self.chapter_activated.emit(
                self.tree.indexOfTopLevelItem(it)))
        split.addWidget(self.tree)

        self.detail = QTextBrowser()
        self.detail.setOpenLinks(False)
        self.detail.setFrameShape(QTextBrowser.NoFrame)
        split.addWidget(self.detail)
        split.setSizes([320, 260])
        lay.addWidget(split, 1)

    def set_index(self, index: Optional[Index]) -> None:
        self.index = index
        self._playing = -1
        self.tree.blockSignals(True)
        self.tree.clear()
        chapters = index.chapters if index else []
        for k, c in enumerate(chapters):
            it = QTreeWidgetItem([format_time(c.start), c.display_title,
                                  format_duration(c.duration)])
            it.setIcon(0, dot_icon(theme.kind_color(c.kind, k).name()))
            it.setToolTip(1, c.reason or c.display_title)
            it.setTextAlignment(2, Qt.AlignRight | Qt.AlignVCenter)
            if c.kind:
                it.setToolTip(0, theme.KIND_LABELS.get(c.kind, c.kind))
            self.tree.addTopLevelItem(it)
        self.tree.blockSignals(False)
        self.copy_btn.setEnabled(bool(chapters))
        if index and chapters:
            kinds = {}
            for c in chapters:
                if c.kind:
                    kinds[c.kind] = kinds.get(c.kind, 0) + 1
            extra = (" · " + ", ".join(f"{n} {k}" for k, n in kinds.items())
                     if kinds else "")
            self.summary.setText(f"{len(chapters)} chapters{extra}")
            self._show_detail(-1)
        else:
            self.summary.setText("No chapters yet")
            self.detail.setHtml(
                f"<p style='color:{theme.MUTED}'>Chapters appear here once "
                f"the session has been analyzed.</p>")

    def select(self, i: int) -> None:
        if 0 <= i < self.tree.topLevelItemCount():
            self.tree.blockSignals(True)
            self.tree.setCurrentItem(self.tree.topLevelItem(i))
            self.tree.scrollToItem(self.tree.topLevelItem(i))
            self.tree.blockSignals(False)
            self._show_detail(i)

    def set_playing(self, i: int) -> None:
        """Mark the chapter under the playhead, without moving the selection."""
        if i == self._playing:
            return
        for k in (self._playing, i):
            it = self.tree.topLevelItem(k) if k >= 0 else None
            if it is not None:
                for col in range(3):
                    f = it.font(col)
                    f.setBold(k == i)
                    it.setFont(col, f)
        self._playing = i

    def _selected(self) -> None:
        it = self.tree.currentItem()
        if it is None:
            return
        i = self.tree.indexOfTopLevelItem(it)
        self._show_detail(i)
        self.chapter_activated.emit(i)

    def _show_detail(self, i: int) -> None:
        if not self.index or not (0 <= i < len(self.index.chapters)):
            self.detail.setHtml(
                f"<p style='color:{theme.MUTED}'>Select a chapter to see the "
                f"evidence behind it.</p>")
            return
        c = self.index.chapters[i]
        e = html.escape
        col = theme.kind_color(c.kind, i).name()
        parts = [f"<h3 style='margin:0'>{e(c.display_title)}</h3>",
                 f"<p style='color:{theme.MUTED};margin-top:4px'>"
                 f"{format_time(c.start)} – {format_time(c.end)} · "
                 f"{format_duration(c.duration)}"]
        if c.kind:
            parts.append(f" · <span style='color:{col}'>● {e(c.kind)}</span>")
        if self.index.is_multi and 0 <= c.source_index < len(self.index.sources):
            parts.append(f"<br>{e(self.index.sources[c.source_index].name)} "
                         f"@ {format_time(c.source_time)}")
        parts.append("</p>")
        if c.reason:
            parts.append(f"<p><b>Why it starts here</b><br>{e(c.reason)}</p>")
        if c.descriptions:
            items = "".join(f"<li>{e(d)}</li>" for d in c.descriptions)
            parts.append(f"<p><b>On screen</b></p>"
                         f"<ul style='margin-left:-20px'>{items}</ul>")
        if c.transcript:
            parts.append(f"<p><b>Said</b><br><i>{e(c.transcript)}</i></p>")
        self.detail.setHtml("".join(parts))

    def _copy(self, with_files: bool) -> None:
        if self.index:
            QGuiApplication.clipboard().setText(
                backend.chapter_list(self.index, show_source=with_files))

    def _save(self) -> None:
        if not self.index:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save chapter list",
            self.index.index_path.replace(backend.INDEX_SUFFIX, ".chapters.txt"),
            "Text (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(backend.chapter_list(
                    self.index, show_source=self.index.is_multi) + "\n")


# ---------------------------------------------------------------------------
# Notes (per-frame observations)
# ---------------------------------------------------------------------------

class _ClickLabel(QLabel):
    clicked = Signal()

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(ev)


class _NotesModel(QAbstractTableModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.notes: List[Note] = []
        self.current = -1
        self._icons = {
            "ok": dot_icon(theme.MUTED, 8),
            "redacted": dot_icon(theme.DANGER, 8),
            "withheld": dot_icon(theme.DANGER, 8, hollow=True),
        }

    def set_notes(self, notes: List[Note]) -> None:
        self.beginResetModel()
        self.notes, self.current = notes, -1
        self.endResetModel()

    def set_current(self, row: int) -> None:
        old, self.current = self.current, row
        for r in (old, row):
            if 0 <= r < len(self.notes):
                self.dataChanged.emit(self.index(r, 0), self.index(r, 1))

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.notes)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 2

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return ["Time", "Note"][section]
        return None

    def data(self, index, role=Qt.DisplayRole):
        n = self.notes[index.row()]
        col = index.column()
        if role == Qt.DisplayRole:
            if col == 0:
                return format_time(n.t)
            if n.withheld:
                return f"({n.missing})"
            return n.description or "(no note)"
        if role == Qt.DecorationRole and col == 0:
            if n.status == "withheld":
                return self._icons["withheld"]
            return self._icons["redacted" if n.redacted else "ok"]
        if role == Qt.ForegroundRole and col == 1 and (n.withheld or not n.description):
            return QColor(theme.FAINT)
        if role == Qt.FontRole and index.row() == self.current:
            f = QFont()
            f.setBold(True)
            return f
        if role == Qt.ToolTipRole and col == 1:
            return n.description
        if role == Qt.UserRole:
            return index.row()
        return None


class _NotesFilter(QSortFilterProxyModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.text = ""
        self.flagged_only = False

    def filterAcceptsRow(self, row, parent) -> bool:
        n: Note = self.sourceModel().notes[row]
        if self.flagged_only and not n.flagged:
            return False
        if not self.text:
            return True
        hay = f"{n.description}\n{n.screen_text}\n{n.transcript}".lower()
        return all(w in hay for w in self.text.lower().split())


class NotesPanel(QWidget):
    note_activated = Signal(int)

    def __init__(self, thumbs: ThumbnailService, parent=None) -> None:
        super().__init__(parent)
        self.thumbs = thumbs
        self.index: Optional[Index] = None
        self._shown = -1
        thumbs.ready.connect(lambda _k: self._shown >= 0 and self._refresh_thumb())
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        bar = QHBoxLayout()
        bar.setContentsMargins(10, 8, 10, 8)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search notes and on-screen text…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        bar.addWidget(self.search, 1)
        self.flagged = QCheckBox("Flagged")
        self.flagged.setToolTip("Only frames where credentials were blacked "
                                "out, or which were withheld")
        self.flagged.toggled.connect(self._filter)
        bar.addWidget(self.flagged)
        lay.addLayout(bar)

        self.unavailable = QLabel("")
        self.unavailable.setWordWrap(True)
        self.unavailable.setObjectName("Subtle")
        self.unavailable.setContentsMargins(14, 10, 14, 10)
        self.unavailable.hide()
        lay.addWidget(self.unavailable)

        split = QSplitter(Qt.Vertical)
        split.setChildrenCollapsible(False)
        self.model = _NotesModel(self)
        self.proxy = _NotesFilter(self)
        self.proxy.setSourceModel(self.model)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table.verticalHeader().setDefaultSectionSize(24)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.selectionModel().currentRowChanged.connect(self._row_changed)
        split.addWidget(self.table)

        insp = QWidget()
        il = QVBoxLayout(insp)
        il.setContentsMargins(12, 10, 12, 10)
        il.setSpacing(6)
        self._revealed = False
        self.thumb = _ClickLabel()
        self.thumb.clicked.connect(self._reveal)
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setMinimumHeight(120)
        self.thumb.setStyleSheet(f"background:{theme.BG}; border-radius:6px;")
        il.addWidget(self.thumb)
        self.meta = QLabel("")
        self.meta.setObjectName("Subtle")
        self.meta.setTextFormat(Qt.RichText)
        il.addWidget(self.meta)
        self.desc = QLabel("")
        self.desc.setWordWrap(True)
        self.desc.setTextInteractionFlags(Qt.TextSelectableByMouse)
        il.addWidget(self.desc)
        self.ocr_toggle = QToolButton()
        self.ocr_toggle.setObjectName("Flat")
        self.ocr_toggle.setCheckable(True)
        self.ocr_toggle.setText("▸ On-screen text (OCR)")
        self.ocr_toggle.toggled.connect(self._toggle_ocr)
        il.addWidget(self.ocr_toggle, 0, Qt.AlignLeft)
        self.ocr = QPlainTextEdit()
        self.ocr.setReadOnly(True)
        self.ocr.setFont(theme.mono_font(10))
        self.ocr.hide()
        il.addWidget(self.ocr, 1)
        il.addStretch(0)
        split.addWidget(insp)
        split.setSizes([300, 320])
        self.split = split
        lay.addWidget(split, 1)
        self._show(-1)

    def set_index(self, index: Optional[Index]) -> None:
        self.index = index
        notes = index.notes if index else []
        self.model.set_notes(notes)
        has = bool(notes)
        self.split.setVisible(has)
        self.search.setEnabled(has)
        self.flagged.setEnabled(has)
        msg = ""
        if index and not has:
            msg = index.notes_unavailable or "No per-frame notes yet."
        elif not index:
            msg = "Frame notes appear here once the session has been analyzed."
        self.unavailable.setText(msg)
        self.unavailable.setVisible(bool(msg))
        if has:
            flagged = sum(1 for n in notes if n.flagged)
            self.flagged.setText(f"Flagged ({flagged})")
        self._show(-1)

    def select(self, i: int) -> None:
        if not self.index or not (0 <= i < len(self.index.notes)):
            return
        pidx = self.proxy.mapFromSource(self.model.index(i, 0))
        if pidx.isValid():
            self.table.selectionModel().blockSignals(True)
            self.table.setCurrentIndex(pidx)
            self.table.scrollTo(pidx, QAbstractItemView.PositionAtCenter)
            self.table.selectionModel().blockSignals(False)
        self._show(i)

    def set_playing(self, i: int, follow: bool) -> None:
        if i == self.model.current:
            return
        self.model.set_current(i)
        if follow and i >= 0:
            pidx = self.proxy.mapFromSource(self.model.index(i, 0))
            if pidx.isValid():
                self.table.scrollTo(pidx, QAbstractItemView.EnsureVisible)

    def focus_search(self) -> None:
        self.search.setFocus()
        self.search.selectAll()

    def _filter(self) -> None:
        self.proxy.text = self.search.text().strip()
        self.proxy.flagged_only = self.flagged.isChecked()
        self.proxy.invalidateFilter()

    def _row_changed(self, cur, _prev) -> None:
        if not cur.isValid():
            return
        i = self.proxy.mapToSource(cur).row()
        self._show(i)
        self.note_activated.emit(i)

    def _toggle_ocr(self, on: bool) -> None:
        self.ocr.setVisible(on)
        self.ocr_toggle.setText(("▾ " if on else "▸ ") + "On-screen text (OCR)")

    def _reveal(self) -> None:
        if not self._revealed and self.index and \
                0 <= self._shown < len(self.index.notes) and \
                self.index.notes[self._shown].flagged:
            self._revealed = True
            self._refresh_thumb()

    def _show(self, i: int) -> None:
        if i != self._shown:
            self._revealed = False
        self._shown = i
        if not self.index or not (0 <= i < len(self.index.notes)):
            self.thumb.setPixmap(QPixmap())
            self.thumb.setText("Select a note")
            self.thumb.setStyleSheet(f"background:{theme.BG}; border-radius:6px;"
                                     f"color:{theme.FAINT}")
            self.meta.setText("")
            self.desc.setText("")
            self.ocr_toggle.hide()
            self.ocr.hide()
            return
        n = self.index.notes[i]
        e = html.escape
        meta = [f"<b style='color:{theme.TEXT}'>{format_time(n.t)}</b>"]
        if self.index.is_multi and 0 <= n.source_index < len(self.index.sources):
            meta.append(f"{e(self.index.sources[n.source_index].name)} @ "
                        f"{format_time(n.source_time)}")
        if n.redacted:
            meta.append(f"<span style='color:{theme.DANGER}'>credentials blacked "
                        f"out ({e(', '.join(sorted(set(n.redacted))))})</span>")
        if n.withheld:
            col = theme.DANGER if n.status == "withheld" else theme.MUTED
            meta.append(f"<span style='color:{col}'>{e(n.missing)}</span>")
        self.meta.setText(" · ".join(meta))
        self.desc.setText(n.description or "")
        if n.transcript:
            self.desc.setText(f"{n.description}\n\n“{n.transcript}”")
        self.ocr_toggle.setVisible(bool(n.screen_text))
        self.ocr.setPlainText(n.screen_text)
        self.ocr.setVisible(self.ocr_toggle.isChecked() and bool(n.screen_text))
        self._refresh_thumb()

    def _refresh_thumb(self) -> None:
        if not self.index or not (0 <= self._shown < len(self.index.notes)):
            return
        n = self.index.notes[self._shown]
        if not (0 <= n.source_index < len(self.index.sources)):
            return
        src = self.index.sources[n.source_index]
        w = max(240, self.thumb.width())
        dpr = self.devicePixelRatioF()
        img = self.thumbs.get(src.path, round(n.source_time, 1), int(w * dpr)) \
            if src.exists else None
        if img is None:
            self.thumb.setPixmap(QPixmap())
            self.thumb.setText("loading…" if src.exists else "video file missing")
            return
        # This image is decoded fresh from the local file, so it is NOT the
        # redacted frame that was uploaded — the credential is right there in
        # the pixels. Flagged frames are therefore blurred until asked for:
        # the "Flagged" filter otherwise puts every secret in the recording
        # front and centre, which is the last thing you want on a shared
        # screen.
        hide = n.flagged and not self._revealed
        if hide:
            img = pixelate(img)
        pm = QPixmap.fromImage(img)
        pm.setDevicePixelRatio(dpr)
        if hide:
            p = QPainter(pm)
            p.fillRect(pm.rect(), QColor(0, 0, 0, 110))
            p.setPen(QColor(theme.TEXT))
            p.drawText(QRect(0, 0, int(pm.width() / dpr), int(pm.height() / dpr)),
                       Qt.AlignCenter,
                       "Credentials were on screen here\nClick to reveal")
            p.end()
        self.thumb.setCursor(Qt.PointingHandCursor if hide else Qt.ArrowCursor)
        self.thumb.setText("")
        self.thumb.setPixmap(pm)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

class RunPanel(QWidget):
    cancel_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._sid = ""
        self._lines_shown = 0
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(8)

        top = QHBoxLayout()
        self.title = QLabel("No run yet")
        self.title.setObjectName("Title")
        top.addWidget(self.title, 1)
        self.elapsed = QLabel("")
        self.elapsed.setObjectName("Subtle")
        top.addWidget(self.elapsed)
        self.cancel = QPushButton("Cancel")
        self.cancel.setObjectName("DangerButton")
        self.cancel.clicked.connect(self.cancel_requested)
        top.addWidget(self.cancel)
        lay.addLayout(top)

        self.stages = QLabel("")
        self.stages.setTextFormat(Qt.RichText)
        lay.addWidget(self.stages)

        self.warnings = QLabel("")
        self.warnings.setWordWrap(True)
        self.warnings.setTextFormat(Qt.RichText)
        lay.addWidget(self.warnings)

        lay.addWidget(section_label("Equivalent command"))
        cmd = QHBoxLayout()
        self.command = QLineEdit()
        self.command.setReadOnly(True)
        self.command.setFont(theme.mono_font(10))
        cmd.addWidget(self.command, 1)
        copy = QToolButton()
        copy.setText("Copy")
        copy.clicked.connect(
            lambda: QGuiApplication.clipboard().setText(self.command.text()))
        cmd.addWidget(copy)
        lay.addLayout(cmd)

        lay.addWidget(section_label("Log"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(theme.mono_font(10))
        self.log.setMaximumBlockCount(5000)
        # Wrapped: the first line is a long command, and an unwrapped view
        # scrolls sideways to show it, leaving every short line off-screen.
        self.log.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        lay.addWidget(self.log, 1)
        self.show_state("", None)

    def show_state(self, session_id: str, st: Optional[RunState]) -> None:
        if session_id != self._sid:
            self._sid = session_id
            self._lines_shown = 0
            self.log.clear()
        if st is None:
            self.title.setText("No run in this window yet")
            self.elapsed.setText("")
            self.cancel.hide()
            self.stages.setText(
                f"<span style='color:{theme.MUTED}'>Analysis output for the "
                f"selected session appears here while it runs.</span>")
            self.warnings.setText("")
            self.command.setText("")
            return

        running = st.active or st.queued
        if st.queued:
            self.title.setText(f"{st.label} — queued")
        elif st.active:
            self.title.setText(st.label)
        elif st.error:
            self.title.setText("Failed")
        else:
            self.title.setText("Finished")
        self.cancel.setVisible(running)
        self.elapsed.setText(format_duration(st.elapsed) if st.started else "")
        self.command.setText(st.command)
        self.command.setCursorPosition(0)

        rows = []
        for key, label in backend.STAGES:
            if key in st.done:
                icon, col, tail = "●", theme.OK, f"{st.done[key]:.0f}s"
            elif st.active and st.stage == key:
                icon, col, tail = "◐", theme.ACCENT, html.escape(st.detail)
            elif key in st.seen:
                icon, col, tail = "●", theme.OK, ""
            else:
                icon, col, tail = "○", theme.FAINT, ""
            rows.append(
                f"<tr><td style='color:{col};padding-right:8px'>{icon}</td>"
                f"<td style='padding-right:14px'>{label}</td>"
                f"<td style='color:{theme.MUTED}'>{tail}</td></tr>")
        self.stages.setText(f"<table cellspacing=2>{''.join(rows)}</table>")

        msgs = []
        if st.error:
            msgs.append(f"<p style='color:{theme.DANGER}'>✕ "
                        f"{html.escape(st.error)}</p>")
        for w in st.warnings:
            col = theme.DANGER if w.startswith("Credentials") else theme.WARN
            extra = (" — rotate whatever was on screen; redaction reduces "
                     "exposure, it does not guarantee it."
                     if w.startswith("Credentials") else "")
            msgs.append(f"<p style='color:{col}'>⚠ {html.escape(w)}{extra}</p>")
        self.warnings.setText("".join(msgs))

        if self._lines_shown > st.total_lines:
            self.log.clear()
            self._lines_shown = 0
        new = st.lines_since(self._lines_shown)
        if new:
            bar = self.log.verticalScrollBar()
            at_end = bar.value() >= bar.maximum() - 4
            self.log.appendPlainText("\n".join(new))
            self._lines_shown = st.total_lines
            if at_end:
                bar.setValue(bar.maximum())
