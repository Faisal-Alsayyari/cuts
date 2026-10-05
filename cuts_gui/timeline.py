"""The per-session timeline: where the chapters are, and which frames have notes.

Lanes, top to bottom:

    ruler     time ticks, an overview of the visible window, the playhead
    files     one block per source file (only when there are several)
    chapters  one block per chapter, coloured by kind, titled when it fits
    notes     one tick per sampled frame; red where credentials were
              blacked out, hollow where the frame was withheld
    frames    a filmstrip of what the footage looks like at each point

Navigation follows video-editor conventions: click or drag anywhere to scrub,
scroll to pan, Ctrl/Cmd-scroll or pinch to zoom around the pointer,
double-click a chapter to zoom to it and empty space to fit everything.

Everything is painted directly rather than built from child widgets: a
two-hour session at a 30s interval is a few hundred note ticks and a few
dozen chapters, and laying those out as widgets would make zooming the
expensive part.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget

from . import theme
from .model import Index, format_duration, format_time
from .thumbs import ThumbnailService

GUTTER = 74          # lane labels
RIGHT_PAD = 10
OVERVIEW_H = 5
RULER_H = 26
FILES_H = 20
CHAPTERS_H = 46
NOTES_H = 30
FRAMES_H = 58
LANE_GAP = 4
MIN_SPAN = 5.0       # seconds — deepest zoom
_NICE_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600,
               7200, 14400]
# Minor tick per major step, chosen so minors land on clock-friendly values.
_MINOR = {15: 5, 30: 10, 60: 15, 120: 30, 300: 60, 600: 120, 900: 300,
          1800: 300, 3600: 900, 7200: 1800, 14400: 3600}


def _nice_step(min_seconds: float) -> float:
    for s in _NICE_STEPS:
        if s >= min_seconds:
            return float(s)
    return float(_NICE_STEPS[-1])


class TimelineView(QWidget):
    # Global seconds the user clicked or dragged to.
    seek_requested = Signal(float)
    # Index of a chapter / note the user picked on the timeline.
    chapter_picked = Signal(int)
    note_picked = Signal(int)

    def __init__(self, thumbs: ThumbnailService, parent=None) -> None:
        super().__init__(parent)
        self.thumbs = thumbs
        self.thumbs.ready.connect(lambda _k: self._frames_visible and self.update())
        self.index: Optional[Index] = None
        self._v0, self._v1 = 0.0, 1.0
        self._playhead = 0.0
        self._hover_x: Optional[float] = None
        self._sel_chapter = -1
        self._sel_note = -1
        self._scrubbing = False
        self._frames_visible = True
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setAttribute(Qt.WA_AcceptTouchEvents, False)
        self._update_height()

    # -- public API -------------------------------------------------------------

    def set_index(self, index: Optional[Index]) -> None:
        self.index = index
        self._sel_chapter = self._sel_note = -1
        self._playhead = 0.0
        self._update_height()
        self.zoom_to_fit()

    def set_playhead(self, t: float, follow: bool = False) -> None:
        if abs(t - self._playhead) < 1e-3:
            return
        self._playhead = t
        if follow:
            span = self._v1 - self._v0
            if t > self._v1 - span * 0.02 or t < self._v0:
                # Page rather than scroll: a continuously sliding timeline is
                # unreadable, and this keeps it still most of the time.
                self._set_view(t - span * 0.1, t + span * 0.9)
                return
        self.update()

    def select_chapter(self, i: int) -> None:
        self._sel_chapter = i
        self.update()

    def select_note(self, i: int, reveal: bool = False) -> None:
        self._sel_note = i
        if reveal and self.index and 0 <= i < len(self.index.notes):
            self.reveal(self.index.notes[i].t)
        self.update()

    def reveal(self, t: float) -> None:
        """Pan just enough to bring `t` into view."""
        span = self._v1 - self._v0
        if not (self._v0 <= t <= self._v1):
            self._set_view(t - span / 2, t + span / 2)

    def zoom_to_fit(self) -> None:
        self._set_view(0.0, self._duration)

    def zoom_to(self, t0: float, t1: float) -> None:
        pad = (t1 - t0) * 0.04
        self._set_view(t0 - pad, t1 + pad)

    def zoom_by(self, factor: float, anchor: Optional[float] = None) -> None:
        anchor = self._playhead if anchor is None else anchor
        span = self._v1 - self._v0
        new = max(MIN_SPAN, min(self._duration, span * factor))
        frac = (anchor - self._v0) / span if span else 0.5
        frac = min(1.0, max(0.0, frac))
        self._set_view(anchor - new * frac, anchor - new * frac + new)

    def set_frames_visible(self, visible: bool) -> None:
        self._frames_visible = visible
        self._update_height()
        self.update()

    # -- geometry ---------------------------------------------------------------

    @property
    def _duration(self) -> float:
        return max(1.0, self.index.duration) if self.index else 1.0

    def _lanes(self) -> List[Tuple[str, float, float]]:
        """(name, top, height) for every lane currently shown."""
        y = OVERVIEW_H + RULER_H + LANE_GAP
        lanes: List[Tuple[str, float, float]] = []
        if self.index and self.index.is_multi:
            lanes.append(("files", y, FILES_H))
            y += FILES_H + LANE_GAP
        lanes.append(("chapters", y, CHAPTERS_H))
        y += CHAPTERS_H + LANE_GAP
        lanes.append(("notes", y, NOTES_H))
        y += NOTES_H + LANE_GAP
        if self._frames_visible and self.thumbs.enabled:
            lanes.append(("frames", y, FRAMES_H))
            y += FRAMES_H + LANE_GAP
        return lanes

    def _update_height(self) -> None:
        lanes = self._lanes()
        _, top, h = lanes[-1]
        self.setFixedHeight(int(top + h + 8))

    def _lane_at(self, y: float) -> str:
        if y < OVERVIEW_H + RULER_H:
            return "ruler"
        for name, top, h in self._lanes():
            if top - LANE_GAP / 2 <= y < top + h + LANE_GAP / 2:
                return name
        return ""

    @property
    def _w(self) -> float:
        return max(1.0, self.width() - GUTTER - RIGHT_PAD)

    def _x(self, t: float) -> float:
        return GUTTER + (t - self._v0) / (self._v1 - self._v0) * self._w

    def _t(self, x: float) -> float:
        return self._v0 + (x - GUTTER) / self._w * (self._v1 - self._v0)

    def _set_view(self, t0: float, t1: float) -> None:
        span = max(MIN_SPAN, min(t1 - t0, self._duration))
        t0 = max(0.0, min(t0, self._duration - span))
        self._v0, self._v1 = t0, t0 + span
        self.update()

    # -- hit testing ------------------------------------------------------------

    def _chapter_at_x(self, x: float) -> int:
        if not self.index:
            return -1
        t = self._t(x)
        for i, c in enumerate(self.index.chapters):
            if c.start <= t < c.end:
                return i
        return -1

    def _note_at_x(self, x: float, radius: float = 6.0) -> int:
        if not self.index or not self.index.notes:
            return -1
        best, best_d = -1, radius + 1
        t = self._t(x)
        # Notes are sorted; binary-search the neighbourhood instead of
        # scanning all of them on every mouse move.
        lo, hi = 0, len(self.index.notes)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.index.notes[mid].t < t:
                lo = mid + 1
            else:
                hi = mid
        for i in range(max(0, lo - 2), min(len(self.index.notes), lo + 2)):
            d = abs(self._x(self.index.notes[i].t) - x)
            if d < best_d:
                best, best_d = i, d
        return best if best_d <= radius else -1

    # -- events -----------------------------------------------------------------

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        if ev.button() != Qt.LeftButton or not self.index:
            return
        x, lane = ev.position().x(), self._lane_at(ev.position().y())
        if x < GUTTER:
            return
        if lane == "ruler" and ev.position().y() < OVERVIEW_H + 2:
            # The overview strip: jump the visible window there.
            frac = (x - GUTTER) / self._w
            t = frac * self._duration
            span = self._v1 - self._v0
            self._set_view(t - span / 2, t + span / 2)
            return
        if lane == "notes":
            i = self._note_at_x(x)
            if i >= 0:
                self._sel_note = i
                self.note_picked.emit(i)
                self.seek_requested.emit(self.index.notes[i].t)
                self.update()
                return
        if lane == "chapters":
            i = self._chapter_at_x(x)
            if i >= 0:
                self._sel_chapter = i
                self.chapter_picked.emit(i)
        self._scrubbing = True
        self.seek_requested.emit(min(max(0.0, self._t(x)), self._duration))

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        x = ev.position().x()
        self._hover_x = x if x >= GUTTER else None
        if self._scrubbing and self.index:
            self.seek_requested.emit(min(max(0.0, self._t(x)), self._duration))
        else:
            self._tooltip(ev)
        self.update()

    def mouseReleaseEvent(self, _ev: QMouseEvent) -> None:
        self._scrubbing = False

    def mouseDoubleClickEvent(self, ev: QMouseEvent) -> None:
        if not self.index:
            return
        lane = self._lane_at(ev.position().y())
        i = self._chapter_at_x(ev.position().x()) if lane == "chapters" else -1
        if i >= 0:
            c = self.index.chapters[i]
            self.zoom_to(c.start, c.end)
        else:
            self.zoom_to_fit()

    def leaveEvent(self, _ev) -> None:
        self._hover_x = None
        self.update()

    def wheelEvent(self, ev: QWheelEvent) -> None:
        if not self.index:
            return
        px, ang = ev.pixelDelta(), ev.angleDelta()
        dx = px.x() if not px.isNull() else ang.x() / 4
        dy = px.y() if not px.isNull() else ang.y() / 4
        if ev.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
            self.zoom_by(math.pow(0.996, dy), self._t(ev.position().x()))
        else:
            # Either axis pans: there is nothing to scroll vertically here,
            # and a plain mouse wheel only has the one axis.
            delta = dx if abs(dx) > abs(dy) else dy
            secs = -delta / self._w * (self._v1 - self._v0)
            self._set_view(self._v0 + secs, self._v1 + secs)
        ev.accept()

    def event(self, ev) -> bool:
        if ev.type() == QEvent.NativeGesture and self.index:
            if ev.gestureType() == Qt.ZoomNativeGesture:
                self.zoom_by(1.0 / (1.0 + ev.value()), self._t(ev.position().x()))
                return True
        return super().event(ev)

    def _tooltip(self, ev: QMouseEvent) -> None:
        if not self.index:
            return
        x, lane = ev.position().x(), self._lane_at(ev.position().y())
        gp = ev.globalPosition().toPoint()
        if lane == "notes":
            i = self._note_at_x(x)
            if i >= 0:
                n = self.index.notes[i]
                body = (n.description or ("<i>withheld — not described</i>"
                                          if n.withheld else "<i>no note</i>"))
                extra = ""
                if n.redacted:
                    extra = (f"<br><span style='color:{theme.DANGER}'>"
                             f"credentials blacked out: "
                             f"{', '.join(sorted(set(n.redacted)))}</span>")
                QToolTip.showText(gp, f"<b>{format_time(n.t)}</b>"
                                      f"<div style='max-width:380px'>{body}"
                                      f"{extra}</div>", self)
                return
        if lane == "chapters":
            i = self._chapter_at_x(x)
            if i >= 0:
                c = self.index.chapters[i]
                kind = f" · {c.kind}" if c.kind else ""
                why = f"<br><span style='color:{theme.MUTED}'>{c.reason}</span>" \
                    if c.reason else ""
                QToolTip.showText(gp, f"<b>{c.display_title}</b><br>"
                                      f"{format_time(c.start)} – {format_time(c.end)}"
                                      f" ({format_duration(c.duration)}){kind}"
                                      f"<div style='max-width:380px'>{why}</div>",
                                  self)
                return
        if lane == "files":
            src, _ = self.index.source_at(self._t(x))
            QToolTip.showText(gp, f"<b>{src.name}</b><br>starts at "
                                  f"{format_time(src.offset)} · "
                                  f"{format_duration(src.duration)}", self)
            return
        QToolTip.hideText()

    # -- painting ---------------------------------------------------------------

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.PANEL))
        small = QFont(self.font())
        small.setPointSizeF(max(8.0, self.font().pointSizeF() - 2))
        p.setFont(small)

        if not self.index:
            p.setPen(QColor(theme.FAINT))
            p.drawText(self.rect(), Qt.AlignCenter, "No timeline")
            return

        lanes = self._lanes()
        self._paint_labels(p, lanes)
        p.save()
        p.setClipRect(QRectF(GUTTER, 0, self._w + 1, self.height()))
        self._paint_ruler(p)
        for name, top, h in lanes:
            getattr(self, f"_paint_{name}")(p, top, h)
        self._paint_joins(p, lanes)
        self._paint_cursor(p)
        p.restore()
        self._paint_overview(p)

    def _paint_labels(self, p: QPainter, lanes) -> None:
        p.setPen(QColor(theme.FAINT))
        names = {"files": "FILES", "chapters": "CHAPTERS",
                 "notes": "NOTES", "frames": "FRAMES"}
        for name, top, h in lanes:
            p.drawText(QRectF(10, top, GUTTER - 14, h),
                       Qt.AlignVCenter | Qt.AlignLeft, names[name])

    def _paint_overview(self, p: QPainter) -> None:
        full = QRectF(GUTTER, 1, self._w, OVERVIEW_H - 2)
        p.fillRect(full, QColor(theme.BORDER))
        if self.index.chapters:
            for k, c in enumerate(self.index.chapters):
                x0 = GUTTER + c.start / self._duration * self._w
                x1 = GUTTER + c.end / self._duration * self._w
                col = theme.kind_color(c.kind, k)
                col.setAlpha(110)
                p.fillRect(QRectF(x0, 1, max(1.0, x1 - x0), OVERVIEW_H - 2), col)
        if self._v1 - self._v0 < self._duration - 0.5:
            x0 = GUTTER + self._v0 / self._duration * self._w
            x1 = GUTTER + self._v1 / self._duration * self._w
            p.setPen(QPen(QColor(theme.TEXT), 1))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(x0, 0.5, max(2.0, x1 - x0), OVERVIEW_H - 1))

    def _paint_ruler(self, p: QPainter) -> None:
        top = OVERVIEW_H
        span = self._v1 - self._v0
        major = _nice_step(span * 90 / self._w)
        minor = _MINOR.get(int(major), major / 5)
        fm = QFontMetrics(p.font())
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.drawLine(QPointF(GUTTER, top + RULER_H - 0.5),
                   QPointF(GUTTER + self._w, top + RULER_H - 0.5))
        t = math.floor(self._v0 / minor) * minor
        while t <= self._v1:
            x = self._x(t)
            is_major = abs((t / major) - round(t / major)) < 1e-6
            h = 8 if is_major else 4
            p.setPen(QPen(QColor(theme.MUTED if is_major else theme.FAINT), 1))
            p.drawLine(QPointF(x, top + RULER_H - h), QPointF(x, top + RULER_H))
            if is_major:
                p.setPen(QColor(theme.MUTED))
                p.drawText(QPointF(x + 3, top + fm.ascent() + 3), format_time(t))
            t += minor

    def _paint_files(self, p: QPainter, top: float, h: float) -> None:
        for s in self.index.sources:
            x0, x1 = self._x(s.offset), self._x(s.end)
            r = QRectF(x0 + 1, top, max(1.0, x1 - x0 - 2), h)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(theme.RAISED if s.index % 2 == 0 else theme.HOVER))
            p.drawRoundedRect(r, 4, 4)
            label = s.name if s.exists else f"{s.name} (missing)"
            p.setPen(QColor(theme.MUTED if s.exists else theme.DANGER))
            vis = r.intersected(QRectF(GUTTER, top, self._w, h))
            text = QFontMetrics(p.font()).elidedText(
                label, Qt.ElideRight, int(vis.width() - 10))
            if vis.width() > 30:
                p.drawText(vis.adjusted(6, 0, -4, 0),
                           Qt.AlignVCenter | Qt.AlignLeft, text)

    def _paint_chapters(self, p: QPainter, top: float, h: float) -> None:
        if not self.index.chapters:
            p.setPen(QColor(theme.FAINT))
            p.drawText(QRectF(GUTTER + 8, top, self._w, h),
                       Qt.AlignVCenter | Qt.AlignLeft,
                       "Not analyzed yet — chapters appear here")
            return
        bold = QFont(p.font())
        bold.setWeight(QFont.DemiBold)
        fm_b, fm = QFontMetrics(bold), QFontMetrics(p.font())
        playing = self.index.chapter_at(self._playhead)
        for k, c in enumerate(self.index.chapters):
            x0, x1 = self._x(c.start), self._x(c.end)
            if x1 < GUTTER or x0 > GUTTER + self._w:
                continue
            r = QRectF(x0 + 0.5, top, max(1.5, x1 - x0 - 1.5), h)
            col = theme.kind_color(c.kind, k)
            selected = k == self._sel_chapter
            fill = QColor(col)
            fill.setAlpha(235 if selected else (175 if k == playing else 120))
            p.setPen(QPen(QColor(theme.TEXT), 1.5) if selected else Qt.NoPen)
            p.setBrush(fill)
            p.drawRoundedRect(r, 5, 5)
            # Kind stripe along the top edge stays saturated however faint the
            # fill, so kind is readable even on an unselected block.
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            stripe = QPainterPath()
            stripe.addRoundedRect(QRectF(r.x(), r.y(), r.width(), 3), 1.5, 1.5)
            p.drawPath(stripe)

            vis = r.intersected(QRectF(GUTTER, top, self._w, h))
            if vis.width() > 28:
                inner = vis.adjusted(7, 5, -5, -3)
                p.setPen(QColor("#ffffff"))
                p.setFont(bold)
                p.drawText(inner, Qt.AlignTop | Qt.AlignLeft,
                           fm_b.elidedText(c.display_title, Qt.ElideRight,
                                           int(inner.width())))
                p.setFont(self._small_font(p))
                p.setPen(QColor(255, 255, 255, 170))
                p.drawText(inner, Qt.AlignBottom | Qt.AlignLeft,
                           fm.elidedText(f"{format_time(c.start)} · "
                                         f"{format_duration(c.duration)}",
                                         Qt.ElideRight, int(inner.width())))

    def _small_font(self, p: QPainter) -> QFont:
        f = QFont(self.font())
        f.setPointSizeF(max(8.0, self.font().pointSizeF() - 2))
        return f

    def _paint_notes(self, p: QPainter, top: float, h: float) -> None:
        notes = self.index.notes
        mid = top + h / 2
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.drawLine(QPointF(GUTTER, mid), QPointF(GUTTER + self._w, mid))
        if not notes:
            # Fall back to the chapters' own sample times, which say where the
            # frames were even when there is no note text to go with them.
            times = [t for c in self.index.chapters for t in c.sample_times]
            p.setPen(QPen(QColor(theme.FAINT), 1.5))
            for t in times:
                if self._v0 <= t <= self._v1:
                    x = self._x(t)
                    p.drawLine(QPointF(x, mid - 4), QPointF(x, mid + 4))
            if not times:
                p.setPen(QColor(theme.FAINT))
                p.drawText(QRectF(GUTTER + 8, top, self._w, h),
                           Qt.AlignVCenter | Qt.AlignLeft,
                           "No per-frame notes for this session")
            return

        hover = self._note_at_x(self._hover_x) if self._hover_x else -1
        for i, n in enumerate(notes):
            if n.t < self._v0 - 1 or n.t > self._v1 + 1:
                continue
            x = self._x(n.t)
            big = i in (self._sel_note, hover)
            half = 11 if big else 7
            if n.redacted or n.withheld:
                col = QColor(theme.DANGER)
            elif n.description:
                col = QColor(theme.TEXT)
                col.setAlpha(235 if big else 150)
            else:
                col = QColor(theme.FAINT)
            if n.withheld:
                p.setPen(QPen(col, 1.2))
                p.setBrush(QColor(theme.PANEL))
                p.drawEllipse(QPointF(x, mid), 3.5, 3.5)
                continue
            p.setPen(QPen(col, 3 if big else 2, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(x, mid - half), QPointF(x, mid + half))
            if n.redacted:
                p.setPen(Qt.NoPen)
                p.setBrush(col)
                p.drawEllipse(QPointF(x, top + 2.5), 2.5, 2.5)
        if 0 <= self._sel_note < len(notes):
            x = self._x(notes[self._sel_note].t)
            p.setPen(QPen(QColor(theme.ACCENT), 2))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(QRectF(x - 5, top + 1, 10, h - 2), 3, 3)

    def _paint_frames(self, p: QPainter, top: float, h: float) -> None:
        # Slots sit on a grid aligned to zero, not to the view's left edge, so
        # panning reuses the thumbnails already decoded instead of requesting
        # a fresh, slightly shifted set for every pixel moved.
        thumb_w = h * 16 / 9
        span = self._v1 - self._v0
        step = _nice_step(span * thumb_w / self._w)
        width_px = int(thumb_w * self.devicePixelRatioF())
        t = math.floor(self._v0 / step) * step
        p.setPen(Qt.NoPen)
        while t < self._v1:
            x0, x1 = self._x(t), self._x(t + step)
            r = QRectF(x0 + 1, top, max(1.0, x1 - x0 - 2), h)
            p.fillRect(r, QColor(theme.BG))
            sample = min(t + step / 2, self._duration - 0.5)
            src, local = self.index.source_at(sample)
            img = self.thumbs.get(src.path, round(local, 1), width_px) \
                if src.exists else None
            if img is not None:
                # Crop to the slot's aspect rather than letterboxing, so the
                # strip reads as continuous footage.
                sw = img.width()
                sh = int(sw * r.height() / r.width()) if r.width() else img.height()
                if sh > img.height():
                    sh = img.height()
                    sw = int(sh * r.width() / r.height())
                sx, sy = (img.width() - sw) / 2, (img.height() - sh) / 2
                p.drawImage(r, img, QRectF(sx, sy, sw, sh))
            t += step

    def _paint_joins(self, p: QPainter, lanes) -> None:
        if not self.index.is_multi:
            return
        bottom = lanes[-1][1] + lanes[-1][2]
        pen = QPen(QColor(theme.WARN), 1, Qt.DashLine)
        p.setPen(pen)
        for s in self.index.sources[1:]:
            x = self._x(s.offset)
            p.drawLine(QPointF(x, OVERVIEW_H + RULER_H), QPointF(x, bottom))

    def _paint_cursor(self, p: QPainter) -> None:
        fm = QFontMetrics(p.font())
        if self._hover_x is not None and not self._scrubbing:
            p.setPen(QPen(QColor(255, 255, 255, 60), 1))
            p.drawLine(QPointF(self._hover_x, OVERVIEW_H),
                       QPointF(self._hover_x, self.height()))
            label = format_time(self._t(self._hover_x))
            w = fm.horizontalAdvance(label) + 10
            r = QRectF(self._hover_x - w / 2, OVERVIEW_H + RULER_H - 17, w, 15)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(theme.HOVER))
            p.drawRoundedRect(r, 3, 3)
            p.setPen(QColor(theme.TEXT))
            p.drawText(r, Qt.AlignCenter, label)

        x = self._x(self._playhead)
        if GUTTER - 1 <= x <= GUTTER + self._w + 1:
            p.setPen(QPen(QColor("#ffffff"), 1.5))
            p.drawLine(QPointF(x, OVERVIEW_H + RULER_H - 6),
                       QPointF(x, self.height()))
            head = QPainterPath()
            y = OVERVIEW_H + RULER_H - 9
            head.moveTo(x - 5, y)
            head.lineTo(x + 5, y)
            head.lineTo(x, y + 6)
            head.closeSubpath()
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#ffffff"))
            p.drawPath(head)
