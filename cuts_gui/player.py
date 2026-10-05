"""Playback over a session's whole timeline, however many files it spans.

The backend reads several files as one continuous recording and reports every
time globally. This does the same for playback: callers seek and observe in
global seconds, and the player works out which file that falls in, loads it
when needed, and rolls on to the next file when one ends. Nothing above this
module needs to know that a timeline can have joins.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

from .model import SourceFile


class TimelinePlayer(QObject):
    position_changed = Signal(float)   # global seconds
    playing_changed = Signal(bool)
    source_changed = Signal(int)       # index into sources
    error = Signal(str)

    def __init__(self, video_output, parent=None) -> None:
        super().__init__(parent)
        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._player.setAudioOutput(self._audio)
        self._player.setVideoOutput(video_output)
        self._sources: List[SourceFile] = []
        self._current = -1
        self._position = 0.0
        # A seek into a file that is still loading has to wait for it.
        self._pending: Optional[float] = None
        self._pending_play = False

        self._player.positionChanged.connect(self._on_position)
        self._player.mediaStatusChanged.connect(self._on_status)
        self._player.playbackStateChanged.connect(
            lambda st: self.playing_changed.emit(
                st == QMediaPlayer.PlayingState))
        self._player.errorOccurred.connect(
            lambda _e, msg: msg and self.error.emit(msg))

    # -- setup ------------------------------------------------------------------

    def set_sources(self, sources: List[SourceFile]) -> None:
        self._player.stop()
        self._player.setSource(QUrl())
        self._sources = list(sources)
        self._current = -1
        self._pending = None
        self._position = 0.0
        if self._sources:
            self._load(0, 0.0, play=False)

    # -- state ------------------------------------------------------------------

    @property
    def position(self) -> float:
        return self._position

    @property
    def is_playing(self) -> bool:
        return self._player.playbackState() == QMediaPlayer.PlayingState

    @property
    def current_source(self) -> Optional[SourceFile]:
        if 0 <= self._current < len(self._sources):
            return self._sources[self._current]
        return None

    @property
    def duration(self) -> float:
        return self._sources[-1].end if self._sources else 0.0

    # -- control ----------------------------------------------------------------

    def play(self) -> None:
        if not self._sources:
            return
        if self._current < 0:
            self.seek(self._position, play=True)
            return
        if self._position >= self.duration - 0.05:
            self.seek(0.0, play=True)
            return
        self._player.play()

    def pause(self) -> None:
        self._pending_play = False
        self._player.pause()

    def toggle(self) -> None:
        self.pause() if self.is_playing else self.play()

    def set_rate(self, rate: float) -> None:
        self._player.setPlaybackRate(rate)

    def set_volume(self, volume: float) -> None:
        self._audio.setVolume(max(0.0, min(1.0, volume)))

    def set_muted(self, muted: bool) -> None:
        self._audio.setMuted(muted)

    def seek(self, t: float, play: Optional[bool] = None) -> None:
        """Move to global time `t`, loading another file if it lies elsewhere."""
        if not self._sources:
            return
        t = max(0.0, min(t, self.duration))
        idx, local = self._locate(t)
        play = self.is_playing if play is None else play
        self._position = t
        self.position_changed.emit(t)
        if idx != self._current or self._pending is not None:
            self._load(idx, local, play)
        else:
            self._player.setPosition(int(local * 1000))
            if play and not self.is_playing:
                self._player.play()

    def step(self, delta: float) -> None:
        self.seek(self._position + delta)

    # -- internals --------------------------------------------------------------

    def _locate(self, t: float):
        for i, s in enumerate(self._sources):
            if t < s.end or i == len(self._sources) - 1:
                return i, max(0.0, t - s.offset)
        return 0, 0.0

    def _load(self, idx: int, local: float, play: bool) -> None:
        src = self._sources[idx]
        self._current = idx
        self.source_changed.emit(idx)
        if not src.exists:
            self._player.stop()
            self._player.setSource(QUrl())
            self._pending = None
            self.error.emit(f"Video file not found: {src.path}")
            return
        self._pending, self._pending_play = local, play
        self._player.setSource(QUrl.fromLocalFile(src.path))

    def _on_status(self, status) -> None:
        if status in (QMediaPlayer.LoadedMedia, QMediaPlayer.BufferedMedia) \
                and self._pending is not None:
            local, play = self._pending, self._pending_play
            self._pending = None
            self._player.setPosition(int(local * 1000))
            # Pausing (rather than leaving it stopped) is what makes a paused
            # seek actually paint the frame.
            self._player.play() if play else self._player.pause()
        elif status == QMediaPlayer.EndOfMedia:
            nxt = self._current + 1
            if nxt < len(self._sources):
                self._load(nxt, 0.0, play=True)
            else:
                self.playing_changed.emit(False)
        elif status == QMediaPlayer.InvalidMedia:
            src = self.current_source
            self.error.emit(f"Cannot play {src.name if src else 'this file'}")

    def _on_position(self, ms: int) -> None:
        src = self.current_source
        if src is None or self._pending is not None:
            return
        self._position = src.offset + ms / 1000.0
        self.position_changed.emit(self._position)
