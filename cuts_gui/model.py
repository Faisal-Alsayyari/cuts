"""The GUI's own vocabulary: plain data, no Qt, no backend imports.

These types are what every widget is written against. `cuts_gui.backend`
translates the backend's on-disk format into them, so a schema change in
`cuts` (a renamed field, a new kind of evidence) is absorbed in that one
translation rather than rippling through the views.

The names are deliberately the user-facing ones rather than the backend's:

  * `Chapter` — what the backend calls an `Event`.
  * `Note`    — what the backend calls an `Observation`: one sampled frame and
                what was written about it. "Which frames have what notes" is
                the question the timeline answers.
  * `Session` — a library entry: one recording, possibly split across several
                files, plus how it should be analyzed. This is GUI state; the
                backend has no notion of a library.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


def format_time(seconds: float) -> str:
    """``M:SS``, or ``H:MM:SS`` past an hour — the chapter-list convention."""
    total = int(max(0.0, seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def format_duration(seconds: float) -> str:
    """Human length: ``42s``, ``7m 32s``, ``1h 04m``."""
    s = int(max(0.0, seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


# ---------------------------------------------------------------------------
# A loaded index
# ---------------------------------------------------------------------------

@dataclass
class SourceFile:
    """One input file's placement on the session's global timeline."""

    path: str
    index: int
    duration: float
    offset: float

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def end(self) -> float:
        return self.offset + self.duration

    @property
    def exists(self) -> bool:
        return bool(self.path) and os.path.exists(self.path)


@dataclass
class Chapter:
    """One titled span of the recording.

    `kind` is the backend's priority signal (pivotal / active / routine /
    idle), or "" when the index was produced by a path that does not assign
    one. `reason` is the model's stated evidence for the boundary.
    """

    start: float
    end: float
    title: str = ""
    kind: str = ""
    reason: str = ""
    descriptions: List[str] = field(default_factory=list)
    transcript: str = ""
    source_index: int = 0
    source_time: float = 0.0
    sample_times: List[float] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def display_title(self) -> str:
        return self.title or "(untitled)"


# Why a note has no description, by the backend's observation status.
_MISSING = {
    "withheld": "withheld — a credential was on screen, never uploaded",
    "undecodable": "no frame at this time",
    "failed": "description failed — re-analyze to retry",
}


@dataclass
class Note:
    """One sampled frame and what was observed about it."""

    t: float
    source_index: int = 0
    source_time: float = 0.0
    description: str = ""
    screen_text: str = ""
    transcript: str = ""
    # Names of credential patterns blacked out of this frame — never content.
    redacted: List[str] = field(default_factory=list)
    # The backend's observation status: ok | withheld | undecodable | failed.
    status: str = "ok"

    @property
    def withheld(self) -> bool:
        """True when the frame has no description for a reason (see `missing`)."""
        return self.status != "ok"

    @property
    def missing(self) -> str:
        """Why this note has no description, or "" if it is not missing one."""
        return _MISSING.get(self.status, "")

    @property
    def flagged(self) -> bool:
        """A credential was on screen: blacked out, or the frame withheld."""
        return bool(self.redacted) or self.status == "withheld"


@dataclass
class Index:
    """Everything the views need about one analyzed session."""

    title: str
    duration: float
    sources: List[SourceFile]
    chapters: List[Chapter]
    notes: List[Note] = field(default_factory=list)
    # Why `notes` is empty, when it is — shown in place of the notes view.
    notes_unavailable: str = ""
    index_path: str = ""
    notes_path: str = ""
    # Run facts worth showing (segmenter, intent, stage times, ...). Opaque to
    # the views beyond display.
    facts: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_multi(self) -> bool:
        return len(self.sources) > 1

    def source_at(self, t: float) -> Tuple[SourceFile, float]:
        """Map a global time to (file, seconds into that file). Clamps."""
        for s in self.sources:
            if s.offset <= t < s.end:
                return s, t - s.offset
        if t < 0 or not self.sources:
            s = self.sources[0]
            return s, 0.0
        s = self.sources[-1]
        return s, max(0.0, min(t - s.offset, s.duration))

    def chapter_at(self, t: float) -> int:
        """Index of the chapter containing `t`, or -1."""
        for i, c in enumerate(self.chapters):
            if c.start <= t < c.end:
                return i
        if self.chapters and t >= self.chapters[-1].start:
            return len(self.chapters) - 1
        return -1

    def note_at(self, t: float) -> int:
        """Index of the most recent note at or before `t`, or -1.

        "Most recent", not "nearest": during playback the caption should show
        what was last observed, not jump ahead to a frame not yet on screen.
        """
        lo, hi, best = 0, len(self.notes) - 1, -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.notes[mid].t <= t + 1e-6:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        return best


# ---------------------------------------------------------------------------
# The library
# ---------------------------------------------------------------------------

class Status(str, Enum):
    NEW = "new"            # added, never analyzed
    QUEUED = "queued"
    RUNNING = "running"
    READY = "ready"        # an index exists and loads
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RunSettings:
    """How a session should be analyzed.

    `None` everywhere means "the backend's default". The GUI deliberately does
    not hard-code what those defaults are — they belong to the backend and
    will change with it — so an unset field is simply not passed.
    """

    intent: str = ""
    interval: Optional[float] = None
    segmenter: Optional[str] = None
    min_chapter: Optional[float] = None
    transcribe: bool = True
    redact: bool = True
    refine: bool = True
    include_screen_text: bool = False
    describe_model: str = ""
    chapter_model: str = ""
    use_cache: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunSettings":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


@dataclass
class Session:
    """One library entry: an ordered set of files analyzed as one timeline."""

    name: str
    files: List[str]
    index_path: str
    settings: RunSettings = field(default_factory=RunSettings)
    status: Status = Status.NEW
    # Last outcome line ("14 chapters", "error: ...") for the library list.
    message: str = ""
    # Set when a run reported blacking out credentials. Sticky across
    # restarts: the user needs to know there is a key to go rotate.
    redaction_warning: str = ""
    # Total length in seconds, cached for the library list (0 = unknown).
    duration: float = 0.0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def is_multi(self) -> bool:
        return len(self.files) > 1

    @property
    def missing_files(self) -> List[str]:
        return [f for f in self.files if not os.path.exists(f)]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["settings"] = self.settings.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Session":
        known = set(cls.__dataclass_fields__)
        kw = {k: v for k, v in d.items() if k in known}
        kw["settings"] = RunSettings.from_dict(d.get("settings", {}))
        try:
            kw["status"] = Status(d.get("status", "new"))
        except ValueError:
            kw["status"] = Status.NEW
        return cls(**kw)
