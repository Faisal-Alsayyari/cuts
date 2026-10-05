"""The adapter: the only module in the GUI that knows anything about `cuts`.

Everything backend-specific is here and nowhere else:

  * **How to run it** — the CLI commands and flags (`analyze_command`,
    `rechapter_command`). The GUI runs the backend as a subprocess and never
    imports it.
  * **Where it writes** — index and observation-cache paths, mirroring the
    CLI's own defaults so a session analyzed here and one analyzed from a
    terminal find each other's output.
  * **What it writes** — `events.json` and `observations.jsonl`, translated
    into `cuts_gui.model` types (`load_index`).
  * **What its log means** — `interpret` maps output lines onto progress
    stages, warnings and results.

When the backend changes, this is the file that changes. The rest of the GUI
is written against `cuts_gui.model` and `Signal`, which do not mention the
backend at all.

Progress comes from parsing log lines, which is the one genuinely fragile
contract here. It is used for display only — which stage is running, what to
warn about — and never for correctness: success is decided by the exit code
and the index file existing, not by having seen a particular line. A
reworded log line degrades the progress display and nothing else. If the
backend ever grows a machine-readable progress stream, `interpret` is the
function to replace.

No Qt in this module, so it can be exercised without a display.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .model import Chapter, Index, Note, RunSettings, Session, SourceFile, Status

# Video extensions offered by the file dialogs and accepted on drop. The
# backend decodes through PyAV/FFmpeg, so this is a convenience filter, not a
# statement of what it can read.
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".m4v", ".webm", ".avi")
INDEX_SUFFIX = ".events.json"
NOTES_SUFFIX = ".observations.jsonl"


# ---------------------------------------------------------------------------
# Locating the backend
# ---------------------------------------------------------------------------

def default_root() -> str:
    """The directory containing the `cuts` package, when the GUI ships beside it.

    Used as the subprocess's working directory, which matters beyond import
    resolution: the CLI looks for `.env` walking upward from its cwd, so this
    is also what lets it find `ANTHROPIC_API_KEY` the same way a terminal run
    in the repo would.
    """
    here = Path(__file__).resolve().parent.parent
    if (here / "cuts" / "__main__.py").exists():
        return str(here)
    return os.getcwd()


def default_python() -> str:
    """The interpreter the GUI itself runs under — normally the project venv."""
    return sys.executable


def credentials_source(root: str) -> str:
    """Where the backend will find an API key: "environment", a .env path, or "".

    Mirrors the CLI's own lookup (a real environment variable, else the first
    .env found walking up from the working directory). Never returns the key.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "environment"
    d = Path(root).resolve()
    for parent in [d, *d.parents]:
        env = parent / ".env"
        if env.is_file():
            try:
                for line in env.read_text(encoding="utf-8").splitlines():
                    key, _, value = line.strip().partition("=")
                    if key.strip().removeprefix("export ").strip() == \
                            "ANTHROPIC_API_KEY" and value.strip().strip("'\""):
                        return str(env)
            except OSError:
                pass
            # Like python-dotenv's find_dotenv: the first .env found is the
            # one that gets loaded, whether or not it has the key.
            return ""
    return ""


@dataclass
class Backend:
    """How to reach the backend: which interpreter, run from where."""

    python: str
    root: str

    def environment(self) -> Dict[str, str]:
        env = dict(os.environ)
        # Without this the child block-buffers stdout into a pipe and the
        # log arrives in one lump when the run ends.
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (self.root, env.get("PYTHONPATH", "")) if p
        )
        return env

    # -- commands -------------------------------------------------------------

    def analyze_command(self, session: Session) -> Tuple[str, List[str]]:
        """Full analysis (both passes) of a session's files into its index."""
        s = session.settings
        args = ["-m", "cuts", "chapters", *session.files,
                "-o", session.index_path, "-v"]
        if s.intent.strip():
            args += ["--intent", s.intent.strip()]
        if s.interval:
            args += ["--interval", _num(s.interval)]
        if s.segmenter:
            args += ["--segmenter", s.segmenter]
        if s.min_chapter is not None:
            args += ["--min-chapter", _num(s.min_chapter)]
        if not s.transcribe:
            args.append("--no-asr")
        if not s.redact:
            args.append("--no-redact")
        if not s.refine:
            args.append("--no-refine")
        if s.include_screen_text:
            args.append("--include-ocr")
        if s.describe_model:
            args += ["--vlm-model", s.describe_model]
        if s.chapter_model:
            args += ["--coalesce-model", s.chapter_model]
        if not s.use_cache:
            args.append("--no-cache")
        return self.python, args

    def rechapter_command(
        self, session: Session, settings: RunSettings
    ) -> Optional[Tuple[str, List[str]]]:
        """Re-decide chapters from cached notes only — no frame is re-read.

        This is the backend's cheap iteration loop (one model call over text),
        so it is what "try a different intent" should cost. Returns None when
        there is no usable cache for this session's files, in which case only
        a full analysis can help.
        """
        notes = find_notes(session.files, session.index_path)
        if notes is None:
            return None
        args = ["-m", "cuts", "coalesce", notes, "-o", session.index_path, "-v"]
        if settings.intent.strip():
            args += ["--intent", settings.intent.strip()]
        if settings.min_chapter is not None:
            args += ["--min-chapter", _num(settings.min_chapter)]
        if settings.include_screen_text:
            args.append("--include-ocr")
        if settings.chapter_model:
            args += ["--coalesce-model", settings.chapter_model]
        return self.python, args


def _num(x: float) -> str:
    return f"{x:g}"


# ---------------------------------------------------------------------------
# Where things live on disk
# ---------------------------------------------------------------------------

def default_index_path(files: Sequence[str]) -> str:
    """Where the CLI would write this session's index if not told otherwise.

    Mirrored rather than chosen, so a session added here picks up an index
    that a terminal run already produced, and vice versa.
    """
    first = Path(files[0])
    if len(files) == 1:
        return str(first.with_suffix(INDEX_SUFFIX))
    return str(first.with_name(f"{first.stem}+{len(files) - 1}{INDEX_SUFFIX}"))


def _notes_candidates(files: Sequence[str], index_path: str = "") -> List[str]:
    """Places a session's observation cache may be: beside its first file,
    or beside its index (for an index whose videos have since moved)."""
    out: List[str] = []
    if files:
        first = Path(files[0])
        out.append(str(first.with_name(first.stem + NOTES_SUFFIX)))
        if index_path:
            out.append(str(Path(index_path).with_name(first.stem + NOTES_SUFFIX)))
    if index_path and index_path.endswith(INDEX_SUFFIX):
        out.append(index_path[: -len(INDEX_SUFFIX)] + NOTES_SUFFIX)
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def find_notes(files: Sequence[str], index_path: str = "") -> Optional[str]:
    """Path to an observation cache that belongs to exactly these files."""
    for p in _notes_candidates(files, index_path):
        if os.path.exists(p) and _notes_match(p, files) is None:
            return p
    return None


def _notes_match(path: str, files: Sequence[str]) -> Optional[str]:
    """None when the cache at `path` was made from `files`, else why not.

    The backend's cache is keyed on its first file's name alone, so one file
    analyzed alone and the same file as the first of three share a cache
    path. Matching on name and size (not the full path) keeps an index usable
    after its videos are moved, which a path comparison would not.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = json.loads(f.readline() or "{}")
    except (OSError, json.JSONDecodeError):
        return "the notes file could not be read"
    cached = head.get("_meta", {}).get("files", [])
    if len(cached) != len(files):
        return (f"the notes file covers {len(cached)} file(s), "
                f"this session has {len(files)}")
    for entry, f in zip(cached, files):
        cpath, csize = entry[0], entry[2] if len(entry) > 2 else None
        if os.path.basename(cpath) != os.path.basename(f):
            return f"the notes file was made from {os.path.basename(cpath)}"
        if csize and os.path.exists(f) and os.path.getsize(f) != csize:
            return f"{os.path.basename(f)} has changed since the notes were taken"
    return None


def _resolve(path: str, search_dirs: Sequence[str]) -> str:
    """`path` if it exists, else a same-named file in one of `search_dirs`.

    Indexes record absolute paths, which go stale the moment a folder is
    moved. Looking beside the index for the same filename recovers the common
    case without asking.
    """
    if not path or os.path.exists(path):
        return path
    name = os.path.basename(path)
    for d in search_dirs:
        cand = os.path.join(d, name)
        if d and os.path.exists(cand):
            return cand
    return path


# ---------------------------------------------------------------------------
# Reading results
# ---------------------------------------------------------------------------

def read_index_sources(index_path: str) -> List[str]:
    """The ordered file list an index was made from, with stale paths resolved."""
    with open(index_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    sources = payload.get("sources") or [{"path": payload.get("video_path", "")}]
    here = os.path.dirname(os.path.abspath(index_path))
    return [_resolve(str(s.get("path", "")), [here]) for s in sources]


def session_from_index(index_path: str) -> Session:
    """A READY library entry for an index produced elsewhere (e.g. the CLI)."""
    index_path = os.path.abspath(index_path)
    files = read_index_sources(index_path)
    name = Path(index_path).name[: -len(INDEX_SUFFIX)] \
        if index_path.endswith(INDEX_SUFFIX) else Path(index_path).stem
    return Session(name=name, files=files, index_path=index_path,
                   status=Status.READY)


def load_index(session: Session) -> Index:
    """Read a session's index and notes into GUI types.

    The session's own file list wins over the paths recorded in the index:
    it is what the user last pointed the GUI at (including any relinking),
    whereas the index records wherever the files were when it was written.

    Raises OSError / ValueError when the index is missing or unreadable.
    """
    with open(session.index_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    here = os.path.dirname(os.path.abspath(session.index_path))
    raw_sources = payload.get("sources") or [{
        "path": payload.get("video_path", ""), "index": 0,
        "duration_sec": payload.get("duration_sec", 0.0), "time_offset": 0.0,
    }]
    sources: List[SourceFile] = []
    if (len(session.files) > len(raw_sources)
            and not any(s.get("path") for s in raw_sources)):
        # An index written without a file layout (the standalone `coalesce`
        # command does this when the original paths no longer exist). The
        # session still knows its files, so lay them out from their own
        # lengths rather than collapsing the timeline to one nameless source.
        from .media import probe
        cursor = 0.0
        for i, path in enumerate(session.files):
            dur = probe(path).duration
            sources.append(SourceFile(path=path, index=i, duration=dur,
                                      offset=cursor))
            cursor += dur
    for i, s in enumerate(raw_sources if not sources else []):
        path = (session.files[i] if len(session.files) == len(raw_sources)
                else _resolve(str(s.get("path", "")), [here]))
        sources.append(SourceFile(
            path=path, index=i,
            duration=float(s.get("duration_sec") or 0.0),
            offset=float(s.get("time_offset") or 0.0),
        ))
    duration = float(payload.get("duration_sec") or 0.0) or \
        sum(s.duration for s in sources)

    chapters = [_chapter(e) for e in payload.get("events", [])]
    chapters.sort(key=lambda c: c.start)

    extra = payload.get("extra") or {}
    facts: Dict[str, Any] = {
        "segmenter": extra.get("segmenter", ""),
        "intent": extra.get("intent", ""),
        "samples": extra.get("n_samples") or extra.get("n_observations"),
        "signals": extra.get("channels", []),
        "titles": extra.get("label_backend", ""),
        "stage_times": extra.get("stage_times", {}),
    }

    notes, notes_path, why_not = _load_notes(session, sources)
    return Index(
        title=str(payload.get("video_id") or session.name),
        duration=duration,
        sources=sources,
        chapters=chapters,
        notes=notes,
        notes_unavailable=why_not,
        index_path=session.index_path,
        notes_path=notes_path,
        facts=facts,
    )


def _chapter(e: Dict[str, Any]) -> Chapter:
    meta = e.get("metadata") or {}
    descriptions = [ln.strip() for ln in str(e.get("vlm_text", "")).split("\n")
                    if ln.strip()]
    return Chapter(
        start=float(e.get("start_time", 0.0)),
        end=float(e.get("end_time", 0.0)),
        title=str(e.get("title", "")).strip(),
        kind=str(meta.get("kind", "")).strip().lower(),
        reason=str(e.get("summary", "")).strip(),
        descriptions=descriptions,
        transcript=str(e.get("transcript_text", "")).strip(),
        source_index=int(e.get("source_index", 0) or 0),
        source_time=float(e.get("source_start_time", 0.0) or 0.0),
        sample_times=[float(t) for t in e.get("sample_times", []) or []],
    )


def _load_notes(
    session: Session, sources: List[SourceFile]
) -> Tuple[List[Note], str, str]:
    """(notes, path, reason-if-empty) for a session."""
    files = [s.path for s in sources] or session.files
    candidates = [p for p in _notes_candidates(files, session.index_path)
                  if os.path.exists(p)]
    if not candidates:
        return [], "", ("No per-frame notes were found for this index. They are "
                        "produced by the default (llm) analysis; an index made "
                        "with the curve segmenter has chapters only.")
    reason = ""
    for path in candidates:
        mismatch = _notes_match(path, files)
        if mismatch:
            reason = f"Per-frame notes exist but don't match: {mismatch}."
            continue
        notes: List[Note] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    if "_meta" in d:
                        continue
                    notes.append(Note(
                        t=float(d.get("t", 0.0)),
                        source_index=int(d.get("source_index", 0) or 0),
                        source_time=float(d.get("source_time", 0.0) or 0.0),
                        description=str(d.get("description", "")).strip(),
                        screen_text=str(d.get("ocr_text", "")),
                        transcript=str(d.get("transcript_text", "")).strip(),
                        redacted=list(d.get("redacted") or []),
                        withheld=bool(d.get("withheld", False)),
                    ))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            reason = f"The per-frame notes file could not be read ({exc})."
            continue
        notes.sort(key=lambda n: n.t)
        return notes, path, ""
    return [], "", reason


def chapter_list(index: Index, show_source: bool = False) -> str:
    """The backend's primary output format — a pasteable chapter list.

        0:00 writing parser
        7:32 compiling

    Mirrors `cuts.schema.to_chapter_list`, including its multi-file form
    (``1:04:11 debugging segfault   [day-2.mp4 @ 4:11]``), so what is copied
    here is byte-for-byte what the CLI prints.
    """
    from .model import format_time

    width = max((len(c.title or "(unlabeled)") for c in index.chapters),
                default=0)
    lines = []
    for c in index.chapters:
        title = c.title or "(unlabeled)"
        line = f"{format_time(c.start)} {title}"
        if show_source and 0 <= c.source_index < len(index.sources):
            name = index.sources[c.source_index].name
            line = (f"{format_time(c.start)} {title:<{width}}"
                    f"   [{name} @ {format_time(c.source_time)}]")
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reading the log
# ---------------------------------------------------------------------------

# Ordered stages of a run, as the GUI presents them. Keys are the GUI's;
# which log lines map to which is `interpret`'s business.
STAGES: List[Tuple[str, str]] = [
    ("observe", "Reading frames"),
    ("transcribe", "Transcribing"),
    ("chapter", "Finding chapters"),
    ("refine", "Refining boundaries"),
    ("label", "Titling"),
]


@dataclass
class Signal:
    """What one log line means to the GUI.

    kind:
      * "stage"   — `stage` has begun; `text` is a detail line for it
      * "done"    — `stage` finished, `value` is its seconds
      * "warning" — something the user should see (`text`); `stage` is
                    "redact" for the credential warning specifically
      * "error"   — the run failed and `text` says why
      * "result"  — `text` is the path of the index the run wrote
    """

    kind: str
    stage: str = ""
    text: str = ""
    value: Optional[float] = None


# Backend timing lines ("  [observe] 233.7s") name its own stages; this maps
# them onto the GUI's.
_TIMED = {"observe": "observe", "signals": "observe", "asr": "transcribe",
          "coalesce": "chapter", "segmentation": "chapter",
          "refine": "refine", "labeling": "label"}

_RULES: List[Tuple[re.Pattern, Any]] = [
    (re.compile(r"^\s*\[(\w+)\] ([\d.]+)s\s*$"),
     lambda m: Signal("done", _TIMED.get(m[1], ""), value=float(m[2]))
     if m[1] in _TIMED else None),
    (re.compile(r"\[observe\] reusing (\d+) cached"),
     lambda m: Signal("stage", "observe", f"Reusing {m[1]} cached frame notes")),
    (re.compile(r"\[observe\] (\d+) shard\(s\), (\d+) samples"),
     lambda m: Signal("stage", "observe",
                      f"Describing {m[2]} frames in {m[1]} shard(s)",
                      value=float(m[2]))),
    (re.compile(r"^\s*\[sample\] every"),
     lambda m: Signal("stage", "observe", "Sampling frames")),
    (re.compile(r"^\s*\[asr\] transcribing"),
     lambda m: Signal("stage", "transcribe", "Transcribing speech")),
    (re.compile(r"^\s*\[asr\] skipped"),
     lambda m: Signal("stage", "transcribe", "No audio — skipped")),
    (re.compile(r"^\s*\[coalesce\] (\d+) observations -> (\S+)"),
     lambda m: Signal("stage", "chapter", f"{m[2]} is reading {m[1]} notes")),
    (re.compile(r"^\s*\[coalesce\] \d+ proposed -> (\d+) chapters"),
     lambda m: Signal("stage", "chapter", f"{m[1]} chapters")),
    (re.compile(r"^\s*(channels:|local minima)"),
     lambda m: Signal("stage", "chapter", "Segmenting similarity curve")),
    (re.compile(r"^\s*\[refine\] (.+)"),
     lambda m: Signal("stage", "refine", m[1])),
    (re.compile(r"^\s*(labeling \d+ events|decoding \d+ frames for labeling)"),
     lambda m: Signal("stage", "label", "Writing chapter titles")),
    (re.compile(r"^\s*\[redact\] (.*blacked out.*)$"),
     lambda m: Signal("warning", "redact", m[1])),
    (re.compile(r"^\[pipeline\] coalescing failed"),
     lambda m: Signal("warning", "",
                      "Chaptering call failed; fell back to the curve "
                      "segmenter over the same notes")),
    (re.compile(r"^\[pipeline\] no ANTHROPIC_API_KEY"),
     lambda m: Signal("warning", "",
                      "No ANTHROPIC_API_KEY: chapters are from the visual "
                      "channel only, with rough titles")),
    (re.compile(r"WARNING:?\s*(.+)"),
     lambda m: Signal("warning", "", m[1])),
    (re.compile(r"^error: (.+)"),
     lambda m: Signal("error", "", m[1])),
    (re.compile(r"^Traceback \(most recent call last\)"),
     lambda m: Signal("error", "", "The backend crashed — see the log")),
    (re.compile(r"^index: (.+)$"),
     lambda m: Signal("result", "", m[1].strip())),
]


def interpret(line: str) -> Optional[Signal]:
    """Map one line of backend output to a `Signal`, or None if uninteresting."""
    for pattern, make in _RULES:
        m = pattern.search(line)
        if m:
            sig = make(m)
            if sig is not None:
                return sig
    return None
