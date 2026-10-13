"""GUI adapter logic and an offscreen smoke test of the main window."""

from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from cuts_gui import backend  # noqa: E402
from cuts_gui.jobs import _LOG_CAP, RunState  # noqa: E402
from cuts_gui.model import Session  # noqa: E402


def test_run_log_keeps_streaming_past_the_cap():
    st = RunState()
    shown = 0
    seen = []
    for i in range(_LOG_CAP * 2 + 7):
        st.add_line(f"line {i}")
        if i % 997 == 0:                      # a viewer polling now and then
            seen += st.lines_since(shown)
            shown = st.total_lines
    seen += st.lines_since(shown)
    assert len(st.log) == _LOG_CAP
    assert seen[-1] == f"line {_LOG_CAP * 2 + 6}"
    assert st.lines_since(st.total_lines) == []


def _write_cache(path, files_meta, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": {"files": files_meta}}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_notes_match_both_cache_formats(tmp_path):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x" * 10)
    new = tmp_path / "new.jsonl"
    old = tmp_path / "old.jsonl"
    _write_cache(new, [["a.mp4", 10]], [])
    _write_cache(old, [[str(video), 123, 10]], [])
    assert backend._notes_match(str(new), [str(video)]) is None
    assert backend._notes_match(str(old), [str(video)]) is None
    _write_cache(new, [["a.mp4", 11]], [])
    assert "changed" in backend._notes_match(str(new), [str(video)])


def test_load_index_reads_new_and_legacy_fields(tmp_path):
    video = tmp_path / "s.mp4"
    video.write_bytes(b"x" * 10)
    idx = tmp_path / "s.events.json"
    idx.write_text(json.dumps({
        "video_id": "s", "video_path": str(video), "duration_sec": 60,
        "sources": [{"path": str(video), "index": 0, "duration_sec": 60,
                     "time_offset": 0}],
        "events": [
            {"start_time": 0, "end_time": 30, "title": "a", "reason": "new"},
            {"start_time": 30, "end_time": 60, "title": "b", "summary": "legacy"},
        ],
        "extra": {"label_backend": "claude"},
    }), encoding="utf-8")
    _write_cache(tmp_path / "s.observations.jsonl", [["s.mp4", 10]], [
        {"t": 0, "source_time": 0, "status": "failed"},
        {"t": 10, "source_time": 10, "withheld": True},           # version 1
        {"t": 20, "source_time": 20, "description": "x", "status": "ok"},
        {"t": 0, "source_time": 0, "description": "retried", "status": "ok"},
    ])
    ix = backend.load_index(Session(name="s", files=[str(video)], index_path=str(idx)))
    assert [c.reason for c in ix.chapters] == ["new", "legacy"]
    assert ix.facts["titles"] == "claude"
    assert [n.status for n in ix.notes] == ["ok", "withheld", "ok"]
    assert ix.notes[0].description == "retried"          # later line wins
    assert ix.notes[1].flagged and ix.notes[1].missing


def test_interpret_new_log_lines():
    assert backend.interpret("  [observe] 6/12 samples").text == "Read 6 of 12 frames"
    sig = backend.interpret("  [segment] 1.5s")
    assert sig.kind == "done" and sig.stage == "chapter"
    assert backend.interpret("error: no Anthropic credentials found").kind == "error"


def test_main_window_opens_a_real_index(three_scenes, fake_llm, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from cuts.cli import main
    from cuts_gui import library

    assert main(["chapters", three_scenes, "--interval", "10", "--no-asr"]) == 0
    monkeypatch.setattr(library, "default_path", lambda: str(tmp_path / "library.json"))

    app = QApplication.instance() or QApplication([])
    from cuts_gui.window import MainWindow

    win = MainWindow()
    win.open_index(three_scenes.replace(".mp4", ".events.json"))
    assert win.index is not None and len(win.index.chapters) == 3
    assert len(win.index.notes) == 12
    win._on_position(45.0)
    assert "working on green" in win.cap_chapter.text()
    win.timeline.resize(1200, 300)
    win.timeline.grab()                     # paints every lane, filmstrip included
    win.runner.shutdown()
    win.thumbs.shutdown()
    win.close()
    app.processEvents()
