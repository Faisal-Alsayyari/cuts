"""CLI commands: index contents, re-chaptering from cache, relabeling."""

from __future__ import annotations

import json
from pathlib import Path

from cuts.cli import main
from cuts.observe import observations_path


def _index(video):
    return json.loads(Path(video).with_suffix(".events.json").read_text(encoding="utf-8"))


def test_chapters_writes_run_facts(three_scenes, fake_llm):
    assert main(["chapters", three_scenes, "--interval", "10", "--no-asr"]) == 0
    idx = _index(three_scenes)
    assert idx["extra"]["segmenter"] == "llm" and idx["extra"]["titles"] == "llm"
    assert idx["events"][0]["reason"]
    assert "start_frame" not in idx["events"][0]
    assert set(idx["sources"][0]) == {"path", "index", "duration_sec", "time_offset"}


def test_chapters_without_credentials_exits_2(three_scenes, no_creds, capsys):
    assert main(["chapters", three_scenes, "--interval", "10"]) == 2
    assert "error: no Anthropic credentials" in capsys.readouterr().err


def test_coalesce_matches_chapters(three_scenes, fake_llm):
    main(["chapters", three_scenes, "--interval", "10", "--no-asr"])
    first = _index(three_scenes)
    calls = fake_llm.count("describe")
    cache = observations_path([three_scenes])
    assert main(["coalesce", cache, "--intent", "something else"]) == 0
    second = _index(three_scenes)
    assert second["video_id"] == first["video_id"] == "session"
    assert second["extra"]["intent"] == "something else"
    assert [e["start_time"] for e in second["events"]] == \
        [e["start_time"] for e in first["events"]]
    # re-chaptering re-reads no frames for Pass 1 (refinement midpoints only)
    assert fake_llm.count("refine") >= 2
    assert fake_llm.count("describe") - calls == fake_llm.count("refine") // 2


def test_label_keeps_extra(three_scenes, fake_llm):
    main(["chapters", three_scenes, "--interval", "10", "--no-asr"])
    assert main(["label", str(Path(three_scenes).with_suffix(".events.json")),
                 "--backend", "heuristic"]) == 0
    idx = _index(three_scenes)
    assert idx["extra"]["segmenter"] == "llm"
    assert idx["extra"]["titles"] == "heuristic"
    assert idx["extra"]["intent"] == ""
