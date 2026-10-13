"""The one pipeline, end to end on synthetic footage with a fake Claude."""

from __future__ import annotations

import pytest

import cuts.pipeline as pipeline
from cuts.pipeline import FALLBACK, PipelineError, run


def test_llm_without_credentials_fails_before_decoding(three_scenes, config,
                                                       no_creds, monkeypatch):
    monkeypatch.setattr(pipeline, "observe",
                        lambda *a, **k: pytest.fail("decoded without credentials"))
    with pytest.raises(PipelineError, match="credentials"):
        run([three_scenes], config)


def test_llm_path(three_scenes, config, fake_llm):
    result = run([three_scenes], config)
    assert result.segmenter == "llm" and result.titles == "llm"
    assert [e.title for e in result.events] == [
        "working on red", "working on green", "working on blue"]
    assert [round(e.start_time) for e in result.events] == [0, 40, 80]
    assert all(e.reason and e.metadata["kind"] == "active" for e in result.events)
    assert result.events[-1].end_time == pytest.approx(result.duration_sec)
    extra = result.extra("x")
    assert extra["segmenter"] == "llm" and extra["titles"] == "llm"
    assert extra["samples"]["described"] == 12
    assert fake_llm.count("refine") == 2          # both detected boundaries looked at


def test_refinement_describes_one_midpoint_per_call(three_scenes, config, fake_llm):
    run([three_scenes], config)
    # midpoint describes come after the coalesce call
    after = fake_llm.calls[[c["kind"] for c in fake_llm.calls].index("coalesce"):]
    for c in after:
        if c["kind"] == "describe":
            images = [b for b in c["request"]["messages"][0]["content"]
                      if b["type"] == "image"]
            assert len(images) == 1


def test_coalesce_failure_falls_back_to_curve(three_scenes, config, fake_llm):
    fake_llm.fail["coalesce"] = 1
    result = run([three_scenes], config)
    assert result.segmenter == FALLBACK
    assert result.titles == "claude"
    assert len(result.events) >= 2
    assert all(e.title.startswith("segment") for e in result.events)


def test_no_labels_still_coalesces(three_scenes, config, fake_llm):
    config.labeling.enabled = False
    result = run([three_scenes], config)
    assert result.segmenter == "llm" and result.events[0].title


def test_curve_without_credentials(three_scenes, config, no_creds, fake_dino):
    config.segmenter = "curve"
    config.segmentation.min_event_sec = 20
    config.segmentation.target_events = 3
    result = run([three_scenes], config)
    assert result.segmenter == "curve" and result.titles == "heuristic"
    assert "dino" in result.channels
    assert [round(e.start_time) for e in result.events] == [0, 40, 80]


def test_curve_with_credentials_uses_both_channels(three_scenes, config, fake_llm, fake_dino):
    config.segmenter = "curve"
    config.segmentation.target_events = 3
    result = run([three_scenes], config)
    assert set(result.channels) == {"dino", "vlm"}
    assert result.titles == "claude"
    assert [round(e.start_time) for e in result.events] == [0, 40, 80]


def test_multi_file_join_is_a_boundary_with_source_offsets(two_files, config, fake_llm):
    result = run(two_files, config)
    starts = [(e.source_index, round(e.source_start_time)) for e in result.events]
    # day2 opens on the same green screen day1 ended on: still a new chapter
    assert (1, 0) in starts
    assert result.events[0].source_path.endswith("day1.mp4")


def test_llm_with_every_description_failed_is_an_error(three_scenes, config, fake_llm):
    fake_llm.fail["describe"] = 100
    with pytest.raises(PipelineError, match="described"):
        run([three_scenes], config)
