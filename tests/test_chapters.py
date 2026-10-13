"""Coalescing validation and the shared event builder (no video, no API)."""

from __future__ import annotations

import pytest

from cuts.chapters import enforce_min_duration, events_from_starts, join_positions
from cuts.coalesce import _validate_and_repair
from cuts.observe import WITHHELD, Observation


def _obs(n, step=30.0, sources=None):
    sources = sources or [0] * n
    return [Observation(t=i * step, source_index=s, source_time=i * step,
                        description=f"d{i}") for i, s in enumerate(sources)]


def _starts(chapters):
    return [c["start"] for c in chapters]


@pytest.mark.parametrize("raw", [
    [{"start": 5, "title": "b"}, {"start": 0, "title": "a"}],                 # unordered
    [{"start": 0, "title": "a"}, {"start": 0, "title": "dup"}],               # duplicate
    [{"start": 0}, {"start": 99}, {"start": -3}],                             # out of range
    [{"start": None}, {"start": "4"}, {"title": "no start"}, {"start": "x"}],  # junk values
    [{"start": 3, "title": "late"}],                                          # missing 0
    [{"start": i} for i in range(20)],                                        # all too short
    [],                                                                       # empty
])
def test_validate_always_yields_a_partition(raw):
    obs = _obs(20)
    out = _validate_and_repair(raw, obs, min_event_sec=60, duration_sec=600)
    starts = _starts(out)
    assert starts[0] == 0
    assert starts == sorted(set(starts))
    assert all(0 <= s < len(obs) for s in starts)
    # min duration holds between consecutive starts, and for the last chapter
    times = [o.t for o in obs]
    for a, b in zip(starts, starts[1:]):
        assert times[b] - times[a] >= 60
    events = events_from_starts(starts, obs, "v", 600, info=out)
    assert events[0].start_time == 0 and events[-1].end_time == 600
    for a, b in zip(events, events[1:]):
        assert a.end_time == b.start_time


def test_validate_keeps_dropped_joins_and_maps_why_to_reason():
    obs = _obs(10, sources=[0] * 5 + [1] * 5)
    joins = join_positions(obs)
    assert joins == [5]
    out = _validate_and_repair(
        [{"start": 0, "title": "a", "kind": "pivotal", "why": "because"}],
        obs, min_event_sec=20, duration_sec=300, locked=joins,
    )
    assert _starts(out) == [0, 5]
    assert out[0]["reason"] == "because"
    assert out[0]["metadata"]["kind"] == "pivotal"
    assert out[1]["reason"].startswith("inserted")


def test_unknown_kind_is_normalized():
    out = _validate_and_repair([{"start": 0, "kind": "bogus"}], _obs(3), 0, 90)
    assert out[0]["metadata"]["kind"] == "active"


def test_min_duration_respects_locked_joins():
    times = [0, 10, 20, 30, 40]
    # 30s survives the walk, then the last chapter (30->50) is too short
    assert enforce_min_duration([1, 2, 3], times, 25, 50) == []
    assert enforce_min_duration([1, 2, 3], times, 25, 55) == [3]
    assert enforce_min_duration([1, 2, 3], times, 25, 50, locked=[1]) == [1]
    # a locked final boundary survives even when the last chapter is short
    assert enforce_min_duration([4], times, 25, 45, locked=[4]) == [4]


def test_events_carry_evidence_and_skip_withheld_representatives():
    obs = _obs(6)
    obs[1].status = WITHHELD
    obs[2].description = obs[3].description = "same"
    ev = events_from_starts([0, 4], obs, "vid", 180,
                            info=[{"title": "t", "reason": "r",
                                   "metadata": {"kind": "active"}}])
    assert [e.event_id for e in ev] == ["0000", "0001"]
    assert ev[0].vlm_text.split("\n") == ["d0", "d1", "same"]
    assert ev[0].metadata["n_withheld"] == 1
    assert 30.0 not in ev[0].representative_times
    assert ev[0].reason == "r" and ev[1].title == ""
