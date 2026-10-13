"""Pass 1: grid, cache, retries, resumption, embeddings, timeline offsets."""

from __future__ import annotations

import pytest

import cuts.observe as observe_mod
from cuts.media import Timeline
from cuts.observe import (
    FAILED,
    OK,
    Work,
    confirm_durations,
    observations_path,
    observe,
    read_observations,
    sample_grid,
)


def _run(paths, config, **work):
    tl = Timeline.build(paths)
    return tl, observe(tl, config, Work(**{"describe": True, "transcribe": False,
                                           **work}))


def test_grid_stays_strictly_inside_each_file(three_scenes, config):
    tl = Timeline.build([three_scenes])
    confirm_durations(tl)
    times = sample_grid(tl, 10.0)[0]
    assert times[0] == 0.0
    assert max(times) < tl.sources[0].duration_sec
    assert len(times) == 12


def test_fresh_run_then_cache_hit(three_scenes, config, fake_llm, capsys):
    _, obs = _run([three_scenes], config)
    first_calls = fake_llm.count("describe")
    assert first_calls > 0
    assert all(o.status == OK and o.description for o in obs)
    assert {o.description.split()[0] for o in obs} == {"red", "green", "blue"}

    _, again = _run([three_scenes], config)
    assert fake_llm.count("describe") == first_calls           # nothing re-described
    assert [o.description for o in again] == [o.description for o in obs]
    assert "reusing 12 cached" in capsys.readouterr().out


def test_failed_descriptions_are_retried_not_cached(three_scenes, config, fake_llm):
    fake_llm.fail["describe"] = 1
    _, obs = _run([three_scenes], config)
    failed = [o for o in obs if o.status == FAILED]
    assert failed and all(not o.description for o in failed)

    cached, _meta = read_observations(observations_path([three_scenes]))
    assert sum(o.status == FAILED for o in cached) == len(failed)

    before = fake_llm.count("describe")
    _, again = _run([three_scenes], config)
    assert all(o.status == OK and o.description for o in again)
    retried = sum(len([b for b in c["request"]["messages"][0]["content"]
                       if b["type"] == "image"])
                  for c in fake_llm.calls[before:] if c["kind"] == "describe")
    assert retried == len(failed)


def test_crashed_shard_does_not_lose_the_rest(three_scenes, config, fake_llm, monkeypatch):
    real = observe_mod.observe_shard
    crashed = {"done": False}

    def flaky(shard, cfg, verbose=False):
        if not crashed["done"]:
            crashed["done"] = True
            raise RuntimeError("worker died")
        return real(shard, cfg, verbose=verbose)

    monkeypatch.setattr(observe_mod, "observe_shard", flaky)
    config.observe.executor = "serial"
    _, obs = _run([three_scenes], config)
    assert sum(o.status == FAILED for o in obs) == 3        # one shard of 3 samples
    assert sum(o.status == OK for o in obs) == 9

    monkeypatch.setattr(observe_mod, "observe_shard", real)
    _, again = _run([three_scenes], config)
    assert all(o.status == OK for o in again)


def test_torn_cache_tail_is_ignored(three_scenes, config, fake_llm):
    _run([three_scenes], config)
    path = observations_path([three_scenes])
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"t": 3.0, "source_ind')
    obs, meta = read_observations(path)
    assert len(obs) == 12 and meta["version"] == observe_mod.CACHE_VERSION


def test_embeddings_are_added_without_redescribing(three_scenes, config, fake_llm, fake_dino):
    _run([three_scenes], config)
    calls = fake_llm.count("describe")
    _, obs = _run([three_scenes], config, embed=True)
    assert fake_llm.count("describe") == calls
    assert fake_dino["n"] == 12
    assert all(o.embedding is not None for o in obs)


def test_model_change_redescribes(three_scenes, config, fake_llm):
    _run([three_scenes], config)
    calls = fake_llm.count("describe")
    config.vlm.model = "another-model"
    _run([three_scenes], config)
    assert fake_llm.count("describe") > calls


def test_no_describe_skips_ocr_and_api(three_scenes, config, no_creds, fake_dino, monkeypatch):
    import cuts.redact as redact

    monkeypatch.setattr(redact, "scrub_frame",
                        lambda *a, **k: pytest.fail("OCR ran with nothing to upload"))
    _, obs = _run([three_scenes], config, describe=False, embed=True)
    assert all(o.status == OK and not o.description for o in obs)
    assert all(o.embedding is not None for o in obs)


def test_cached_and_fresh_runs_agree_on_offsets(two_files, config, fake_llm):
    tl1, fresh = _run(two_files, config)
    tl2, cached = _run(two_files, config)
    assert [s.time_offset for s in tl1.sources] == [s.time_offset for s in tl2.sources]
    assert [o.t for o in fresh] == [o.t for o in cached]
    second = [o for o in cached if o.source_index == 1]
    assert second[0].t == pytest.approx(tl2.sources[1].time_offset)


def test_multi_file_cache_does_not_collide_with_single(two_files):
    assert observations_path(two_files[:1]).endswith("day1.observations.jsonl")
    assert observations_path(two_files).endswith("day1+1.observations.jsonl")


def test_cache_survives_moving_the_videos(three_scenes, config, fake_llm, tmp_path):
    import shutil

    _run([three_scenes], config)
    calls = fake_llm.count("describe")
    moved_dir = tmp_path / "moved"
    moved_dir.mkdir()
    moved = shutil.move(three_scenes, moved_dir / "session.mp4")
    shutil.move(observations_path([three_scenes]), moved_dir / "session.observations.jsonl")
    _run([str(moved)], config)
    assert fake_llm.count("describe") == calls
