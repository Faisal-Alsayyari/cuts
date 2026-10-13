"""Credentials never reach a request: pixels, grounding text, or transcript."""

from __future__ import annotations

import base64
import json

import cv2
import numpy as np

import cuts.observe as observe_mod
from cuts.config import OCRConfig, RedactConfig
from cuts.media import Timeline
from cuts.observe import WITHHELD, Work, observe
from cuts.redact import find_secret, scrub_text
from cuts.signals.ocr import ocr_lines

from .conftest import SECRET


def test_scrub_text():
    text, hits = scrub_text(f"so the key is {SECRET} okay", RedactConfig())
    assert "sk-proj" not in text and "[REDACTED]" in text
    assert hits
    assert scrub_text("token = tokens[i]", RedactConfig()) == ("token = tokens[i]", [])
    off = RedactConfig(enabled=False)
    assert scrub_text(SECRET, off) == (SECRET, [])


def test_nothing_uploaded_contains_the_secret(secret_video, config, fake_llm):
    tl = Timeline.build([secret_video])
    obs = observe(tl, config, Work(describe=True, transcribe=False))
    assert any(o.redacted for o in obs), "the planted key was never detected"

    for call in fake_llm.calls:
        content = call["request"]["messages"][0]["content"]
        for block in content if isinstance(content, list) else []:
            if block["type"] == "text":
                assert "sk-proj" not in block["text"]
            else:
                img = cv2.imdecode(np.frombuffer(
                    base64.b64decode(block["source"]["data"]), np.uint8),
                    cv2.IMREAD_COLOR)
                for line in ocr_lines(img, OCRConfig()) or []:
                    assert find_secret(line.text) is None, line.text

    cache = open(observe_mod.observations_path([secret_video]), encoding="utf-8").read()
    assert "sk-proj" not in cache
    assert all(o.status != WITHHELD or not o.description for o in obs)


def test_spoken_secret_is_scrubbed(three_scenes, config, fake_llm, monkeypatch):
    import cuts.signals.asr as asr

    monkeypatch.setattr(observe_mod, "has_audio", lambda p: True)
    monkeypatch.setattr(asr, "transcribe", lambda *a, **k: [
        (0.0, 15.0, f"my key is {SECRET}"), (20.0, 35.0, "now building"),
    ])
    config.asr.enabled = True
    tl = Timeline.build([three_scenes])
    obs = observe(tl, config, Work(describe=True, transcribe=True))
    said = " ".join(o.transcript_text for o in obs)
    assert "now building" in said and "[REDACTED]" in said and "sk-proj" not in said

    raw = open(observe_mod.observations_path([three_scenes]), encoding="utf-8").read()
    assert "sk-proj" not in raw
    meta = json.loads(raw.splitlines()[0])["_meta"]
    assert meta["asr"]["model"] == config.asr.model
