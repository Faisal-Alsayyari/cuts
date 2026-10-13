"""Shared fixtures: synthetic recordings and a scripted stand-in for Claude.

Nothing here touches the network. `fake_llm` replaces the client that
`cuts.llm` hands out, and answers each kind of request the pipeline makes
(describe, coalesce, label, refine) from what is actually in the request —
descriptions are derived from the colour of the uploaded JPEG, so the whole
pipeline can run end to end and produce meaningful chapters.
"""

from __future__ import annotations

import base64
import json
import re
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import av
import cv2
import numpy as np
import pytest

import cuts.llm as llm

# ---------------------------------------------------------------------------
# Synthetic videos
# ---------------------------------------------------------------------------

COLORS: Dict[str, Tuple[int, int, int]] = {   # BGR
    "red": (40, 40, 200),
    "green": (40, 180, 40),
    "blue": (200, 60, 40),
    "gray": (110, 110, 110),
}

SECRET = "OPENAI_API_KEY=sk-proj-AbC123xyzQWERTY7890"


def make_video(
    path, scenes: Sequence[Tuple[str, float, Sequence[str]]],
    fps: int = 2, size=(640, 360),
) -> str:
    """Encode an h264 mp4 of solid-colour scenes, each with lines of text."""
    w, h = size
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    stream.codec_context.gop_size = fps * 2      # keyframe every 2s: seekable
    for color, seconds, lines in scenes:
        img = np.full((h, w, 3), COLORS[color], np.uint8)
        for i, line in enumerate(lines):
            cv2.putText(img, line, (20, 60 + 50 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)
        for _ in range(int(seconds * fps)):
            frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return str(path)


@pytest.fixture
def three_scenes(tmp_path) -> str:
    """120s: red editor, green terminal, blue browser — 40s each."""
    return make_video(tmp_path / "session.mp4", [
        ("red", 40, ["parser.py", "def tokenize(src):"]),
        ("green", 40, ["$ cargo build", "error[E0308]"]),
        ("blue", 40, ["docs.rs", "serde::Deserialize"]),
    ])


@pytest.fixture
def two_files(tmp_path) -> List[str]:
    """Two recordings; the second opens on the same screen the first ended on."""
    a = make_video(tmp_path / "day1.mp4", [
        ("red", 40, ["parser.py"]), ("green", 40, ["$ cargo test"]),
    ])
    b = make_video(tmp_path / "day2.mp4", [
        ("green", 40, ["$ cargo test"]), ("blue", 40, ["docs.rs"]),
    ])
    return [a, b]


@pytest.fixture
def secret_video(tmp_path) -> str:
    """A .env file on screen in the middle scene."""
    return make_video(tmp_path / "leak.mp4", [
        ("gray", 30, ["def parse_tokens(src):"]),
        ("gray", 30, [".env", SECRET]),
        ("gray", 30, ["token = tokens[i]"]),
    ])


# ---------------------------------------------------------------------------
# A stand-in for the Anthropic client
# ---------------------------------------------------------------------------

def jpeg_color(b64: str) -> str:
    """Name of the palette colour nearest the image's mean."""
    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8),
                       cv2.IMREAD_COLOR)
    mean = img.reshape(-1, 3).mean(axis=0)
    return min(COLORS, key=lambda c: float(np.sum((np.array(COLORS[c]) - mean) ** 2)))


def _response(payload: Any = None, stop_reason: str = "end_turn", text: Optional[str] = None):
    if text is None:
        text = json.dumps(payload)
    return SimpleNamespace(
        stop_reason=stop_reason, stop_details=None,
        content=[SimpleNamespace(type="text", text=text)],
    )


class FakeClient:
    """Answers pipeline requests; scriptable to fail.

    `fail` maps a request kind ("describe", "coalesce", "label", "refine")
    to how many of the next such requests should fail, and `failure` says
    how ("error", "refusal", "max_tokens", "badjson").
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.fail: Dict[str, int] = {}
        self.failure = "error"
        self.messages = SimpleNamespace(create=self._create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create_beta))

    def count(self, kind: str) -> int:
        return sum(1 for c in self.calls if c["kind"] == kind)

    def _create_beta(self, betas=None, fallbacks=None, **request):
        return self._create(**request)

    def _create(self, **request):
        kind = _kind(request["system"])
        self.calls.append({"kind": kind, "request": request})
        if self.fail.get(kind, 0) > 0:
            self.fail[kind] -= 1
            if self.failure == "error":
                raise RuntimeError("simulated API failure")
            if self.failure == "badjson":
                return _response(text="not json")
            return _response({}, stop_reason=self.failure)
        return _response(HANDLERS[kind](request))


def _kind(system: str) -> str:
    if system.startswith("You are scouting footage from a screen-recorded coding session to help"):
        return "describe"
    if "pin down" in system:
        return "refine"
    if system.startswith("You are indexing"):
        return "coalesce"
    if system.startswith("You write chapter titles"):
        return "label"
    raise AssertionError(f"unexpected system prompt: {system[:60]}")


def _describe(request) -> Dict[str, Any]:
    content = request["messages"][0]["content"]
    images = [b["source"]["data"] for b in content if b["type"] == "image"]
    return {"descriptions": [
        {"frame": i, "description": f"{c} screen, editing {c}.py in an editor"}
        for i, c in enumerate(jpeg_color(b) for b in images)
    ]}


_OBS_LINE = re.compile(r"^\[(\d+)\] \S+\s+(.*)$")


def _coalesce(request) -> Dict[str, Any]:
    chapters, last = [], None
    for line in request["messages"][0]["content"].splitlines():
        m = _OBS_LINE.match(line)
        if not m:
            continue
        idx, desc = int(m[1]), m[2]
        topic = desc.split()[0] if not desc.startswith("(") else last
        if topic != last:
            chapters.append({"start": idx, "title": f"working on {topic}",
                             "kind": "active", "why": f"obs {idx} turns {topic}"})
            last = topic
    return {"chapters": chapters}


def _label(request) -> Dict[str, Any]:
    out = []
    for block in request["messages"][0]["content"]:
        if block["type"] == "text":
            m = re.match(r"^\[(\d{4})\]", block["text"])
            if m:
                out.append({"event_id": m[1], "title": f"segment {m[1]}"})
    return {"chapters": out}


HANDLERS: Dict[str, Callable] = {
    "describe": _describe,
    "coalesce": _coalesce,
    "label": _label,
    "refine": lambda r: {"transition": "unclear"},
}


class _NoNetwork:
    def __getattr__(self, name):
        raise AssertionError("a test reached for the real Anthropic client")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Every test starts with a client that refuses to be used."""
    monkeypatch.setattr(llm, "_client", _NoNetwork())


@pytest.fixture
def fake_llm(monkeypatch) -> FakeClient:
    fake = FakeClient()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_client", fake)
    llm.configure(4)
    return fake


@pytest.fixture
def no_creds(monkeypatch):
    monkeypatch.setattr(llm, "has_credentials", lambda: False)

    def _boom():
        raise AssertionError("an API call was attempted without credentials")
    monkeypatch.setattr(llm, "client", _boom)


@pytest.fixture
def fake_dino(monkeypatch):
    """DINOv2 stand-in: the image's mean colour, L2-normalized."""
    import cuts.signals.dino as dino

    calls = {"n": 0}

    def embed_images(images, config, device="cpu", batch_size=16, verbose=False):
        vecs = []
        for img in images:
            calls["n"] += 1
            v = img.reshape(-1, 3).mean(axis=0).astype(np.float32) + 1.0
            vecs.append(v / np.linalg.norm(v))
        return np.asarray(vecs, dtype=np.float32)

    monkeypatch.setattr(dino, "embed_images", embed_images)
    return calls


@pytest.fixture
def config():
    """Small, fast settings for the synthetic fixtures."""
    from cuts.config import CutsConfig

    cfg = CutsConfig()
    cfg.device = "cpu"
    cfg.sampling.sample_interval_sec = 10.0
    cfg.observe.shard_sec = 30.0
    cfg.observe.workers = 2
    cfg.asr.enabled = False
    return cfg
