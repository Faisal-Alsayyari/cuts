"""Black out credentials in a frame before it leaves the machine.

Every sampled frame in this pipeline is uploaded to an external API. A screen
recording of real work will contain credentials sooner or later — a `.env`
file open in an editor, a terminal echoing a token, a dashboard showing a live
key — and nothing else in the pipeline looks at a frame before sending it.
This module is that look.

The approach is deliberately cheap: OCR already runs on every sampled frame to
ground the description call (`cuts.signals.ocr`), and the engine returns a
bounding box alongside every line it reads. So the credential check costs one
regex pass over text we already have, and a match already carries the pixel
coordinates needed to paint over it. No extra model, no extra decode.

**Redaction is two-sided, and both sides matter.** The pixels are filled solid
black, *and* the matched line is replaced in the grounding text — that text is
sent to the API verbatim in the prompt, so scrubbing only the image would ship
the key as a string instead of as pixels.

Solid fill, never blur: blur on high-contrast monospace text is not reliably
irreversible, and the point of this pass is that the secret is gone.

**What this does not do.** It catches what OCR actually read. A key too small
or too low-contrast to OCR, split across two boxes, or in a font the engine
mangles is not seen here and is not redacted. Patterns are anchored on
distinctive prefixes (`sk-`, `AKIA`, `ghp_`) and kept loose about the body
precisely because OCR drops and doubles characters — in testing it read
`OPENAI_API_KEY` as `OPENAl_API_KEY` — but that only buys robustness where the
prefix survived. This reduces exposure; it does not guarantee its absence, and
it is not a substitute for keeping credentials off screen while recording.
"""

from __future__ import annotations

import re
from typing import List, Optional, Pattern, Tuple

import numpy as np

from .config import OCRConfig, RedactConfig
from .signals.ocr import OCRLine, join_lines, ocr_lines

# Named so a hit can be reported ("openai_key") without ever echoing the
# matched string. Body lengths are deliberately loose — a real AWS key id is
# AKIA + exactly 16 chars, but OCR routinely drops or doubles one, and a
# credential that only *nearly* matches is still a credential.
_BUILTIN_PATTERNS: List[Tuple[str, str]] = [
    ("anthropic_key",  r"sk-ant-[A-Za-z0-9_\-]{16,}"),
    ("openai_key",     r"sk-(?:proj-)?[A-Za-z0-9_\-]{16,}"),
    ("aws_access_key", r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA)[0-9A-Z]{12,20}\b"),
    ("github_token",   r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    ("google_key",     r"\bAIza[0-9A-Za-z_\-]{30,}"),
    ("slack_token",    r"\bxox[baprs]-[A-Za-z0-9\-]{10,}"),
    ("stripe_key",     r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}"),
    ("hf_token",       r"\bhf_[A-Za-z0-9]{30,}"),
    ("jwt",            r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
    ("private_key",    r"-----\s?BEGIN[ A-Z]*PRIVATE KEY"),
    # The `.env` case, which is the common real leak. Requires an assignment
    # and a value of some length, so ordinary code (`token = tokens[i]`,
    # `password: str`) does not trip it.
    #
    # The leading `[A-Za-z0-9_\-]*` is load-bearing and easy to get wrong: a
    # `\b` there does NOT match inside `DATABASE_PASSWORD`, because `_` is a
    # word character, so the most common real shapes (`DATABASE_PASSWORD=`,
    # `MY_API_KEY=`, `APP_CLIENT_SECRET=`) would all slip through. The
    # *trailing* `\b` is what still keeps `get_secret_name(` and
    # `api_key_field.render()` from matching, since `_` blocks it there too.
    ("assignment",
     r"(?i)[A-Za-z0-9_\-]*(?:api[_\-]?key|secret[_\-]?key|secret"
     r"|access[_\-]?token|auth[_\-]?token|client[_\-]?secret"
     r"|password|passwd|bearer)\b"
     r"\s*[=:]\s*[\"']?[A-Za-z0-9_\-./+]{8,}"),
    # SCREAMING_SNAKE env vars ending in a credential-ish word, which is what
    # a `.env` file or an exported shell variable actually looks like and what
    # the pattern above misses when the name is not literally "API_KEY"
    # (`SUPABASE_ANON_KEY`, `GITHUB_PAT`, `STRIPE_SECRET`). Deliberately broad
    # enough to also catch an uppercase constant like `CACHE_KEY = "profile"` —
    # over-redaction costs one blacked-out line, under-redaction costs a key.
    ("env_assignment",
     r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*_"
     r"(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|PAT)\b"
     r"\s*[=:]\s*[\"']?[A-Za-z0-9_\-./+]{8,}"),
    # `Authorization: Bearer <token>` — separated by whitespace, not `=`, so
    # the assignment pattern above does not see it. The 16-char floor keeps
    # prose ("bearer tokens are issued by...") from matching.
    ("bearer_token", r"(?i)\bbearer\s+[\"']?[A-Za-z0-9_\-./+]{16,}"),
]

SECRET_PATTERNS: List[Tuple[str, Pattern[str]]] = [
    (name, re.compile(pat)) for name, pat in _BUILTIN_PATTERNS
]


def _patterns(config: RedactConfig) -> List[Tuple[str, Pattern[str]]]:
    """Built-in patterns plus any the user configured.

    A bad user regex is reported and skipped rather than raised: losing one
    extra pattern is a smaller failure than aborting a multi-hour run.
    """
    if not config.extra_patterns:
        return SECRET_PATTERNS
    extra: List[Tuple[str, Pattern[str]]] = []
    for i, pat in enumerate(config.extra_patterns):
        try:
            extra.append((f"custom_{i}", re.compile(pat)))
        except re.error as exc:
            print(f"[redact] ignoring invalid extra_patterns[{i}]: {exc}")
    return SECRET_PATTERNS + extra


def find_secret(
    text: str, patterns: Optional[List[Tuple[str, Pattern[str]]]] = None
) -> Optional[str]:
    """Return the name of the first credential pattern `text` matches.

    Returns the pattern *name*, never the matched text, so callers can log a
    hit without re-leaking what they just redacted.
    """
    for name, pat in (patterns if patterns is not None else SECRET_PATTERNS):
        if pat.search(text):
            return name
    return None


def _fill(img: np.ndarray, box: Tuple[int, int, int, int], pad: int) -> None:
    """Fill one box solid black, padded and clamped to the frame."""
    import cv2

    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    cv2.rectangle(
        img,
        (max(0, x1 - pad), max(0, y1 - pad)),
        (min(w - 1, x2 + pad), min(h - 1, y2 + pad)),
        (0, 0, 0),
        -1,
    )


def redact_lines(
    img: np.ndarray,
    lines: List[OCRLine],
    ocr_config: OCRConfig,
    config: RedactConfig,
) -> Tuple[str, bool, List[str]]:
    """Black out matching lines in `img` (in place) and scrub them from the text.

    Split out from `scrub_frame` so callers that already hold OCR lines — and
    tests that want to supply their own — do not have to re-run the engine.

    Returns `(grounding_text, safe, hits)`. `safe=False` means a credential
    was found that could not be tied to a box, so there is nothing to paint
    over and the frame must not be uploaded at all.
    """
    patterns = _patterns(config)
    hits: List[str] = []
    clean: List[OCRLine] = []
    untouched: List[str] = []

    for line in lines:
        name = find_secret(line.text, patterns)
        if name is None:
            clean.append(line)
            untouched.append(line.text)
            continue
        hits.append(name)
        _fill(img, line.box, config.pad_px)
        clean.append(OCRLine(config.placeholder, line.box, line.score))

    # A credential split across two lines matches their concatenation but
    # neither line alone, so no single box covers it and there is nothing to
    # paint over. Checked over the lines we did *not* touch — checking the
    # original text would re-match every line just redacted and condemn the
    # frame for secrets that are already gone. Mostly this catches the
    # assignment pattern, whose `\s*[=:]\s*` spans a line break.
    unlocalized = None
    if config.skip_frame_on_unlocalized:
        unlocalized = find_secret("\n".join(untouched), patterns)
        if unlocalized is not None:
            hits.append(unlocalized)

    text = join_lines(clean, ocr_config)
    return text, unlocalized is None, hits


def scrub_frame(
    img: np.ndarray,
    ocr_config: OCRConfig,
    config: RedactConfig,
    verbose: bool = False,
) -> Tuple[str, bool, List[str]]:
    """OCR `img`, black out credentials **in place**, return grounding text.

    The single chokepoint every upload path goes through. `img` is mutated,
    which is intended: a redacted frame should stay redacted for whatever
    else touches it downstream.

    Returns `(grounding_text, safe, hits)`:
      * `grounding_text` — what to send as the OCR hint, secrets replaced.
      * `safe` — False means **do not upload this frame**, either because OCR
        failed (so nothing about the frame can be vouched for) or because a
        credential could not be localized to a box.
      * `hits` — names of the patterns that matched, for reporting. Never the
        matched text.

    With `config.enabled=False` this is exactly the old `ocr_image` path: OCR
    and join, no pixels touched, always safe.
    """
    lines = ocr_lines(img, ocr_config, verbose=verbose)
    if lines is None:
        # OCR failed. With redaction off this is the long-standing
        # best-effort behaviour (no grounding text, frame still sent). With it
        # on, an unreadable frame is one we cannot vouch for, and the whole
        # point of this pass is not to send those.
        return "", not config.enabled, []
    if not config.enabled:
        return join_lines(lines, ocr_config), True, []
    return redact_lines(img, lines, ocr_config, config)


if __name__ == "__main__":
    # Standalone debug: render a frame with planted credentials, scrub it, and
    # prove the secrets are gone from both the text and the pixels — including
    # after the resize + JPEG round trip a real upload goes through.
    import cv2

    from .config import CutsConfig
    from .media import resize_long_edge

    cfg = CutsConfig()
    frame = np.full((1080, 1920, 3), 30, np.uint8)
    planted = [
        "OPENAI_API_KEY=sk-proj-AbC123xyzQWERTY7890",
        "AKIAIOSFODNN7EXAMPLE",
        "def parse_tokens(src):",            # benign, must survive
        "token = tokens[i]",                 # benign, must survive
    ]
    for i, line in enumerate(planted):
        cv2.putText(frame, line, (60, 200 + i * 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)

    text, safe, hits = scrub_frame(frame, cfg.ocr, cfg.redact, verbose=True)
    print(f"safe={safe}  hits={hits}")
    print(f"grounding text:\n{text}\n")

    sent = cv2.imdecode(cv2.imencode(
        ".jpg", resize_long_edge(frame, cfg.vlm.frame_long_edge),
        [int(cv2.IMWRITE_JPEG_QUALITY), 85],
    )[1], cv2.IMREAD_COLOR)
    readback = ocr_lines(sent, cfg.ocr) or []
    print("re-OCR of what would actually be uploaded:")
    for ln in readback:
        leak = find_secret(ln.text)
        print(f"  {'LEAK ' + leak if leak else 'ok   '}  {ln.text!r}")
