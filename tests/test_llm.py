"""The shared request path: every failure mode raises LLMError."""

from __future__ import annotations

import pytest

import cuts.llm as llm
from cuts.signals.vlm import decide_refinement, describe_frames
from cuts.config import VLMConfig

DESCRIBE_SYSTEM = "You are scouting footage from a screen-recorded coding session to help index"


def _call():
    return llm.structured_call(model="m", system="You are indexing x", content="hi",
                               schema={}, max_tokens=10)


@pytest.mark.parametrize("failure", ["error", "refusal", "max_tokens", "badjson"])
def test_structured_call_raises(fake_llm, failure):
    fake_llm.failure = failure
    fake_llm.fail["coalesce"] = 1
    with pytest.raises(llm.LLMError):
        _call()


def test_structured_call_returns_json(fake_llm):
    assert "chapters" in _call()


def test_fallbacks_retry_without_beta_on_bad_request(fake_llm, monkeypatch):
    import anthropic

    class _Bad(anthropic.BadRequestError):
        def __init__(self):
            Exception.__init__(self, "no beta")

    def reject(**kw):
        raise _Bad()

    monkeypatch.setattr(fake_llm.beta.messages, "create", reject)
    data = llm.structured_call(model="m", system="You write chapter titles", content=[],
                               schema={}, max_tokens=10, fallbacks=True)
    assert data == {"chapters": []}


def test_describe_failure_is_none_not_empty(fake_llm):
    fake_llm.fail["describe"] = 1
    assert describe_frames(["x", "y"], VLMConfig()) == [None, None]


def test_refinement_failure_is_unclear(fake_llm):
    fake_llm.fail["refine"] = 1
    assert decide_refinement("a", "m", "b", "m", 10) == "unclear"


def test_has_credentials_honours_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert llm.has_credentials()
