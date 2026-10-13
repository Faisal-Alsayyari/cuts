"""The one place `cuts` talks to the Claude API.

Every model call in the pipeline — frame descriptions, coalescing, boundary
refinement, chapter titles — is a single structured-output request, so they
all go through `structured_call`. Centralizing it buys three things that were
otherwise duplicated (and slightly different) in four modules:

  * **One credential check.** `has_credentials` asks the SDK itself, so an
    `ant auth login` profile counts the same as `ANTHROPIC_API_KEY`.
  * **One client and one request cap.** A single `anthropic.Anthropic` is
    shared by every thread (it is thread-safe and pools connections), and a
    global semaphore bounds requests in flight across every shard and stage,
    rather than each thread pool picking its own number.
  * **One definition of "failed".** SDK errors, refusals, truncation, a
    missing text block and unparseable JSON all raise `LLMError`. Each caller
    then decides what failure means for it — coalescing falls back to the
    curve segmenter, labeling to heuristic titles, refinement to "keep the
    boundary", descriptions to a `failed` observation that the next run
    retries.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional, Union

# The server-side refusal fallback beta, used where a refusal would otherwise
# cost a whole stage (see labeling). "default" routes by refusal category.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

_client_lock = threading.Lock()
_client: Any = None
_semaphore = threading.BoundedSemaphore(8)


class LLMError(RuntimeError):
    """A request that did not produce a usable structured response."""


def configure(max_concurrency: int) -> None:
    """Set the global cap on requests in flight. Call before work starts."""
    global _semaphore
    _semaphore = threading.BoundedSemaphore(max(1, int(max_concurrency)))


def has_credentials() -> bool:
    """True when the SDK is installed and can resolve a credential.

    Covers every source the SDK accepts — `ANTHROPIC_API_KEY`,
    `ANTHROPIC_AUTH_TOKEN`, an `ant auth login` profile, workload identity —
    by constructing a client and asking it, rather than checking one
    environment variable and disagreeing with the SDK.
    """
    try:
        import anthropic
    except ImportError:
        return False
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    try:
        c = anthropic.Anthropic()
    except Exception:
        return False
    return bool(c.api_key or c.auth_token or getattr(c, "credentials", None))


def client():
    """The process-wide client, created on first use."""
    global _client
    with _client_lock:
        if _client is None:
            import anthropic

            _client = anthropic.Anthropic()
        return _client


def structured_call(
    *,
    model: str,
    system: str,
    content: Union[str, List[Dict[str, Any]]],
    schema: Dict[str, Any],
    max_tokens: int,
    effort: Optional[str] = None,
    fallbacks: bool = False,
) -> Dict[str, Any]:
    """One request whose response is constrained to `schema`. Returns the JSON.

    Raises `LLMError` on anything short of a parsed JSON object: an SDK
    exception (after the SDK's own retries), a refusal, a `max_tokens` stop,
    a response with no text block, or text that is not a JSON object.
    """
    output_config: Dict[str, Any] = {
        "format": {"type": "json_schema", "schema": schema},
    }
    if effort:
        output_config["effort"] = effort
    request: Dict[str, Any] = dict(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": content}],
        output_config=output_config,
    )

    try:
        with _semaphore:
            response = _send(request, fallbacks)
    except Exception as exc:
        raise LLMError(f"{model} request failed: {exc!r}") from exc

    if response.stop_reason == "refusal":
        detail = getattr(response, "stop_details", None)
        category = getattr(detail, "category", None) if detail else None
        raise LLMError(f"{model} refused the request ({category})")
    if response.stop_reason == "max_tokens":
        raise LLMError(f"{model} response hit max_tokens={max_tokens}")
    text = next((b.text for b in response.content if b.type == "text"), "")
    if not text:
        raise LLMError(f"{model} response contained no text block")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"{model} response was not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMError(f"{model} response was not a JSON object")
    return data


def _send(request: Dict[str, Any], fallbacks: bool):
    c = client()
    if fallbacks:
        import anthropic

        try:
            return c.beta.messages.create(
                betas=[_FALLBACK_BETA], fallbacks="default", **request,
            )
        except anthropic.BadRequestError:
            # The fallbacks beta is not available to this account or SDK
            # version; the request itself is still valid without it.
            pass
    return c.messages.create(**request)
