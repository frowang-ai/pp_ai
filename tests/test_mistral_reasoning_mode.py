"""Python port of `packages/ai/test/mistral-reasoning-mode.test.ts`."""

from __future__ import annotations

import dataclasses
from typing import Any

from pi_ai.compat import stream_simple
from pi_ai.providers.all import get_builtin_model
from pi_ai.types import Context, Model, SimpleStreamOptions, UserMessage, now_ms


def make_context() -> Context:
    return Context(messages=[UserMessage(content="Hello", timestamp=now_ms())])


async def capture_payload(model: Model, options: SimpleStreamOptions | None = None) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    # Port 9 (discard) refuses immediately: the payload is captured before the
    # request fails, exactly as in the TypeScript test.
    payload_capture_model = dataclasses.replace(model, base_url="http://127.0.0.1:9")

    request_options = dataclasses.replace(options) if options is not None else SimpleStreamOptions()
    request_options.api_key = "fake-key"

    def on_payload(payload: dict[str, Any], _model: Model) -> dict[str, Any]:
        captured["payload"] = payload
        return payload

    request_options.on_payload = on_payload

    await stream_simple(payload_capture_model, make_context(), request_options).result()

    if "payload" not in captured:
        raise AssertionError("Expected payload to be captured before request failure")
    return captured["payload"]


async def test_uses_reasoning_effort_for_mistral_small_4() -> None:
    payload = await capture_payload(
        get_builtin_model("mistral", "mistral-small-2603"),
        SimpleStreamOptions(reasoning="medium"),
    )
    assert payload.get("reasoningEffort") == "high"
    assert payload.get("promptMode") is None


async def test_omits_reasoning_controls_for_mistral_small_4_when_thinking_is_off() -> None:
    payload = await capture_payload(get_builtin_model("mistral", "mistral-small-2603"))
    assert payload.get("reasoningEffort") is None
    assert payload.get("promptMode") is None


async def test_uses_prompt_mode_for_magistral_reasoning_models() -> None:
    payload = await capture_payload(
        get_builtin_model("mistral", "magistral-medium-latest"),
        SimpleStreamOptions(reasoning="medium"),
    )
    assert payload.get("promptMode") == "reasoning"
    assert payload.get("reasoningEffort") is None


async def test_uses_reasoning_effort_for_mistral_medium_35() -> None:
    payload = await capture_payload(
        get_builtin_model("mistral", "mistral-medium-3.5"),
        SimpleStreamOptions(reasoning="medium"),
    )
    assert payload.get("reasoningEffort") == "high"
    assert payload.get("promptMode") is None


async def test_omits_reasoning_controls_for_mistral_medium_35_when_thinking_is_off() -> None:
    payload = await capture_payload(get_builtin_model("mistral", "mistral-medium-3.5"))
    assert payload.get("reasoningEffort") is None
    assert payload.get("promptMode") is None


async def test_uses_the_session_id_as_prompt_cache_key() -> None:
    payload = await capture_payload(
        get_builtin_model("mistral", "mistral-large-latest"),
        SimpleStreamOptions(session_id="session-123"),
    )
    assert payload.get("promptCacheKey") == "session-123"


async def test_omits_prompt_cache_key_when_cache_retention_is_disabled() -> None:
    payload = await capture_payload(
        get_builtin_model("mistral", "mistral-large-latest"),
        SimpleStreamOptions(session_id="session-123", cache_retention="none"),
    )
    assert payload.get("promptCacheKey") is None
