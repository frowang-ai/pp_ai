"""Python port of `packages/ai/test/anthropic-thinking-disable.test.ts`.

The trailing `describe.skipIf(!process.env.ANTHROPIC_API_KEY)` E2E block makes
live Anthropic calls; it is declared below as an explicit skip so the gap stays
visible in `pytest -rs` output rather than only in prose.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from pi_ai.compat import stream_simple
from pi_ai.providers.all import get_builtin_model
from pi_ai.types import Context, Model, SimpleStreamOptions, UserMessage


class PayloadCaptured(Exception):
    def __init__(self) -> None:
        super().__init__("payload captured")


def make_payload_capture_context() -> Context:
    return Context(messages=[UserMessage(content="Hello")])


async def capture_payload(model: Model, options: SimpleStreamOptions | None = None) -> dict[str, Any]:
    captured: dict[str, Any] | None = None

    def on_payload(payload: dict[str, Any], request_model: Model) -> None:
        nonlocal captured
        captured = payload
        raise PayloadCaptured()

    payload_capture_model = dataclasses.replace(model, base_url="http://127.0.0.1:9")
    request_options = dataclasses.replace(options or SimpleStreamOptions(), api_key="fake-key", on_payload=on_payload)

    await stream_simple(payload_capture_model, make_payload_capture_context(), request_options).result()

    assert captured is not None, "Expected payload to be captured before request failure"
    return captured


async def test_sends_thinking_disabled_for_budget_based_reasoning_models():
    payload = await capture_payload(get_builtin_model("anthropic", "claude-sonnet-4-5"))
    assert payload["thinking"] == {"type": "disabled"}
    assert "output_config" not in payload


async def test_sends_thinking_disabled_for_adaptive_reasoning_models():
    payload = await capture_payload(get_builtin_model("anthropic", "claude-opus-4-6"))
    assert payload["thinking"] == {"type": "disabled"}
    assert "output_config" not in payload


async def test_sends_thinking_disabled_for_claude_opus_4_8():
    payload = await capture_payload(get_builtin_model("anthropic", "claude-opus-4-8"))
    assert payload["thinking"] == {"type": "disabled"}
    assert "output_config" not in payload


async def test_omits_thinking_disabled_for_claude_fable_5():
    payload = await capture_payload(get_builtin_model("anthropic", "claude-fable-5"))
    assert "thinking" not in payload
    assert "output_config" not in payload


async def test_uses_adaptive_thinking_for_claude_opus_4_8_when_reasoning_enabled():
    payload = await capture_payload(
        get_builtin_model("anthropic", "claude-opus-4-8"), SimpleStreamOptions(reasoning="high")
    )
    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": "high"}


async def test_uses_adaptive_thinking_for_claude_sonnet_5_when_reasoning_enabled():
    payload = await capture_payload(
        get_builtin_model("anthropic", "claude-sonnet-5"), SimpleStreamOptions(reasoning="high")
    )
    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": "high"}


async def test_maps_xhigh_reasoning_to_effort_xhigh_for_claude_opus_4_8():
    payload = await capture_payload(
        get_builtin_model("anthropic", "claude-opus-4-8"), SimpleStreamOptions(reasoning="xhigh")
    )
    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": "xhigh"}


@pytest.mark.skip(
    reason="Live Anthropic call: TypeScript gates the whole describe block on ANTHROPIC_API_KEY, "
    "and the assertions are about a real streamed response, not a request payload."
)
async def test_disables_thinking_for_claude_reasoning_models_e2e():
    """`it("disables thinking for Claude reasoning models")` in the E2E block.

    Streams `claude-sonnet-4-5` with reasoning off and asserts the response
    carries no thinking events (`thinkingEventCount == 0`), no thinking
    characters, no `thinking` content block, and that the text still contains at
    least 35 "pong" repetitions (i.e. the model kept working with thinking off).
    """
