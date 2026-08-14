"""Python port of `packages/ai/test/anthropic-empty-thinking-signature-compat.test.ts`."""

from __future__ import annotations

from typing import Any

import pytest
from pi_ai.compat import stream_simple
from pi_ai.providers.all import get_builtin_model
from pi_ai.types import (
    AssistantMessage,
    Context,
    Model,
    ModelCost,
    SimpleStreamOptions,
    ThinkingContent,
    Usage,
    UserMessage,
)


class PayloadCaptured(Exception):
    def __init__(self) -> None:
        super().__init__("payload captured")


def make_model(allow_empty_signature: bool | None = None) -> Model:
    return Model(
        id="mimo-v2.5-pro",
        name="MiMo-V2.5-Pro",
        api="anthropic-messages",
        provider="xiaomi-token-plan-ams",
        base_url="http://127.0.0.1:9/anthropic",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=1048576,
        max_tokens=1024,
        compat={} if allow_empty_signature is None else {"allowEmptySignature": allow_empty_signature},
    )


def make_context(
    thinking_signature: str,
    thinking: str = "internal reasoning",
    provider: str = "xiaomi-token-plan-ams",
    model: str = "mimo-v2.5-pro",
) -> Context:
    assistant = AssistantMessage(
        content=[ThinkingContent(thinking=thinking, thinking_signature=thinking_signature)],
        provider=provider,
        api="anthropic-messages",
        model=model,
        usage=Usage(),
        stop_reason="stop",
    )
    return Context(
        messages=[
            UserMessage(content="first"),
            assistant,
            UserMessage(content="second"),
        ]
    )


async def capture_payload(model: Model, context: Context) -> dict[str, Any]:
    captured: dict[str, Any] | None = None

    def on_payload(payload: dict[str, Any], request_model: Model) -> None:
        nonlocal captured
        captured = payload
        raise PayloadCaptured()

    await stream_simple(model, context, SimpleStreamOptions(api_key="fake-key", on_payload=on_payload)).result()

    assert captured is not None, "Expected payload capture before request"
    return captured


def _assistant_content(payload: dict[str, Any]) -> list[dict[str, Any]]:
    assistant = next(message for message in payload["messages"] if message["role"] == "assistant")
    return assistant["content"]


async def test_converts_empty_signature_thinking_to_text_by_default():
    payload = await capture_payload(make_model(), make_context(""))
    assert _assistant_content(payload) == [{"type": "text", "text": "internal reasoning"}]


async def test_preserves_empty_thinking_text_when_the_signature_is_present():
    payload = await capture_payload(make_model(), make_context("signed-thinking", ""))
    assert _assistant_content(payload) == [{"type": "thinking", "thinking": "", "signature": "signed-thinking"}]


async def test_preserves_empty_signature_thinking_when_allow_empty_signature_is_enabled():
    payload = await capture_payload(make_model(True), make_context(" "))
    assert _assistant_content(payload) == [{"type": "thinking", "thinking": "internal reasoning", "signature": ""}]


@pytest.mark.parametrize("model_id", ["k3"])
async def test_allows_empty_signatures_for_kimi_coding(model_id: str):
    model = get_builtin_model("kimi-coding", model_id)
    assert model.compat.get("allowEmptySignature") is True

    payload = await capture_payload(model, make_context(" ", "internal reasoning", "kimi-coding", model_id))
    assert _assistant_content(payload) == [{"type": "thinking", "thinking": "internal reasoning", "signature": ""}]
