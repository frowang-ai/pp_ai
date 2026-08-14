"""Python port of `packages/ai/test/anthropic-force-adaptive-thinking.test.ts`."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from pi_ai.compat import stream_simple
from pi_ai.providers.all import get_builtin_model
from pi_ai.types import Context, Model, ModelCost, SimpleStreamOptions, UserMessage


class PayloadCaptured(Exception):
    def __init__(self) -> None:
        super().__init__("payload captured")


def make_context() -> Context:
    return Context(messages=[UserMessage(content="Hello")])


def make_custom_model(compat: dict[str, Any] | None = None) -> Model:
    return Model(
        # Id intentionally does not match any built-in adaptive substring. This
        # mirrors corporate proxy schemes such as `anthropic--claude-opus-latest`.
        id="vendor--claude-opus-latest",
        name="Vendor Proxy Opus Latest",
        api="anthropic-messages",
        provider="vendor-proxy",
        base_url="http://127.0.0.1:9",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=200000,
        max_tokens=32000,
        compat=compat or {},
    )


async def capture_payload(model: Model, options: SimpleStreamOptions | None = None) -> dict[str, Any]:
    captured: dict[str, Any] | None = None

    def on_payload(payload: dict[str, Any], request_model: Model) -> None:
        nonlocal captured
        captured = payload
        raise PayloadCaptured()

    payload_capture_model = dataclasses.replace(model, base_url="http://127.0.0.1:9")
    request_options = dataclasses.replace(options or SimpleStreamOptions(), api_key="fake-key", on_payload=on_payload)

    await stream_simple(payload_capture_model, make_context(), request_options).result()

    assert captured is not None, "Expected payload to be captured before request failure"
    return captured


async def test_sends_legacy_thinking_payload_for_custom_model_ids_by_default():
    payload = await capture_payload(make_custom_model(), SimpleStreamOptions(reasoning="medium"))

    assert payload["thinking"]["type"] == "enabled"
    assert "output_config" not in payload


async def test_sends_adaptive_thinking_payload_when_compat_force_adaptive_thinking_is_true():
    payload = await capture_payload(
        make_custom_model({"forceAdaptiveThinking": True}), SimpleStreamOptions(reasoning="medium")
    )

    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": "medium"}


async def test_uses_adaptive_thinking_with_native_xhigh_effort_for_claude_fable_5():
    payload = await capture_payload(
        get_builtin_model("anthropic", "claude-fable-5"), SimpleStreamOptions(reasoning="xhigh")
    )

    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": "xhigh"}


@pytest.mark.parametrize(
    ("model_id", "reasoning", "effort"),
    [
        ("kimi-for-coding", "medium", "medium"),
        ("k3", "max", "max"),
        ("kimi-for-coding-highspeed", "medium", "medium"),
    ],
)
async def test_uses_adaptive_thinking_effort_without_a_token_budget_for_kimi_coding(
    model_id: str, reasoning: str, effort: str
):
    payload = await capture_payload(
        get_builtin_model("kimi-coding", model_id), SimpleStreamOptions(reasoning=reasoning)
    )

    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": effort}


async def test_allows_built_in_adaptive_models_to_opt_out_with_compat_force_adaptive_thinking_false():
    model = dataclasses.replace(
        get_builtin_model("anthropic", "claude-opus-4-8"), compat={"forceAdaptiveThinking": False}
    )
    payload = await capture_payload(model, SimpleStreamOptions(reasoning="medium"))

    assert payload["thinking"]["type"] == "enabled"
    assert "output_config" not in payload


async def test_preserves_thinking_type_disabled_when_reasoning_is_off_regardless_of_override():
    payload = await capture_payload(make_custom_model({"forceAdaptiveThinking": True}))

    assert payload["thinking"] == {"type": "disabled"}
    assert "output_config" not in payload
