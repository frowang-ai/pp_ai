"""Python port of `packages/ai/test/anthropic-temperature-compat.test.ts`."""

from __future__ import annotations

import dataclasses
from typing import Any

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
        id="vendor--claude-opus-4-7",
        name="Vendor Proxy Opus 4.7",
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


async def test_omits_temperature_for_claude_opus_4_7():
    payload = await capture_payload(
        get_builtin_model("anthropic", "claude-opus-4-7"), SimpleStreamOptions(temperature=0)
    )
    assert "temperature" not in payload


async def test_omits_temperature_for_claude_opus_4_8():
    payload = await capture_payload(
        get_builtin_model("anthropic", "claude-opus-4-8"), SimpleStreamOptions(temperature=0)
    )
    assert "temperature" not in payload


async def test_omits_default_temperature_for_claude_opus_4_7():
    payload = await capture_payload(
        get_builtin_model("anthropic", "claude-opus-4-7"), SimpleStreamOptions(temperature=1)
    )
    assert "temperature" not in payload


async def test_keeps_temperature_for_claude_opus_4_6():
    payload = await capture_payload(
        get_builtin_model("anthropic", "claude-opus-4-6"), SimpleStreamOptions(temperature=0)
    )
    assert payload["temperature"] == 0


async def test_keeps_temperature_for_claude_sonnet_4_6():
    payload = await capture_payload(
        get_builtin_model("anthropic", "claude-sonnet-4-6"), SimpleStreamOptions(temperature=0)
    )
    assert payload["temperature"] == 0


async def test_omits_temperature_for_custom_models_with_supports_temperature_disabled():
    payload = await capture_payload(
        make_custom_model({"supportsTemperature": False}), SimpleStreamOptions(temperature=0)
    )
    assert "temperature" not in payload
