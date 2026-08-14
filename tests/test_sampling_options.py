"""Python port of `packages/ai/test/sampling-options.test.ts`."""

from __future__ import annotations

import dataclasses
from typing import Any

from pi_ai.compat import stream_simple
from pi_ai.types import Context, Model, ModelCost, SimpleStreamOptions, UserMessage, now_ms


class PayloadCaptured(Exception):
    """Raised from the payload hook to stop the request before it is sent."""


def make_context() -> Context:
    return Context(messages=[UserMessage(content="Hello", timestamp=now_ms())])


def make_completions_model(**overrides: Any) -> Model:
    defaults: dict[str, Any] = dict(
        id="custom-model",
        name="Custom Model",
        api="openai-completions",
        provider="custom-provider",
        base_url="http://127.0.0.1:9/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128000,
        max_tokens=16384,
    )
    defaults.update(overrides)
    return Model(**defaults)


def make_anthropic_model() -> Model:
    return Model(
        id="vendor--claude",
        name="Vendor Proxy Claude",
        api="anthropic-messages",
        provider="vendor-proxy",
        base_url="http://127.0.0.1:9",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=200000,
        max_tokens=32000,
    )


async def capture_payload(model: Model, options: SimpleStreamOptions | None = None) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    request_options = dataclasses.replace(options) if options is not None else SimpleStreamOptions()
    request_options.api_key = "fake-key"

    def on_payload(payload: dict[str, Any], _model: Model) -> None:
        captured["payload"] = payload
        raise PayloadCaptured()

    request_options.on_payload = on_payload

    await stream_simple(model, make_context(), request_options).result()

    if "payload" not in captured:
        raise AssertionError("Expected payload to be captured before request failure")
    return captured["payload"]


async def test_merges_stream_option_sampling_params_into_the_request_body() -> None:
    payload = await capture_payload(
        make_completions_model(),
        SimpleStreamOptions(sampling_params={"top_p": 0.95, "top_k": 0, "min_p": 0}),
    )
    assert payload["top_p"] == 0.95
    assert payload["top_k"] == 0
    assert payload["min_p"] == 0


async def test_omits_sampling_params_when_neither_options_nor_model_set_them() -> None:
    payload = await capture_payload(make_completions_model())
    assert "temperature" not in payload
    assert "top_p" not in payload


async def test_applies_model_level_sampling_params() -> None:
    payload = await capture_payload(make_completions_model(sampling_params={"temperature": 1, "top_p": 0.95}))
    assert payload["temperature"] == 1
    assert payload["top_p"] == 0.95


async def test_merges_stream_option_keys_over_model_level_keys() -> None:
    payload = await capture_payload(
        make_completions_model(sampling_params={"top_p": 0.95, "min_p": 0.05}),
        SimpleStreamOptions(sampling_params={"top_p": 0.5}),
    )
    assert payload["top_p"] == 0.5
    assert payload["min_p"] == 0.05


async def test_overrides_named_request_fields() -> None:
    payload = await capture_payload(
        make_completions_model(),
        SimpleStreamOptions(temperature=0, sampling_params={"temperature": 1}),
    )
    assert payload["temperature"] == 1


async def test_is_ignored_by_non_openai_compatible_apis() -> None:
    payload = await capture_payload(
        make_anthropic_model(),
        SimpleStreamOptions(sampling_params={"top_p": 0.9, "top_k": 40}),
    )
    assert "top_p" not in payload
    assert "top_k" not in payload
