"""Python port of `packages/ai/test/cache-retention.test.ts`.

The four `it.skipIf(!process.env.ANTHROPIC_API_KEY)` /
`!process.env.OPENAI_API_KEY` cases are ported *offline*: TypeScript gates them
on a credential only because they call `compat.stream()` without an explicit
`apiKey`, but every assertion is made on the payload captured before the
request is sent, so supplying a fake key preserves them exactly.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Iterator
from typing import Any

import pytest
from pi_ai.api.anthropic_messages import AnthropicOptions
from pi_ai.api.anthropic_messages import stream as stream_anthropic
from pi_ai.api.openai_completions import OpenAICompletionsOptions
from pi_ai.api.openai_completions import stream as stream_openai_completions
from pi_ai.api.openai_responses import OpenAIResponsesOptions
from pi_ai.api.openai_responses import stream as stream_openai_responses
from pi_ai.providers.all import get_builtin_model
from pi_ai.types import Context, Model, ModelCost, StreamOptions, UserMessage


class PayloadCaptured(Exception):
    def __init__(self) -> None:
        super().__init__("payload captured")


@pytest.fixture(autouse=True)
def clear_cache_retention_env() -> Iterator[None]:
    original = os.environ.pop("PI_CACHE_RETENTION", None)
    try:
        yield
    finally:
        os.environ.pop("PI_CACHE_RETENTION", None)
        if original is not None:
            os.environ["PI_CACHE_RETENTION"] = original


def make_context() -> Context:
    return Context(
        system_prompt="You are a helpful assistant.",
        messages=[UserMessage(content="Hello")],
    )


async def capture_payload(stream_fn: Any, model: Model, options: StreamOptions) -> dict[str, Any]:
    captured: dict[str, Any] | None = None

    def on_payload(payload: dict[str, Any], request_model: Model) -> None:
        nonlocal captured
        captured = payload
        raise PayloadCaptured()

    await stream_fn(model, make_context(), dataclasses.replace(options, on_payload=on_payload)).result()

    assert captured is not None, "Expected payload to be captured before request failure"
    return captured


# --- Anthropic ------------------------------------------------------------


async def test_anthropic_uses_default_cache_ttl_when_pi_cache_retention_is_not_set():
    # TypeScript gates this on `ANTHROPIC_API_KEY` because it lets `compat.stream()`
    # resolve an ambient credential. The payload is captured before any network
    # I/O, so passing a fake key ports it offline with the same assertion.
    payload = await capture_payload(
        stream_anthropic,
        get_builtin_model("anthropic", "claude-haiku-4-5"),
        AnthropicOptions(api_key="fake-key"),
    )
    assert payload["system"] is not None
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}


async def test_anthropic_uses_1h_cache_ttl_when_pi_cache_retention_is_long():
    os.environ["PI_CACHE_RETENTION"] = "long"

    payload = await capture_payload(
        stream_anthropic,
        get_builtin_model("anthropic", "claude-haiku-4-5"),
        AnthropicOptions(api_key="fake-key"),
    )
    assert payload["system"] is not None
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


async def test_anthropic_adds_ttl_for_non_anthropic_base_url_by_default():
    os.environ["PI_CACHE_RETENTION"] = "long"
    proxy_model = dataclasses.replace(
        get_builtin_model("anthropic", "claude-haiku-4-5"),
        base_url="https://my-proxy.example.com/v1",
    )

    payload = await capture_payload(stream_anthropic, proxy_model, AnthropicOptions(api_key="fake-key"))
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


async def test_anthropic_omits_ttl_when_supports_long_cache_retention_is_false():
    proxy_model = dataclasses.replace(
        get_builtin_model("anthropic", "claude-haiku-4-5"),
        base_url="https://my-proxy.example.com/v1",
        compat={"supportsLongCacheRetention": False},
    )

    payload = await capture_payload(
        stream_anthropic,
        proxy_model,
        AnthropicOptions(api_key="fake-key", cache_retention="long"),
    )
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}


async def test_anthropic_omits_cache_control_when_cache_retention_is_none():
    payload = await capture_payload(
        stream_anthropic,
        get_builtin_model("anthropic", "claude-haiku-4-5"),
        AnthropicOptions(api_key="fake-key", cache_retention="none"),
    )
    assert "cache_control" not in payload["system"][0]


async def test_anthropic_adds_cache_control_to_string_user_messages():
    payload = await capture_payload(
        stream_anthropic,
        get_builtin_model("anthropic", "claude-haiku-4-5"),
        AnthropicOptions(api_key="fake-key"),
    )
    last_message = payload["messages"][-1]
    assert isinstance(last_message["content"], list)
    assert last_message["content"][-1]["cache_control"] == {"type": "ephemeral"}


async def test_anthropic_sets_1h_cache_ttl_when_cache_retention_is_long():
    payload = await capture_payload(
        stream_anthropic,
        get_builtin_model("anthropic", "claude-haiku-4-5"),
        AnthropicOptions(api_key="fake-key", cache_retention="long"),
    )
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


# --- OpenAI Responses -----------------------------------------------------


async def test_responses_does_not_set_prompt_cache_retention_when_pi_cache_retention_is_not_set():
    # Ported offline for the same reason as the Anthropic default-TTL case above:
    # TypeScript's `OPENAI_API_KEY` gate only exists to resolve a credential.
    payload = await capture_payload(
        stream_openai_responses,
        get_builtin_model("openai", "gpt-4o-mini"),
        OpenAIResponsesOptions(api_key="fake-key"),
    )
    assert "prompt_cache_retention" not in payload


async def test_responses_sets_prompt_cache_retention_to_24h_when_pi_cache_retention_is_long():
    os.environ["PI_CACHE_RETENTION"] = "long"

    payload = await capture_payload(
        stream_openai_responses,
        get_builtin_model("openai", "gpt-4o-mini"),
        OpenAIResponsesOptions(api_key="fake-key"),
    )
    assert payload["prompt_cache_retention"] == "24h"


async def test_responses_sets_prompt_cache_retention_for_non_openai_base_url_by_default():
    os.environ["PI_CACHE_RETENTION"] = "long"
    proxy_model = dataclasses.replace(
        get_builtin_model("openai", "gpt-4o-mini"),
        base_url="https://my-proxy.example.com/v1",
    )

    payload = await capture_payload(stream_openai_responses, proxy_model, OpenAIResponsesOptions(api_key="fake-key"))
    assert payload["prompt_cache_retention"] == "24h"


async def test_responses_omits_prompt_cache_retention_when_compat_disables_it():
    model = dataclasses.replace(
        get_builtin_model("openai", "gpt-4o-mini"),
        compat={"supportsLongCacheRetention": False},
    )

    payload = await capture_payload(
        stream_openai_responses,
        model,
        OpenAIResponsesOptions(api_key="fake-key", cache_retention="long", session_id="session-compat-false"),
    )
    assert "prompt_cache_retention" not in payload


async def test_responses_omits_cache_key_and_disables_implicit_writes_when_none():
    payload = await capture_payload(
        stream_openai_responses,
        get_builtin_model("openai", "gpt-5.6-sol"),
        OpenAIResponsesOptions(api_key="fake-key", cache_retention="none", session_id="session-1"),
    )
    assert "prompt_cache_key" not in payload
    assert "prompt_cache_retention" not in payload
    assert payload["prompt_cache_options"] == {"mode": "explicit"}


async def test_responses_omits_prompt_cache_options_for_models_that_reject_it():
    payload = await capture_payload(
        stream_openai_responses,
        get_builtin_model("openai", "gpt-4o-mini"),
        OpenAIResponsesOptions(api_key="fake-key", cache_retention="none", session_id="session-1"),
    )
    assert "prompt_cache_key" not in payload
    assert "prompt_cache_options" not in payload


async def test_responses_sets_prompt_cache_retention_when_cache_retention_is_long():
    payload = await capture_payload(
        stream_openai_responses,
        get_builtin_model("openai", "gpt-4o-mini"),
        OpenAIResponsesOptions(api_key="fake-key", cache_retention="long", session_id="session-2"),
    )
    assert payload["prompt_cache_key"] == "session-2"
    assert payload["prompt_cache_retention"] == "24h"


# --- OpenAI Completions ---------------------------------------------------


def make_completions_model(compat: dict[str, Any] | None = None) -> Model:
    return Model(
        id="test-model",
        name="Test Model",
        api="openai-completions",
        provider="test-openai-completions",
        base_url="https://my-proxy.example.com/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128000,
        max_tokens=4096,
        compat=compat or {},
    )


async def test_completions_sets_prompt_cache_retention_for_non_openai_base_url_by_default():
    payload = await capture_payload(
        stream_openai_completions,
        make_completions_model(),
        OpenAICompletionsOptions(api_key="fake-key", cache_retention="long", session_id="session-completions"),
    )
    assert payload["prompt_cache_key"] == "session-completions"
    assert payload["prompt_cache_retention"] == "24h"


async def test_completions_omits_prompt_cache_retention_when_compat_disables_it():
    payload = await capture_payload(
        stream_openai_completions,
        make_completions_model({"supportsLongCacheRetention": False}),
        OpenAICompletionsOptions(
            api_key="fake-key",
            cache_retention="long",
            session_id="session-completions-false",
        ),
    )
    assert "prompt_cache_key" not in payload
    assert "prompt_cache_retention" not in payload


@pytest.mark.parametrize(
    ("provider", "model_id"),
    [
        ("opencode", "deepseek-v4-flash"),
        ("opencode", "deepseek-v4-pro"),
        ("opencode", "kimi-k2.5"),
        ("opencode", "kimi-k2.6"),
        ("opencode", "minimax-m2.7"),
        ("opencode-go", "kimi-k2.6"),
    ],
)
async def test_completions_omits_long_cache_retention_for_opencode_models(provider: str, model_id: str):
    model = get_builtin_model(provider, model_id)
    payload = await capture_payload(
        stream_openai_completions,
        model,
        OpenAICompletionsOptions(
            api_key="fake-key",
            cache_retention="long",
            session_id="session-opencode-long-cache-unsupported",
        ),
    )
    assert model.compat.get("supportsLongCacheRetention") is False
    assert "prompt_cache_key" not in payload
    assert "prompt_cache_retention" not in payload
