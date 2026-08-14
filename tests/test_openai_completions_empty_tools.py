"""Python port of `packages/ai/test/openai-completions-empty-tools.test.ts`.

Empty tools arrays must NOT be serialized as `tools: []` -- some
OpenAI-compatible backends (e.g. DashScope / Aliyun Qwen via compatible-mode)
reject the request with `"[] is too short - 'tools'"` (HTTP 400) when
`--no-tools` produces an empty array.

TypeScript mocks the `openai` SDK and reads back the constructed client's
options; this port records the `httpx` request instead, so the "client options"
assertions become assertions on the request URL and headers.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import httpx
import pytest
from pi_ai.compat import register_builtin_api_providers, stream_simple
from pi_ai.providers.all import get_builtin_model
from pi_ai.types import (
    AssistantMessage,
    Context,
    Cost,
    Message,
    Model,
    SimpleStreamOptions,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
    now_ms,
)

register_builtin_api_providers()

_CHUNK: dict[str, Any] = {
    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    "usage": {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
}


@pytest.fixture
def cloudflare_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cf-token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account-id")
    monkeypatch.setenv("CLOUDFLARE_GATEWAY_ID", "gateway-id")


def openai_model(**overrides: object) -> Model:
    base = get_builtin_model("openai", "gpt-4o-mini")
    assert base is not None
    fields: dict[str, object] = {"compat": {}, "api": "openai-completions"}
    fields.update(overrides)
    return dataclasses.replace(base, **fields)


async def capture(
    model: Model,
    messages: list[Message],
    tools: list[object] | None = None,
    options: SimpleStreamOptions | None = None,
    system_prompt: str | None = None,
) -> httpx.Request:
    captured: dict[str, httpx.Request] = {}
    body = f"data: {json.dumps(_CHUNK)}\n\ndata: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    context = Context(messages=messages)
    if tools is not None:
        context.tools = tools
    if system_prompt is not None:
        context.system_prompt = system_prompt

    await stream_simple(model, context, options, client=client).result()

    assert "request" in captured, "no request was made"
    return captured["request"]


def params_of(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)


def hi() -> list[Message]:
    return [UserMessage(content="hi", timestamp=now_ms())]


async def test_omits_tools_field_when_context_tools_is_an_empty_array() -> None:
    request = await capture(openai_model(), hi(), tools=[], options=SimpleStreamOptions(api_key="test"))
    assert "tools" not in params_of(request)


async def test_omits_tools_field_when_context_tools_is_undefined() -> None:
    request = await capture(openai_model(), hi(), options=SimpleStreamOptions(api_key="test"))
    assert "tools" not in params_of(request)


async def test_sends_default_max_tokens() -> None:
    model = openai_model()
    request = await capture(model, hi(), options=SimpleStreamOptions(api_key="test"))
    params = params_of(request)

    assert "max_tokens" not in params
    assert params["max_completion_tokens"] == model.max_tokens


async def test_sends_explicit_max_tokens() -> None:
    request = await capture(openai_model(), hi(), options=SimpleStreamOptions(api_key="test", max_tokens=1234))
    params = params_of(request)

    assert "max_tokens" not in params
    assert params["max_completion_tokens"] == 1234


async def test_clamps_default_max_tokens_to_remaining_context() -> None:
    model = openai_model(context_window=10000, max_tokens=8000)
    request = await capture(
        model,
        [UserMessage(content="x" * 8000, timestamp=now_ms())],
        options=SimpleStreamOptions(api_key="test"),
    )
    params = params_of(request)

    assert "max_tokens" not in params
    assert params["max_completion_tokens"] == 3904


async def test_clamps_explicit_max_tokens_to_remaining_context() -> None:
    model = openai_model(context_window=10000, max_tokens=8000)
    request = await capture(
        model,
        [UserMessage(content="x" * 8000, timestamp=now_ms())],
        options=SimpleStreamOptions(api_key="test", max_tokens=7000),
    )
    params = params_of(request)

    assert "max_tokens" not in params
    assert params["max_completion_tokens"] == 3904


async def test_uses_conservative_openai_compatible_fields_for_cloudflare_compat_models(
    cloudflare_env: None,
) -> None:
    model = get_builtin_model("cloudflare-ai-gateway", "workers-ai/@cf/moonshotai/kimi-k2.6")
    assert model is not None

    request = await capture(
        model,
        hi(),
        options=SimpleStreamOptions(max_tokens=1234, reasoning="high"),
        system_prompt="You are helpful.",
    )
    params = params_of(request)

    assert params["messages"][0]["role"] == "system"
    assert params["max_tokens"] == 1234
    assert "max_completion_tokens" not in params
    assert "reasoning_effort" not in params
    assert "store" not in params

    assert str(request.url).startswith(
        "https://gateway.ai.cloudflare.com/v1/account-id/gateway-id/compat/chat/completions"
    )
    # TypeScript asserts `defaultHeaders.Authorization` is null (the SDK drops
    # null-valued headers); the port drops them while building the request, so
    # the observable equivalent is that no authorization header was sent.
    assert "authorization" not in request.headers
    assert request.headers["cf-aig-authorization"] == "Bearer cf-token"


async def test_resolves_cloudflare_ai_gateway_base_url_through_provider_auth(cloudflare_env: None) -> None:
    model = get_builtin_model("cloudflare-ai-gateway", "workers-ai/@cf/moonshotai/kimi-k2.6")
    assert model is not None

    request = await capture(model, hi())

    assert str(request.url).startswith(
        "https://gateway.ai.cloudflare.com/v1/account-id/gateway-id/compat/chat/completions"
    )


async def test_preserves_inline_upstream_authorization_for_cloudflare_byok_requests(
    cloudflare_env: None,
) -> None:
    model = get_builtin_model("cloudflare-ai-gateway", "gpt-5.1")
    assert model is not None

    request = await capture(model, hi(), options=SimpleStreamOptions(headers={"Authorization": "Bearer upstream-key"}))

    assert request.headers["authorization"] == "Bearer upstream-key"
    assert request.headers["cf-aig-authorization"] == "Bearer cf-token"


async def test_sends_session_affinity_headers_for_workers_ai_through_cloudflare_ai_gateway(
    cloudflare_env: None,
) -> None:
    model = get_builtin_model("cloudflare-ai-gateway", "workers-ai/@cf/moonshotai/kimi-k2.6")
    assert model is not None

    options = SimpleStreamOptions()
    options.session_id = "session-1"
    request = await capture(model, hi(), options=options)

    assert request.headers["session_id"] == "session-1"
    assert request.headers["x-client-request-id"] == "session-1"
    assert request.headers["x-session-affinity"] == "session-1"


async def test_still_emits_tools_for_anthropic_litellm_proxy_when_conversation_has_tool_history() -> None:
    timestamp = now_ms()
    messages: list[Message] = [
        UserMessage(content="use the tool", timestamp=timestamp),
        AssistantMessage(
            content=[ToolCall(id="t1", name="noop", arguments={})],
            stop_reason="toolUse",
            usage=Usage(
                input=0,
                output=0,
                cache_read=0,
                cache_write=0,
                total_tokens=0,
                cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
            ),
            api="openai-completions",
            provider="openai",
            model="gpt-4o-mini",
            timestamp=timestamp,
        ),
        ToolResultMessage(
            tool_call_id="t1",
            tool_name="noop",
            content=[TextContent(text="done")],
            is_error=False,
            timestamp=timestamp,
        ),
    ]

    request = await capture(openai_model(), messages, tools=[], options=SimpleStreamOptions(api_key="test"))
    params = params_of(request)

    assert isinstance(params["tools"], list)
    assert params["tools"] == []
