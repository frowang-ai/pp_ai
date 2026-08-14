"""Python port of `packages/ai/test/openai-responses-compat.test.ts`.

TypeScript spies on the global `fetch` to read back the request headers and
uses the `onPayload` hook for the request body; this port captures the
`httpx.Request` through `httpx.MockTransport` and uses the same `on_payload`
hook.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from pi_ai.api.openai_responses import OpenAIResponsesOptions
from pi_ai.api.openai_responses import stream as stream_openai_responses
from pi_ai.providers.all import get_builtin_model
from pi_ai.types import Context, Model, Tool, UserMessage, now_ms

DONE_BODY = "data: [DONE]\n\n"


def model_of(builtin_provider: str, model_id: str, **overrides: object) -> Model:
    model = get_builtin_model(builtin_provider, model_id)
    assert model is not None, f"missing built-in model {builtin_provider}/{model_id}"
    return dataclasses.replace(model, **overrides) if overrides else model


@dataclass
class Captured:
    request: httpx.Request
    payload: dict[str, Any] | None

    def header(self, name: str) -> str | None:
        return self.request.headers.get(name)

    @property
    def session_id(self) -> str | None:
        return self.header("session_id")

    @property
    def client_request_id(self) -> str | None:
        return self.header("x-client-request-id")

    @property
    def x_session_id(self) -> str | None:
        return self.header("x-session-id")


async def capture(
    options: OpenAIResponsesOptions | None = None,
    model: Model | None = None,
    context: Context | None = None,
    sse: str = DONE_BODY,
) -> Captured:
    model = model or model_of("openai", "gpt-5.4")
    options = options or OpenAIResponsesOptions()
    if options.api_key is None:
        options = dataclasses.replace(options, api_key="test-key")

    recorded: dict[str, httpx.Request] = {}
    payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded["request"] = request
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    user_on_payload = options.on_payload

    def on_payload(payload: dict[str, Any], captured_model: Model) -> None:
        payloads.append(payload)
        if user_on_payload is not None:
            user_on_payload(payload, captured_model)

    options = dataclasses.replace(options, on_payload=on_payload)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    context = context or Context(
        system_prompt="sys",
        messages=[UserMessage(content="hi", timestamp=now_ms())],
    )

    stream = stream_openai_responses(model, context, options, client=client)
    async for event in stream:
        if event.type in ("done", "error"):
            break

    assert "request" in recorded, "no request was made"
    return Captured(request=recorded["request"], payload=payloads[0] if payloads else None)


def body_of(captured: Captured) -> dict[str, Any]:
    return json.loads(captured.request.content)


async def test_omits_reasoning_when_no_reasoning_is_requested() -> None:
    captured = await capture(model=model_of("github-copilot", "gpt-5-mini"))
    assert captured.payload is not None
    assert "reasoning" not in captured.payload


async def test_forwards_required_tool_choice() -> None:
    context = Context(
        messages=[UserMessage(content="Do not call ping. Respond with text instead.", timestamp=now_ms())],
        tools=[
            Tool(
                name="ping",
                description="Ping",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            )
        ],
    )
    captured = await capture(OpenAIResponsesOptions(tool_choice="required"), context=context)

    assert captured.payload is not None
    assert captured.payload["tool_choice"] == "required"
    assert captured.payload["tools"][0]["name"] == "ping"


@pytest.mark.parametrize(
    "model_id",
    [
        "gpt-5.1",
        "gpt-5.2",
        "gpt-5.3-codex",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.5",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
    ],
)
async def test_sends_none_reasoning_effort_when_no_reasoning_is_requested(model_id: str) -> None:
    captured = await capture(model=model_of("openai", model_id))
    assert captured.payload is not None
    assert captured.payload["reasoning"]["effort"] == "none"


@pytest.mark.parametrize(
    "model_id",
    ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro", "gpt-5.2-pro", "gpt-5.4-pro", "gpt-5.5-pro"],
)
async def test_omits_reasoning_effort_when_off_is_unsupported(model_id: str) -> None:
    captured = await capture(model=model_of("openai", model_id))
    assert captured.payload is not None
    assert "reasoning" not in captured.payload


async def test_sets_cache_affinity_headers_for_official_openai_requests_with_a_session_id() -> None:
    captured = await capture(OpenAIResponsesOptions(session_id="session-123"))
    assert captured.session_id == "session-123"
    assert captured.client_request_id == "session-123"


async def test_clamps_prompt_cache_key_to_openais_64_character_limit() -> None:
    captured = await capture(OpenAIResponsesOptions(session_id="x" * 67))
    assert captured.payload is not None
    assert captured.payload["prompt_cache_key"] == "x" * 64


async def test_sets_cache_affinity_headers_for_proxy_requests_with_a_session_id() -> None:
    proxy_model = model_of("openai", "gpt-5.4", provider="opencode", base_url="https://proxy.example.com/v1")
    captured = await capture(OpenAIResponsesOptions(session_id="session-123"), model=proxy_model)
    assert captured.session_id == "session-123"
    assert captured.client_request_id == "session-123"


async def test_uses_openrouter_session_affinity_header_when_configured() -> None:
    proxy_model = model_of(
        "openai",
        "gpt-5.4",
        provider="proxy",
        base_url="https://proxy.example.com/v1",
        compat={"sessionAffinityFormat": "openrouter"},
    )
    captured = await capture(OpenAIResponsesOptions(session_id="session-proxy"), model=proxy_model)

    assert captured.session_id is None
    assert captured.client_request_id is None
    assert captured.x_session_id == "session-proxy"
    assert captured.payload is not None
    assert "session_id" not in captured.payload
    assert captured.payload["prompt_cache_key"] == "session-proxy"


async def test_auto_detects_openrouter_session_affinity_for_openrouter_endpoints() -> None:
    openrouter_model = model_of("openai", "gpt-5.4", provider="openrouter", base_url="https://openrouter.ai/api/v1")
    captured = await capture(OpenAIResponsesOptions(session_id="session-openrouter"), model=openrouter_model)

    assert captured.session_id is None
    assert captured.client_request_id is None
    assert captured.x_session_id == "session-openrouter"
    assert captured.payload is not None
    assert "session_id" not in captured.payload
    assert captured.payload["prompt_cache_key"] == "session-openrouter"


async def test_uses_openai_no_session_format_when_configured() -> None:
    proxy_model = model_of(
        "openai",
        "gpt-5.4",
        provider="proxy",
        base_url="https://proxy.example.com/v1",
        compat={"sessionAffinityFormat": "openai-nosession"},
    )
    captured = await capture(OpenAIResponsesOptions(session_id="session-proxy"), model=proxy_model)

    assert captured.session_id is None
    assert captured.client_request_id == "session-proxy"
    assert captured.x_session_id is None
    assert captured.payload is not None
    assert "session_id" not in captured.payload
    assert captured.payload["prompt_cache_key"] == "session-proxy"


async def test_uses_openai_no_session_format_for_opencode_responses_models() -> None:
    model = model_of("opencode", "gpt-5.4")
    captured = await capture(OpenAIResponsesOptions(session_id="session-opencode"), model=model)

    assert model.compat.get("sessionAffinityFormat") == "openai-nosession"
    assert captured.session_id is None
    assert captured.client_request_id == "session-opencode"
    assert captured.x_session_id is None
    assert captured.payload is not None
    assert captured.payload["prompt_cache_key"] == "session-opencode"


async def test_can_omit_openai_session_id_header_while_preserving_other_affinity_data() -> None:
    proxy_model = model_of(
        "openai",
        "gpt-5.4",
        provider="opencode",
        base_url="https://proxy.example.com/v1",
        compat={"sessionAffinityFormat": "openai-nosession"},
    )
    captured = await capture(OpenAIResponsesOptions(session_id="session-123"), model=proxy_model)

    assert captured.session_id is None
    assert captured.client_request_id == "session-123"
    assert captured.payload is not None
    assert captured.payload["prompt_cache_key"] == "session-123"


async def test_lets_explicit_headers_override_the_default_cache_affinity_headers() -> None:
    captured = await capture(
        OpenAIResponsesOptions(
            session_id="session-123",
            headers={"session_id": "override-session", "x-client-request-id": "override-request"},
        )
    )
    assert captured.session_id == "override-session"
    assert captured.client_request_id == "override-request"


async def test_omits_cache_affinity_headers_when_cache_retention_is_none() -> None:
    captured = await capture(OpenAIResponsesOptions(cache_retention="none", session_id="session-123"))
    assert captured.session_id is None
    assert captured.client_request_id is None


@pytest.mark.parametrize(
    ("model_id", "service_tier", "multiplier"),
    [("gpt-5.4", "priority", 2.0), ("gpt-5.5", "priority", 2.5), ("gpt-5.5", "flex", 0.5)],
)
async def test_applies_service_tier_cost_multipliers(model_id: str, service_tier: str, multiplier: float) -> None:
    model = model_of("openai", model_id)
    token_count = 100_000
    token_scale = token_count / 1_000_000
    event = {
        "type": "response.completed",
        "response": {
            "status": "completed",
            "service_tier": service_tier,
            "usage": {
                "input_tokens": token_count,
                "output_tokens": token_count,
                "total_tokens": token_count * 2,
                "input_tokens_details": {"cached_tokens": 0},
            },
        },
    }
    sse = f"data: {json.dumps(event)}\n\n"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    context = Context(system_prompt="sys", messages=[UserMessage(content="hi", timestamp=now_ms())])
    stream = stream_openai_responses(
        model,
        context,
        OpenAIResponsesOptions(api_key="test-key", service_tier=service_tier),
        client=client,
    )
    result = await stream.result()

    assert result.usage.cost.input == pytest.approx(model.cost.input * multiplier * token_scale)
    assert result.usage.cost.output == pytest.approx(model.cost.output * multiplier * token_scale)
    assert result.usage.cost.total == pytest.approx((model.cost.input + model.cost.output) * multiplier * token_scale)
