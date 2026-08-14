"""Python port of `packages/ai/test/fireworks-models.test.ts`.

The integration block uses a real `node:http` server to capture the outbound
Anthropic request; this port captures the same request through an
`httpx.MockTransport`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from pi_ai.api.anthropic_messages import AnthropicOptions
from pi_ai.api.anthropic_messages import stream as stream_anthropic
from pi_ai.compat import stream_simple
from pi_ai.env_api_keys import find_env_keys, get_env_api_key
from pi_ai.providers.all import get_builtin_model, get_builtin_models
from pi_ai.types import Context, Model, ModelCost, SimpleStreamOptions, Tool, UserMessage


class PayloadCaptured(Exception):
    def __init__(self) -> None:
        super().__init__("payload captured")


async def capture_payload(model: Model, options: SimpleStreamOptions) -> dict[str, Any]:
    captured: dict[str, Any] | None = None

    def on_payload(payload: dict[str, Any], _model: Model) -> None:
        nonlocal captured
        captured = payload
        raise PayloadCaptured()

    options.on_payload = on_payload
    await stream_simple(model, Context(messages=[UserMessage(content="test")]), options).result()
    assert captured is not None
    return captured


def test_registers_the_default_kimi_k2_6_model_via_anthropic_messages():
    model = get_builtin_model("fireworks", "accounts/fireworks/models/kimi-k2p6")

    assert model is not None
    assert model.api == "anthropic-messages"
    assert model.provider == "fireworks"
    assert model.base_url == "https://api.fireworks.ai/inference"
    assert model.reasoning is True
    assert model.input == ["text", "image"]
    assert model.context_window == 262000
    assert model.max_tokens == 262000
    assert model.cost == ModelCost(input=0.95, output=4, cache_read=0.16, cache_write=0)


def test_registers_the_fire_pass_turbo_router_model():
    model = next(
        (
            candidate
            for candidate in get_builtin_models("fireworks")
            if candidate.id.startswith("accounts/fireworks/routers/") and candidate.id.endswith("-turbo")
        ),
        None,
    )

    assert model is not None
    assert model.api == "anthropic-messages"
    assert model.base_url == "https://api.fireworks.ai/inference"
    assert model.input == ["text", "image"]


def test_aligns_glm_5_2_fast_with_glm_5_2_config():
    base = get_builtin_model("fireworks", "accounts/fireworks/models/glm-5p2")
    fast = get_builtin_model("fireworks", "accounts/fireworks/routers/glm-5p2-fast")

    assert fast.api == base.api
    assert fast.base_url == base.base_url
    assert fast.compat == base.compat
    assert fast.thinking_level_map == base.thinking_level_map


@pytest.mark.parametrize("model_id", ["accounts/fireworks/models/glm-5p2", "accounts/fireworks/routers/glm-5p2-fast"])
async def test_omits_unsupported_long_cache_retention(model_id: str):
    model = get_builtin_model("fireworks", model_id)
    payload = await capture_payload(
        model,
        SimpleStreamOptions(
            api_key="test-fireworks-key",
            cache_retention="long",
            session_id="test-fireworks-session",
        ),
    )
    assert "prompt_cache_retention" not in payload


async def test_routes_kimi_k3_through_openai_completions_with_native_effort_controls():
    base = get_builtin_model("fireworks", "accounts/fireworks/models/kimi-k3")
    fast = get_builtin_model("fireworks", "accounts/fireworks/routers/kimi-k3-fast")
    compat = {
        "supportsStore": False,
        "supportsDeveloperRole": False,
        "requiresReasoningContentOnAssistantMessages": True,
        "thinkingFormat": "openai",
        "deferredToolsMode": "kimi",
        "sendSessionAffinityHeaders": True,
        "supportsLongCacheRetention": False,
    }
    thinking_level_map = {
        "off": None,
        "minimal": None,
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": None,
        "max": "max",
    }

    assert base.api == "openai-completions"
    assert base.base_url == "https://api.fireworks.ai/inference/v1"
    assert base.compat == compat
    assert base.thinking_level_map == thinking_level_map
    assert fast.api == base.api
    assert fast.base_url == base.base_url
    assert fast.compat == compat
    assert fast.thinking_level_map == thinking_level_map

    payload = await capture_payload(base, SimpleStreamOptions(api_key="test-fireworks-key", reasoning="max"))
    assert payload["reasoning_effort"] == "max"


def test_resolves_fireworks_api_key_from_the_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-fireworks-key")

    assert find_env_keys("fireworks") == ["FIREWORKS_API_KEY"]
    assert get_env_api_key("fireworks") == "test-fireworks-key"


def test_sets_fireworks_specific_compat_for_session_affinity_and_tool_fields():
    model = get_builtin_model("fireworks", "accounts/fireworks/models/kimi-k2p6")

    assert model.compat
    assert model.compat["sendSessionAffinityHeaders"] is True
    assert model.compat["supportsEagerToolInputStreaming"] is False
    assert model.compat["supportsCacheControlOnTools"] is False
    assert model.compat["supportsLongCacheRetention"] is False


# --- Integration tests for Fireworks Anthropic session affinity and tool compat ---


@dataclass
class CapturedRequest:
    headers: dict[str, str] = field(default_factory=dict)
    body: dict[str, Any] = field(default_factory=dict)


TOOL = Tool(
    name="lookup",
    description="Look up a value",
    parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
)

FIREWORKS_ANTHROPIC_COMPAT: dict[str, Any] = {
    "sendSessionAffinityHeaders": True,
    "supportsEagerToolInputStreaming": False,
    "supportsCacheControlOnTools": False,
    "supportsLongCacheRetention": False,
}


def create_fireworks_model(compat: dict[str, Any] | None = None) -> Model:
    return Model(
        id="accounts/fireworks/models/kimi-k2p6",
        name="Kimi K2.6",
        api="anthropic-messages",
        provider="fireworks",
        base_url="http://127.0.0.1:9999",
        reasoning=True,
        input=["text", "image"],
        cost=ModelCost(input=0.95, output=4, cache_read=0.16, cache_write=0),
        context_window=262000,
        max_tokens=262000,
        compat=FIREWORKS_ANTHROPIC_COMPAT if compat is None else compat,
    )


def create_anthropic_model() -> Model:
    return Model(
        id="claude-opus-4-8",
        name="Claude Opus 4.8",
        api="anthropic-messages",
        provider="anthropic",
        base_url="http://127.0.0.1:9999",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=200000,
        max_tokens=32000,
    )


def create_context(tools: list[Tool] | None = None) -> Context:
    tools = [TOOL] if tools is None else tools
    return Context(messages=[UserMessage(content="Use the tool")], tools=tools or None)


async def capture_anthropic_request(
    model: Model,
    context: Context,
    session_id: str | None = None,
    cache_retention: str = "short",
) -> CapturedRequest:
    captured = CapturedRequest()

    def handler(request: httpx.Request) -> httpx.Response:
        captured.headers = {name.lower(): value for name, value in request.headers.items()}
        captured.body = json.loads(request.content)
        return httpx.Response(200, text="", headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    event_stream = stream_anthropic(
        model,
        context,
        AnthropicOptions(api_key="test-key", cache_retention=cache_retention, session_id=session_id),
        client=client,
    )
    async for event in event_stream:
        if event.type in ("done", "error"):
            break

    assert captured.body, "Anthropic request was not captured"
    return captured


def get_tools(body: dict[str, Any]) -> list[dict[str, Any]]:
    tools = body["tools"]
    assert isinstance(tools, list)
    return tools


async def test_sends_x_session_affinity_header_for_fireworks_models():
    request = await capture_anthropic_request(
        create_fireworks_model(), create_context(), session_id="fireworks-session-1"
    )
    assert request.headers["x-session-affinity"] == "fireworks-session-1"


async def test_omits_x_session_affinity_header_for_native_anthropic_models():
    request = await capture_anthropic_request(
        create_anthropic_model(), create_context(), session_id="anthropic-session-1"
    )
    assert "x-session-affinity" not in request.headers


async def test_omits_x_session_affinity_header_when_cache_retention_is_none():
    request = await capture_anthropic_request(
        create_fireworks_model(),
        create_context(),
        session_id="fireworks-session-2",
        cache_retention="none",
    )
    assert "x-session-affinity" not in request.headers


async def test_omits_cache_control_on_tools_for_fireworks_models():
    request = await capture_anthropic_request(create_fireworks_model(), create_context())
    tools = get_tools(request.body)
    assert "cache_control" not in tools[-1]


async def test_omits_eager_input_streaming_on_tools_for_fireworks_models():
    request = await capture_anthropic_request(create_fireworks_model(), create_context())
    for tool in get_tools(request.body):
        assert "eager_input_streaming" not in tool


async def test_sends_cache_control_on_tools_for_native_anthropic_models():
    request = await capture_anthropic_request(create_anthropic_model(), create_context())
    tools = get_tools(request.body)
    last_tool_cache_control = tools[-1].get("cache_control")
    assert last_tool_cache_control is not None
    assert last_tool_cache_control["type"] == "ephemeral"


async def test_sends_eager_input_streaming_on_tools_for_native_anthropic_models():
    request = await capture_anthropic_request(create_anthropic_model(), create_context())
    tools = get_tools(request.body)
    assert tools[0]["eager_input_streaming"] is True
