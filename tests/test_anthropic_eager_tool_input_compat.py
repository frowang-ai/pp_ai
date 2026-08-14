"""Python port of `packages/ai/test/anthropic-eager-tool-input-compat.test.ts`.

TypeScript spins up a real `node:http` server on 127.0.0.1 and reads the
request it receives; this port captures the same request through an
`httpx.MockTransport`, which returns the identical empty SSE response.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx
from pi_ai.api.anthropic_messages import AnthropicOptions
from pi_ai.api.anthropic_messages import stream as stream_anthropic
from pi_ai.types import (
    Context,
    JsonSchemaConstrainedSampling,
    Model,
    ModelCost,
    Tool,
    UserMessage,
)


@dataclass
class CapturedRequest:
    headers: dict[str, str] = field(default_factory=dict)
    body: dict[str, Any] = field(default_factory=dict)


def create_model(compat: dict[str, Any] | None = None) -> Model:
    return Model(
        id="claude-opus-4-8",
        name="Claude Opus 4.8",
        api="anthropic-messages",
        provider="test-anthropic",
        base_url="http://127.0.0.1:9999",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=200000,
        max_tokens=32000,
        compat={"forceAdaptiveThinking": True, **(compat or {})},
    )


TOOL = Tool(
    name="lookup",
    description="Look up a value",
    parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
)

SCHEMA_COMPATIBILITY_TOOL = Tool(
    name="lookup",
    description="Look up a value",
    parameters={
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
        "title": "LookupInput",
    },
)

STRICT_TOOL = Tool(
    name="lookup",
    description="Look up a value",
    parameters={
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
        "title": "StrictLookupInput",
    },
    constrained_sampling=JsonSchemaConstrainedSampling(strict="prefer"),
)


def create_context(tools: list[Tool] | None = None) -> Context:
    tools = [TOOL] if tools is None else tools
    return Context(messages=[UserMessage(content="Use the tool")], tools=tools or None)


async def capture_anthropic_request(compat: dict[str, Any] | None, context: Context) -> CapturedRequest:
    captured = CapturedRequest()

    def handler(request: httpx.Request) -> httpx.Response:
        captured.headers = {name.lower(): value for name, value in request.headers.items()}
        captured.body = json.loads(request.content)
        return httpx.Response(200, text="", headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    event_stream = stream_anthropic(
        create_model(compat),
        context,
        AnthropicOptions(api_key="test-key", cache_retention="none"),
        client=client,
    )
    async for event in event_stream:
        if event.type in ("done", "error"):
            break

    assert captured.body, "Anthropic request was not captured"
    return captured


def get_first_tool(body: dict[str, Any]) -> dict[str, Any]:
    tools = body["tools"]
    assert isinstance(tools, list) and isinstance(tools[0], dict)
    return tools[0]


def get_first_tool_input_schema(body: dict[str, Any]) -> dict[str, Any]:
    input_schema = get_first_tool(body)["input_schema"]
    assert isinstance(input_schema, dict)
    return input_schema


async def test_sends_per_tool_eager_input_streaming_by_default():
    request = await capture_anthropic_request(None, create_context())

    assert get_first_tool(request.body)["eager_input_streaming"] is True
    assert "anthropic-beta" not in request.headers


async def test_uses_legacy_fine_grained_beta_when_eager_tool_input_streaming_disabled():
    request = await capture_anthropic_request({"supportsEagerToolInputStreaming": False}, create_context())

    assert "eager_input_streaming" not in get_first_tool(request.body)
    assert request.headers["anthropic-beta"] == "fine-grained-tool-streaming-2025-05-14"


async def test_does_not_send_legacy_fine_grained_beta_when_there_are_no_tools():
    request = await capture_anthropic_request({"supportsEagerToolInputStreaming": False}, create_context([]))

    assert "tools" not in request.body
    assert "anthropic-beta" not in request.headers


async def test_only_sends_the_full_input_schema_for_strict_json_schema_tools():
    legacy_request = await capture_anthropic_request(
        {"supportsStrictTools": True}, create_context([SCHEMA_COMPATIBILITY_TOOL])
    )
    assert get_first_tool_input_schema(legacy_request.body) == {
        "type": "object",
        "properties": SCHEMA_COMPATIBILITY_TOOL.parameters["properties"],
        "required": SCHEMA_COMPATIBILITY_TOOL.parameters["required"],
    }

    strict_request = await capture_anthropic_request({"supportsStrictTools": True}, create_context([STRICT_TOOL]))
    assert get_first_tool(strict_request.body)["strict"] is True
    input_schema = get_first_tool_input_schema(strict_request.body)
    assert input_schema["additionalProperties"] is False
    assert input_schema["title"] == "StrictLookupInput"
