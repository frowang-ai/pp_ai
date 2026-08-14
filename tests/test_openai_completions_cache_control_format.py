"""Python port of `packages/ai/test/openai-completions-cache-control-format.test.ts`."""

from __future__ import annotations

import json
from typing import Any

import httpx
from pi_ai.api.openai_completions import OpenAICompletionsOptions, stream
from pi_ai.providers.all import get_builtin_model
from pi_ai.types import (
    AssistantMessage,
    Context,
    Cost,
    Message,
    Model,
    ModelCost,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
    now_ms,
)

_CHUNK: dict[str, Any] = {
    "id": "chatcmpl-test",
    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    "usage": {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
}


async def capture_payload(
    model: Model,
    options: OpenAICompletionsOptions | None = None,
    messages: list[Message] | None = None,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    body = f"data: {json.dumps(_CHUNK)}\n\ndata: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = json.loads(request.content)
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    stream_options = options or OpenAICompletionsOptions()
    stream_options.api_key = "test-key"

    context = Context(
        system_prompt="System prompt",
        messages=messages if messages is not None else [UserMessage(content="Hello", timestamp=now_ms())],
        tools=[
            Tool(
                name="read",
                description="Read a file",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            )
        ],
    )

    await stream(model, context, stream_options, client=client).result()

    if "params" not in captured:
        raise AssertionError("Expected payload to be captured")
    return captured["params"]


def get_instruction_message(params: dict[str, Any]) -> dict[str, Any] | None:
    return next((m for m in params["messages"] if m["role"] in ("system", "developer")), None)


def expect_anthropic_cache_markers(params: dict[str, Any]) -> None:
    instruction_message = get_instruction_message(params)
    assert instruction_message is not None
    assert isinstance(instruction_message["content"], list)
    assert instruction_message["content"][0]["cache_control"] == {"type": "ephemeral"}

    assert len(params["tools"]) == 1
    assert params["tools"][0]["cache_control"] == {"type": "ephemeral"}

    last_message = params["messages"][-1]
    assert last_message["role"] == "user"
    assert isinstance(last_message["content"], list)
    assert last_message["content"][0]["cache_control"] == {"type": "ephemeral"}


def custom_qwen_model() -> Model:
    return Model(
        id="custom-qwen",
        name="Custom Qwen",
        api="openai-completions",
        provider="openrouter",
        base_url="https://example.com/v1",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128000,
        max_tokens=32000,
        compat={"cacheControlFormat": "anthropic"},
    )


def openrouter_sonnet() -> Model:
    model = get_builtin_model("openrouter", "anthropic/claude-sonnet-4")
    assert model is not None
    return model


async def test_applies_anthropic_style_cache_markers_when_model_compat_enables_them() -> None:
    params = await capture_payload(custom_qwen_model())
    expect_anthropic_cache_markers(params)


async def test_preserves_anthropic_style_cache_markers_for_openrouter_anthropic_models() -> None:
    params = await capture_payload(openrouter_sonnet())
    expect_anthropic_cache_markers(params)


async def test_moves_the_conversation_cache_marker_to_a_tool_result() -> None:
    model = openrouter_sonnet()
    timestamp = now_ms()
    params = await capture_payload(
        model,
        None,
        [
            UserMessage(content="Read the file", timestamp=timestamp),
            AssistantMessage(
                content=[ToolCall(id="call_1", name="read", arguments={"path": "README.md"})],
                api="openai-completions",
                provider="openrouter",
                model=model.id,
                usage=Usage(
                    input=0,
                    output=0,
                    cache_read=0,
                    cache_write=0,
                    total_tokens=0,
                    cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
                ),
                stop_reason="toolUse",
                timestamp=timestamp,
            ),
            ToolResultMessage(
                tool_call_id="call_1",
                tool_name="read",
                content=[TextContent(text="file contents")],
                is_error=False,
                timestamp=timestamp,
            ),
        ],
    )

    user_message = next(m for m in params["messages"] if m["role"] == "user")
    assert user_message["content"] == "Read the file"

    tool_message = params["messages"][-1]
    assert tool_message["role"] == "tool"
    assert isinstance(tool_message["content"], list)
    assert tool_message["content"][0]["cache_control"] == {"type": "ephemeral"}


async def test_omits_anthropic_style_cache_markers_when_cache_retention_is_none() -> None:
    params = await capture_payload(custom_qwen_model(), OpenAICompletionsOptions(cache_retention="none"))
    instruction_message = get_instruction_message(params)

    assert instruction_message is not None
    assert not isinstance(instruction_message["content"], list)
    assert params["tools"][0].get("cache_control") is None
    assert isinstance(params["messages"][-1]["content"], str)
