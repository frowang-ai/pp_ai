"""Python port of `packages/ai/test/openai-completions-tool-choice.test.ts`.

The TypeScript test mocks the `openai` SDK module and inspects `mockState.lastParams`
(the object handed to `chat.completions.create`) plus the chunks the fake stream
yields. The Python port has no SDK layer: requests go out over `httpx`, so the
equivalent fake is an `httpx.MockTransport` that serves the same chunk list as an
SSE body, and the payload is captured through the `on_payload` hook (the same
object `build_params` produced, before serialization).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

import httpx
import pytest
from pi_ai.api.openai_completions import (
    OpenAICompletionsOptions,
    ResolvedCompat,
    convert_messages,
)
from pi_ai.compat import stream, stream_simple
from pi_ai.providers.all import get_builtin_model
from pi_ai.types import (
    AssistantMessage,
    Context,
    Message,
    Model,
    ModelCost,
    SimpleStreamOptions,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    now_ms,
)

_DEFAULT_CHUNKS: list[dict[str, Any] | None] = [
    {
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }
]

LOCAL_OPENAI_COMPLETIONS_MODEL = Model(
    id="",
    name="",
    api="openai-completions",
    provider="local-vllm",
    base_url="http://localhost:8000/v1",
    reasoning=True,
    input=["text"],
    cost=ModelCost(),
    context_window=128_000,
    max_tokens=8_192,
)


@dataclass
class SimpleOptionsWithToolChoice(SimpleStreamOptions):
    """`SimpleStreamOptions` plus the `toolChoice` the TS test smuggles in.

    TypeScript passes `{ apiKey, toolChoice }` through an
    `as unknown as Parameters<typeof streamSimple>[2]` cast; `stream_simple`
    reads the field with `getattr(options, "tool_choice", None)`.
    """

    tool_choice: str | dict[str, Any] | None = None


def sse_client(chunks: list[dict[str, Any] | None] | None = None) -> httpx.AsyncClient:
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in (chunks if chunks is not None else _DEFAULT_CHUNKS))
    body += "data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def user_context(text: str = "Hi", **kwargs: Any) -> Context:
    return Context(messages=[UserMessage(content=text, timestamp=now_ms())], **kwargs)


def builtin_model(provider: str, model_id: str) -> Model:
    model = get_builtin_model(provider, model_id)
    assert model is not None, f"missing built-in model {provider}/{model_id}"
    return model


def without_compat(provider: str, model_id: str) -> Model:
    """TypeScript's `{ ...baseModel, api: "openai-completions" }` with `compat` dropped."""
    return replace(builtin_model(provider, model_id), compat={}, api="openai-completions")


async def capture_simple_params(
    model: Model,
    reasoning: str | None = None,
    *,
    context: Context | None = None,
    tools: list[Tool] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    max_tokens: int | None = None,
    cache_retention: str | None = None,
    session_id: str | None = None,
    chunks: list[dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def on_payload(params: dict[str, Any], _model: Model) -> None:
        captured["params"] = params
        return None

    options = SimpleOptionsWithToolChoice(
        api_key="test",
        reasoning=reasoning,
        tool_choice=tool_choice,
        max_tokens=max_tokens,
        cache_retention=cache_retention,
        session_id=session_id,
        on_payload=on_payload,
    )
    request_context = context if context is not None else user_context()
    if tools is not None:
        request_context = replace(request_context, tools=tools)

    await stream_simple(model, request_context, options, client=sse_client(chunks)).result()
    assert "params" in captured, "expected the payload hook to fire"
    return captured["params"]


async def run_simple(
    model: Model,
    context: Context,
    chunks: list[dict[str, Any] | None] | None = None,
) -> AssistantMessage:
    return await stream_simple(model, context, SimpleStreamOptions(api_key="test"), client=sse_client(chunks)).result()


def ping_tool() -> Tool:
    return Tool(
        name="ping",
        description="Ping tool",
        parameters={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
    )


def read_tool() -> Tool:
    return Tool(
        name="read",
        description="Read a file",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    )


async def test_forwards_tool_choice_from_simple_options_to_payload() -> None:
    model = without_compat("openai", "gpt-4o-mini")
    params = await capture_simple_params(
        model,
        context=user_context("Call ping with ok=true"),
        tools=[ping_tool()],
        tool_choice="required",
    )

    assert params["tool_choice"] == "required"
    assert isinstance(params["tools"], list)
    assert len(params["tools"]) > 0


async def test_omits_strict_when_compat_disables_strict_mode() -> None:
    model = replace(without_compat("openai", "gpt-4o-mini"), compat={"supportsStrictMode": False})
    params = await capture_simple_params(
        model,
        context=user_context("Call ping with ok=true"),
        tools=[ping_tool()],
    )

    tool = params["tools"][0]["function"]
    assert tool
    assert "strict" not in tool


async def test_maps_groq_qwen_reasoning_levels_to_default_reasoning_effort() -> None:
    model = builtin_model("groq", "qwen/qwen3.6-27b")
    params = await capture_simple_params(model, "medium")

    assert params["reasoning_effort"] == "default"


async def test_keeps_normal_reasoning_effort_for_groq_models_without_compat_mapping() -> None:
    model = builtin_model("groq", "openai/gpt-oss-20b")
    params = await capture_simple_params(model, "medium")

    assert params["reasoning_effort"] == "medium"


async def test_enables_tool_stream_for_supported_zai_models_with_tools() -> None:
    model = builtin_model("zai", "glm-5.2")
    params = await capture_simple_params(
        model,
        context=user_context("Call ping with ok=true"),
        tools=[ping_tool()],
    )

    assert params["tool_stream"] is True


def test_stores_zai_tool_stream_support_in_model_compat_metadata() -> None:
    assert builtin_model("zai", "glm-4.7").compat.get("zaiToolStream") is True
    assert builtin_model("zai", "glm-5-turbo").compat.get("zaiToolStream") is True
    assert builtin_model("zai", "glm-5.2").compat.get("zaiToolStream") is True


def test_stores_zai_glm_5_2_effort_metadata() -> None:
    for provider in ("zai", "zai-coding-cn"):
        model = builtin_model(provider, "glm-5.2")
        assert model.compat.get("supportsReasoningEffort") is True
        assert model.thinking_level_map == {
            "minimal": None,
            "low": "high",
            "medium": "high",
            "high": "high",
            "max": "max",
        }


@pytest.mark.parametrize(
    ("reasoning", "effort"),
    [("low", "high"), ("medium", "high"), ("high", "high"), ("max", "max")],
)
async def test_maps_zai_glm_5_2_thinking_levels_to_reasoning_effort(reasoning: str, effort: str) -> None:
    model = builtin_model("zai", "glm-5.2")
    params = await capture_simple_params(model, reasoning)

    assert params["thinking"] == {"type": "enabled", "clear_thinking": False}
    assert params["reasoning_effort"] == effort


async def test_preserves_zai_thinking_when_replaying_reasoning_content() -> None:
    model = builtin_model("zai", "glm-5.2")
    assistant_message = AssistantMessage(
        api="openai-completions",
        provider="zai",
        model="glm-5.2",
        content=[
            ThinkingContent(thinking="prior reasoning", thinking_signature="reasoning_content"),
            ToolCall(id="call_1", name="read", arguments={"path": "README.md"}),
        ],
        stop_reason="toolUse",
        timestamp=now_ms(),
    )
    tool_result = ToolResultMessage(
        tool_call_id="call_1",
        tool_name="read",
        content=[TextContent(text="contents")],
        is_error=False,
        timestamp=now_ms(),
    )
    messages: list[Message] = [
        UserMessage(content="Read README.md", timestamp=now_ms()),
        assistant_message,
        tool_result,
        UserMessage(content="Continue", timestamp=now_ms()),
    ]

    params = await capture_simple_params(model, "high", context=Context(messages=messages))

    replayed = next(message for message in params["messages"] if message["role"] == "assistant")
    assert replayed["reasoning_content"] == "prior reasoning"
    assert params["thinking"] == {"type": "enabled", "clear_thinking": False}


async def test_omits_zai_glm_5_2_reasoning_effort_when_thinking_is_off() -> None:
    model = builtin_model("zai", "glm-5.2")
    params = await capture_simple_params(model)

    assert params["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in params


async def test_respects_explicit_zai_tool_stream_compat_override() -> None:
    base = builtin_model("zai", "glm-5.2")
    model = replace(base, compat={**base.compat, "zaiToolStream": True})
    params = await capture_simple_params(
        model,
        context=user_context("Call ping with ok=true"),
        tools=[ping_tool()],
    )

    assert params["tool_stream"] is True


async def test_omits_tool_stream_when_no_tools_are_provided() -> None:
    model = builtin_model("zai", "glm-5.2")
    params = await capture_simple_params(model)

    assert "tool_stream" not in params


async def test_maps_non_standard_provider_finish_reason_values_to_stop_reason_error() -> None:
    chunks: list[dict[str, Any] | None] = [
        {"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]},
        {
            "choices": [{"delta": {}, "finish_reason": "network_error"}],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        },
    ]

    model = builtin_model("zai", "glm-5.2")
    response = await run_simple(model, user_context(), chunks)

    assert response.stop_reason == "error"
    assert response.error_message == "Provider finish_reason: network_error"


async def test_ignores_null_stream_chunks_from_openai_compatible_providers() -> None:
    chunks: list[dict[str, Any] | None] = [
        None,
        {"id": "chatcmpl-test", "choices": [{"delta": {"content": "OK"}, "finish_reason": None}]},
        {
            "id": "chatcmpl-test",
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        },
    ]

    model = without_compat("openai", "gpt-4o-mini")
    response = await run_simple(model, user_context("Reply with exactly OK"), chunks)

    assert response.stop_reason == "stop"
    assert response.error_message is None
    assert response.response_id == "chatcmpl-test"
    assert response.usage.total_tokens == 4
    assert response.content == [TextContent(text="OK")]


async def test_errors_when_a_stream_ends_after_only_null_finish_reason_chunks() -> None:
    chunks: list[dict[str, Any] | None] = [
        {"id": "chatcmpl-truncated", "choices": [{"delta": {"content": "partial answer"}, "finish_reason": None}]},
        {"id": "chatcmpl-truncated", "choices": [{"delta": {"content": "partial answer"}, "finish_reason": None}]},
    ]

    model = without_compat("openai", "gpt-4o-mini")
    response = await run_simple(model, user_context("Reply with a longer sentence"), chunks)

    assert response.stop_reason == "error"
    assert response.error_message == "Stream ended without finish_reason"


async def test_accepts_streams_without_finish_reason_when_compat_disables_it() -> None:
    chunks: list[dict[str, Any] | None] = [
        {
            "id": "chatcmpl-no-finish-reason",
            "choices": [{"delta": {"content": "complete answer"}, "finish_reason": None}],
        }
    ]

    model = replace(without_compat("openai", "gpt-4o-mini"), compat={"supportsFinishReason": False})
    response = await run_simple(model, user_context("Reply with a complete answer"), chunks)

    assert response.stop_reason == "stop"
    assert response.error_message is None
    assert response.content == [TextContent(text="complete answer")]


async def test_ignores_empty_custom_objects_on_function_tool_call_deltas() -> None:
    chunks: list[dict[str, Any] | None] = [
        {
            "id": "chatcmpl-empty-custom",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "read", "arguments": '{"path":"README.md"}'},
                                "custom": {},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
    ]

    model = without_compat("openai", "gpt-4o-mini")
    context = Context(
        messages=[UserMessage(content="Read README.md", timestamp=now_ms())],
        tools=[read_tool()],
    )
    response = await run_simple(model, context, chunks)

    assert response.content == [ToolCall(id="call_1", name="read", arguments={"path": "README.md"})]


async def test_coalesces_tool_call_deltas_by_stable_index_when_ids_change() -> None:
    chunks: list[dict[str, Any] | None] = [
        {
            "id": "chatcmpl-kimi-bad-stream",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "functions.read:0",
                                "type": "function",
                                "function": {"name": "read", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-kimi-bad-stream",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "chatcmpl-tool-a",
                                "type": "function",
                                "function": {"name": None, "arguments": '{"path":"README'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-kimi-bad-stream",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "chatcmpl-tool-b",
                                "type": "function",
                                "function": {"name": None, "arguments": '.md"}'},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        },
    ]

    model = without_compat("openai", "gpt-4o-mini")
    context = Context(
        messages=[UserMessage(content="Read README.md", timestamp=now_ms())],
        tools=[read_tool()],
    )
    event_stream = stream_simple(model, context, SimpleStreamOptions(api_key="test"), client=sse_client(chunks))

    tool_call_content_indexes: list[int] = []
    async for event in event_stream:
        if event.type in ("toolcall_start", "toolcall_delta", "toolcall_end"):
            tool_call_content_indexes.append(event.content_index)

    response = await event_stream.result()
    assert response.stop_reason == "toolUse"
    assert tool_call_content_indexes == [0, 0, 0, 0, 0]
    assert len(response.content) == 1
    # TS also asserts the finished tool call carries no `streamIndex`/`partialArgs`
    # scratch fields; `ToolCall` is a fixed-field dataclass here, so the equality
    # check below covers that.
    assert response.content[0] == ToolCall(id="functions.read:0", name="read", arguments={"path": "README.md"})


async def test_accumulates_mixed_content_reasoning_and_parallel_tool_call_deltas() -> None:
    chunks: list[dict[str, Any] | None] = [
        {
            "id": "chatcmpl-mixed-deltas",
            "choices": [
                {
                    "delta": {
                        "content": "answer 1",
                        "reasoning_content": "think 1",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "tc_read_initial",
                                "type": "function",
                                "function": {"name": "read", "arguments": '{"path":"README'},
                            },
                            {
                                "index": 1,
                                "id": "tc_grep_initial",
                                "type": "function",
                                "function": {"name": "grep", "arguments": '{"pattern":"TODO'},
                            },
                            {
                                "id": "tc_list_no_index",
                                "type": "function",
                                "function": {"name": "list", "arguments": '{"path":"packages'},
                            },
                            {
                                "id": "tc_write_no_index",
                                "type": "function",
                                "function": {"name": "write", "arguments": '{"path":"out'},
                            },
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-mixed-deltas",
            "choices": [
                {
                    "delta": {
                        "content": " answer 2",
                        "tool_calls": [
                            {
                                "index": 1,
                                "id": "tc_grep_changed",
                                "type": "function",
                                "function": {"arguments": '","path":"src'},
                            },
                            {
                                "id": "tc_write_no_index",
                                "type": "function",
                                "function": {"arguments": '.txt","content":"ok"}'},
                            },
                            {
                                "id": "tc_list_no_index",
                                "type": "function",
                                "function": {"arguments": '/ai"}'},
                            },
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-mixed-deltas",
            "choices": [
                {
                    "delta": {
                        "content": "\n",
                        "reasoning_content": " think 2",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "tc_read_changed",
                                "type": "function",
                                "function": {"arguments": '.md"}'},
                            },
                            {"index": 1, "type": "function", "function": {"arguments": '"}'}},
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        },
    ]

    model = without_compat("openai", "gpt-4o-mini")
    tools = [
        read_tool(),
        Tool(
            name="grep",
            description="Search a file",
            parameters={
                "type": "object",
                "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                "required": ["pattern", "path"],
            },
        ),
        Tool(
            name="list",
            description="List a directory",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        ),
        Tool(
            name="write",
            description="Write a file",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        ),
    ]
    context = Context(
        messages=[UserMessage(content="Think, answer, and use tools.", timestamp=now_ms())],
        tools=tools,
    )
    event_stream = stream_simple(model, context, SimpleStreamOptions(api_key="test"), client=sse_client(chunks))

    event_types: list[str] = []
    tool_events_by_content_index: dict[int, list[str]] = {}
    async for event in event_stream:
        event_types.append(event.type)
        if event.type in ("toolcall_start", "toolcall_delta", "toolcall_end"):
            tool_events_by_content_index.setdefault(event.content_index, []).append(event.type)

    response = await event_stream.result()
    assert response.stop_reason == "toolUse"
    assert event_types.count("text_start") == 1
    assert event_types.count("text_delta") == 3
    assert event_types.count("text_end") == 1
    assert event_types.count("thinking_start") == 1
    assert event_types.count("thinking_delta") == 2
    assert event_types.count("thinking_end") == 1
    assert event_types.count("toolcall_start") == 4
    assert event_types.count("toolcall_delta") == 9
    assert event_types.count("toolcall_end") == 4
    assert tool_events_by_content_index[2] == [
        "toolcall_start",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_end",
    ]
    assert tool_events_by_content_index[3] == [
        "toolcall_start",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_end",
    ]
    assert tool_events_by_content_index[4] == [
        "toolcall_start",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_end",
    ]
    assert tool_events_by_content_index[5] == [
        "toolcall_start",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_end",
    ]

    assert len(response.content) == 6
    assert response.content[0] == TextContent(text="answer 1 answer 2\n")
    assert response.content[1] == ThinkingContent(thinking="think 1 think 2", thinking_signature="reasoning_content")
    assert response.content[2] == ToolCall(id="tc_read_initial", name="read", arguments={"path": "README.md"})
    assert response.content[3] == ToolCall(
        id="tc_grep_initial", name="grep", arguments={"pattern": "TODO", "path": "src"}
    )
    assert response.content[4] == ToolCall(id="tc_list_no_index", name="list", arguments={"path": "packages/ai"})
    assert response.content[5] == ToolCall(
        id="tc_write_no_index", name="write", arguments={"path": "out.txt", "content": "ok"}
    )


async def test_uses_system_messages_for_non_openai_anthropic_openrouter_reasoning_models() -> None:
    model = builtin_model("openrouter", "deepseek/deepseek-v4-pro")
    params = await capture_simple_params(model, context=user_context(system_prompt="Follow instructions."))

    assert params["messages"][0]["role"] == "system"


async def test_keeps_developer_messages_for_openai_and_anthropic_openrouter_models() -> None:
    for model_id in ("openai/gpt-5.2-codex", "anthropic/claude-sonnet-4.5"):
        model = builtin_model("openrouter", model_id)
        params = await capture_simple_params(model, context=user_context(system_prompt="Follow instructions."))

        assert params["messages"][0]["role"] == "developer"


async def test_keeps_developer_messages_for_openai_reasoning_model_instructions() -> None:
    model = without_compat("openai", "gpt-5.5")
    params = await capture_simple_params(model, context=user_context(system_prompt="Follow instructions."))

    assert params["messages"][0]["role"] == "developer"


def test_stores_openrouter_kimi_k2_6_reasoning_replay_compat() -> None:
    # The `:free` variant is delisted from the OpenRouter API; the generator
    # override matches any listed `moonshotai/kimi-k2.6*` variant.
    model = builtin_model("openrouter", "moonshotai/kimi-k2.6")
    assert model.compat.get("supportsDeveloperRole") is False
    assert model.compat.get("requiresReasoningContentOnAssistantMessages") is True


def test_stores_xiaomi_mimo_reasoning_replay_compat() -> None:
    for provider in ("xiaomi", "xiaomi-token-plan-cn", "xiaomi-token-plan-ams", "xiaomi-token-plan-sgp"):
        model = builtin_model(provider, "mimo-v2.5-pro")
        assert model.compat.get("requiresReasoningContentOnAssistantMessages") is True
        assert model.compat.get("thinkingFormat") == "deepseek"
        assert "maxTokensField" not in model.compat
        assert "supportsDeveloperRole" not in model.compat


def test_stores_qwen_token_plan_reasoning_replay_compat() -> None:
    for provider in ("qwen-token-plan", "qwen-token-plan-cn", "qwen-token-plan-individual"):
        model = builtin_model(provider, "qwen3.7-max")
        assert model.compat.get("thinkingFormat") == "qwen"
        assert "requiresReasoningContentOnAssistantMessages" not in model.compat
        assert model.compat.get("supportsDeveloperRole") is False
        assert model.compat.get("supportsStore") is False


async def test_replays_xiaomi_mimo_tool_calls_with_empty_reasoning_content() -> None:
    model = builtin_model("xiaomi", "mimo-v2.5-pro")
    assistant_message = AssistantMessage(
        api="openai-completions",
        provider="xiaomi",
        model="mimo-v2.5-pro",
        content=[ToolCall(id="call_1", name="read", arguments={"path": "README.md"})],
        stop_reason="toolUse",
        timestamp=now_ms(),
    )
    tool_result = ToolResultMessage(
        tool_call_id="call_1",
        tool_name="read",
        content=[TextContent(text="contents")],
        is_error=False,
        timestamp=now_ms(),
    )
    messages: list[Message] = [
        UserMessage(content="Read README.md", timestamp=now_ms()),
        assistant_message,
        tool_result,
    ]

    params = await capture_simple_params(model, "high", context=Context(messages=messages))

    replayed = next(message for message in params["messages"] if message["role"] == "assistant")
    assert replayed["reasoning_content"] == ""
    assert params["thinking"] == {"type": "enabled"}
    assert params["reasoning_effort"] == "high"


async def test_normalizes_opencode_go_reasoning_deltas_to_reasoning_content() -> None:
    chunks: list[dict[str, Any] | None] = [
        {
            "id": "chatcmpl-opencode-go-reasoning",
            "choices": [{"delta": {"reasoning": "think"}, "finish_reason": "stop"}],
        }
    ]

    model = without_compat("opencode-go", "kimi-k2.6")
    response = await run_simple(model, user_context("Use reasoning."), chunks)

    assert response.content == [ThinkingContent(thinking="think", thinking_signature="reasoning_content")]


async def test_keeps_non_opencode_go_reasoning_deltas_on_the_original_field() -> None:
    chunks: list[dict[str, Any] | None] = [
        {"id": "chatcmpl-reasoning", "choices": [{"delta": {"reasoning": "think"}, "finish_reason": "stop"}]}
    ]

    model = without_compat("openai", "gpt-4o-mini")
    response = await run_simple(model, user_context("Use reasoning."), chunks)

    assert response.content == [ThinkingContent(thinking="think", thinking_signature="reasoning")]


def test_replays_opencode_go_reasoning_thinking_blocks_as_reasoning_content() -> None:
    model = without_compat("opencode-go", "kimi-k2.6")
    context = Context(
        messages=[
            AssistantMessage(
                api="openai-completions",
                provider="opencode-go",
                model="kimi-k2.6",
                content=[
                    ThinkingContent(thinking="think", thinking_signature="reasoning"),
                    ToolCall(id="call_1", name="read", arguments={"path": "README.md"}),
                ],
                stop_reason="stop",
                timestamp=now_ms(),
            )
        ]
    )
    compat = ResolvedCompat(
        supports_store=False,
        supports_developer_role=False,
        supports_reasoning_effort=True,
        supports_usage_in_streaming=True,
        supports_finish_reason=True,
        max_tokens_field="max_completion_tokens",
        requires_tool_result_name=False,
        requires_assistant_after_tool_result=False,
        requires_thinking_as_text=False,
        requires_reasoning_content_on_assistant_messages=False,
        thinking_format="openai",
        open_router_routing={},
        vercel_gateway_routing={},
        chat_template_kwargs={},
        chat_template_args={},
        zai_tool_stream=False,
        supports_strict_mode=True,
        supports_openai_grammar_tools=False,
        send_session_affinity_headers=False,
        session_affinity_format="openai",
        supports_long_cache_retention=True,
    )

    messages = convert_messages(model, context, compat)

    assert messages[0]["role"] == "assistant"
    assert messages[0]["reasoning_content"] == "think"
    assert "reasoning" not in messages[0]


async def test_sends_thinking_disabled_for_opencode_go_kimi_when_thinking_is_off() -> None:
    model = builtin_model("opencode-go", "kimi-k2.6")
    params = await capture_simple_params(model)

    assert params["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in params


async def test_sends_thinking_enabled_for_opencode_go_kimi_when_thinking_is_enabled() -> None:
    model = builtin_model("opencode-go", "kimi-k2.6")
    params = await capture_simple_params(model, "high")

    assert params["thinking"] == {"type": "enabled"}
    assert "reasoning_effort" not in params


async def test_omits_disabled_thinking_for_moonshot_kimi_k2_7_code_models() -> None:
    for provider in ("moonshotai", "moonshotai-cn"):
        model = builtin_model(provider, "kimi-k2.7-code")
        params = await capture_simple_params(model)

        assert "thinking" not in params
        assert "reasoning_effort" not in params


async def test_keeps_disabled_thinking_for_moonshot_kimi_k2_6_when_thinking_is_off() -> None:
    model = builtin_model("moonshotai-cn", "kimi-k2.6")
    params = await capture_simple_params(model)

    assert params["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in params


async def test_sends_max_tokens_for_opencode_completions_models() -> None:
    for provider in ("opencode-go", "opencode"):
        model = builtin_model(provider, "kimi-k2.6")
        assert model.compat.get("maxTokensField") == "max_tokens"

        params = await capture_simple_params(model, max_tokens=123)

        assert params["max_tokens"] == 123
        assert "max_completion_tokens" not in params


async def test_sends_max_tokens_for_builtin_and_custom_deepseek_models() -> None:
    custom_model = replace(
        LOCAL_OPENAI_COMPLETIONS_MODEL,
        id="custom-deepseek-model",
        name="Custom DeepSeek Model",
        provider="custom-deepseek",
        base_url="https://api.deepseek.com",
    )
    native_models = [
        builtin_model("deepseek", "deepseek-v4-flash"),
        builtin_model("deepseek", "deepseek-v4-pro"),
    ]
    for model in native_models:
        assert model.compat.get("maxTokensField") == "max_tokens"

    for model in [*native_models, custom_model]:
        params = await capture_simple_params(model, max_tokens=123)

        assert params["max_tokens"] == 123
        assert "max_completion_tokens" not in params


async def test_sends_max_tokens_for_zai_completions_models() -> None:
    for model_id in ("glm-5-turbo", "glm-5.2"):
        model = builtin_model("zai", model_id)
        assert model.compat.get("maxTokensField") == "max_tokens"

        params = await capture_simple_params(model, max_tokens=123)

        assert params["max_tokens"] == 123
        assert "max_completion_tokens" not in params


async def test_omits_reasoning_effort_for_opencode_grok_build() -> None:
    model = builtin_model("opencode", "grok-build-0.1")
    params = await capture_simple_params(model, "high")

    assert "reasoning_effort" not in params


async def test_does_not_double_count_reasoning_tokens_in_completion_usage() -> None:
    chunks: list[dict[str, Any] | None] = [
        {
            "id": "chatcmpl-reasoning-usage",
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 33,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 21},
            },
        }
    ]

    model = without_compat("openai", "gpt-4o-mini")
    response = await run_simple(model, user_context("Use reasoning."), chunks)

    assert response.usage.input == 10
    assert response.usage.output == 33
    assert response.usage.total_tokens == 43


async def test_preserves_prompt_tokens_details_cache_fields_from_chunk_usage() -> None:
    chunks: list[dict[str, Any] | None] = [
        {"id": "chatcmpl-cache-write", "choices": [{"delta": {"content": "OK"}, "finish_reason": None}]},
        {
            "id": "chatcmpl-cache-write",
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 50, "cache_write_tokens": 30},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        },
    ]

    model = without_compat("openai", "gpt-4o-mini")
    response = await run_simple(model, user_context("Reply with exactly OK"), chunks)

    # cached_tokens is documented as cache reads; cache_write_tokens is separate.
    assert response.usage.input == 20
    assert response.usage.cache_read == 50
    assert response.usage.cache_write == 30
    assert response.usage.total_tokens == 105


async def test_preserves_prompt_tokens_details_cache_fields_from_choice_usage_fallback() -> None:
    chunks: list[dict[str, Any] | None] = [
        {
            "id": "chatcmpl-cache-write-choice",
            "choices": [{"delta": {"content": "OK"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-cache-write-choice",
            "choices": [
                {
                    "delta": {},
                    "finish_reason": "stop",
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 5,
                        "prompt_tokens_details": {"cached_tokens": 50, "cache_write_tokens": 30},
                        "completion_tokens_details": {"reasoning_tokens": 0},
                    },
                }
            ],
        },
    ]

    model = without_compat("openai", "gpt-4o-mini")
    response = await run_simple(model, user_context("Reply with exactly OK"), chunks)

    # cached_tokens is documented as cache reads; cache_write_tokens is separate.
    assert response.usage.input == 20
    assert response.usage.cache_read == 50
    assert response.usage.cache_write == 30
    assert response.usage.total_tokens == 105


async def test_uses_openrouter_reasoning_object_instead_of_reasoning_effort() -> None:
    model = builtin_model("openrouter", "deepseek/deepseek-r1")
    params = await capture_simple_params(model, "high")

    assert params["reasoning"] == {"effort": "high"}
    assert "reasoning_effort" not in params


@pytest.mark.parametrize(("reasoning", "expected"), [("high", True), (None, False)])
async def test_uses_configurable_chat_template_boolean_thinking_kwargs(reasoning: str | None, expected: bool) -> None:
    model = replace(
        LOCAL_OPENAI_COMPLETIONS_MODEL,
        id="deepseek-ai/DeepSeek-V3.1",
        name="DeepSeek V3.1 via vLLM",
        compat={
            "thinkingFormat": "chat-template",
            "supportsReasoningEffort": False,
            "chatTemplateKwargs": {"thinking": {"$var": "thinking.enabled"}},
        },
    )

    params = await capture_simple_params(model, reasoning)

    assert params["chat_template_kwargs"] == {"thinking": expected}
    assert "thinking" not in params
    assert "reasoning_effort" not in params


@pytest.mark.parametrize(("reasoning", "expected"), [("high", True), (None, False)])
async def test_uses_qwen_chat_template_thinking_kwargs(reasoning: str | None, expected: bool) -> None:
    model = replace(
        LOCAL_OPENAI_COMPLETIONS_MODEL,
        id="Qwen/Qwen3-Coder",
        name="Qwen3 Coder via vLLM",
        compat={"thinkingFormat": "qwen-chat-template", "supportsReasoningEffort": False},
    )

    params = await capture_simple_params(model, reasoning)

    assert params["chat_template_kwargs"] == {
        "enable_thinking": expected,
        "preserve_thinking": True,
    }
    assert "reasoning_effort" not in params


async def test_uses_configurable_chat_template_effort_kwargs_with_static_kwargs() -> None:
    model = replace(
        LOCAL_OPENAI_COMPLETIONS_MODEL,
        id="unsloth/gpt-oss-120b-GGUF",
        name="GPT OSS via vLLM",
        thinking_level_map={"xhigh": "max"},
        compat={
            "thinkingFormat": "chat-template",
            "supportsReasoningEffort": False,
            "chatTemplateKwargs": {
                "preserve_thinking": True,
                "reasoning_effort": {"$var": "thinking.effort", "omitWhenOff": True},
            },
        },
    )

    params = await capture_simple_params(model, "xhigh")

    assert params["chat_template_kwargs"] == {"preserve_thinking": True, "reasoning_effort": "max"}
    assert "reasoning_effort" not in params


async def test_uses_ant_ling_compatibility_metadata() -> None:
    model = builtin_model("ant-ling", "Ring-2.6-1T")

    assert model.compat.get("supportsStore") is False
    assert model.compat.get("supportsDeveloperRole") is False
    assert model.compat.get("supportsReasoningEffort") is False
    assert model.compat.get("maxTokensField") == "max_tokens"
    assert model.compat.get("thinkingFormat") == "ant-ling"
    assert model.compat.get("supportsLongCacheRetention") is False
    assert "supportsStrictMode" not in model.compat
    assert "requiresReasoningContentOnAssistantMessages" not in model.compat

    params = await capture_simple_params(
        model,
        "high",
        context=user_context(system_prompt="Follow instructions."),
        max_tokens=123,
        cache_retention="long",
        session_id="ant-ling-session",
    )

    assert params["max_tokens"] == 123
    assert "max_completion_tokens" not in params
    assert params["messages"][0]["role"] == "system"
    assert params["reasoning"] == {"effort": "high"}
    assert "reasoning_effort" not in params
    assert "store" not in params
    assert "prompt_cache_key" not in params
    assert "prompt_cache_retention" not in params


async def test_omits_ant_ling_reasoning_for_unmapped_efforts_and_non_reasoning_models() -> None:
    ring = builtin_model("ant-ling", "Ring-2.6-1T")
    captured: dict[str, Any] = {}

    def on_payload(params: dict[str, Any], _model: Model) -> None:
        captured["params"] = params
        return None

    await stream(
        ring,
        user_context(),
        OpenAICompletionsOptions(api_key="test", reasoning_effort="medium", on_payload=on_payload),
        client=sse_client(),
    ).result()

    assert "reasoning" not in captured["params"]

    ling = builtin_model("ant-ling", "Ling-2.6-flash")
    params = await capture_simple_params(ling, "high")

    assert "reasoning" not in params
