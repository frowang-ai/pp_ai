"""Python port of `packages/ai/test/deferred-tools.test.ts`.

Named with a `_ported` suffix because `tests/test_deferred_tools.py` already
exists in this repo and unit-tests `split_deferred_tools` directly; this file is
the port of the TypeScript test of the same name.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from pi_ai.api.openai_completions import ResolvedCompat
from pi_ai.api.openai_completions import convert_messages as convert_completions_messages
from pi_ai.compat import stream_simple
from pi_ai.providers.all import get_builtin_model
from pi_ai.types import (
    AssistantMessage,
    Context,
    ImageContent,
    Model,
    ModelCost,
    SimpleStreamOptions,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pi_ai.utils.estimate import estimate_context_tokens


class PayloadCaptured(Exception):
    pass


def make_tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=f"The {name} tool",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
    )


def make_user_message(timestamp: int) -> UserMessage:
    return UserMessage(content="Hello", timestamp=timestamp)


def make_assistant_tool_call() -> AssistantMessage:
    return AssistantMessage(
        content=[ToolCall(id="call_1", name="base_tool", arguments={})],
        api="anthropic-messages",
        provider="anthropic",
        model="claude-opus-4-6",
        usage=Usage(),
        stop_reason="toolUse",
        timestamp=2,
    )


def make_tool_result(added_tool_names: list[str]) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id="call_1",
        tool_name="base_tool",
        content=[TextContent(text="done")],
        added_tool_names=added_tool_names,
        is_error=False,
        timestamp=3,
    )


def make_context(tools: list[Tool], added_tool_names: list[str] | None = None) -> Context:
    return Context(
        messages=[
            make_user_message(1),
            make_assistant_tool_call(),
            make_tool_result(["late_tool"] if added_tool_names is None else added_tool_names),
            make_user_message(4),
        ],
        tools=tools,
    )


def make_kimi_model(deferred_tools_mode: str | None = None) -> Model:
    return Model(
        id="deferred-tools-model",
        name="Deferred Tools Model",
        api="openai-completions",
        provider="moonshotai",
        base_url="http://127.0.0.1:9/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128000,
        max_tokens=4096,
        compat={"deferredToolsMode": deferred_tools_mode} if deferred_tools_mode else {},
    )


async def capture_payload(model: Model, context: Context, api_key: str = "fake-key") -> dict[str, Any]:
    captured: dict[str, Any] | None = None

    def on_payload(payload: dict[str, Any], request_model: Model) -> None:
        nonlocal captured
        captured = payload
        raise PayloadCaptured()

    await stream_simple(
        dataclasses.replace(model, base_url="http://127.0.0.1:9"),
        context,
        SimpleStreamOptions(api_key=api_key, on_payload=on_payload),
    ).result()

    assert captured is not None, "Expected payload capture"
    return captured


def find_anthropic_tool_result_content(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for message in payload["messages"]:
        content = message["content"]
        if not isinstance(content, str) and any(block.get("type") == "tool_result" for block in content):
            return content
    raise AssertionError("No tool result in payload")


def find_anthropic_tool_result(payload: dict[str, Any]) -> dict[str, Any]:
    for block in find_anthropic_tool_result_content(payload):
        if block.get("type") == "tool_result":
            return block
    raise AssertionError("No tool result in payload")


def openai_tool_names(payload: dict[str, Any]) -> list[str]:
    return [tool.get("name") or tool.get("function", {}).get("name") or "" for tool in payload.get("tools") or []]


def assert_matches(actual: Any, expected: Any) -> None:
    """Equivalent of vitest `toMatchObject`."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"expected dict, got {actual!r}"
        for key, value in expected.items():
            assert key in actual, f"missing key {key!r} in {actual!r}"
            assert_matches(actual[key], value)
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"expected list, got {actual!r}"
        assert len(actual) == len(expected), f"length mismatch: {actual!r} vs {expected!r}"
        for actual_item, expected_item in zip(actual, expected, strict=True):
            assert_matches(actual_item, expected_item)
    else:
        assert actual == expected


async def test_loads_an_anthropic_tool_at_its_tool_result_marker():
    context = make_context([make_tool("base_tool"), make_tool("late_tool")])
    payload = await capture_payload(get_builtin_model("anthropic", "claude-opus-4-6"), context)

    assert_matches(payload["tools"], [{"name": "base_tool"}, {"name": "late_tool", "defer_loading": True}])
    assert find_anthropic_tool_result(payload)["content"] == [{"type": "tool_reference", "tool_name": "late_tool"}]


async def test_preserves_tool_output_as_sibling_content_after_emitting_references():
    context = make_context([make_tool("base_tool"), make_tool("late_tool")])
    assistant = context.messages[1]
    assert isinstance(assistant, AssistantMessage)
    assistant.content = [
        ToolCall(id="call_1", name="base_tool", arguments={}),
        ToolCall(id="call_2", name="base_tool", arguments={}),
    ]
    first_result = context.messages[2]
    assert isinstance(first_result, ToolResultMessage)
    first_result.content = [
        TextContent(text="work completed"),
        ImageContent(mime_type="image/png", data="aW1hZ2U="),
    ]
    second_result = make_tool_result([])
    second_result.tool_call_id = "call_2"
    second_result.content = [TextContent(text="second result")]
    context.messages.insert(3, second_result)

    payload = await capture_payload(get_builtin_model("anthropic", "claude-opus-4-6"), context)

    assert_matches(
        find_anthropic_tool_result_content(payload),
        [
            {
                "type": "tool_result",
                "tool_use_id": "call_1",
                "content": [{"type": "tool_reference", "tool_name": "late_tool"}],
            },
            {"type": "tool_result", "tool_use_id": "call_2", "content": "second result"},
            {"type": "text", "text": "work completed"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="}},
        ],
    )


async def test_loads_a_tool_introduced_by_openai_history_after_switching_to_anthropic():
    context = make_context([make_tool("base_tool"), make_tool("late_tool")])
    assistant = context.messages[1]
    assert isinstance(assistant, AssistantMessage)
    assistant.api = "openai-responses"
    assistant.provider = "openai"
    assistant.model = "gpt-5.4"

    payload = await capture_payload(get_builtin_model("anthropic", "claude-opus-4-8"), context)

    assert_matches(payload["tools"], [{"name": "base_tool"}, {"name": "late_tool", "defer_loading": True}])
    assert find_anthropic_tool_result(payload)["content"] == [{"type": "tool_reference", "tool_name": "late_tool"}]


async def test_does_not_resurrect_a_marked_tool_missing_from_context_tools():
    context = make_context([make_tool("base_tool")])
    payload = await capture_payload(get_builtin_model("anthropic", "claude-opus-4-6"), context)

    assert [tool["name"] for tool in payload["tools"]] == ["base_tool"]
    content = find_anthropic_tool_result(payload)["content"]
    assert not (isinstance(content, list) and any(block.get("type") == "tool_reference" for block in content))


async def test_keeps_a_tool_immediate_when_it_was_used_before_its_marker():
    context = make_context([make_tool("base_tool"), make_tool("late_tool")])
    assistant = context.messages[1]
    assert isinstance(assistant, AssistantMessage)
    assistant.content = [ToolCall(id="call_1", name="late_tool", arguments={})]

    payload = await capture_payload(get_builtin_model("anthropic", "claude-opus-4-6"), context)

    assert [tool["name"] for tool in payload["tools"]] == ["base_tool", "late_tool"]
    assert all(not tool.get("defer_loading") for tool in payload["tools"])


async def test_normalizes_oauth_names_before_checking_prior_tool_usage():
    context = make_context([make_tool("base_tool"), make_tool("read")], ["read"])
    assistant = context.messages[1]
    assert isinstance(assistant, AssistantMessage)
    assistant.content = [ToolCall(id="call_1", name="Read", arguments={})]

    payload = await capture_payload(get_builtin_model("anthropic", "claude-opus-4-6"), context, "sk-ant-oat-fake")

    assert [tool["name"] for tool in payload["tools"]] == ["base_tool", "Read"]
    assert all(not tool.get("defer_loading") for tool in payload["tools"])
    content = find_anthropic_tool_result(payload)["content"]
    assert not (isinstance(content, list) and any(block.get("type") == "tool_reference" for block in content))


async def test_matches_oauth_canonicalized_markers_to_active_tools():
    context = make_context([make_tool("base_tool"), make_tool("read")], ["Read"])
    payload = await capture_payload(get_builtin_model("anthropic", "claude-opus-4-6"), context, "sk-ant-oat-fake")

    assert_matches(payload["tools"], [{"name": "base_tool"}, {"name": "Read", "defer_loading": True}])
    content = find_anthropic_tool_result(payload)["content"]
    assert isinstance(content, list)
    assert any(block.get("type") == "tool_reference" and block.get("tool_name") == "Read" for block in content)


async def test_deduplicates_active_tools_after_oauth_canonicalization():
    read_tool = make_tool("Read")
    read_tool.description = "Canonical definition"
    context = Context(messages=[make_user_message(1)], tools=[make_tool("read"), read_tool])

    payload = await capture_payload(get_builtin_model("anthropic", "claude-opus-4-6"), context, "sk-ant-oat-fake")

    assert_matches(payload["tools"], [{"name": "Read", "description": "Canonical definition"}])


@pytest.mark.parametrize("model_id", ["claude-haiku-4-5", "claude-sonnet-4-20250514"])
async def test_uses_the_normal_tool_list_when_anthropic_tool_references_unsupported(model_id: str):
    context = make_context([make_tool("base_tool"), make_tool("late_tool")])
    if model_id == "claude-haiku-4-5":
        model = get_builtin_model("anthropic", "claude-haiku-4-5")
    else:
        model = dataclasses.replace(get_builtin_model("anthropic", "claude-opus-4-6"), id=model_id)

    payload = await capture_payload(model, context)

    assert [tool["name"] for tool in payload["tools"]] == ["base_tool", "late_tool"]
    assert all(not tool.get("defer_loading") for tool in payload["tools"])


async def test_keeps_one_immediate_anthropic_tool_when_every_current_tool_is_marked():
    context = make_context([make_tool("late_tool")])
    payload = await capture_payload(get_builtin_model("anthropic", "claude-opus-4-6"), context)

    assert_matches(payload["tools"], [{"name": "late_tool"}])
    assert payload["tools"][0].get("defer_loading") is None
    content = find_anthropic_tool_result(payload)["content"]
    assert not (isinstance(content, list) and any(block.get("type") == "tool_reference" for block in content))


async def test_supports_explicit_anthropic_compatibility_overrides():
    model = dataclasses.replace(
        get_builtin_model("anthropic", "claude-opus-4-6"),
        provider="anthropic-proxy",
        compat={"supportsToolReferences": True},
    )
    context = make_context([make_tool("base_tool"), make_tool("late_tool")])
    payload = await capture_payload(model, context)

    late = next(tool for tool in payload["tools"] if tool["name"] == "late_tool")
    assert late["defer_loading"] is True


async def test_serializes_kimi_deferred_tools_as_system_tool_definitions():
    context = make_context([make_tool("base_tool"), make_tool("late_tool")])
    payload = await capture_payload(make_kimi_model("kimi"), context)

    assert [tool["function"]["name"] for tool in payload["tools"]] == ["base_tool"]
    tool_result_index = next(index for index, message in enumerate(payload["messages"]) if message["role"] == "tool")
    system_tool_index = next(index for index, message in enumerate(payload["messages"]) if "tools" in message)
    assert tool_result_index >= 0
    assert system_tool_index > tool_result_index
    assert [tool["function"]["name"] for tool in payload["messages"][system_tool_index]["tools"]] == ["late_tool"]


def test_emits_kimi_deferred_schemas_after_all_tool_results_in_a_batch():
    context = make_context([make_tool("base_tool"), make_tool("late_tool"), make_tool("later_tool")])
    second_result = make_tool_result(["later_tool"])
    second_result.tool_call_id = "call_2"
    context.messages.insert(3, second_result)

    messages = convert_completions_messages(
        make_kimi_model("kimi"),
        context,
        ResolvedCompat(
            supports_store=False,
            supports_developer_role=False,
            supports_reasoning_effort=False,
            supports_usage_in_streaming=True,
            supports_finish_reason=True,
            max_tokens_field="max_tokens",
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
            supports_strict_mode=False,
            supports_openai_grammar_tools=False,
            cache_control_format=None,
            send_session_affinity_headers=False,
            deferred_tools_mode="kimi",
            session_affinity_format="openai",
            supports_long_cache_retention=False,
        ),
    )

    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "system",
        "user",
    ]
    assert [tool["function"]["name"] for tool in messages[4]["tools"]] == ["late_tool", "later_tool"]


async def test_leaves_openai_completions_tools_unchanged_without_kimi_mode():
    context = make_context([make_tool("base_tool"), make_tool("late_tool")])
    payload = await capture_payload(make_kimi_model(), context)

    assert [tool["function"]["name"] for tool in payload["tools"]] == ["base_tool", "late_tool"]
    assert not any("tools" in message for message in payload["messages"])


async def test_loads_an_openai_responses_tool_through_additional_tools():
    context = make_context([make_tool("base_tool"), make_tool("late_tool")])
    payload = await capture_payload(get_builtin_model("openai", "gpt-5.4"), context)
    additional_tools = next(item for item in payload["input"] if item.get("type") == "additional_tools")

    assert openai_tool_names(payload) == ["base_tool"]
    assert_matches(additional_tools, {"role": "developer"})
    assert_matches(additional_tools["tools"], [{"type": "function", "name": "late_tool"}])
    assert all(tool.get("defer_loading") is None for tool in additional_tools["tools"])
    assert not any(item.get("type") == "tool_search_call" for item in payload["input"])
    assert not any(item.get("type") == "tool_search_output" for item in payload["input"])


async def test_preserves_an_additional_tools_marker_after_the_loaded_tool_is_used():
    context = make_context([make_tool("base_tool"), make_tool("late_tool")])
    late_call = make_assistant_tool_call()
    late_call.content = [ToolCall(id="call_late|fc_late", name="late_tool", arguments={})]
    late_call.api = "openai-responses"
    late_call.provider = "openai"
    late_call.model = "gpt-5.4"
    late_result = make_tool_result(["late_tool"])
    late_result.tool_call_id = "call_late|fc_late"
    late_result.tool_name = "late_tool"
    context.messages[3:3] = [late_call, late_result]

    payload = await capture_payload(get_builtin_model("openai", "gpt-5.4"), context)
    additional_tool_indexes = [
        index for index, item in enumerate(payload["input"]) if item.get("type") == "additional_tools"
    ]
    late_call_index = next(
        index
        for index, item in enumerate(payload["input"])
        if item.get("type") == "function_call" and item.get("name") == "late_tool"
    )

    assert len(additional_tool_indexes) == 1
    assert additional_tool_indexes[0] < late_call_index
    assert openai_tool_names(payload) == ["base_tool"]


async def test_falls_back_to_client_tool_search_when_additional_tools_is_unsupported():
    model = dataclasses.replace(
        get_builtin_model("openai", "gpt-5.4"),
        provider="openai-proxy",
        compat={"supportsAdditionalTools": False, "supportsToolSearch": True},
    )
    context = make_context([make_tool("base_tool"), make_tool("late_tool")])
    payload = await capture_payload(model, context)

    search_call = next(item for item in payload["input"] if item.get("type") == "tool_search_call")
    search_output = next(item for item in payload["input"] if item.get("type") == "tool_search_output")

    assert openai_tool_names(payload) == ["base_tool"]
    assert_matches(search_call, {"execution": "client", "status": "completed"})
    assert search_output["call_id"] == search_call["call_id"]
    assert_matches(search_output["tools"], [{"type": "function", "name": "late_tool", "defer_loading": True}])
    assert not any(item.get("type") == "additional_tools" for item in payload["input"])


@pytest.mark.parametrize("model_id", ["gpt-5.2", "gpt-5.4-nano", "gpt-5.5-pro"])
async def test_uses_the_normal_tool_list_for_unsupported_openai_model(model_id: str):
    context = make_context([make_tool("base_tool"), make_tool("late_tool")])
    payload = await capture_payload(get_builtin_model("openai", model_id), context)

    assert openai_tool_names(payload) == ["base_tool", "late_tool"]
    assert not any(item.get("type") == "tool_search_output" for item in payload["input"])


async def test_uses_the_normal_tool_list_when_openai_tool_search_is_explicitly_disabled():
    model = dataclasses.replace(
        get_builtin_model("openai", "gpt-5.4"),
        provider="openai-proxy",
        compat={"supportsToolSearch": False},
    )
    context = make_context([make_tool("base_tool"), make_tool("late_tool")])
    payload = await capture_payload(model, context)

    assert openai_tool_names(payload) == ["base_tool", "late_tool"]
    assert not any(item.get("type") == "tool_search_output" for item in payload["input"])


@pytest.mark.skip(
    reason="This port deliberately omits the `openai-codex-responses` provider "
    "(`pi_ai.api.openai_codex_responses.stream` raises NotImplementedError; see the README's "
    "list of omissions), so there is no Python code path to exercise."
)
async def test_selects_additional_tools_tool_search_or_top_level_tools_for_codex_models():
    """`it("selects additional tools, tool search, or top-level tools for Codex models")`.

    Asserts that `openai-codex/gpt-5.6-sol` defers via an `additional_tools`
    input item (tools list narrowed to `base_tool`, no `tool_search_output`),
    that `gpt-5.4` defers via `tool_search_output` instead, and that
    `gpt-5.3-codex-spark` uses neither and keeps both tools top-level.
    """


async def test_leaves_providers_without_deferred_loading_unchanged():
    context = make_context([make_tool("base_tool"), make_tool("late_tool")])
    payload = await capture_payload(get_builtin_model("groq", "llama-3.3-70b-versatile"), context)
    assert openai_tool_names(payload) == ["base_tool", "late_tool"]


def test_counts_definitions_marked_after_the_latest_usage_checkpoint():
    assistant = make_assistant_tool_call()
    assistant.content = [TextContent(text="done")]
    assistant.usage = Usage(input=50, output=50, total_tokens=100)
    assistant.stop_reason = "stop"

    plain = estimate_context_tokens(Context(messages=[assistant, make_user_message(4)], tools=[]))
    late_tool = make_tool("late_tool")
    late_tool.description = "x" * 4000
    marked = estimate_context_tokens(Context(messages=[assistant, make_tool_result(["late_tool"])], tools=[late_tool]))

    assert marked.tokens > plain.tokens + 500
    assert marked.trailing_tokens > plain.trailing_tokens + 500
