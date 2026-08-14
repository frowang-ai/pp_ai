from pi_ai.types import AssistantMessage, Context, TextContent, Tool, ToolCall, ToolResultMessage, UserMessage
from pi_ai.utils.deferred_tools import split_deferred_tools


def _tool(name: str, description: str | None = None) -> Tool:
    return Tool(name=name, description=description or name)


def _assistant_tool_call(*names: str) -> AssistantMessage:
    return AssistantMessage(
        content=[ToolCall(id=f"call_{index}", name=name) for index, name in enumerate(names, start=1)],
        stop_reason="stop",
    )


def _tool_result(*added_tool_names: str) -> ToolResultMessage:
    return ToolResultMessage(added_tool_names=list(added_tool_names))


def test_split_deferred_tools_disabled_returns_deduplicated_immediate_tools():
    context = Context(
        tools=[
            _tool("read", "first definition"),
            _tool("Read", "canonical definition"),
            _tool("echo"),
        ]
    )

    split = split_deferred_tools(context, enabled=False, normalize_name=str.lower)

    assert [tool.name for tool in split.immediate] == ["Read", "echo"]
    assert [tool.description for tool in split.immediate] == ["canonical definition", "echo"]
    assert split.deferred == {}


def test_split_deferred_tools_enabled_uses_custom_name_normalization():
    context = Context(
        messages=[_tool_result("read")],
        tools=[_tool("Read", "canonical definition"), _tool("write")],
    )

    split = split_deferred_tools(context, enabled=True, normalize_name=str.lower)

    assert [tool.name for tool in split.immediate] == ["write"]
    assert list(split.deferred) == ["read"]
    assert split.deferred["read"].name == "Read"


def test_split_deferred_tools_keeps_used_tools_immediate_when_marker_comes_after_call():
    context = Context(
        messages=[
            _assistant_tool_call("late_tool"),
            _tool_result("late_tool"),
        ],
        tools=[_tool("base_tool"), _tool("late_tool")],
    )

    split = split_deferred_tools(context, enabled=True)

    assert [tool.name for tool in split.immediate] == ["base_tool", "late_tool"]
    assert split.deferred == {}


def test_split_deferred_tools_still_defers_tool_when_call_happens_after_marker():
    context = Context(
        messages=[
            _tool_result("late_tool"),
            _assistant_tool_call("late_tool"),
        ],
        tools=[_tool("base_tool"), _tool("late_tool")],
    )

    split = split_deferred_tools(context, enabled=True)

    assert [tool.name for tool in split.immediate] == ["base_tool"]
    assert list(split.deferred) == ["late_tool"]
    assert split.deferred["late_tool"].name == "late_tool"


def test_split_deferred_tools_collects_markers_from_multiple_tool_results():
    context = Context(
        messages=[
            _tool_result("late_tool"),
            _assistant_tool_call("base_tool"),
            _tool_result("later_tool"),
        ],
        tools=[_tool("base_tool"), _tool("late_tool"), _tool("later_tool")],
    )

    split = split_deferred_tools(context, enabled=True)

    assert [tool.name for tool in split.immediate] == ["base_tool"]
    assert list(split.deferred) == ["late_tool", "later_tool"]


def test_split_deferred_tools_duplicate_tool_names_use_last_definition_without_reordering():
    context = Context(
        tools=[
            _tool("dup", "first definition"),
            _tool("other", "other tool"),
            _tool("dup", "last definition"),
        ]
    )

    split = split_deferred_tools(context, enabled=False)

    assert [tool.name for tool in split.immediate] == ["dup", "other"]
    assert [tool.description for tool in split.immediate] == ["last definition", "other tool"]


def test_split_deferred_tools_ignores_markers_for_missing_tools():
    context = Context(
        messages=[_tool_result("missing_tool")],
        tools=[_tool("base_tool")],
    )

    split = split_deferred_tools(context, enabled=True)

    assert [tool.name for tool in split.immediate] == ["base_tool"]
    assert split.deferred == {}


def test_split_deferred_tools_ignores_non_tool_blocks_and_unrelated_messages():
    context = Context(
        messages=[
            AssistantMessage(content=[TextContent(text="note")], stop_reason="stop"),
            UserMessage(content="hello"),
            ToolResultMessage(),
        ],
        tools=[_tool("base_tool")],
    )

    split = split_deferred_tools(context, enabled=True)

    assert [tool.name for tool in split.immediate] == ["base_tool"]
    assert split.deferred == {}
