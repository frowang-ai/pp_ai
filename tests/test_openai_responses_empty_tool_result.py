"""Python port of `packages/ai/test/openai-responses-empty-tool-result.test.ts`."""

from __future__ import annotations

from pi_ai.api.openai_responses_shared import convert_responses_messages
from pi_ai.providers.all import get_builtin_model
from pi_ai.types import (
    AssistantMessage,
    Context,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    now_ms,
)

ALLOWED_TOOL_CALL_PROVIDERS = {"openai", "openai-codex", "opencode"}


def build_empty_tool_result(tool_call_id: str, timestamp: int) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=tool_call_id,
        tool_name="bash",
        content=[TextContent(text="")],
        is_error=False,
        timestamp=timestamp,
    )


def test_uses_no_tool_output_placeholder_for_empty_tool_results_without_images() -> None:
    model = get_builtin_model("openai", "gpt-4o-mini")
    assert model is not None
    now = now_ms()
    assistant = AssistantMessage(
        content=[ToolCall(id="tool-1", name="bash", arguments={"command": "true"})],
        api=model.api,
        provider=model.provider,
        model=model.id,
        stop_reason="toolUse",
        timestamp=now,
    )
    context = Context(
        messages=[
            UserMessage(content="Run the command", timestamp=now - 1),
            assistant,
            build_empty_tool_result("tool-1", now + 1),
        ]
    )

    input_items = convert_responses_messages(model, context, ALLOWED_TOOL_CALL_PROVIDERS)
    function_call_output = next(
        (item for item in input_items if item.get("type") == "function_call_output"),
        None,
    )

    assert function_call_output is not None
    assert function_call_output["output"] == "(no tool output)"
    assert "see attached image" not in function_call_output["output"]
