"""Python port of `packages/ai/test/openai-completions-tool-result-images.test.ts`."""

from __future__ import annotations

import dataclasses
import time
from typing import Any

from pi_ai import (
    AssistantMessage,
    Context,
    Cost,
    ImageContent,
    Model,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pi_ai.api.openai_completions import ResolvedCompat, convert_messages
from pi_ai.providers.all import get_builtin_model

EMPTY_USAGE = Usage(
    input=0,
    output=0,
    cache_read=0,
    cache_write=0,
    total_tokens=0,
    cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
)

# TS spells out every `Required<OpenAICompletionsCompat>` field; the port's
# `ResolvedCompat` already defaults to the same values, so only the fields that
# differ from the dataclass defaults are set here.
COMPAT = ResolvedCompat(cache_control_format="anthropic")


def openai_completions_model() -> Model:
    base = get_builtin_model("openai", "gpt-4o-mini")
    assert base is not None
    return dataclasses.replace(base, api="openai-completions", input=["text", "image"], compat={})


def build_tool_result(tool_call_id: str, timestamp: float) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=tool_call_id,
        tool_name="read",
        content=[
            TextContent(text="Read image file [image/png]"),
            ImageContent(data="ZmFrZQ==", mime_type="image/png"),
        ],
        is_error=False,
        timestamp=timestamp,
    )


def build_empty_tool_result(tool_call_id: str, timestamp: float) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=tool_call_id,
        tool_name="bash",
        content=[TextContent(text="")],
        is_error=False,
        timestamp=timestamp,
    )


def test_batches_tool_result_images_after_consecutive_tool_results() -> None:
    model = openai_completions_model()
    now = time.time() * 1000

    assistant_message = AssistantMessage(
        content=[
            ToolCall(id="tool-1", name="read", arguments={"path": "img-1.png"}),
            ToolCall(id="tool-2", name="read", arguments={"path": "img-2.png"}),
        ],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=EMPTY_USAGE,
        stop_reason="toolUse",
        timestamp=now,
    )

    context = Context(
        messages=[
            UserMessage(content="Read the images", timestamp=now - 2),
            assistant_message,
            build_tool_result("tool-1", now + 1),
            build_tool_result("tool-2", now + 2),
        ]
    )

    messages = convert_messages(model, context, COMPAT)
    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "tool", "user"]

    image_message = messages[-1]
    assert image_message["role"] == "user"
    assert isinstance(image_message["content"], list)

    content: list[dict[str, Any]] = image_message["content"]
    image_parts = [part for part in content if part.get("type") == "image_url"]
    assert len(image_parts) == 2


def test_uses_no_tool_output_placeholder_for_empty_tool_results_without_images() -> None:
    model = openai_completions_model()
    now = time.time() * 1000

    assistant_message = AssistantMessage(
        content=[ToolCall(id="tool-1", name="bash", arguments={"command": "true"})],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=EMPTY_USAGE,
        stop_reason="toolUse",
        timestamp=now,
    )

    context = Context(
        messages=[
            UserMessage(content="Run the command", timestamp=now - 1),
            assistant_message,
            build_empty_tool_result("tool-1", now + 1),
        ]
    )

    messages = convert_messages(model, context, COMPAT)
    tool_message = next((m for m in messages if m["role"] == "tool"), None)
    assert tool_message is not None
    assert tool_message["content"] == "(no tool output)"
    assert "see attached image" not in tool_message["content"]
