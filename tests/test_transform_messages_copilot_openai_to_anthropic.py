"""Python port of `packages/ai/test/transform-messages-copilot-openai-to-anthropic.test.ts`."""

from __future__ import annotations

import json
import re
import time

from pi_ai.api.transform_messages import transform_messages
from pi_ai.types import (
    AssistantMessage,
    Content,
    Cost,
    Message,
    Model,
    ModelCost,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)

EMPTY_USAGE = Usage(
    input=0,
    output=0,
    cache_read=0,
    cache_write=0,
    total_tokens=0,
    cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
)


def anthropic_normalize_tool_call_id(tool_call_id: str, _model: Model, _source: AssistantMessage) -> str:
    """Normalize function matching what anthropic.py uses."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", tool_call_id)[:64]


def make_copilot_claude_model() -> Model:
    return Model(
        id="claude-sonnet-4.6",
        name="Claude Sonnet 4.6",
        api="anthropic-messages",
        provider="github-copilot",
        base_url="https://api.individual.githubcopilot.com",
        reasoning=True,
        input=["text", "image"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128000,
        max_tokens=16000,
    )


def make_assistant_message(content: list[Content]) -> AssistantMessage:
    return AssistantMessage(
        content=content,
        api="openai-responses",
        provider="github-copilot",
        model="gpt-5",
        usage=EMPTY_USAGE,
        stop_reason="toolUse",
        timestamp=time.time() * 1000,
    )


def test_converts_thinking_blocks_to_plain_text_when_source_model_differs() -> None:
    model = make_copilot_claude_model()
    messages: list[Message] = [
        UserMessage(content="hello", timestamp=time.time() * 1000),
        AssistantMessage(
            content=[
                ThinkingContent(thinking="Let me think about this...", thinking_signature="reasoning_content"),
                TextContent(text="Hi there!"),
            ],
            api="openai-completions",
            provider="github-copilot",
            model="gpt-4o",
            usage=EMPTY_USAGE,
            stop_reason="stop",
            timestamp=time.time() * 1000,
        ),
    ]

    result = transform_messages(messages, model, anthropic_normalize_tool_call_id)
    assistant_msg = next(m for m in result if m.role == "assistant")

    text_blocks = [block for block in assistant_msg.content if block.type == "text"]
    thinking_blocks = [block for block in assistant_msg.content if block.type == "thinking"]
    assert len(thinking_blocks) == 0
    assert len(text_blocks) >= 2


def test_removes_thought_signature_from_tool_calls_when_migrating_between_models() -> None:
    model = make_copilot_claude_model()
    messages: list[Message] = [
        UserMessage(content="run a command", timestamp=time.time() * 1000),
        AssistantMessage(
            content=[
                ToolCall(
                    id="call_123",
                    name="bash",
                    arguments={"command": "ls"},
                    thought_signature=json.dumps(
                        {"type": "reasoning.encrypted", "id": "call_123", "data": "encrypted"}
                    ),
                )
            ],
            api="openai-responses",
            provider="github-copilot",
            model="gpt-5",
            usage=EMPTY_USAGE,
            stop_reason="toolUse",
            timestamp=time.time() * 1000,
        ),
        ToolResultMessage(
            tool_call_id="call_123",
            tool_name="bash",
            content=[TextContent(text="output")],
            is_error=False,
            timestamp=time.time() * 1000,
        ),
    ]

    result = transform_messages(messages, model, anthropic_normalize_tool_call_id)
    assistant_msg = next(m for m in result if m.role == "assistant")
    tool_call = next(block for block in assistant_msg.content if block.type == "toolCall")

    assert tool_call.thought_signature is None


def test_adds_synthetic_tool_results_for_trailing_orphaned_tool_calls() -> None:
    model = make_copilot_claude_model()
    messages: list[Message] = [
        UserMessage(content="read the file", timestamp=time.time() * 1000),
        make_assistant_message([ToolCall(id="call_123|fc_123", name="read", arguments={"path": "README.md"})]),
    ]

    result = transform_messages(messages, model, anthropic_normalize_tool_call_id)
    last_message = result[-1]

    assert last_message.role == "toolResult"
    assert last_message.tool_call_id == "call_123_fc_123"
    assert last_message.tool_name == "read"
    assert last_message.is_error is True
    assert last_message.content == [TextContent(text="No result provided")]


def test_adds_synthetic_results_only_for_trailing_tool_calls_still_missing_results() -> None:
    model = make_copilot_claude_model()
    messages: list[Message] = [
        UserMessage(content="run commands", timestamp=time.time() * 1000),
        make_assistant_message(
            [
                ToolCall(id="call_1|fc_1", name="read", arguments={"path": "README.md"}),
                ToolCall(id="call_2|fc_2", name="bash", arguments={"command": "pwd"}),
            ]
        ),
        ToolResultMessage(
            tool_call_id="call_1|fc_1",
            tool_name="read",
            content=[TextContent(text="done")],
            is_error=False,
            timestamp=time.time() * 1000,
        ),
    ]

    result = transform_messages(messages, model, anthropic_normalize_tool_call_id)
    synthetic_results = [m for m in result if m.role == "toolResult" and m.is_error]

    assert len(synthetic_results) == 1
    assert synthetic_results[0].tool_call_id == "call_2_fc_2"
    assert synthetic_results[0].tool_name == "bash"
    assert synthetic_results[0].content == [TextContent(text="No result provided")]
