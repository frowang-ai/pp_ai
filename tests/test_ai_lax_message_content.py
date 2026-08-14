"""Python port of `packages/ai/test/lax-message-content.test.ts`.

The Message types require `content` to always be present, but untyped callers
(custom tools, hand-built histories, old session files) can violate that
contract. `transform_messages` is the choke point before every provider request
and is intentionally lax: it normalizes null/missing content to an empty list
(issues #6259, #6276).
"""

from __future__ import annotations

from pi_ai.api.transform_messages import transform_messages
from pi_ai.types import (
    AssistantMessage,
    Message,
    Model,
    ModelCost,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def make_text_only_model() -> Model:
    """Text-only model so the image downgrade path runs, which was the primary
    crash site for null tool result content."""
    return Model(
        id="test-model",
        name="Test Model",
        api="openai-completions",
        provider="openai",
        base_url="https://example.invalid/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128000,
        max_tokens=16000,
    )


def test_normalizes_null_missing_content_to_an_empty_list_instead_of_crashing():
    user = UserMessage(content="placeholder")
    user.content = None  # type: ignore[assignment]
    assistant = AssistantMessage(
        content=[],
        api="openai-completions",
        provider="openai",
        model="test-model",
        usage=Usage(),
        stop_reason="stop",
    )
    assistant.content = None  # type: ignore[assignment]
    tool_result = ToolResultMessage(tool_call_id="call_1", tool_name="web_search", is_error=False)
    tool_result.content = None  # type: ignore[assignment]

    messages: list[Message] = [user, assistant, tool_result]
    result = transform_messages(messages, make_text_only_model())

    assert len(result) == 3
    for msg in result:
        assert msg.content == []
