"""Python port of `packages/ai/test/openai-responses-message-id.test.ts`."""

from __future__ import annotations

from pi_ai.api.openai_responses_shared import convert_responses_messages
from pi_ai.providers.all import get_builtin_model
from pi_ai.types import AssistantMessage, Context, TextContent, ThinkingContent, UserMessage, now_ms

ALLOWED_TOOL_CALL_PROVIDERS = {"openai", "openai-codex", "opencode"}


def test_generates_unique_fallback_message_ids_for_multiple_text_blocks() -> None:
    model = get_builtin_model("openai-codex", "gpt-5.5")
    assert model is not None
    assistant = AssistantMessage(
        content=[
            ThinkingContent(thinking="private reasoning"),
            TextContent(text="visible answer"),
        ],
        api="anthropic-messages",
        provider="anthropic",
        model="claude-opus-4-8",
        stop_reason="stop",
        timestamp=now_ms() - 1000,
    )
    context = Context(
        system_prompt="You are concise.",
        messages=[UserMessage(content="hello", timestamp=now_ms() - 2000), assistant],
    )

    input_items = convert_responses_messages(model, context, ALLOWED_TOOL_CALL_PROVIDERS)
    message_ids = [
        item["id"] for item in input_items if item.get("type") == "message" and isinstance(item.get("id"), str)
    ]

    assert message_ids == ["msg_pi_1", "msg_pi_1_1"]
    assert len(set(message_ids)) == len(message_ids)
