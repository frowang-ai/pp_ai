"""Python port of `packages/ai/test/openai-responses-foreign-toolcall-id.test.ts`."""

from __future__ import annotations

import re

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
from pi_ai.utils.hash import short_hash

COPILOT_RAW_TOOL_CALL_ID = (
    "call_4VnzVawQXPB9MgYib7CiQFEY|I9b95oN1wD/cHXKTw3PpRkL6KkCtzTJhUxMouMWYwHeTo2j3htzfSk7YPx2vifiIM4g3A8XXyOj8q4Bt"
    "6SLUG7gqY1E3ELkrkVQNHglRfUmWj84lqxJY+Puieb3VKyX0FB+83TUzn91cDMF/4gzt990IzqVrc+nIb9RRscRD070Du16q1glydVjWR0SBJs"
    "E6TbY/esOjFpqplogQqrajm1eI++f3eLi73R6q7hVusY0QbeFySVxABCjhN0lXB04caBe1rzHjYzul6MAXj7uq+0r17VLq+yrtyYhN12wkmFqH"
    "eqTyEei6EFPbMy24Nc+IbJlkP0OCg02W+gOnyBFcbi2ctvJFSOhSjt1CqBdqCnnhwUqXjbWiT0wh3DmLScRgTHmGkaI+oAcQQjfic65nxj+TnE"
    "kReA=="
)

ALLOWED_TOOL_CALL_PROVIDERS = {"openai", "openai-codex", "opencode"}


def test_hashes_foreign_copilot_tool_item_ids_into_a_bounded_codex_safe_shape() -> None:
    model = get_builtin_model("openai-codex", "gpt-5.5")
    assert model is not None
    assistant = AssistantMessage(
        content=[
            ToolCall(
                id=COPILOT_RAW_TOOL_CALL_ID,
                name="edit",
                arguments={"path": "src/styles/app.css"},
            )
        ],
        api="openai-responses",
        provider="github-copilot",
        model="gpt-5.5",
        stop_reason="toolUse",
        timestamp=now_ms() - 2000,
    )
    tool_result = ToolResultMessage(
        tool_call_id=COPILOT_RAW_TOOL_CALL_ID,
        tool_name="edit",
        content=[TextContent(text="ok")],
        is_error=False,
        timestamp=now_ms() - 1000,
    )
    context = Context(
        system_prompt="You are concise.",
        messages=[
            UserMessage(content="Use the tool.", timestamp=now_ms() - 3000),
            assistant,
            tool_result,
        ],
    )

    input_items = convert_responses_messages(model, context, ALLOWED_TOOL_CALL_PROVIDERS)
    function_call = next((item for item in input_items if item.get("type") == "function_call"), None)

    assert function_call is not None
    expected_item_id = f"fc_{short_hash(COPILOT_RAW_TOOL_CALL_ID.split('|', 1)[1])}"
    assert function_call["id"] == expected_item_id
    assert len(function_call["id"]) <= 64
    assert re.match(r"^fc_[A-Za-z0-9]+$", function_call["id"])
