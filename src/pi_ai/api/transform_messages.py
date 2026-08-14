"""Cross-provider message normalization.

Python port of `packages/ai/src/api/transform-messages.ts`. Runs before every
provider request: it downgrades images for text-only models, drops or rewrites
assistant content that a different model cannot replay, and synthesizes tool
results for orphaned tool calls so the wire format stays valid.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence

from ..types import (
    AssistantMessage,
    Message,
    Model,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserContent,
    now_ms,
)

NON_VISION_USER_IMAGE_PLACEHOLDER = "(image omitted: model does not support images)"
NON_VISION_TOOL_IMAGE_PLACEHOLDER = "(tool image omitted: model does not support images)"

NormalizeToolCallId = Callable[[str, Model, AssistantMessage], str]


def _replace_images_with_placeholder(content: Sequence[UserContent], placeholder: str) -> list[TextContent]:
    result: list[TextContent] = []
    previous_was_placeholder = False

    for block in content:
        if block.type == "image":
            if not previous_was_placeholder:
                result.append(TextContent(text=placeholder))
            previous_was_placeholder = True
            continue

        result.append(block)
        previous_was_placeholder = block.text == placeholder

    return result


def _downgrade_unsupported_images(messages: list[Message], model: Model) -> list[Message]:
    if "image" in model.input:
        return messages

    downgraded: list[Message] = []
    for msg in messages:
        if msg.role == "user" and not isinstance(msg.content, str):
            replacement = copy.copy(msg)
            replacement.content = _replace_images_with_placeholder(msg.content, NON_VISION_USER_IMAGE_PLACEHOLDER)
            downgraded.append(replacement)
        elif msg.role == "toolResult":
            replacement = copy.copy(msg)
            replacement.content = _replace_images_with_placeholder(msg.content, NON_VISION_TOOL_IMAGE_PLACEHOLDER)
            downgraded.append(replacement)
        else:
            downgraded.append(msg)
    return downgraded


def transform_messages(
    messages: Sequence[Message],
    model: Model,
    normalize_tool_call_id: NormalizeToolCallId | None = None,
) -> list[Message]:
    """Normalize a message history for ``model``.

    ``normalize_tool_call_id`` rewrites tool call ids for providers with
    stricter id rules; the same mapping is applied to the matching tool results.
    """
    tool_call_id_map: dict[str, str] = {}

    normalized_messages: list[Message] = []
    for msg in messages:
        if getattr(msg, "content", ()) is None:
            replacement = copy.copy(msg)
            replacement.content = []
            normalized_messages.append(replacement)
        else:
            normalized_messages.append(msg)

    image_aware_messages = _downgrade_unsupported_images(normalized_messages, model)

    transformed: list[Message] = []
    for msg in image_aware_messages:
        if msg.role == "user":
            transformed.append(msg)
            continue

        if msg.role == "toolResult":
            normalized_id = tool_call_id_map.get(msg.tool_call_id)
            if normalized_id and normalized_id != msg.tool_call_id:
                replacement = copy.copy(msg)
                replacement.tool_call_id = normalized_id
                transformed.append(replacement)
            else:
                transformed.append(msg)
            continue

        assistant = msg
        is_same_model = (
            assistant.provider == model.provider and assistant.api == model.api and assistant.model == model.id
        )

        transformed_content: list[object] = []
        for block in assistant.content:
            if block.type == "thinking":
                # Redacted thinking is opaque encrypted content, only valid for the
                # same model. Dropping it cross-model avoids API errors.
                if block.redacted:
                    if is_same_model:
                        transformed_content.append(block)
                    continue
                # Same model: keep signed thinking blocks even when the text is
                # empty (OpenAI encrypted reasoning) because replay needs them.
                if is_same_model and block.thinking_signature:
                    transformed_content.append(block)
                    continue
                if not block.thinking or not block.thinking.strip():
                    continue
                if is_same_model:
                    transformed_content.append(block)
                else:
                    transformed_content.append(TextContent(text=block.thinking))
                continue

            if block.type == "text":
                transformed_content.append(block if is_same_model else TextContent(text=block.text))
                continue

            if block.type == "toolCall":
                tool_call = block
                if not is_same_model and tool_call.thought_signature:
                    tool_call = copy.copy(tool_call)
                    tool_call.thought_signature = None

                if not is_same_model and normalize_tool_call_id is not None:
                    normalized_id = normalize_tool_call_id(block.id, model, assistant)
                    if normalized_id != block.id:
                        tool_call_id_map[block.id] = normalized_id
                        tool_call = copy.copy(tool_call)
                        tool_call.id = normalized_id

                transformed_content.append(tool_call)
                continue

            transformed_content.append(block)

        replacement = copy.copy(assistant)
        replacement.content = transformed_content  # type: ignore[assignment]
        transformed.append(replacement)

    # Second pass: insert synthetic tool results for orphaned tool calls. This
    # preserves thinking signatures and satisfies provider requirements.
    result: list[Message] = []
    pending_tool_calls: list[ToolCall] = []
    existing_tool_result_ids: set[str] = set()

    def insert_synthetic_tool_results() -> None:
        nonlocal pending_tool_calls, existing_tool_result_ids
        if not pending_tool_calls:
            return
        for tool_call in pending_tool_calls:
            if tool_call.id not in existing_tool_result_ids:
                result.append(
                    ToolResultMessage(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        content=[TextContent(text="No result provided")],
                        is_error=True,
                        timestamp=now_ms(),
                    )
                )
        pending_tool_calls = []
        existing_tool_result_ids = set()

    for msg in transformed:
        if msg.role == "assistant":
            insert_synthetic_tool_results()

            # Skip errored/aborted assistant turns entirely: they may carry
            # partial content whose replay makes providers reject the request.
            if msg.stop_reason in ("error", "aborted"):
                continue

            tool_calls = [block for block in msg.content if block.type == "toolCall"]
            if tool_calls:
                pending_tool_calls = tool_calls
                existing_tool_result_ids = set()

            result.append(msg)
        elif msg.role == "toolResult":
            existing_tool_result_ids.add(msg.tool_call_id)
            result.append(msg)
        elif msg.role == "user":
            # A user message interrupts the tool flow.
            insert_synthetic_tool_results()
            result.append(msg)
        else:
            result.append(msg)

    insert_synthetic_tool_results()

    return result
