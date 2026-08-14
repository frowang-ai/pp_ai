"""Context token estimation.

Python port of `packages/ai/src/utils/estimate.ts`. Token counts here are a
cheap heuristic (characters / 4, rounded up), not a real tokenizer; they exist
to keep context usage roughly tracked without depending on a per-model tokenizer.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..types import (
    AssistantMessage,
    Context,
    ImageContent,
    Message,
    TextContent,
    Tool,
    Usage,
)

CHARS_PER_TOKEN = 4
ESTIMATED_IMAGE_CHARS = 4800


@dataclass
class ContextUsageEstimate:
    """Result of :func:`estimate_context_tokens`."""

    tokens: int
    """Estimated total context tokens."""
    usage_tokens: int
    """Tokens reported by the most recent applicable assistant usage block."""
    trailing_tokens: int
    """Estimated tokens after the most recent applicable assistant usage block."""
    last_usage_index: int | None
    """Index of the applicable message that provided usage, or None when none exists."""


def calculate_context_tokens(usage: Usage) -> int:
    return usage.total_tokens or usage.input + usage.output + usage.cache_read + usage.cache_write


def _json_default(value: Any) -> Any:
    """`json.dumps` default hook: expand dataclass instances via `dataclasses.asdict`.

    `dataclasses.asdict` already recurses into nested dataclasses/lists/dicts,
    so this makes `json.dumps` able to serialize e.g. a `list[Tool]` the same
    way `JSON.stringify` walks arbitrary nested object properties.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _safe_json_stringify(value: Any) -> str:
    try:
        return json.dumps(value, separators=(",", ":"), default=_json_default)
    except (TypeError, ValueError):
        return "[unserializable]"


def _estimate_text_and_image_content_chars(content: str | Sequence[TextContent | ImageContent]) -> int:
    if isinstance(content, str):
        return len(content)

    chars = 0
    for block in content:
        chars += len(block.text) if block.type == "text" else ESTIMATED_IMAGE_CHARS
    return chars


def estimate_text_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def estimate_text_and_image_content_tokens(content: str | Sequence[TextContent | ImageContent]) -> int:
    return math.ceil(_estimate_text_and_image_content_chars(content) / CHARS_PER_TOKEN)


def estimate_message_tokens(message: Message) -> int:
    chars = 0

    if message.role == "user" or message.role == "toolResult":
        return estimate_text_and_image_content_tokens(message.content)

    for block in message.content:
        if block.type == "text":
            chars += len(block.text)
        elif block.type == "thinking":
            chars += len(block.thinking)
        else:
            chars += len(block.name) + len(_safe_json_stringify(block.arguments))
    return math.ceil(chars / CHARS_PER_TOKEN)


def _get_last_assistant_usage_info(messages: Sequence[Message]) -> tuple[Usage, int] | None:
    latest_prefix_timestamp = float("-inf")
    usage_info: tuple[Usage, int] | None = None

    for index, message in enumerate(messages):
        if isinstance(message, AssistantMessage):
            # A newer prefix message was inserted after this response (for example, a
            # compaction summary), so its usage cannot describe the current prefix.
            usage_applies_to_prefix = message.timestamp >= latest_prefix_timestamp
            if (
                usage_applies_to_prefix
                and message.stop_reason != "aborted"
                and message.stop_reason != "error"
                and calculate_context_tokens(message.usage) > 0
            ):
                usage_info = (message.usage, index)
        latest_prefix_timestamp = max(latest_prefix_timestamp, message.timestamp)

    return usage_info


def _estimate_messages(messages: Sequence[Message]) -> ContextUsageEstimate:
    usage_info = _get_last_assistant_usage_info(messages)
    if usage_info is not None:
        usage, index = usage_info
        usage_tokens = calculate_context_tokens(usage)
        trailing_tokens = sum(estimate_message_tokens(m) for m in messages[index + 1 :])
        return ContextUsageEstimate(
            tokens=usage_tokens + trailing_tokens,
            usage_tokens=usage_tokens,
            trailing_tokens=trailing_tokens,
            last_usage_index=index,
        )

    tokens = sum(estimate_message_tokens(m) for m in messages)
    return ContextUsageEstimate(tokens=tokens, usage_tokens=0, trailing_tokens=tokens, last_usage_index=None)


def _estimate_tools_tokens(tools: Sequence[Tool] | None) -> int:
    if not tools:
        return 0
    return estimate_text_tokens(_safe_json_stringify(tools))


def estimate_context_tokens(context: Context | Sequence[Message]) -> ContextUsageEstimate:
    if not isinstance(context, Context):
        return _estimate_messages(context)

    estimate = _estimate_messages(context.messages)
    if estimate.last_usage_index is not None:
        added_names: set[str] = set()
        for message in context.messages[estimate.last_usage_index + 1 :]:
            if message.role == "toolResult":
                added_names.update(message.added_tool_names or [])
        added_tool_tokens = _estimate_tools_tokens([tool for tool in (context.tools or []) if tool.name in added_names])
        return ContextUsageEstimate(
            tokens=estimate.tokens + added_tool_tokens,
            usage_tokens=estimate.usage_tokens,
            trailing_tokens=estimate.trailing_tokens + added_tool_tokens,
            last_usage_index=estimate.last_usage_index,
        )

    prefix_tokens = (
        estimate_text_tokens(context.system_prompt) if context.system_prompt else 0
    ) + _estimate_tools_tokens(context.tools)

    return ContextUsageEstimate(
        tokens=estimate.tokens + prefix_tokens,
        usage_tokens=estimate.usage_tokens,
        trailing_tokens=estimate.trailing_tokens + prefix_tokens,
        last_usage_index=estimate.last_usage_index,
    )
