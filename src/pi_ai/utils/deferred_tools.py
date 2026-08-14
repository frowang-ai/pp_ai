"""Split a tool set into immediately-visible and deferred (transcript-loaded) tools.

Python port of `packages/ai/src/utils/deferred-tools.ts`. Providers that support
tool search/reference (currently Anthropic) can omit a tool's full definition
from the request once its name has already appeared in the conversation
transcript via a `toolResult.added_tool_names` entry, deferring it to load
on demand through a `tool_reference` block instead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..types import Context, Tool

ToolNameNormalizer = Callable[[str], str]


def _identity_tool_name(name: str) -> str:
    return name


@dataclass
class DeferredToolSplit:
    """Result of :func:`split_deferred_tools`."""

    immediate: list[Tool] = field(default_factory=list)
    deferred: dict[str, Tool] = field(default_factory=dict)


def split_deferred_tools(
    context: Context,
    enabled: bool,
    normalize_name: ToolNameNormalizer = _identity_tool_name,
) -> DeferredToolSplit:
    """Split current tools into prefix and transcript-loaded definitions."""
    unique_tools: dict[str, Tool] = {}
    for tool in context.tools or []:
        unique_tools[normalize_name(tool.name)] = tool
    if not enabled:
        return DeferredToolSplit(immediate=list(unique_tools.values()), deferred={})

    deferred_names: set[str] = set()
    used_names: set[str] = set()
    for message in context.messages:
        if message.role == "assistant":
            for block in message.content:
                if block.type == "toolCall":
                    used_names.add(normalize_name(block.name))
        elif message.role == "toolResult":
            for name in message.added_tool_names or []:
                normalized_name = normalize_name(name)
                if normalized_name not in used_names:
                    deferred_names.add(normalized_name)

    immediate: list[Tool] = []
    deferred: dict[str, Tool] = {}
    for name, tool in unique_tools.items():
        if name in deferred_names:
            deferred[name] = tool
        else:
            immediate.append(tool)
    return DeferredToolSplit(immediate=immediate, deferred=deferred)
