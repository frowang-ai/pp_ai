"""Text helpers.

Python port of `packages/ai/src/utils/text.ts`.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..types import Content


def content_text(content: str | Sequence[Content], separator: str = "\n") -> str:
    """Extract and join the text blocks of a message content value."""
    if isinstance(content, str):
        return content
    return separator.join(block.text for block in content if block.type == "text")
