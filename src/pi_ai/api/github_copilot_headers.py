"""GitHub Copilot dynamic request headers.

Python port of `packages/ai/src/api/github-copilot-headers.ts`. Wired into
`openai_completions.py` and `anthropic_messages.py` when `model.provider ==
"github-copilot"`.
"""

from __future__ import annotations

from ..types import Message


def infer_copilot_initiator(messages: list[Message]) -> str:
    """Copilot expects `X-Initiator` to indicate whether the request is user-initiated
    or agent-initiated (e.g. follow-up after assistant/tool messages).
    """
    last = messages[-1] if messages else None
    return "agent" if last is not None and last.role != "user" else "user"


def has_copilot_vision_input(messages: list[Message]) -> bool:
    """Copilot requires the `Copilot-Vision-Request` header when sending images."""
    for msg in messages:
        if msg.role == "user" and not isinstance(msg.content, str) and any(c.type == "image" for c in msg.content):
            return True
        if msg.role == "toolResult" and any(c.type == "image" for c in msg.content):
            return True
    return False


def build_copilot_dynamic_headers(messages: list[Message], has_images: bool) -> dict[str, str]:
    headers = {
        "X-Initiator": infer_copilot_initiator(messages),
        "Openai-Intent": "conversation-edits",
    }
    if has_images:
        headers["Copilot-Vision-Request"] = "true"
    return headers
