"""Redacted, structured diagnostics attached to assistant messages on failure.

Python port of `packages/ai/src/utils/diagnostics.ts`.

The TypeScript `AssistantMessageDiagnostic` carries a `type` tag, a `timestamp`,
an optional structured `error` (name/message/stack/code) and optional
free-form `details`. The already-ported `pi_ai.types.AssistantMessageDiagnostic`
instead has a flatter shape (`kind`, `message`, `detail`, `timestamp`) with no
dedicated `error` sub-object, so `create_assistant_message_diagnostic` folds
the extracted error's message (and its `code`, when present) into that shape
rather than reproducing the nested TypeScript structure.
"""

from __future__ import annotations

from typing import Any

from ..types import AssistantMessage, AssistantMessageDiagnostic, now_ms


def format_thrown_value(value: Any) -> str:
    if isinstance(value, BaseException):
        return str(value) or type(value).__name__
    if isinstance(value, str):
        return value
    return str(value)


def create_assistant_message_diagnostic(
    kind: str, error: Any, details: dict[str, Any] | None = None
) -> AssistantMessageDiagnostic:
    """Build a diagnostic entry from a thrown value, matching `createAssistantMessageDiagnostic`."""
    detail = dict(details) if details else {}
    code = getattr(error, "code", None)
    if code is not None and "code" not in detail:
        detail["code"] = code
    return AssistantMessageDiagnostic(
        kind=kind,
        message=format_thrown_value(error),
        detail=detail or None,
        timestamp=now_ms(),
    )


def append_assistant_message_diagnostic(message: AssistantMessage, diagnostic: AssistantMessageDiagnostic) -> None:
    message.diagnostics.append(diagnostic)
