"""Tests for `pi_ai.utils.diagnostics` — construction/redaction helpers for
`AssistantMessageDiagnostic`.

Ported concept from `packages/ai/src/utils/diagnostics.ts`. The
`AssistantMessageDiagnostic` type itself was already ported onto
`pi_ai.types.AssistantMessageDiagnostic` before this session; this test
covers the construction helpers in `pi_ai.utils.diagnostics`.
"""

from __future__ import annotations

from pi_ai.types import AssistantMessage, Cost, Usage
from pi_ai.utils.diagnostics import (
    append_assistant_message_diagnostic,
    create_assistant_message_diagnostic,
    format_thrown_value,
)


def test_format_thrown_value_uses_the_exception_message() -> None:
    assert format_thrown_value(ValueError("boom")) == "boom"


def test_format_thrown_value_falls_back_to_the_type_name_for_empty_messages() -> None:
    assert format_thrown_value(ValueError()) == "ValueError"


def test_format_thrown_value_passes_through_strings() -> None:
    assert format_thrown_value("already a string") == "already a string"


def test_format_thrown_value_stringifies_other_values() -> None:
    assert format_thrown_value(404) == "404"


def test_create_assistant_message_diagnostic_captures_kind_and_message() -> None:
    diagnostic = create_assistant_message_diagnostic("stream-error", ValueError("network down"))

    assert diagnostic.kind == "stream-error"
    assert diagnostic.message == "network down"
    assert diagnostic.timestamp > 0


def test_create_assistant_message_diagnostic_folds_error_code_into_detail() -> None:
    class CodedError(Exception):
        code = "ECONNRESET"

    diagnostic = create_assistant_message_diagnostic("stream-error", CodedError("reset"))

    assert diagnostic.detail == {"code": "ECONNRESET"}


def test_create_assistant_message_diagnostic_merges_explicit_details_with_error_code() -> None:
    class CodedError(Exception):
        code = "ETIMEDOUT"

    diagnostic = create_assistant_message_diagnostic("stream-error", CodedError("timeout"), {"attempt": 2})

    assert diagnostic.detail == {"attempt": 2, "code": "ETIMEDOUT"}


def test_create_assistant_message_diagnostic_without_details_or_code_has_none_detail() -> None:
    diagnostic = create_assistant_message_diagnostic("stream-error", ValueError("plain"))

    assert diagnostic.detail is None


def test_append_assistant_message_diagnostic_appends_to_the_message() -> None:
    message = AssistantMessage(api="test-api", provider="test", model="m1", usage=Usage(cost=Cost()))
    diagnostic = create_assistant_message_diagnostic("stream-error", ValueError("boom"))

    append_assistant_message_diagnostic(message, diagnostic)

    assert message.diagnostics == [diagnostic]
