"""Python port of `packages/ai/test/error-body.test.ts`.

Named with a `_ported` suffix because `tests/test_error_body.py` already exists
in this repo; this file is the port of the TypeScript test of the same name.

The TypeScript test synthesizes one error object per JavaScript SDK shape
(Mistral `statusCode`/`body`, `openai` `status`/`error`, `@google/genai`
`status` + body-in-message, AWS Bedrock `$metadata.httpStatusCode` /
`$response.statusCode` / `$response.body`). This port probes the equivalent
Python attribute names that `pi_ai.utils.error_body` documents:
`status_code`, `status`, `body`, `error`, and `response.status_code` /
`response.body`. The Bedrock `$metadata` / `$response` prefix has no Python
analogue, so those cases use a `response` object instead.
"""

from __future__ import annotations

import json
from typing import Any

from pi_ai.utils.error_body import (
    MAX_PROVIDER_ERROR_BODY_CHARS,
    format_provider_error,
    normalize_provider_error,
)


class SdkError(Exception):
    """Stand-in for `Object.assign(new Error(message), {...})`."""

    def __init__(self, message: str, **attrs: Any) -> None:
        super().__init__(message)
        for key, value in attrs.items():
            setattr(self, key, value)


class SdkResponse:
    def __init__(self, **attrs: Any) -> None:
        for key, value in attrs.items():
            setattr(self, key, value)


# --- normalize_provider_error ---------------------------------------------


def test_extracts_status_and_body_from_a_mistral_shaped_error():
    error = SdkError(
        "Mistral request failed",
        status_code=403,
        body='{"error":"blocked by gateway WAF"}',
    )

    norm = normalize_provider_error(error)

    assert norm.status == 403
    assert norm.body == '{"error":"blocked by gateway WAF"}'
    assert norm.message_carries_body is False


def test_reads_the_parsed_body_off_an_openai_api_error_when_the_message_is_opaque():
    error = SdkError("403 status code (no body)", status=403, error={"error": "blocked by gateway WAF"})

    norm = normalize_provider_error(error)

    assert norm.status == 403
    assert norm.body == '{"error": "blocked by gateway WAF"}'
    assert norm.message_carries_body is False


def test_preserves_the_message_when_the_sdk_already_folds_the_body_into_it():
    body = {"error": {"code": 403, "message": "Permission denied"}}
    error = SdkError(json.dumps(body), status=403)

    norm = normalize_provider_error(error)

    assert norm.status == 403
    assert norm.message_carries_body is True
    assert norm.message == json.dumps(body)


def test_extracts_status_and_body_from_a_bedrock_shaped_service_exception():
    error = SdkError(
        "UnknownError",
        response=SdkResponse(status_code=403, body='{"message":"blocked by gateway WAF"}'),
    )

    norm = normalize_provider_error(error)

    assert norm.status == 403
    assert norm.body == '{"message":"blocked by gateway WAF"}'
    assert norm.message_carries_body is False


def test_ignores_a_response_stream_instead_of_serializing_its_internals():
    class ResponseStream:
        def read(self) -> bytes:
            return b""

    error = SdkError(
        "Invocation of model ID anthropic.claude-opus-5 with on-demand throughput isn't supported.",
        response=SdkResponse(status_code=400, body=ResponseStream()),
    )

    norm = normalize_provider_error(error)

    assert norm.status == 400
    assert norm.body is None
    assert "on-demand throughput isn't supported" in norm.message
    assert norm.message_carries_body is True


def test_ignores_a_class_instance_response_body_instead_of_serializing_it():
    # Not every SDK response wrapper is a stream: SDK-specific wrapper classes
    # have no `read`, but serializing them still yields internals-noise that
    # would replace the real message.
    class SdkHttpResponseBody:
        def __init__(self) -> None:
            self.locked = False
            self.state: dict[str, Any] = {"storedError": None}

    error = SdkError(
        "Input is too long for requested model.",
        response=SdkResponse(status_code=400, body=SdkHttpResponseBody()),
    )

    norm = normalize_provider_error(error)

    assert norm.status == 400
    assert norm.body is None
    assert "Input is too long" in norm.message
    assert norm.message_carries_body is True


def test_ignores_a_class_instance_error_field_instead_of_serializing_it():
    class SdkInnerError:
        def __init__(self) -> None:
            self.code = "EPROTO"
            self.internal_state: dict[str, Any] = {}

    error = SdkError("TLS handshake failed", status=502, error=SdkInnerError())

    norm = normalize_provider_error(error)

    assert norm.body is None
    assert norm.message == "TLS handshake failed"
    assert norm.message_carries_body is True


def test_still_surfaces_a_plain_parsed_json_body_object():
    error = SdkError(
        "400 status code (no body)",
        status=400,
        error={"message": "schema validation failed", "field": "tools[0]"},
    )

    norm = normalize_provider_error(error)

    assert norm.body == '{"message": "schema validation failed", "field": "tools[0]"}'
    assert norm.message_carries_body is False


def test_json_stringifies_a_non_error_thrown_value():
    norm = normalize_provider_error({"reason": "boom"})

    assert norm.status is None
    assert norm.body is None
    assert norm.message == '{"reason": "boom"}'
    assert norm.message_carries_body is False


def test_treats_an_empty_parsed_body_object_as_no_body():
    error = SdkError("403 status code (no body)", status=403, error={})

    norm = normalize_provider_error(error)

    assert norm.body is None
    assert norm.message_carries_body is True


def test_truncates_the_body_at_the_cap():
    long_body = "x" * (MAX_PROVIDER_ERROR_BODY_CHARS + 50)
    error = SdkError("failed", status_code=500, body=long_body)

    norm = normalize_provider_error(error)

    assert norm.body is not None
    assert "... [truncated 50 chars]" in norm.body
    assert len(norm.body) < len(long_body)


def test_sets_message_carries_body_when_the_message_already_contains_the_body():
    error = SdkError("500: upstream exploded", status_code=500, body="upstream exploded")

    norm = normalize_provider_error(error)

    assert norm.message_carries_body is True


# --- format_provider_error -------------------------------------------------


def test_format_surfaces_status_and_body_without_a_prefix():
    norm = normalize_provider_error(
        SdkError("403 status code (no body)", status=403, error={"error": "blocked by gateway WAF"})
    )

    formatted = format_provider_error(norm)

    assert "403" in formatted
    assert "blocked by gateway WAF" in formatted
    assert formatted != "403 status code (no body)"


def test_format_applies_a_provider_prefix_with_status_and_body():
    norm = normalize_provider_error(
        SdkError("403 status code (no body)", status=403, error={"error": "blocked by gateway WAF"})
    )

    assert (
        format_provider_error(norm, "OpenAI API error") == 'OpenAI API error (403): {"error": "blocked by gateway WAF"}'
    )


def test_format_preserves_the_message_when_it_already_carries_the_body():
    body = json.dumps({"error": {"message": "Permission denied"}})
    norm = normalize_provider_error(SdkError(body, status=403))

    assert format_provider_error(norm, "OpenAI API error") == f"OpenAI API error (403): {body}"


def test_format_returns_the_bare_message_for_a_non_error_value():
    norm = normalize_provider_error({"reason": "boom"})

    assert format_provider_error(norm) == '{"reason": "boom"}'
