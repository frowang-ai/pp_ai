"""Shared normalization for provider HTTP error objects.

Python port of `packages/ai/src/utils/error-body.ts`.

Endpoints behind a proxy/gateway may return a non-2xx response whose body the
provider SDK cannot fold into the exception message. The SDK exception object
still carries the HTTP status and the raw/parsed body, but under SDK-specific
attribute names. Provider except blocks that read only `str(error)` therefore
drop the body and surface opaque messages like `"403 status code (no body)"`.

`normalize_provider_error` probes the known SDK attribute shapes and returns a
struct each provider composes into its display string. The
`message_carries_body` flag captures the case where the SDK already folded the
body into the message, so providers can preserve it without double-printing.

The TypeScript SDK field names (`statusCode`, `$metadata.httpStatusCode`,
`$response.statusCode`, ...) come from the Mistral, `openai`, `@google/genai`,
and AWS Bedrock JS SDKs. This port instead probes attribute names used by
their closest Python equivalents plus `httpx.HTTPStatusError` (`response.
status_code`, `response.text`), since pi_ai's provider integrations are HTTP
based: `status_code`, `status`, `body`, `error`, and `response`
(`response.status_code`, `response.body`, `response.text`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

MAX_PROVIDER_ERROR_BODY_CHARS = 4000


@dataclass
class NormalizedProviderError:
    """Normalized shape of a provider HTTP error."""

    message: str
    """`str(error)`, or `safe_json_stringify(error)` for a non-exception throw."""
    status: int | None = None
    """HTTP status code, when one could be extracted from the SDK error object."""
    body: str | None = None
    """Raw HTTP body reason, already trimmed and truncated to the cap."""
    message_carries_body: bool = False
    """True when `message` already contains the body (no separate body to add)."""


def normalize_provider_error(error: object) -> NormalizedProviderError:
    if not isinstance(error, BaseException):
        return NormalizedProviderError(message=safe_json_stringify(error), message_carries_body=False)

    status = _extract_status(error)
    body = _extract_body(error)
    message = str(error)
    message_carries_body = body is None or body in message

    return NormalizedProviderError(
        status=status,
        body=body,
        message=message,
        message_carries_body=message_carries_body,
    )


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _extract_status(error: BaseException) -> int | None:
    """Probe the HTTP status, first numeric hit wins.

    Order: `status_code` (Mistral-style) -> `status` (`openai`, `@google/genai`
    style) -> `response.status_code` (`httpx.HTTPStatusError` and Bedrock-style
    `$response.statusCode`).
    """
    status = _as_int(getattr(error, "status_code", None))
    if status is not None:
        return status
    status = _as_int(getattr(error, "status", None))
    if status is not None:
        return status
    response = getattr(error, "response", None)
    if response is not None:
        status = _as_int(getattr(response, "status_code", None))
        if status is not None:
            return status
    return None


def _extract_body(error: BaseException) -> str | None:
    """Probe the raw body reason, first usable hit wins.

    Order: `body` string -> `error` parsed JSON body object (`openai` SDK's
    `this.error`) -> `response.body`/`response.text` (`httpx` and Bedrock
    style). Empty objects and unread response streams are treated as no body
    so they do not surface as `"{}"` or serialized stream internals. The
    chosen body is truncated to the cap.
    """
    body_text = _pick_body_text(error)
    if body_text is None:
        return None
    trimmed = body_text.strip()
    if not trimmed:
        return None
    return truncate_error_text(trimmed, MAX_PROVIDER_ERROR_BODY_CHARS)


def _pick_body_text(error: BaseException) -> str | None:
    body = getattr(error, "body", None)
    if isinstance(body, str):
        return body

    error_field = getattr(error, "error", None)
    if _is_plain_non_empty_object(error_field):
        return safe_json_stringify(error_field)

    response = getattr(error, "response", None)
    if response is None:
        return None

    response_body = getattr(response, "body", None)
    if isinstance(response_body, str):
        return response_body
    if _is_readable_stream_like(response_body):
        return None
    if _is_plain_non_empty_object(response_body):
        return safe_json_stringify(response_body)

    # `httpx.Response.text` (already-read body); guarded because accessing
    # `.text` on a not-yet-read streaming response raises.
    try:
        response_text = getattr(response, "text", None)
    except Exception:
        return None
    if isinstance(response_text, str) and response_text:
        return response_text

    return None


def _is_readable_stream_like(value: Any) -> bool:
    """True for stream-like objects that should not be stringified as a body."""
    return hasattr(value, "read") and not isinstance(value, (str, bytes))


def _is_plain_non_empty_object(value: Any) -> bool:
    """Only a plain, non-empty `dict` counts as an HTTP body.

    SDK error attributes can hold class instances instead of parsed bodies.
    Stringifying an arbitrary class instance produces useless noise (or an
    exception) that would then REPLACE the exception message in the composed
    display string, discarding the one useful string (the real message is
    where the SDK puts the deserialized exception text). Requiring `type(value)
    is dict` excludes such instances while still accepting parsed JSON bodies,
    which are plain dicts by construction.
    """
    return type(value) is dict and len(value) > 0


def format_provider_error(norm: NormalizedProviderError, prefix: str | None = None) -> str:
    """Compose a display string from a normalized error.

    When the message already carries the body, or no body/status was
    extracted, the message is returned unchanged. Otherwise the status and
    body are surfaced, with an optional provider prefix.

    - no prefix: `"<status>: <body>"`
    - prefix:    `"<prefix> (<status>): <body>"`
    """
    if norm.message_carries_body or norm.status is None or norm.body is None:
        if prefix is not None and norm.status is not None:
            return f"{prefix} ({norm.status}): {norm.message}"
        return norm.message
    if prefix is not None:
        return f"{prefix} ({norm.status}): {norm.body}"
    return f"{norm.status}: {norm.body}"


def truncate_error_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [truncated {len(text) - max_chars} chars]"


def safe_json_stringify(value: Any) -> str:
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
