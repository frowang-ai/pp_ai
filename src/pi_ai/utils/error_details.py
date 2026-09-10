"""Structured classification of provider request failures.

pi_ai-specific extension (no TypeScript counterpart). Provider streams swallow
exceptions into a terminal ``error`` event whose ``error_message`` string is
the only place the HTTP status survived. :func:`build_error_details` distills
the same exception into :class:`pi_ai.types.AssistantMessageErrorDetails` so
callers can drive failover logic (retry on rate limit, fail over on auth
errors, ...) without parsing that string.

Classification reuses the existing probing helpers rather than re-deriving
them: the status and body come from
:func:`pi_ai.utils.error_body.normalize_provider_error`, and the retry delay
is read from the same ``retry-after-ms``/``retry-after`` headers the retry
classifier in :mod:`pi_ai.utils.provider_retry` honors.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from ..types import AssistantMessageErrorDetails, ErrorKind
from .error_body import normalize_provider_error
from .provider_retry import ProviderRequestAbortError


def build_error_details(error: BaseException, *, aborted: bool = False) -> AssistantMessageErrorDetails:
    """Distill ``error`` into an :class:`AssistantMessageErrorDetails`.

    ``aborted`` should mirror the provider's own abort check
    (``options.signal.aborted``); it wins over every other classification,
    matching how the ``stop_reason`` is chosen. A status-less transport
    failure (``httpx.TransportError``: connect/read/timeout) is classified
    ``network``; anything without a recognizable status is ``unknown``.
    """
    if aborted or isinstance(error, ProviderRequestAbortError):
        return AssistantMessageErrorDetails(kind="aborted")

    normalized = normalize_provider_error(error)
    retry_after_ms = _retry_after_ms(getattr(error, "headers", None))

    if normalized.status is None:
        kind: ErrorKind = "network" if isinstance(error, httpx.TransportError) else "unknown"
        return AssistantMessageErrorDetails(kind=kind, body=normalized.body)

    return AssistantMessageErrorDetails(
        kind=_classify_status(normalized.status),
        status=normalized.status,
        body=normalized.body,
        retry_after_ms=retry_after_ms,
    )


def _classify_status(status: int) -> ErrorKind:
    if status in (401, 403):
        return "auth"
    if status == 402:
        return "quota"
    if status == 429:
        return "rate_limit"
    if status >= 500:
        return "server"
    return "unknown"


def _retry_after_ms(headers: Any) -> float | None:
    """Parse the server-requested retry delay, mirroring provider_retry's header precedence.

    Only plain numeric values are honored; HTTP-date ``retry-after`` values
    need a clock to resolve and are left to the retry classifier itself.
    """
    if not isinstance(headers, Mapping):
        return None
    lowered = {str(key).lower(): value for key, value in headers.items()}

    retry_after_ms = lowered.get("retry-after-ms")
    if retry_after_ms:
        try:
            return float(retry_after_ms)
        except (TypeError, ValueError):
            pass

    retry_after = lowered.get("retry-after")
    if retry_after:
        try:
            return float(retry_after) * 1000
        except (TypeError, ValueError):
            pass

    return None
