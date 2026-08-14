"""Assistant-call retry policy and transient-error classification.

Python port of `packages/ai/src/utils/retry.ts`.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace

from ..types import AssistantMessage
from .abort import AbortSignal


def _build_provider_error_pattern(patterns: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile("|".join(patterns), re.IGNORECASE)


_NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN = _build_provider_error_pattern(
    (
        # OpenCode Go/free-tier limits returned as 429 JSON error types by OpenCode's
        # Zen API. These are subscription/account limits, not transient throttles.
        "GoUsageLimitError",
        "FreeUsageLimitError",
        # OpenCode Go subscription-limit text asks users to enable available-balance
        # usage after rolling/weekly/monthly limits are reached.
        "Monthly usage limit reached",
        "available balance",
        # Generic quota/budget/billing exhaustion. `insufficient_quota` is OpenAI's
        # quota/billing error code; the other strings cover common gateway wording.
        "insufficient_quota",
        "out of budget",
        "quota exceeded",
        "billing",
    )
)

_RETRYABLE_PROVIDER_ERROR_PATTERN = _build_provider_error_pattern(
    (
        # Generic provider load, HTTP status, and server-side transient failures.
        "overloaded",
        "rate.?limit",
        "too many requests",
        "429",
        "500",
        "502",
        "503",
        "504",
        "524",
        "service.?unavailable",
        "server.?error",
        "internal.?error",
        # Wrapper/provider text for transient upstream failures, including OpenRouter
        # "Provider returned error" responses (#2264).
        "provider.?returned.?error",
        "exceeded request buffer limit while retrying upstream",
        # Network, proxy, and fetch transport failures. This includes OpenAI Codex
        # raw-fetch failures such as "upstream connect", "connection refused", and
        # "reset before headers" (#733), plus OpenRouter connection drops (#3317).
        "network.?error",
        "connection.?error",
        "connection.?refused",
        "connection.?lost",
        "other side closed",
        "fetch failed",
        "getaddrinfo",
        "ENOTFOUND",
        "EAI_AGAIN",
        "upstream.?connect",
        "reset before headers",
        "socket hang up",
        "socket connection was closed",
        "timed? out",
        "timeout",
        "terminated",
        # WebSocket transports can report close/error text instead of HTTP/fetch text.
        "websocket.?closed",
        "websocket.?error",
        # Premature stream endings from SDKs and transports. Anthropic can throw
        # "stream ended without ..." and "Anthropic stream ended before message_stop"
        # (#4433); Bedrock/Smithy can throw an HTTP/2 no-response error (#3594).
        "ended without",
        "stream ended before message_stop",
        "stream ended before a terminal response event",
        "http2 request did not get a response",
        # Provider-requested retry delay cap failures should flow through the outer
        # retry policy so callers can surface/abort the backoff (#1123).
        "retry delay",
        # Explicit retry guidance emitted mid-stream by OpenAI Responses and Bedrock
        # stream exceptions (#6019).
        "you can retry your request",
        "try your request again",
        "please retry your request",
        # gRPC based providers (e.g. NVIDIA NIM)
        "ResourceExhausted",
    )
)


@dataclass
class RetryPolicy:
    """Retry policy: bounded attempts with exponential backoff (`base_delay_ms * 2^(attempt-1)`).

    Matches `settings.retry` (`enabled`, `maxRetries`, `baseDelayMs`) in coding-agent;
    kept here so the classifier and the policy-driven retry loop live together and stay
    reusable by the SDK and other callers.
    """

    enabled: bool
    max_retries: int
    """Max retry attempts (0 = no retries). The initial call never counts as a retry."""
    base_delay_ms: float
    """Base delay in ms. Per-attempt delay is `base_delay_ms * 2^(attempt-1)` before jitter."""


@dataclass
class RetryCallbacks:
    """Optional callbacks emitted by :func:`retry_assistant_call` around each retry."""

    on_retry_scheduled: Callable[[int, int, float, str], Awaitable[None] | None] | None = None
    """Emitted before the backoff sleep of each retry attempt (1-indexed)."""
    on_retry_attempt_start: Callable[[], Awaitable[None] | None] | None = None
    """Emitted after the backoff sleep, immediately before the retried call starts."""
    on_retry_finished: Callable[[bool, int, str | None], Awaitable[None] | None] | None = None
    """Emitted once when the loop ends: success if a later call completed normally."""


class RetrySleepAbortError(Exception):
    def __init__(self) -> None:
        super().__init__("Aborted")


async def _maybe_await(result: Awaitable[None] | None) -> None:
    if result is not None:
        await result


async def _sleep(ms: float, signal: AbortSignal | None) -> None:
    if signal is not None and signal.aborted:
        raise RetrySleepAbortError()

    sleep_task = asyncio.ensure_future(asyncio.sleep(ms / 1000))
    if signal is None:
        await sleep_task
        return

    abort_task = asyncio.ensure_future(signal.wait())
    try:
        done, _pending = await asyncio.wait({sleep_task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
        if sleep_task in done:
            return
        raise RetrySleepAbortError()
    finally:
        for task in (sleep_task, abort_task):
            if not task.done():
                task.cancel()


async def retry_assistant_call(
    produce: Callable[[], Awaitable[AssistantMessage]],
    policy: RetryPolicy | None,
    signal: AbortSignal | None,
    callbacks: RetryCallbacks | None = None,
) -> AssistantMessage:
    """Run a single assistant-producing call with bounded retry on transient errors.

    Behavior:
    - A successful response is returned immediately. Aborts are terminal and never
      retried, but reported as unsuccessful if they happen after a retry was scheduled.
      Aborts during the backoff sleep are normalized to an aborted `AssistantMessage`
      too, so callers do not need to care when cancellation happened.
    - A non-retryable error (per :func:`is_retryable_assistant_error`, including quota/
      billing exhaustion) is returned immediately so deterministic errors fail fast.
    - Otherwise retries up to `max_retries` times with exponential backoff, emitting
      `on_retry_scheduled` before each sleep, `on_retry_attempt_start` after each sleep
      before the retried call starts, and `on_retry_finished` once at the end (whether
      the loop ends in success, exhausted retries, or an aborted backoff).

    When `policy` is None or disabled, the first response is returned unchanged
    (equivalent to calling `produce()` directly).
    """
    max_attempts = policy.max_retries if policy is not None and policy.enabled else 0

    attempt = 0
    last_retry: tuple[int, str] | None = None
    while True:
        response = await produce()

        # Abort: terminal but not successful. Never retry an aborted message.
        if response.stop_reason == "aborted":
            if last_retry is not None and callbacks is not None and callbacks.on_retry_finished is not None:
                await _maybe_await(callbacks.on_retry_finished(False, last_retry[0], None))
            return response

        # Success: non-error, non-abort responses return as-is.
        if response.stop_reason != "error":
            if last_retry is not None and callbacks is not None and callbacks.on_retry_finished is not None:
                await _maybe_await(callbacks.on_retry_finished(True, last_retry[0], None))
            return response

        # Non-retryable, or budget exhausted: return the final error message.
        if attempt >= max_attempts or not is_retryable_assistant_error(response):
            if last_retry is not None and callbacks is not None and callbacks.on_retry_finished is not None:
                await _maybe_await(callbacks.on_retry_finished(False, last_retry[0], response.error_message))
            return response

        attempt += 1
        error_message = response.error_message or "Unknown error"
        last_retry = (attempt, error_message)
        delay_ms = policy.base_delay_ms * 2 ** (attempt - 1)  # type: ignore[union-attr]
        if callbacks is not None and callbacks.on_retry_scheduled is not None:
            await _maybe_await(callbacks.on_retry_scheduled(attempt, max_attempts, delay_ms, error_message))

        # Normalize aborts during retry backoff to the same AssistantMessage shape as
        # provider stream aborts, so callers do not need to care when cancellation happened.
        try:
            await _sleep(delay_ms, signal)
        except RetrySleepAbortError:
            if callbacks is not None and callbacks.on_retry_finished is not None:
                await _maybe_await(callbacks.on_retry_finished(False, attempt, error_message))
            return replace(response, stop_reason="aborted", error_message=None)

        if callbacks is not None and callbacks.on_retry_attempt_start is not None:
            await _maybe_await(callbacks.on_retry_attempt_start())


def is_retryable_assistant_error(message: AssistantMessage) -> bool:
    """Classify whether a failed assistant message looks like a transient provider or transport error.

    This does not implement retry policy. Callers should first handle context
    overflow separately, then apply their own retry budget, backoff, and reporting
    before restarting the assistant turn.
    """
    if message.stop_reason != "error" or not message.error_message:
        return False
    error_message = message.error_message
    if _NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN.search(error_message):
        return False
    return bool(_RETRYABLE_PROVIDER_ERROR_PATTERN.search(error_message))
