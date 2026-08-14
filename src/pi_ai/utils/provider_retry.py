"""Interruptible retry wrapper for raw provider SDK requests.

Python port of `packages/ai/src/utils/provider-retry.ts`. Reproduces the retry
behavior used by the OpenAI and Anthropic SDKs while making the backoff sleep
interruptible via :class:`pi_ai.utils.abort.AbortSignal`. Their built-in retry
timers ignore the request's abort signal, so callers must invoke the SDK with
`max_retries=0` and wrap the request with this helper.
"""

from __future__ import annotations

import asyncio
import email.utils
import math
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TypeVar

from .abort import AbortSignal

T = TypeVar("T")

_DEFAULT_MAX_RETRY_DELAY_MS = 60_000


@dataclass(frozen=True)
class RetryClock:
    """The single time source the retry loop reads.

    TypeScript drives this loop's tests with `vi.useFakeTimers()`, which
    replaces `Date.now` and `setTimeout` together. asyncio has no global
    equivalent, so `time` and `sleep` are read from here: a test can supply a
    virtual pair and get an exact backoff schedule instead of waiting real
    seconds and depending on machine load.
    """

    time: Callable[[], float] = time.time
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep


REAL_CLOCK = RetryClock()


@dataclass
class ProviderRetryOptions:
    max_retries: int = 0
    max_retry_delay_ms: float | None = None
    signal: AbortSignal | None = None
    clock: RetryClock = REAL_CLOCK


class ProviderRequestAbortError(Exception):
    """Raised when the request is aborted, mirroring the JS `AbortError`."""

    def __init__(self, message: str = "Request aborted") -> None:
        super().__init__(message)


class ProviderError(Exception):
    """A provider HTTP error carrying an optional status and response headers."""

    def __init__(self, message: str, status: int | None = None, headers: Mapping[str, str] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.headers = headers


def _is_provider_error(error: BaseException) -> bool:
    if not isinstance(error, Exception):
        return False
    status = getattr(error, "status", None)
    headers = getattr(error, "headers", None)
    if status is not None and not isinstance(status, int):
        return False
    if headers is not None and not isinstance(headers, Mapping):
        return False
    return hasattr(error, "status") and hasattr(error, "headers")


def _is_retryable_provider_error(error: BaseException) -> bool:
    """Mirrors the pinned OpenAI/Anthropic SDK retry policy; review when either SDK is upgraded."""
    headers = getattr(error, "headers", None) or {}
    should_retry = headers.get("x-should-retry") if isinstance(headers, Mapping) else None
    if should_retry == "true":
        return True
    if should_retry == "false":
        return False

    status = getattr(error, "status", None)
    if status is None:
        return True
    return status in (408, 409, 429) or (isinstance(status, int) and status >= 500)


def _validate_server_retry_delay_ms(
    delay_ms: float, max_retry_delay_ms: float | None, provider_error_message: str
) -> float:
    max_delay_ms = max_retry_delay_ms if max_retry_delay_ms is not None else _DEFAULT_MAX_RETRY_DELAY_MS
    if max_delay_ms > 0 and delay_ms > max_delay_ms:
        raise RuntimeError(
            f"Server requested {_ceil_seconds(delay_ms)}s retry delay "
            f"(max: {_ceil_seconds(max_delay_ms)}s). {provider_error_message}"
        )
    return delay_ms


def _ceil_seconds(ms: float) -> int:
    return math.ceil(ms / 1000)


def _get_retry_delay_ms(
    error: BaseException, retry_index: int, max_retry_delay_ms: float | None, clock: RetryClock = REAL_CLOCK
) -> float:
    headers = getattr(error, "headers", None) or {}
    message = str(error)

    retry_after_ms = headers.get("retry-after-ms") if isinstance(headers, Mapping) else None
    if retry_after_ms:
        try:
            value = float(retry_after_ms)
        except ValueError:
            value = float("nan")
        if value == value:  # not NaN
            return _validate_server_retry_delay_ms(value, max_retry_delay_ms, message)

    retry_after = headers.get("retry-after") if isinstance(headers, Mapping) else None
    if retry_after:
        try:
            seconds = float(retry_after)
            delay_ms = seconds * 1000
        except ValueError:
            try:
                parsed = email.utils.parsedate_to_datetime(retry_after)
            except (TypeError, ValueError, IndexError):
                delay_ms = float("nan")
            else:
                delay_ms = parsed.timestamp() * 1000 - clock.time() * 1000
        return _validate_server_retry_delay_ms(delay_ms, max_retry_delay_ms, message)

    exponential_delay = min(0.5 * 2**retry_index, 8) * 1000
    return exponential_delay * (1 - random.random() * 0.25)


async def _abortable_sleep(ms: float, signal: AbortSignal | None, clock: RetryClock = REAL_CLOCK) -> None:
    if signal is not None and signal.aborted:
        raise ProviderRequestAbortError()

    sleep_task = asyncio.ensure_future(clock.sleep(max(0.0, ms) / 1000))
    if signal is None:
        await sleep_task
        return

    abort_task = asyncio.ensure_future(signal.wait())
    try:
        done, _pending = await asyncio.wait({sleep_task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
        if sleep_task in done:
            return
        raise ProviderRequestAbortError()
    finally:
        for task in (sleep_task, abort_task):
            if not task.done():
                task.cancel()


async def retry_provider_request(
    request: Callable[[], Awaitable[T]],
    options: ProviderRetryOptions | None = None,
) -> T:
    """Retry ``request`` on retryable :class:`ProviderError`\\ s with backoff.

    Provider-requested delays above `max_retry_delay_ms` fail immediately
    (60 seconds by default); set it to zero to disable the limit.
    """
    opts = options if options is not None else ProviderRetryOptions()
    max_retries = opts.max_retries
    retries_remaining = max_retries

    while True:
        try:
            # Each retry is a fresh SDK request, so a per-request retry counter stays zero.
            return await request()
        except Exception as error:
            if opts.signal is not None and opts.signal.aborted:
                raise ProviderRequestAbortError() from error
            if retries_remaining <= 0 or not _is_provider_error(error) or not _is_retryable_provider_error(error):
                raise

            retry_index = max_retries - retries_remaining
            retries_remaining -= 1
            await _abortable_sleep(
                _get_retry_delay_ms(error, retry_index, opts.max_retry_delay_ms, opts.clock), opts.signal, opts.clock
            )
