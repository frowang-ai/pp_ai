"""Shared OAuth device-code polling flow.

Python port of `packages/ai/src/auth/oauth/device-code.ts`.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from ...utils.abort import AbortSignal

CANCEL_MESSAGE = "Login cancelled"
TIMEOUT_MESSAGE = "Device flow timed out"
SLOW_DOWN_TIMEOUT_MESSAGE = (
    "Device flow timed out after one or more slow_down responses. This is often caused by "
    "clock drift in WSL or VM environments. Please sync or restart the VM clock and try again."
)
MINIMUM_INTERVAL_MS = 1000
# RFC 8628 section 3.2: if the authorization server omits `interval`, the client must use 5 seconds.
DEFAULT_POLL_INTERVAL_SECONDS = 5
# RFC 8628 section 3.5: `slow_down` means the polling interval must increase by 5 seconds.
SLOW_DOWN_INTERVAL_INCREMENT_MS = 5000

T = TypeVar("T")


@dataclass(frozen=True)
class DeviceCodeClock:
    """The single time source the device-code flow reads.

    TypeScript drives this flow's tests with `vi.useFakeTimers()`, which
    replaces `Date.now` and `setTimeout` together from one virtual clock, so
    upstream's deadline check and its inter-poll wait can never disagree.
    asyncio has no global equivalent: `time.monotonic()` and `asyncio.sleep()`
    are independent, and a test that fakes only one of them decides the
    timeout partly from real elapsed time, which makes it depend on machine
    load. Reading both from one injected object restores upstream's guarantee
    and lets a test make the flow fully deterministic.
    """

    monotonic: Callable[[], float] = time.monotonic
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep


REAL_CLOCK = DeviceCodeClock()


@dataclass
class DeviceCodePollResult(Generic[T]):
    """One dataclass covers every `OAuthDeviceCodePollResult<T>` variant.

    ``status`` selects which fields apply: ``pending`` (none), ``slow_down``
    (``interval_seconds``), ``failed`` (``message``), or ``complete``
    (``value``).
    """

    status: Literal["pending", "slow_down", "failed", "complete"]
    interval_seconds: float | None = None
    message: str | None = None
    value: T | None = None


class DeviceCodeError(Exception):
    """Raised when the device-code flow fails, is cancelled, or times out."""


async def _abortable_sleep(
    seconds: float, signal: AbortSignal, cancel_message: str, clock: DeviceCodeClock = REAL_CLOCK
) -> None:
    if signal.aborted:
        raise DeviceCodeError(cancel_message)

    sleep_task: asyncio.Task[None] = asyncio.ensure_future(clock.sleep(seconds))
    abort_task: asyncio.Task[None] = asyncio.ensure_future(signal.wait())
    try:
        done, _pending = await asyncio.wait({sleep_task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
        if sleep_task in done:
            return
        raise DeviceCodeError(cancel_message)
    finally:
        for task in (sleep_task, abort_task):
            if not task.done():
                task.cancel()


async def poll_oauth_device_code_flow(
    poll: Callable[[], Awaitable[DeviceCodePollResult[T]]],
    signal: AbortSignal,
    *,
    interval_seconds: float | None = None,
    expires_in_seconds: float | None = None,
    wait_before_first_poll: bool = False,
    clock: DeviceCodeClock = REAL_CLOCK,
) -> T:
    """Poll ``poll`` on an interval until it reports ``complete`` or ``failed``.

    Raises :class:`DeviceCodeError` on cancellation, an explicit ``failed``
    result, or timeout (using a distinct message when one or more
    ``slow_down`` responses preceded the timeout).

    Both the deadline and the waits between polls come from ``clock``, so the
    schedule is decided by one time source rather than by real elapsed time.
    """
    deadline = clock.monotonic() + expires_in_seconds if expires_in_seconds is not None else math.inf
    interval_ms = max(MINIMUM_INTERVAL_MS, math.floor((interval_seconds or DEFAULT_POLL_INTERVAL_SECONDS) * 1000))

    slow_down_responses = 0
    if wait_before_first_poll:
        remaining_s = deadline - clock.monotonic()
        if remaining_s > 0:
            await _abortable_sleep(min(interval_ms / 1000, remaining_s), signal, CANCEL_MESSAGE, clock)

    while clock.monotonic() < deadline:
        if signal.aborted:
            raise DeviceCodeError(CANCEL_MESSAGE)

        result = await poll()
        if result.status == "complete":
            return result.value  # type: ignore[return-value]
        if result.status == "failed":
            raise DeviceCodeError(result.message or "Device flow failed")
        if result.status == "slow_down":
            slow_down_responses += 1
            # Use the server-provided interval when given (GitHub reports the new required minimum
            # in `interval`); trusting only a client-tracked value risks polling early forever under
            # WSL/VM clock drift. Otherwise apply RFC 8628 section 3.5: increase by 5 seconds.
            if (
                result.interval_seconds is not None
                and math.isfinite(result.interval_seconds)
                and result.interval_seconds > 0
            ):
                interval_ms = max(MINIMUM_INTERVAL_MS, math.floor(result.interval_seconds * 1000))
            else:
                interval_ms = max(MINIMUM_INTERVAL_MS, interval_ms + SLOW_DOWN_INTERVAL_INCREMENT_MS)

        remaining_s = deadline - clock.monotonic()
        if remaining_s <= 0:
            break

        await _abortable_sleep(min(interval_ms / 1000, remaining_s), signal, CANCEL_MESSAGE, clock)

    raise DeviceCodeError(SLOW_DOWN_TIMEOUT_MESSAGE if slow_down_responses > 0 else TIMEOUT_MESSAGE)
