"""Python port of `packages/ai/test/oauth-device-code.test.ts`.

TypeScript uses vitest fake timers and asserts on `Date.now()` at each poll.
`vi.useFakeTimers()` replaces `Date.now` and `setTimeout` from one virtual
clock; asyncio has no global equivalent, so the flow reads both its deadline
and its waits from an injected `DeviceCodeClock`. These tests pass a virtual
one that records each requested delay and advances instead of sleeping, which
pins exactly the schedule the TS test asserts through
`advanceTimersByTimeAsync` without consulting real elapsed time.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest
from pi_ai.auth.oauth.device_code import (
    DeviceCodeClock,
    DeviceCodeError,
    DeviceCodePollResult,
    poll_oauth_device_code_flow,
)
from pi_ai.utils.abort import AbortController

NEVER_ABORTED_SIGNAL = AbortController().signal


@dataclass
class VirtualClock:
    """Records `poll` calls and sleep durations in order, and never really waits."""

    now: float = 0.0
    events: list[object] = field(default_factory=list)

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.events.append(seconds)
        self.now += seconds

    def as_device_code_clock(self) -> DeviceCodeClock:
        return DeviceCodeClock(monotonic=self.monotonic, sleep=self.sleep)


@pytest.fixture
def clock() -> VirtualClock:
    return VirtualClock()


async def test_polls_immediately_and_returns_the_completed_value(clock: VirtualClock) -> None:
    polls = 0

    async def poll() -> DeviceCodePollResult[str]:
        nonlocal polls
        polls += 1
        clock.events.append("poll")
        if polls == 1:
            return DeviceCodePollResult(status="pending")
        return DeviceCodePollResult(status="complete", value="token")

    result = await poll_oauth_device_code_flow(
        poll, NEVER_ABORTED_SIGNAL, interval_seconds=2, expires_in_seconds=30, clock=clock.as_device_code_clock()
    )
    assert result == "token"
    # First poll happens immediately (no leading sleep), the second 2s later.
    assert clock.events == ["poll", 2.0, "poll"]


async def test_can_wait_before_the_first_poll(clock: VirtualClock) -> None:
    async def poll() -> DeviceCodePollResult[str]:
        clock.events.append("poll")
        return DeviceCodePollResult(status="complete", value="token")

    result = await poll_oauth_device_code_flow(
        poll,
        NEVER_ABORTED_SIGNAL,
        interval_seconds=2,
        expires_in_seconds=30,
        wait_before_first_poll=True,
        clock=clock.as_device_code_clock(),
    )
    assert result == "token"
    assert clock.events == [2.0, "poll"]


async def test_increases_the_interval_by_5_seconds_after_slow_down_without_a_server_interval(
    clock: VirtualClock,
) -> None:
    results = [
        DeviceCodePollResult[str](status="slow_down"),
        DeviceCodePollResult[str](status="complete", value="token"),
    ]

    async def poll() -> DeviceCodePollResult[str]:
        clock.events.append("poll")
        if not results:
            raise AssertionError("Unexpected extra poll")
        return results.pop(0)

    result = await poll_oauth_device_code_flow(
        poll, NEVER_ABORTED_SIGNAL, interval_seconds=2, expires_in_seconds=900, clock=clock.as_device_code_clock()
    )
    assert result == "token"
    assert clock.events == ["poll", 7.0, "poll"]


async def test_honors_a_server_provided_slow_down_interval(clock: VirtualClock) -> None:
    results = [
        DeviceCodePollResult[str](status="slow_down", interval_seconds=30),
        DeviceCodePollResult[str](status="complete", value="token"),
    ]

    async def poll() -> DeviceCodePollResult[str]:
        clock.events.append("poll")
        if not results:
            raise AssertionError("Unexpected extra poll")
        return results.pop(0)

    result = await poll_oauth_device_code_flow(
        poll, NEVER_ABORTED_SIGNAL, interval_seconds=2, expires_in_seconds=900, clock=clock.as_device_code_clock()
    )
    assert result == "token"
    assert clock.events == ["poll", 30.0, "poll"]


async def test_cancels_an_in_flight_wait() -> None:
    controller = AbortController()

    async def poll() -> DeviceCodePollResult[str]:
        return DeviceCodePollResult(status="pending")

    task = asyncio.ensure_future(
        poll_oauth_device_code_flow(poll, controller.signal, interval_seconds=5, expires_in_seconds=30)
    )
    await asyncio.sleep(0)
    controller.abort()
    with pytest.raises(DeviceCodeError, match="Login cancelled"):
        await asyncio.wait_for(task, timeout=5)
