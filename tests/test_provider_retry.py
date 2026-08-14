"""Tests for `pi_ai.utils.provider_retry`.

Includes the Python port of `packages/ai/test/provider-retry.test.ts`.

TypeScript drives the backoff with `vi.useFakeTimers()`. asyncio has no global
equivalent, so `retry_provider_request` reads its wait *and* its `retry-after`
date arithmetic from an injected `RetryClock`; `retry_options()` supplies a
virtual one so the schedule is exact and no test spends real seconds backing
off. The two abort-during-backoff cases keep the real clock on purpose: they
need a genuinely pending sleep for the signal to interrupt.
"""

import asyncio
import math

import pi_ai.utils.provider_retry as provider_retry_module
import pytest
from pi_ai.utils.abort import AbortSignal
from pi_ai.utils.provider_retry import (
    ProviderError,
    ProviderRequestAbortError,
    ProviderRetryOptions,
    RetryClock,
    retry_provider_request,
)


class VirtualClock:
    """The single time source the retry loop reads; its waits are instant."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def as_retry_clock(self) -> RetryClock:
        return RetryClock(time=self.time, sleep=self.sleep)


def retry_options(**kwargs) -> ProviderRetryOptions:
    """`ProviderRetryOptions` whose backoff is instant unless a clock is given."""
    kwargs.setdefault("clock", VirtualClock().as_retry_clock())
    return ProviderRetryOptions(**kwargs)


async def test_returns_result_on_first_success():
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        return "ok"

    result = await retry_provider_request(request)
    assert result == "ok"
    assert calls == 1


async def test_retries_retryable_status_error_until_success():
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ProviderError("server error", status=500)
        return "ok"

    result = await retry_provider_request(request, retry_options(max_retries=5))
    assert result == "ok"
    assert calls == 3


async def test_raises_immediately_for_non_provider_error():
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        raise ValueError("not a provider error")

    with pytest.raises(ValueError, match="not a provider error"):
        await retry_provider_request(request, retry_options(max_retries=3))
    assert calls == 1


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503])
async def test_retryable_status_codes(status):
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderError("err", status=status)
        return "ok"

    result = await retry_provider_request(request, retry_options(max_retries=1))
    assert result == "ok"


@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_non_retryable_status_codes_raise_immediately(status):
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        raise ProviderError("err", status=status)

    with pytest.raises(ProviderError):
        await retry_provider_request(request, retry_options(max_retries=3))
    assert calls == 1


async def test_undefined_status_is_treated_as_retryable():
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderError("err", status=None)
        return "ok"

    result = await retry_provider_request(request, retry_options(max_retries=1))
    assert result == "ok"


async def test_x_should_retry_header_true_forces_retry_of_non_retryable_status():
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderError("err", status=400, headers={"x-should-retry": "true"})
        return "ok"

    result = await retry_provider_request(request, retry_options(max_retries=1))
    assert result == "ok"


async def test_x_should_retry_header_false_prevents_retry_of_retryable_status():
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        raise ProviderError("err", status=500, headers={"x-should-retry": "false"})

    with pytest.raises(ProviderError):
        await retry_provider_request(request, retry_options(max_retries=3))
    assert calls == 1


async def test_exhausts_max_retries_and_raises_last_error():
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        raise ProviderError("still failing", status=500)

    with pytest.raises(ProviderError, match="still failing"):
        await retry_provider_request(request, retry_options(max_retries=2))
    assert calls == 3  # initial + 2 retries


async def test_retry_after_ms_header_used_for_delay():
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderError("err", status=500, headers={"retry-after-ms": "5"})
        return "ok"

    result = await retry_provider_request(request, retry_options(max_retries=1))
    assert result == "ok"


async def test_retry_after_ms_header_above_cap_raises():
    async def request():
        raise ProviderError("err", status=500, headers={"retry-after-ms": "999999"})

    with pytest.raises(RuntimeError, match="retry delay"):
        await retry_provider_request(request, retry_options(max_retries=1, max_retry_delay_ms=1000))


async def test_retry_after_seconds_header_used_for_delay():
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderError("err", status=500, headers={"retry-after": "0.01"})
        return "ok"

    result = await retry_provider_request(request, retry_options(max_retries=1))
    assert result == "ok"


async def test_signal_aborted_before_call_raises_abort_error():
    signal = AbortSignal()
    signal.abort()

    async def request():
        raise ProviderError("err", status=500)

    with pytest.raises(ProviderRequestAbortError):
        await retry_provider_request(request, retry_options(max_retries=3, signal=signal))


async def test_signal_aborted_during_backoff_raises_abort_error():
    signal = AbortSignal()
    slept: list[float] = []

    async def request():
        raise ProviderError("err", status=500, headers={"retry-after-ms": "5000"})

    # Aborting from a timed task would be a guess about when the loop reaches
    # its backoff; aborting from inside the clock's own sleep is the exact
    # moment, and costs no wall-clock time under parallel test load.
    async def sleep(seconds: float) -> None:
        slept.append(seconds)
        signal.abort()
        await asyncio.Event().wait()

    with pytest.raises(ProviderRequestAbortError):
        await retry_provider_request(
            request,
            ProviderRetryOptions(max_retries=3, signal=signal, clock=RetryClock(sleep=sleep)),
        )

    assert slept == [5.0]


async def test_max_retry_delay_ms_zero_disables_cap():
    signal = AbortSignal()
    slept: list[float] = []

    async def request():
        raise ProviderError("err", status=500, headers={"retry-after-ms": "999999999"})

    async def sleep(seconds: float) -> None:
        slept.append(seconds)
        signal.abort()
        await asyncio.Event().wait()

    # Cap disabled: the provider's ~999999s delay must reach the sleep unchanged
    # instead of raising the "exceeds cap" RuntimeError.
    with pytest.raises(ProviderRequestAbortError):
        await retry_provider_request(
            request,
            ProviderRetryOptions(max_retries=1, max_retry_delay_ms=0, signal=signal, clock=RetryClock(sleep=sleep)),
        )

    assert slept == [999999.999]


class _StatusOnlyError(Exception):
    def __init__(self, status):
        super().__init__("status only")
        self.status = status
        self.headers = {}


class _HeadersOnlyError(Exception):
    def __init__(self, headers):
        super().__init__("headers only")
        self.status = 500
        self.headers = headers


def test_is_provider_error_rejects_non_exception_and_invalid_fields():
    assert provider_retry_module._is_provider_error(KeyboardInterrupt()) is False
    assert provider_retry_module._is_provider_error(_StatusOnlyError("500")) is False
    assert provider_retry_module._is_provider_error(_HeadersOnlyError([("x", "y")])) is False
    assert provider_retry_module._is_provider_error(ProviderError("ok", status=500, headers={})) is True


def test_get_retry_delay_ms_uses_exponential_backoff_with_jitter_and_cap(monkeypatch):
    monkeypatch.setattr(provider_retry_module.random, "random", lambda: 0.0)
    error = ProviderError("server error", status=500)

    assert provider_retry_module._get_retry_delay_ms(error, retry_index=0, max_retry_delay_ms=None) == 500
    assert provider_retry_module._get_retry_delay_ms(error, retry_index=10, max_retry_delay_ms=None) == 8000

    monkeypatch.setattr(provider_retry_module.random, "random", lambda: 1.0)
    assert provider_retry_module._get_retry_delay_ms(error, retry_index=0, max_retry_delay_ms=None) == 375


def test_get_retry_delay_ms_falls_back_from_invalid_retry_after_ms_to_retry_after_header():
    error = ProviderError(
        "server error",
        status=500,
        headers={"retry-after-ms": "not-a-number", "retry-after": "0.25"},
    )

    assert provider_retry_module._get_retry_delay_ms(error, retry_index=0, max_retry_delay_ms=None) == 250


def test_get_retry_delay_ms_parses_http_date_retry_after():
    error = ProviderError("server error", status=500, headers={"retry-after": "Thu, 01 Jan 1970 00:16:42 GMT"})

    delay_ms = provider_retry_module._get_retry_delay_ms(
        error, retry_index=0, max_retry_delay_ms=None, clock=VirtualClock(now=1000.0).as_retry_clock()
    )
    assert delay_ms == 2000


def test_get_retry_delay_ms_returns_nan_for_invalid_retry_after_date():
    error = ProviderError("server error", status=500, headers={"retry-after": "not a date"})

    delay_ms = provider_retry_module._get_retry_delay_ms(error, retry_index=0, max_retry_delay_ms=None)
    assert math.isnan(delay_ms)


async def test_invalid_retry_after_date_retries_instead_of_raising():
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderError("err", status=500, headers={"retry-after": "not a date"})
        return "ok"

    result = await retry_provider_request(request, retry_options(max_retries=1))

    assert result == "ok"
    assert calls == 2


async def test_abortable_sleep_raises_when_signal_is_already_aborted():
    signal = AbortSignal()
    signal.abort()

    with pytest.raises(ProviderRequestAbortError):
        await provider_retry_module._abortable_sleep(1, signal)


async def test_abortable_sleep_returns_when_delay_completes_before_abort():
    await provider_retry_module._abortable_sleep(1, AbortSignal())


# --------------------------------------------------------------------------
# Ported from `packages/ai/test/provider-retry.test.ts`
#
# vitest drives those cases with fake timers and asserts the request has not
# been retried until the clock reaches the requested delay. Python has no
# equivalent global clock, so `_abortable_sleep` is replaced with a recorder:
# the assertion "the retry waited exactly the delay the provider asked for"
# is checked against the recorded delay instead of a virtual clock.
# --------------------------------------------------------------------------


@pytest.fixture
def clock() -> VirtualClock:
    """A virtual clock whose recorded `sleeps` are the backoff schedule, in seconds."""
    return VirtualClock()


async def test_ts_retries_retryable_provider_errors(clock: VirtualClock):
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderError("Provider error: 429", status=429, headers={"retry-after-ms": "1000"})
        return "ok"

    assert await retry_provider_request(request, retry_options(max_retries=1, clock=clock.as_retry_clock())) == "ok"
    assert calls == 2
    # The provider asked for 1000ms via `retry-after-ms`.
    assert clock.sleeps == [1.0]


async def test_ts_does_not_retry_errors_the_provider_marks_as_non_retryable():
    calls = 0
    error = ProviderError("Provider error: 429", status=429, headers={"x-should-retry": "false"})

    async def request():
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(ProviderError) as raised:
        await retry_provider_request(request, retry_options(max_retries=2))
    assert raised.value is error
    assert calls == 1


async def test_ts_rejects_a_provider_requested_retry_delay_above_the_limit():
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        raise ProviderError("Provider error: 429", status=429, headers={"retry-after": "277403"})

    with pytest.raises(RuntimeError, match=r"Server requested 277403s retry delay \(max: 1s\)"):
        await retry_provider_request(request, retry_options(max_retries=1, max_retry_delay_ms=1000))
    assert calls == 1


async def test_ts_allows_disabling_the_provider_requested_retry_delay_cap(clock: VirtualClock):
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderError("Provider error: 429", status=429, headers={"retry-after": "2"})
        return "ok"

    result = await retry_provider_request(
        request, retry_options(max_retries=1, max_retry_delay_ms=0, clock=clock.as_retry_clock())
    )
    assert result == "ok"
    assert calls == 2
    # `retry-after: 2` seconds, with the cap disabled.
    assert clock.sleeps == [2.0]


async def test_ts_aborts_a_provider_requested_retry_delay():
    calls = 0
    signal = AbortSignal()
    slept: list[float] = []

    async def request():
        nonlocal calls
        calls += 1
        raise ProviderError("Provider error: 429", status=429, headers={"retry-after": "277403"})

    # TS aborts from inside a fake timer. Aborting from the injected clock's
    # sleep is the same instant without depending on real scheduling.
    async def sleep(seconds: float) -> None:
        slept.append(seconds)
        signal.abort()
        await asyncio.Event().wait()

    with pytest.raises(ProviderRequestAbortError):
        await retry_provider_request(
            request,
            ProviderRetryOptions(max_retries=2, max_retry_delay_ms=0, signal=signal, clock=RetryClock(sleep=sleep)),
        )
    assert calls == 1
    assert slept == [277403.0]
    # TS also asserts `vi.getTimerCount()` is 1 during the backoff and 0 after the abort.
    # There is no analogue: `_abortable_sleep` waits on an `asyncio` future rather than a
    # registered timer, and asyncio exposes no global timer registry to count. The
    # recorded `slept` entry above is the closest equivalent -- it proves the backoff was
    # entered with the provider's delay, and the raise proves it ended on the abort.
