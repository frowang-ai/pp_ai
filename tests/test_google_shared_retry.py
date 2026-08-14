"""Python port of `packages/ai/test/google-shared-retry.test.ts`."""

from __future__ import annotations

import pi_ai.utils.provider_retry as provider_retry_module
import pytest
from pi_ai.api.google_shared import retry_google_request
from pi_ai.types import StreamOptions
from pi_ai.utils.abort import AbortSignal


class _GoogleApiError(Exception):
    """Shaped like `@google/genai`'s `ApiError`: has `status`, but no `headers`."""

    def __init__(self, status: int) -> None:
        super().__init__(f"got status: {status}")
        self.status = status


class _Request:
    """A callable recording its call count, with a scripted result sequence."""

    def __init__(self, *results: object) -> None:
        self._results = list(results)
        self.calls = 0

    async def __call__(self) -> object:
        self.calls += 1
        result = self._results[min(self.calls - 1, len(self._results) - 1)]
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.fixture(autouse=True)
def _instant_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for vitest's fake timers: the backoff sleep resolves at once.

    `retry_google_request` takes a `StreamOptions`, which carries no clock, so
    the backoff cannot be injected the way `ProviderRetryOptions.clock` allows.
    Replacing the sleep is safe here because the retry policy has no deadline:
    it counts attempts, never elapsed time, so nothing consults a real clock.
    """

    async def _sleep(
        ms: float,
        signal: AbortSignal | None = None,
        clock: provider_retry_module.RetryClock = provider_retry_module.REAL_CLOCK,
    ) -> None:
        return None

    monkeypatch.setattr(provider_retry_module, "_abortable_sleep", _sleep)


async def test_retries_a_headers_less_sdk_error_with_a_retryable_status():
    request = _Request(_GoogleApiError(429), "ok")

    assert await retry_google_request(request, StreamOptions(max_retries=1)) == "ok"
    assert request.calls == 2


async def test_does_not_retry_when_max_retries_is_unset():
    error = _GoogleApiError(429)
    request = _Request(error)

    with pytest.raises(_GoogleApiError) as excinfo:
        await retry_google_request(request)
    assert excinfo.value is error
    assert request.calls == 1


async def test_does_not_retry_a_non_retryable_status():
    error = _GoogleApiError(400)
    request = _Request(error)

    with pytest.raises(_GoogleApiError) as excinfo:
        await retry_google_request(request, StreamOptions(max_retries=2))
    assert excinfo.value is error
    assert request.calls == 1
