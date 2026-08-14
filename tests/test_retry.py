"""Tests for `pi_ai.utils.retry`.

Includes the Python port of `packages/ai/test/retry.test.ts`.
"""

import asyncio

import pytest
from pi_ai.providers.faux import faux_assistant_message
from pi_ai.types import AssistantMessage, TextContent
from pi_ai.utils.abort import AbortSignal
from pi_ai.utils.retry import (
    RetryCallbacks,
    RetryPolicy,
    is_retryable_assistant_error,
    retry_assistant_call,
)


def _error_message(text: str) -> AssistantMessage:
    return AssistantMessage(stop_reason="error", error_message=text)


@pytest.mark.parametrize(
    "text",
    [
        "rate limit exceeded",
        "rate-limit exceeded",
        "ratelimit exceeded",
        "429 too many requests",
        "500 Internal Server Error",
        "502 Bad Gateway",
        "503 Service Unavailable",
        "connection refused",
        "getaddrinfo ENOTFOUND",
        "socket hang up",
        "request timed out",
        "request timeout",
        "the operation was terminated",
        "websocket closed unexpectedly",
        "stream ended without a final response",
        "Anthropic stream ended before message_stop",
        "please try your request again",
        "ResourceExhausted: quota",
        "overloaded_error",
        "Provider returned error",
    ],
)
def test_is_retryable_assistant_error_true_for_transient_patterns(text):
    assert is_retryable_assistant_error(_error_message(text)) is True


@pytest.mark.parametrize(
    "text",
    [
        "insufficient_quota",
        "quota exceeded",
        "out of budget",
        "billing issue",
        "Monthly usage limit reached",
        "GoUsageLimitError",
        "FreeUsageLimitError",
        "please check your available balance",
    ],
)
def test_is_retryable_assistant_error_false_for_non_retryable_limit_patterns(text):
    assert is_retryable_assistant_error(_error_message(text)) is False


def test_is_retryable_assistant_error_false_for_unrelated_error():
    assert is_retryable_assistant_error(_error_message("invalid api key")) is False


def test_is_retryable_assistant_error_false_when_stop_reason_is_not_error():
    message = AssistantMessage(stop_reason="stop", error_message="rate limit exceeded")
    assert is_retryable_assistant_error(message) is False


def test_is_retryable_assistant_error_false_when_no_error_message():
    message = AssistantMessage(stop_reason="error", error_message=None)
    assert is_retryable_assistant_error(message) is False


def test_non_retryable_pattern_takes_precedence_over_retryable_pattern():
    # "insufficient_quota" also loosely resembles overload language in some
    # providers; the non-retryable pattern must win when both could match.
    message = _error_message("insufficient_quota: rate limit also applies")
    assert is_retryable_assistant_error(message) is False


async def test_retry_assistant_call_returns_success_without_retry():
    async def produce():
        return AssistantMessage(stop_reason="stop")

    policy = RetryPolicy(enabled=True, max_retries=3, base_delay_ms=1)
    result = await retry_assistant_call(produce, policy, None)
    assert result.stop_reason == "stop"


async def test_retry_assistant_call_returns_aborted_immediately_without_retry():
    calls = 0

    async def produce():
        nonlocal calls
        calls += 1
        return AssistantMessage(stop_reason="aborted")

    policy = RetryPolicy(enabled=True, max_retries=3, base_delay_ms=1)
    result = await retry_assistant_call(produce, policy, None)
    assert result.stop_reason == "aborted"
    assert calls == 1


async def test_retry_assistant_call_returns_immediately_for_non_retryable_error():
    calls = 0

    async def produce():
        nonlocal calls
        calls += 1
        return _error_message("insufficient_quota")

    policy = RetryPolicy(enabled=True, max_retries=3, base_delay_ms=1)
    result = await retry_assistant_call(produce, policy, None)
    assert result.stop_reason == "error"
    assert calls == 1


async def test_retry_assistant_call_retries_until_success():
    calls = 0

    async def produce():
        nonlocal calls
        calls += 1
        if calls < 3:
            return _error_message("rate limit exceeded")
        return AssistantMessage(stop_reason="stop")

    policy = RetryPolicy(enabled=True, max_retries=5, base_delay_ms=1)
    result = await retry_assistant_call(produce, policy, None)
    assert result.stop_reason == "stop"
    assert calls == 3


async def test_retry_assistant_call_exhausts_retries_and_returns_final_error():
    calls = 0

    async def produce():
        nonlocal calls
        calls += 1
        return _error_message("rate limit exceeded")

    policy = RetryPolicy(enabled=True, max_retries=2, base_delay_ms=1)
    result = await retry_assistant_call(produce, policy, None)
    assert result.stop_reason == "error"
    assert calls == 3  # initial + 2 retries


async def test_retry_assistant_call_disabled_policy_returns_first_response():
    calls = 0

    async def produce():
        nonlocal calls
        calls += 1
        return _error_message("rate limit exceeded")

    policy = RetryPolicy(enabled=False, max_retries=5, base_delay_ms=1)
    result = await retry_assistant_call(produce, policy, None)
    assert calls == 1
    assert result.stop_reason == "error"


async def test_retry_assistant_call_none_policy_returns_first_response():
    async def produce():
        return _error_message("rate limit exceeded")

    result = await retry_assistant_call(produce, None, None)
    assert result.stop_reason == "error"


async def test_retry_assistant_call_invokes_callbacks_around_retries():
    calls = 0
    scheduled = []
    attempt_starts = 0
    finished = []

    async def produce():
        nonlocal calls
        calls += 1
        if calls < 2:
            return _error_message("rate limit exceeded")
        return AssistantMessage(stop_reason="stop")

    async def on_retry_scheduled(attempt, max_attempts, delay_ms, error_message):
        scheduled.append((attempt, max_attempts, error_message))

    async def on_retry_attempt_start():
        nonlocal attempt_starts
        attempt_starts += 1

    async def on_retry_finished(success, attempt, final_error):
        finished.append((success, attempt, final_error))

    callbacks = RetryCallbacks(
        on_retry_scheduled=on_retry_scheduled,
        on_retry_attempt_start=on_retry_attempt_start,
        on_retry_finished=on_retry_finished,
    )
    policy = RetryPolicy(enabled=True, max_retries=3, base_delay_ms=1)
    result = await retry_assistant_call(produce, policy, None, callbacks)

    assert result.stop_reason == "stop"
    assert scheduled == [(1, 3, "rate limit exceeded")]
    assert attempt_starts == 1
    assert finished == [(True, 1, None)]


async def test_retry_assistant_call_aborts_during_backoff_returns_aborted_message():
    signal = AbortSignal()

    async def produce():
        return _error_message("rate limit exceeded")

    async def on_retry_scheduled(attempt, max_attempts, delay_ms, error_message):
        signal.abort()

    callbacks = RetryCallbacks(on_retry_scheduled=on_retry_scheduled)
    policy = RetryPolicy(enabled=True, max_retries=3, base_delay_ms=1000)
    result = await retry_assistant_call(produce, policy, signal, callbacks)

    assert result.stop_reason == "aborted"
    assert result.error_message is None


async def test_retry_assistant_call_exponential_backoff_delay_values():
    delays = []

    async def produce():
        return _error_message("rate limit exceeded")

    async def on_retry_scheduled(attempt, max_attempts, delay_ms, error_message):
        delays.append(delay_ms)
        if attempt >= 2:
            # Abort after capturing two scheduled delays to keep the test fast.
            raise asyncio.CancelledError()

    callbacks = RetryCallbacks(on_retry_scheduled=on_retry_scheduled)
    policy = RetryPolicy(enabled=True, max_retries=5, base_delay_ms=10)

    with pytest.raises(asyncio.CancelledError):
        await retry_assistant_call(produce, policy, None, callbacks)

    assert delays == [10, 20]


@pytest.mark.parametrize(
    "text",
    [
        "An error occurred while processing your request. You can retry your request, or contact us through our help center at help.openai.com if the error persists.",
        '{"message":"The system encountered an unexpected error during processing. Try your request again."}',
        "OpenAI Responses stream ended before a terminal response event",
        "Error: exceeded request buffer limit while retrying upstream",
        "The socket connection was closed unexpectedly.",
        "upstream connect error or disconnect/reset before headers",
        "websocket error: connection lost",
        "Network error: other side closed",
        "connect EAI_AGAIN api.example.com",
        "HTTP2 request did not get a response",
        "524 status code (no body)",
        "retry delay exceeded temporary cap",
        "Please retry your request after the proxy recovers.",
    ],
)
def test_is_retryable_assistant_error_true_for_upstream_provider_messages(text):
    assert is_retryable_assistant_error(_error_message(text)) is True


@pytest.mark.parametrize(
    "text",
    [
        "429 quota exceeded",
        "Incorrect API key provided: sk-invalid",
        "AuthenticationError: invalid x-api-key",
        '{"error":{"type":"invalid_request_error","message":"Unrecognized request argument supplied: temperature"}}',
    ],
)
def test_is_retryable_assistant_error_false_for_non_transient_provider_messages(text):
    assert is_retryable_assistant_error(_error_message(text)) is False


def test_non_retryable_limit_pattern_beats_explicit_retry_guidance():
    message = _error_message("429 quota exceeded; please retry your request later")
    assert is_retryable_assistant_error(message) is False


async def test_retry_assistant_call_accepts_synchronous_callbacks():
    events = []
    calls = 0

    async def produce():
        nonlocal calls
        calls += 1
        if calls == 1:
            return _error_message("request timed out")
        return AssistantMessage(stop_reason="stop")

    def on_retry_scheduled(attempt, max_attempts, delay_ms, error_message):
        events.append(("scheduled", attempt, max_attempts, delay_ms, error_message))

    def on_retry_attempt_start():
        events.append(("attempt-start",))

    def on_retry_finished(success, attempt, final_error):
        events.append(("finished", success, attempt, final_error))

    policy = RetryPolicy(enabled=True, max_retries=2, base_delay_ms=1)
    result = await retry_assistant_call(
        produce,
        policy,
        None,
        RetryCallbacks(
            on_retry_scheduled=on_retry_scheduled,
            on_retry_attempt_start=on_retry_attempt_start,
            on_retry_finished=on_retry_finished,
        ),
    )

    assert result.stop_reason == "stop"
    assert calls == 2
    assert events == [
        ("scheduled", 1, 2, 1, "request timed out"),
        ("attempt-start",),
        ("finished", True, 1, None),
    ]


async def test_retry_assistant_call_reports_aborted_retried_call_as_unsuccessful():
    calls = 0
    finished = []

    async def produce():
        nonlocal calls
        calls += 1
        if calls == 1:
            return _error_message("terminated")
        return AssistantMessage(stop_reason="aborted")

    async def on_retry_finished(success, attempt, final_error):
        finished.append((success, attempt, final_error))

    result = await retry_assistant_call(
        produce,
        RetryPolicy(enabled=True, max_retries=3, base_delay_ms=1),
        None,
        RetryCallbacks(on_retry_finished=on_retry_finished),
    )

    assert result.stop_reason == "aborted"
    assert calls == 2
    assert finished == [(False, 1, None)]


async def test_retry_assistant_call_reports_final_error_when_retry_budget_is_exhausted():
    calls = 0
    finished = []

    async def produce():
        nonlocal calls
        calls += 1
        return _error_message("terminated")

    async def on_retry_finished(success, attempt, final_error):
        finished.append((success, attempt, final_error))

    result = await retry_assistant_call(
        produce,
        RetryPolicy(enabled=True, max_retries=2, base_delay_ms=1),
        None,
        RetryCallbacks(on_retry_finished=on_retry_finished),
    )

    assert result.stop_reason == "error"
    assert calls == 3
    assert finished == [(False, 2, "terminated")]


async def test_retry_assistant_call_completes_backoff_with_signal_and_retries():
    calls = 0
    signal = AbortSignal()

    async def produce():
        nonlocal calls
        calls += 1
        if calls == 1:
            return _error_message("connection error")
        return AssistantMessage(stop_reason="stop")

    result = await retry_assistant_call(
        produce,
        RetryPolicy(enabled=True, max_retries=2, base_delay_ms=1),
        signal,
    )

    assert result.stop_reason == "stop"
    assert calls == 2


async def test_retry_assistant_call_aborts_while_backoff_sleep_is_running():
    calls = 0
    signal = AbortSignal()
    finished = []

    async def produce():
        nonlocal calls
        calls += 1
        return _error_message("terminated")

    async def on_retry_finished(success, attempt, final_error):
        finished.append((success, attempt, final_error))

    # `on_retry_scheduled` fires immediately before the backoff sleep starts, so
    # scheduling the abort with `call_soon` lands it while `_sleep` is pending on
    # its abort race -- the same instant a timed task was guessing at, minus the
    # dependence on wall-clock scheduling under parallel test load.
    async def on_retry_scheduled(*_args):
        asyncio.get_running_loop().call_soon(signal.abort)

    result = await retry_assistant_call(
        produce,
        RetryPolicy(enabled=True, max_retries=5, base_delay_ms=1000),
        signal,
        RetryCallbacks(on_retry_finished=on_retry_finished, on_retry_scheduled=on_retry_scheduled),
    )

    assert result.stop_reason == "aborted"
    assert result.error_message is None
    assert calls == 1
    assert finished == [(False, 1, "terminated")]


# --------------------------------------------------------------------------
# Ported from `packages/ai/test/retry.test.ts`
# --------------------------------------------------------------------------

OPENAI_EXPLICIT_RETRY_MESSAGE = (
    "An error occurred while processing your request. You can retry your request, or contact us "
    "through our help center at help.openai.com if the error persists. Please include the request "
    "ID req_******** in your message."
)
BEDROCK_EXPLICIT_RETRY_MESSAGE = (
    '{"message":"The system encountered an unexpected error during processing. Try your request again."}'
)
NVIDIA_NIM_RESOURCE_EXHAUSTED_MESSAGE = "ResourceExhausted: Worker local total request limit reached (288/48)"
BUN_FETCH_SOCKET_CLOSED_MESSAGE = (
    "The socket connection was closed unexpectedly. For more information, pass `verbose: true` in "
    "the second argument to fetch()"
)
OPENAI_RESPONSES_EARLY_EOF_MESSAGE = "OpenAI Responses stream ended before a terminal response event"
WRAPPED_DNS_LOOKUP_ERROR = (
    "The pending stream has been canceled (caused by: getaddrinfo ENOTFOUND bedrock-runtime.us-east-1.amazonaws.com)"
)


def _faux_error(text: str):
    return faux_assistant_message("", stop_reason="error", error_message=text)


def test_ts_matches_explicit_provider_retry_guidance():
    assert is_retryable_assistant_error(_faux_error(OPENAI_EXPLICIT_RETRY_MESSAGE)) is True
    assert is_retryable_assistant_error(_faux_error(BEDROCK_EXPLICIT_RETRY_MESSAGE)) is True
    assert is_retryable_assistant_error(_faux_error(NVIDIA_NIM_RESOURCE_EXHAUSTED_MESSAGE)) is True


def test_ts_matches_bun_fetch_socket_drop_wording():
    assert is_retryable_assistant_error(_faux_error(BUN_FETCH_SOCKET_CLOSED_MESSAGE)) is True


def test_ts_matches_upstream_request_buffer_exhaustion_wording():
    assert is_retryable_assistant_error(_faux_error("Error: exceeded request buffer limit while retrying upstream"))


@pytest.mark.parametrize(
    "error_message",
    [
        WRAPPED_DNS_LOOKUP_ERROR,
        "connect ENOTFOUND api.example.com",
        "EAI_AGAIN api.example.com",
        "getaddrinfo failed for api.example.com",
    ],
)
def test_ts_matches_dns_transport_failure_wording(error_message):
    assert is_retryable_assistant_error(_faux_error(error_message)) is True


def test_ts_matches_openai_responses_streams_that_end_before_terminal_events():
    assert is_retryable_assistant_error(_faux_error(OPENAI_RESPONSES_EARLY_EOF_MESSAGE)) is True


def test_ts_keeps_provider_limit_errors_non_retryable():
    assert is_retryable_assistant_error(_faux_error("429 quota exceeded")) is False


def test_ts_classifies_assistant_error_messages():
    assert is_retryable_assistant_error(_faux_error("overloaded_error")) is True
    assert is_retryable_assistant_error(_faux_error("524 status code (no body)")) is True
    assert is_retryable_assistant_error(faux_assistant_message("not an error")) is False


_TS_DISABLED = RetryPolicy(enabled=False, max_retries=3, base_delay_ms=0)
_TS_ENABLED = RetryPolicy(enabled=True, max_retries=3, base_delay_ms=0)


async def test_ts_returns_a_successful_response_immediately_without_retrying():
    calls = 0

    async def produce():
        nonlocal calls
        calls += 1
        return faux_assistant_message("ok")

    result = await retry_assistant_call(produce, _TS_ENABLED, None)
    assert result.content == [TextContent(text="ok")]
    assert calls == 1


async def test_ts_does_not_retry_an_aborted_message():
    calls = 0
    scheduled = []

    async def produce():
        nonlocal calls
        calls += 1
        return faux_assistant_message("", stop_reason="aborted")

    result = await retry_assistant_call(
        produce,
        _TS_ENABLED,
        None,
        RetryCallbacks(on_retry_scheduled=lambda *args: scheduled.append(args)),
    )
    assert result.stop_reason == "aborted"
    assert calls == 1
    assert scheduled == []


async def test_ts_does_not_retry_a_non_retryable_error():
    calls = 0
    scheduled = []
    finished = []

    async def produce():
        nonlocal calls
        calls += 1
        return _faux_error("insufficient_quota")

    result = await retry_assistant_call(
        produce,
        _TS_ENABLED,
        None,
        RetryCallbacks(
            on_retry_scheduled=lambda *args: scheduled.append(args),
            on_retry_finished=lambda *args: finished.append(args),
        ),
    )
    assert result.stop_reason == "error"
    assert calls == 1
    assert scheduled == []
    assert finished == []


async def test_ts_retries_a_transient_error_up_to_max_retries_then_returns_the_final_error():
    calls = 0
    scheduled = []
    finished = []

    async def produce():
        nonlocal calls
        calls += 1
        return _faux_error("terminated")

    result = await retry_assistant_call(
        produce,
        _TS_ENABLED,
        None,
        RetryCallbacks(
            on_retry_scheduled=lambda *args: scheduled.append(args),
            on_retry_finished=lambda *args: finished.append(args),
        ),
    )
    assert result.stop_reason == "error"
    assert calls == 4  # 1 initial + 3 retries
    assert len(scheduled) == 3
    assert finished == [(False, 3, "terminated")]


async def test_ts_stops_retrying_once_a_call_succeeds():
    calls = 0
    finished = []

    async def produce():
        nonlocal calls
        calls += 1
        if calls < 3:
            return _faux_error("terminated")
        return faux_assistant_message("recovered")

    result = await retry_assistant_call(
        produce,
        _TS_ENABLED,
        None,
        RetryCallbacks(on_retry_finished=lambda *args: finished.append(args)),
    )
    assert result.content == [TextContent(text="recovered")]
    assert calls == 3
    assert finished == [(True, 2, None)]


async def test_ts_reports_an_aborted_retried_call_as_unsuccessful():
    calls = 0
    finished = []

    async def produce():
        nonlocal calls
        calls += 1
        if calls == 1:
            return _faux_error("terminated")
        return faux_assistant_message("", stop_reason="aborted")

    result = await retry_assistant_call(
        produce,
        _TS_ENABLED,
        None,
        RetryCallbacks(on_retry_finished=lambda *args: finished.append(args)),
    )
    assert result.stop_reason == "aborted"
    assert calls == 2
    assert finished == [(False, 1, None)]


async def test_ts_does_not_retry_when_policy_is_disabled():
    calls = 0
    scheduled = []
    finished = []

    async def produce():
        nonlocal calls
        calls += 1
        return _faux_error("terminated")

    result = await retry_assistant_call(
        produce,
        _TS_DISABLED,
        None,
        RetryCallbacks(
            on_retry_scheduled=lambda *args: scheduled.append(args),
            on_retry_finished=lambda *args: finished.append(args),
        ),
    )
    assert result.stop_reason == "error"
    assert calls == 1
    assert scheduled == []
    assert finished == []


async def test_ts_emits_on_retry_attempt_start_after_backoff_before_each_retried_call():
    events = []
    calls = 0

    async def produce():
        nonlocal calls
        events.append(f"produce:{calls}")
        calls += 1
        if calls < 3:
            return _faux_error("terminated")
        return faux_assistant_message("recovered")

    def on_retry_scheduled(attempt, _max_attempts, _delay_ms, _error_message):
        events.append(f"retry:{attempt}")

    def on_retry_attempt_start():
        events.append("attempt-start")

    result = await retry_assistant_call(
        produce,
        _TS_ENABLED,
        None,
        RetryCallbacks(on_retry_scheduled=on_retry_scheduled, on_retry_attempt_start=on_retry_attempt_start),
    )
    assert result.content == [TextContent(text="recovered")]
    assert events == [
        "produce:0",
        "retry:1",
        "attempt-start",
        "produce:1",
        "retry:2",
        "attempt-start",
        "produce:2",
    ]


async def test_ts_aborts_backoff_sleep_via_signal_and_reports_it_as_unsuccessful():
    signal = AbortSignal()
    calls = 0
    finished = []

    async def produce():
        nonlocal calls
        calls += 1
        return _faux_error("terminated")

    # TS aborts from a fake timer inside the backoff. `call_soon` from
    # `on_retry_scheduled` is the asyncio equivalent: it fires once `_sleep` is
    # already waiting, with no real delay to be starved by parallel load.
    async def on_retry_scheduled(*_args):
        asyncio.get_running_loop().call_soon(signal.abort)

    policy = RetryPolicy(enabled=True, max_retries=5, base_delay_ms=10_000)
    result = await retry_assistant_call(
        produce,
        policy,
        signal,
        RetryCallbacks(
            on_retry_finished=lambda *args: finished.append(args),
            on_retry_scheduled=on_retry_scheduled,
        ),
    )
    assert result.stop_reason == "aborted"
    assert result.error_message is None
    assert calls == 1
    assert finished == [(False, 1, "terminated")]
