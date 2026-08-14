"""Tests for `pi_ai.utils.overflow`, ported from
`packages/ai/test/overflow.test.ts` and extended to cover every provider
pattern in `packages/ai/src/utils/overflow.ts`.

The module is a pure function of an `AssistantMessage`; nothing here touches
the filesystem, the network or the environment.
"""

from __future__ import annotations

import pytest
from pi_ai.types import AssistantMessage, Cost, Usage
from pi_ai.utils.overflow import is_context_overflow, is_recoverable_length


def usage(input: int = 0, output: int = 0, cache_read: int = 0, cache_write: int = 0) -> Usage:
    return Usage(
        input=input,
        output=output,
        cache_read=cache_read,
        cache_write=cache_write,
        total_tokens=input + output + cache_read + cache_write,
        cost=Cost(),
    )


def error_message(text: str) -> AssistantMessage:
    return AssistantMessage(
        api="openai-completions",
        provider="ollama",
        model="qwen3.5:35b",
        content=[],
        usage=usage(),
        stop_reason="error",
        error_message=text,
        timestamp=1,
    )


def length_stop(input: int, cache_read: int, output: int, cache_write: int = 0) -> AssistantMessage:
    return AssistantMessage(
        api="openai-completions",
        provider="test-provider",
        model="test-model",
        content=[],
        usage=usage(input=input, output=output, cache_read=cache_read, cache_write=cache_write),
        stop_reason="length",
        timestamp=1,
    )


OVERFLOW_ERRORS = [
    "prompt is too long: 213462 tokens > 200000 maximum",
    '413 {"error":{"type":"request_too_large","message":"Request exceeds the maximum size"}}',
    "Input is too long for requested model.",
    "Your input exceeds the context window of this model",
    "Requested token count exceeds the model's maximum context length of 131072 tokens.",
    "Error: 400 Input length (265330) exceeds model's maximum context length (262144).",
    "The input token count (1196265) exceeds the maximum number of tokens allowed (1048575)",
    "This model's maximum prompt length is 131072 but the request contains 537812 tokens",
    "Please reduce the length of the messages or completion",
    "This endpoint's maximum context length is 262144 tokens. However, you requested about 300000",
    "Input length 131393 exceeds the maximum allowed input length of 131040 tokens.",
    "400 The input (516368 tokens) is longer than the model's context length (262144 tokens).",
    "prompt token count of 300000 exceeds the limit of 128000",
    "the request exceeds the available context size, try increasing it",
    "tokens to keep from the initial prompt is greater than the context length",
    "invalid params, context window exceeds limit",
    "Your request exceeded model token limit: 262144 (requested: 300000)",
    "Prompt contains 300000 tokens ... too large for model with 131072 maximum context length",
    "400 Prompt has 256468 tokens, but the configured context size is 256000 tokens",
    "Prompt has 5,958,968 tokens, but the configured context size is 256,000 tokens",
    "model_context_window_exceeded",
    "400 `prompt too long; exceeded max context length by 100918 tokens`",
    "Range of input length should be [1, 129024]",
    "context_length_exceeded",
    "context length exceeded",
    "Request contains too many tokens",
    "token limit exceeded",
    "400 status code (no body)",
    "413 (no body)",
]


@pytest.mark.parametrize("text", OVERFLOW_ERRORS)
def test_detects_provider_overflow_errors(text: str):
    assert is_context_overflow(error_message(text)) is True


@pytest.mark.parametrize(
    "text",
    [
        "500 `model runner crashed unexpectedly`",
        "Throttling error: Too many tokens, please wait before trying again.",
        "Service unavailable: The service is temporarily unavailable.",
        "Rate limit exceeded, please retry after 30 seconds.",
        "Too many requests. Please slow down.",
        "500 internal server error",
        "context length is fine",
    ],
)
def test_does_not_treat_non_overflow_errors_as_overflow(text: str):
    assert is_context_overflow(error_message(text), 200000) is False


def test_non_overflow_pattern_wins_over_an_overflow_match():
    # "Too many tokens" alone matches an overflow pattern; the throttling prefix
    # must veto it.
    assert is_context_overflow(error_message("Too many tokens in request")) is True
    assert is_context_overflow(error_message("Throttling error: Too many tokens in request")) is False


def test_error_stop_without_a_message_is_not_overflow():
    message = error_message("prompt is too long")
    message.error_message = None
    assert is_context_overflow(message, 200000) is False


def test_overflow_message_on_a_non_error_stop_reason_is_ignored():
    message = error_message("prompt is too long")
    message.stop_reason = "stop"
    assert is_context_overflow(message) is False


def test_detects_silent_overflow_when_usage_exceeds_the_context_window():
    message = AssistantMessage(
        api="openai-completions",
        provider="zai",
        model="glm-4.6",
        content=[],
        usage=usage(input=100000, cache_read=101000, output=50),
        stop_reason="stop",
        timestamp=1,
    )
    assert is_context_overflow(message, 200000) is True
    assert is_context_overflow(message, 300000) is False
    # Without a context window there is nothing to compare against.
    assert is_context_overflow(message) is False


def test_usage_exactly_at_the_context_window_is_not_silent_overflow():
    message = AssistantMessage(
        api="openai-completions",
        provider="zai",
        model="glm-4.6",
        content=[],
        usage=usage(input=200000, output=10),
        stop_reason="stop",
        timestamp=1,
    )
    assert is_context_overflow(message, 200000) is False


def test_detects_xiaomi_style_length_stop_overflow():
    message = length_stop(input=58, cache_read=1048512, output=0)
    assert is_context_overflow(message, 1048576) is True


def test_length_stop_overflow_needs_a_context_window():
    message = length_stop(input=58, cache_read=1048512, output=0)
    assert is_context_overflow(message) is False


def test_length_stop_at_the_99_percent_threshold():
    # 99% of 200000 is exactly 198000, and the check is inclusive.
    assert is_context_overflow(length_stop(input=198000, cache_read=0, output=0), 200000) is True
    assert is_context_overflow(length_stop(input=197999, cache_read=0, output=0), 200000) is False


def test_does_not_treat_normal_length_stops_with_output_as_overflow():
    assert is_context_overflow(length_stop(input=1000, cache_read=0, output=4096), 200000) is False
    # Even a filled context is fine as long as the model produced output.
    assert is_context_overflow(length_stop(input=199000, cache_read=0, output=1), 200000) is False


def test_does_not_treat_zero_output_length_stops_far_below_context_as_overflow():
    assert is_context_overflow(length_stop(input=100, cache_read=0, output=0), 200000) is False


def test_recoverable_length_below_the_desired_output_limit():
    message = length_stop(input=3, cache_read=253584, output=16, cache_write=25554)
    assert is_recoverable_length(message, 128000) is True


def test_not_recoverable_when_the_desired_output_limit_was_reached():
    assert is_recoverable_length(length_stop(input=4062, cache_read=0, output=1024), 1024) is False


def test_zero_output_length_stops_are_recoverable_without_context_metadata():
    assert is_recoverable_length(length_stop(input=100, cache_read=0, output=0), 128000) is True


def test_not_recoverable_without_a_positive_desired_output():
    assert is_recoverable_length(length_stop(input=100, cache_read=0, output=0), 0) is False
    assert is_recoverable_length(length_stop(input=100, cache_read=0, output=0), -1) is False


def test_not_recoverable_for_other_stop_reasons():
    message = length_stop(input=100, cache_read=0, output=10)
    message.stop_reason = "stop"
    assert is_recoverable_length(message, 128000) is False
