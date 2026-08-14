"""Extra coverage tests for pi_ai.utils.overflow (lines 118-120, 145)."""

from __future__ import annotations

from pi_ai import AssistantMessage, Usage
from pi_ai.utils.overflow import is_context_overflow, is_recoverable_length


def _msg(
    stop_reason: str, *, input: int = 0, cache_read: int = 0, output: int = 0, error_message: str | None = None
) -> AssistantMessage:
    return AssistantMessage(
        api="openai-completions",
        provider="test",
        model="test-model",
        content=[],
        usage=Usage(input=input, output=output, cache_read=cache_read, total_tokens=input + cache_read + output),
        stop_reason=stop_reason,
        error_message=error_message,
    )


# Lines 118-120: silent overflow (stop_reason="stop", input > context_window)
def test_silent_overflow_detected_when_input_exceeds_context_window() -> None:
    # z.ai style: stop_reason is "stop" but tokens used exceed context window
    msg = _msg("stop", input=200_000, cache_read=5_000)
    assert is_context_overflow(msg, 200_000) is True


def test_silent_overflow_not_detected_when_within_context_window() -> None:
    msg = _msg("stop", input=100_000)
    assert is_context_overflow(msg, 200_000) is False


def test_silent_overflow_not_checked_without_context_window() -> None:
    # context_window=None should skip Case 2
    msg = _msg("stop", input=999_999)
    assert is_context_overflow(msg, None) is False


# Line 145: is_recoverable_length returns False when desired_max_output is 0
def test_recoverable_length_false_when_desired_max_output_is_zero() -> None:
    msg = _msg("length", output=0)
    assert is_recoverable_length(msg, 0) is False


def test_recoverable_length_false_for_non_length_stop_reason() -> None:
    msg = _msg("stop", output=10)
    assert is_recoverable_length(msg, 1024) is False
