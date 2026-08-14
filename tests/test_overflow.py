"""Tests ported from packages/ai/test/overflow.ts and context-overflow.test.ts."""

from __future__ import annotations

from pi_ai import AssistantMessage, Usage
from pi_ai.utils.overflow import is_context_overflow, is_recoverable_length


def make_error_message(error_message: str) -> AssistantMessage:
    return AssistantMessage(
        api="openai-completions",
        provider="ollama",
        model="qwen3.5:35b",
        content=[],
        usage=Usage(),
        stop_reason="error",
        error_message=error_message,
    )


def make_length_stop_message(
    *,
    input: int,
    cache_read: int,
    output: int,
    cache_write: int = 0,
    api: str = "openai-completions",
    provider: str = "test-provider",
    model: str = "test-model",
) -> AssistantMessage:
    usage = Usage(
        input=input,
        output=output,
        cache_read=cache_read,
        cache_write=cache_write,
        total_tokens=input + cache_read + cache_write + output,
    )
    return AssistantMessage(
        api=api,
        provider=provider,
        model=model,
        content=[],
        usage=usage,
        stop_reason="length",
    )


def test_detects_explicit_ollama_prompt_too_long_errors() -> None:
    message = make_error_message("400 `prompt too long; exceeded max context length by 100918 tokens`")
    assert is_context_overflow(message, 32768) is True


def test_detects_together_ai_context_length_errors() -> None:
    message = make_error_message(
        "400 The input (516368 tokens) is longer than the model's context length (262144 tokens)."
    )
    assert is_context_overflow(message, 262144) is True


def test_detects_litellm_wrapped_openai_maximum_context_length_errors() -> None:
    message = make_error_message(
        "Error: 503 litellm.ServiceUnavailableError: litellm.MidStreamFallbackError: "
        "litellm.APIConnectionError: APIConnectionError: OpenAIException - Requested token "
        "count exceeds the model's maximum context length of 131072 tokens."
    )
    assert is_context_overflow(message, 131072) is True


def test_detects_openai_compatible_parenthesized_maximum_context_length_errors() -> None:
    message = make_error_message("Error: 400 Input length (265330) exceeds model's maximum context length (262144).")
    assert is_context_overflow(message, 262144) is True


def test_detects_openrouter_poolside_maximum_allowed_input_length_errors() -> None:
    message = make_error_message(
        "Provider returned error: Input length 131393 exceeds the maximum allowed input length of 131040 tokens."
    )
    assert is_context_overflow(message, 131072) is True


def test_detects_ds4_configured_context_size_errors() -> None:
    message = make_error_message("400 Prompt has 256468 tokens, but the configured context size is 256000 tokens")
    assert is_context_overflow(message, 256000) is True

    comma_message = make_error_message("Prompt has 5,958,968 tokens, but the configured context size is 256,000 tokens")
    assert is_context_overflow(comma_message, 256000) is True


def test_does_not_treat_generic_non_overflow_ollama_errors_as_overflow() -> None:
    message = make_error_message("500 `model runner crashed unexpectedly`")
    assert is_context_overflow(message, 32768) is False


def test_does_not_treat_bedrock_throttling_too_many_tokens_as_overflow() -> None:
    # Bedrock returns this for HTTP 429 rate limiting, NOT context overflow.
    message = make_error_message("Throttling error: Too many tokens, please wait before trying again.")
    assert is_context_overflow(message, 200000) is False


def test_does_not_treat_bedrock_service_unavailable_as_overflow() -> None:
    message = make_error_message("Service unavailable: The service is temporarily unavailable.")
    assert is_context_overflow(message, 200000) is False


def test_does_not_treat_generic_rate_limit_errors_as_overflow() -> None:
    message = make_error_message("Rate limit exceeded, please retry after 30 seconds.")
    assert is_context_overflow(message, 200000) is False


def test_does_not_treat_http_429_style_errors_as_overflow() -> None:
    message = make_error_message("Too many requests. Please slow down.")
    assert is_context_overflow(message, 200000) is False


def test_detects_xiaomi_style_overflow_length_stop_zero_output_filled_context() -> None:
    message = make_length_stop_message(
        input=58,
        cache_read=1_048_512,
        output=0,
        provider="xiaomi",
        model="mimo-v2.5-pro",
    )
    assert is_context_overflow(message, 1_048_576) is True


def test_treats_a_length_stop_below_the_desired_output_limit_as_recoverable() -> None:
    message = make_length_stop_message(
        input=3,
        cache_read=253_584,
        cache_write=25_554,
        output=16,
        api="openai-responses",
        provider="openai",
        model="gpt-5.6-sol",
    )
    assert is_recoverable_length(message, 128_000) is True


def test_does_not_recover_a_length_stop_that_reached_the_desired_output_limit() -> None:
    message = make_length_stop_message(input=4062, cache_read=0, output=1024)
    assert is_recoverable_length(message, 1024) is False


def test_treats_zero_output_length_stops_as_recoverable_without_context_metadata() -> None:
    message = make_length_stop_message(input=100, cache_read=0, output=0)
    assert is_recoverable_length(message, 128_000) is True


def test_does_not_treat_normal_length_stops_with_output_as_context_overflow() -> None:
    message = make_length_stop_message(input=1000, cache_read=0, output=4096)
    assert is_context_overflow(message, 200_000) is False


def test_does_not_treat_zero_output_length_stops_far_below_context_as_context_overflow() -> None:
    message = make_length_stop_message(input=100, cache_read=0, output=0)
    assert is_context_overflow(message, 200_000) is False
