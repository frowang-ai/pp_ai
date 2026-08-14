"""Python port of `packages/ai/test/context-estimate.test.ts`."""

from __future__ import annotations

from pi_ai.api.simple_options import build_base_options
from pi_ai.types import (
    AssistantMessage,
    Context,
    Model,
    ModelCost,
    TextContent,
    Usage,
    UserMessage,
)
from pi_ai.utils.estimate import ContextUsageEstimate, estimate_context_tokens


def create_usage(total_tokens: int) -> Usage:
    return Usage(input=total_tokens, output=0, cache_read=0, cache_write=0, total_tokens=total_tokens)


def create_assistant(timestamp: int, total_tokens: int) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text="kept")],
        api="openai-responses",
        provider="openai",
        model="test-model",
        usage=create_usage(total_tokens),
        stop_reason="stop",
        timestamp=timestamp,
    )


MODEL = Model(
    id="test-model",
    name="Test Model",
    api="openai-responses",
    provider="openai",
    base_url="https://api.openai.com/v1",
    reasoning=False,
    input=["text"],
    cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    context_window=10_000,
    max_tokens=8_000,
)


def test_ignores_stale_assistant_usage_after_a_newer_message_is_inserted_before_it():
    context = Context(
        system_prompt="system",
        messages=[
            UserMessage(content="summary", timestamp=200),
            create_assistant(100, 9_500),
            UserMessage(content="x" * 4_000, timestamp=300),
        ],
    )

    assert estimate_context_tokens(context) == ContextUsageEstimate(
        tokens=1_005,
        usage_tokens=0,
        trailing_tokens=1_005,
        last_usage_index=None,
    )
    assert build_base_options(MODEL, context).max_tokens == 4_899


def test_uses_assistant_usage_again_after_a_response_to_the_inserted_context():
    context = Context(
        messages=[
            UserMessage(content="summary", timestamp=200),
            create_assistant(100, 9_500),
            UserMessage(content="new prompt", timestamp=300),
            create_assistant(400, 2_000),
            UserMessage(content="tail", timestamp=500),
        ],
    )

    assert estimate_context_tokens(context) == ContextUsageEstimate(
        tokens=2_001,
        usage_tokens=2_000,
        trailing_tokens=1,
        last_usage_index=3,
    )
