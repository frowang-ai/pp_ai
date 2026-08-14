"""Tests for `pi_ai.legacy_api_aliases` — the deprecated flat-name aliases
for each api module's `stream`/`stream_simple` functions.

Ported concept from `packages/ai/src/legacy-api-aliases.ts`; there is no
dedicated TypeScript test file for this module (it's exercised indirectly),
so this test simply confirms every alias resolves to the correct underlying
api module function.
"""

from __future__ import annotations

from pi_ai import legacy_api_aliases as aliases
from pi_ai.api import (
    anthropic_messages,
    azure_openai_responses,
    google_generative_ai,
    google_vertex,
    mistral_conversations,
    openai_completions,
    openai_responses,
    pi_messages,
)


def test_aliases_reference_the_underlying_api_module_functions() -> None:
    assert aliases.stream_anthropic is anthropic_messages.stream
    assert aliases.stream_simple_anthropic is anthropic_messages.stream_simple

    assert aliases.stream_azure_openai_responses is azure_openai_responses.stream
    assert aliases.stream_simple_azure_openai_responses is azure_openai_responses.stream_simple

    assert aliases.stream_google is google_generative_ai.stream
    assert aliases.stream_simple_google is google_generative_ai.stream_simple

    assert aliases.stream_google_vertex is google_vertex.stream
    assert aliases.stream_simple_google_vertex is google_vertex.stream_simple

    assert aliases.stream_mistral is mistral_conversations.stream
    assert aliases.stream_simple_mistral is mistral_conversations.stream_simple

    assert aliases.stream_openai_completions is openai_completions.stream
    assert aliases.stream_simple_openai_completions is openai_completions.stream_simple

    assert aliases.stream_openai_responses is openai_responses.stream
    assert aliases.stream_simple_openai_responses is openai_responses.stream_simple

    assert aliases.stream_pi_messages is pi_messages.stream
    assert aliases.stream_simple_pi_messages is pi_messages.stream_simple


def test_bedrock_and_codex_aliases_are_not_generated() -> None:
    assert not hasattr(aliases, "stream_bedrock")
    assert not hasattr(aliases, "stream_openai_codex_responses")
