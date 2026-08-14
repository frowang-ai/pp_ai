"""Deprecated aliases for the per-API `stream`/`stream_simple` functions.

Python port of `packages/ai/src/legacy-api-aliases.ts`. TypeScript names each
alias after the api module (`streamAnthropic`, `streamSimpleAnthropic`, ...);
this module keeps that mapping so code migrating from the old flat import
surface has a direct equivalent. New code should import `stream`/`stream_simple`
directly from the api module (e.g. `pi_ai.api.anthropic_messages.stream`).

`bedrock-converse-stream` and `openai-codex-responses` are out of scope for
this port (see the package README), so no aliases are generated for them.
"""

from __future__ import annotations

from .api import (
    anthropic_messages,
    azure_openai_responses,
    google_generative_ai,
    google_vertex,
    mistral_conversations,
    openai_completions,
    openai_responses,
    pi_messages,
)

# anthropic-messages
stream_anthropic = anthropic_messages.stream
"""Deprecated. Use `pi_ai.api.anthropic_messages.stream`."""
stream_simple_anthropic = anthropic_messages.stream_simple
"""Deprecated. Use `pi_ai.api.anthropic_messages.stream_simple`."""

# azure-openai-responses
stream_azure_openai_responses = azure_openai_responses.stream
"""Deprecated. Use `pi_ai.api.azure_openai_responses.stream`."""
stream_simple_azure_openai_responses = azure_openai_responses.stream_simple
"""Deprecated. Use `pi_ai.api.azure_openai_responses.stream_simple`."""

# google-generative-ai
stream_google = google_generative_ai.stream
"""Deprecated. Use `pi_ai.api.google_generative_ai.stream`."""
stream_simple_google = google_generative_ai.stream_simple
"""Deprecated. Use `pi_ai.api.google_generative_ai.stream_simple`."""

# google-vertex
stream_google_vertex = google_vertex.stream
"""Deprecated. Use `pi_ai.api.google_vertex.stream`."""
stream_simple_google_vertex = google_vertex.stream_simple
"""Deprecated. Use `pi_ai.api.google_vertex.stream_simple`."""

# mistral-conversations
stream_mistral = mistral_conversations.stream
"""Deprecated. Use `pi_ai.api.mistral_conversations.stream`."""
stream_simple_mistral = mistral_conversations.stream_simple
"""Deprecated. Use `pi_ai.api.mistral_conversations.stream_simple`."""

# openai-completions
stream_openai_completions = openai_completions.stream
"""Deprecated. Use `pi_ai.api.openai_completions.stream`."""
stream_simple_openai_completions = openai_completions.stream_simple
"""Deprecated. Use `pi_ai.api.openai_completions.stream_simple`."""

# openai-responses
stream_openai_responses = openai_responses.stream
"""Deprecated. Use `pi_ai.api.openai_responses.stream`."""
stream_simple_openai_responses = openai_responses.stream_simple
"""Deprecated. Use `pi_ai.api.openai_responses.stream_simple`."""

# pi-messages
stream_pi_messages = pi_messages.stream
"""Deprecated. Use `pi_ai.api.pi_messages.stream`."""
stream_simple_pi_messages = pi_messages.stream_simple
"""Deprecated. Use `pi_ai.api.pi_messages.stream_simple`."""

__all__ = [
    "stream_anthropic",
    "stream_azure_openai_responses",
    "stream_google",
    "stream_google_vertex",
    "stream_mistral",
    "stream_openai_completions",
    "stream_openai_responses",
    "stream_pi_messages",
    "stream_simple_anthropic",
    "stream_simple_azure_openai_responses",
    "stream_simple_google",
    "stream_simple_google_vertex",
    "stream_simple_mistral",
    "stream_simple_openai_completions",
    "stream_simple_openai_responses",
    "stream_simple_pi_messages",
]
