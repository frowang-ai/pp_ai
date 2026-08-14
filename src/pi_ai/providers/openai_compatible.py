"""Generic OpenAI Chat Completions provider factory.

Python port of the shared shape of the OpenAI-compatible provider factories in
`packages/ai/src/providers/` (`groq.ts`, `cerebras.ts`, `deepseek.ts`,
`together.ts`, ... all wrap `createProvider` around `openAICompletionsApi()`).

The built-in providers themselves now live in one module each, matching the
TypeScript layout, and read the generated catalog:
:mod:`pi_ai.providers.groq`, :mod:`pi_ai.providers.cerebras`,
:mod:`pi_ai.providers.deepseek`, and so on. What remains here is
:func:`openai_compatible_provider`, which has no TypeScript counterpart: it
lets a caller point pi at any OpenAI Chat Completions compatible endpoint
(a local llama.cpp server, an internal gateway) with a hand-written catalog.

``OPENAI_COMPLETIONS_MODELS`` stays as a tiny hand-written catalog for that
purpose. The real OpenAI catalog is :data:`pi_ai.providers.openai.OPENAI_MODELS`,
which is generated and served over the Responses API, as in `openai.ts`.
"""

from __future__ import annotations

from ..api import openai_completions
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..registry import Provider, create_provider
from ..types import Model, ModelCost


def _model(
    model_id: str,
    name: str,
    context_window: int,
    max_tokens: int,
    cost: ModelCost,
    reasoning: bool = False,
    input_modalities: list[str] | None = None,
    thinking_level_map: dict[str, str | None] | None = None,
) -> Model:
    return Model(
        id=model_id,
        name=name,
        api="openai-completions",
        provider="",
        base_url="",
        reasoning=reasoning,
        thinking_level_map=thinking_level_map or {},
        input=input_modalities or ["text"],
        cost=cost,
        context_window=context_window,
        max_tokens=max_tokens,
    )


def openai_compatible_provider(
    provider_id: str,
    name: str,
    base_url: str,
    env_vars: list[str],
    models: list[Model],
    headers: dict[str, str] | None = None,
) -> Provider:
    """Build a provider for any OpenAI Chat Completions compatible endpoint."""
    return create_provider(
        id=provider_id,
        name=name,
        auth=ProviderAuth(api_key=env_api_key_auth(f"{name} API key", env_vars)),
        api=openai_completions,
        models=models,
        base_url=base_url,
        headers=headers,
    )


OPENAI_COMPLETIONS_MODELS = [
    _model(
        "gpt-4o-mini",
        "GPT-4o mini",
        context_window=128_000,
        max_tokens=16_384,
        cost=ModelCost(input=0.15, output=0.6, cache_read=0.075),
        input_modalities=["text", "image"],
    ),
    _model(
        "gpt-4o",
        "GPT-4o",
        context_window=128_000,
        max_tokens=16_384,
        cost=ModelCost(input=2.5, output=10.0, cache_read=1.25),
        input_modalities=["text", "image"],
    ),
]
