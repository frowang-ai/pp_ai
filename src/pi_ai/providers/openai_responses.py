"""OpenAI Responses API provider factory.

Python port of `packages/ai/src/providers/openai.ts`. The TypeScript catalog is
generated from live provider metadata (`scripts/generate-models.ts` plus
`providers/data/openai.json`, gitignored and not checked into this repository);
this port ships a small hand-written catalog of three current OpenAI models
reachable through the Responses API (`/v1/responses`), with ids, context
windows, output limits and prices taken from OpenAI's published model and
pricing pages (https://platform.openai.com/docs/pricing).

Named distinctly from :func:`pi_ai.providers.openai_compatible.openai_provider`
(Chat Completions) so both APIs can coexist for the ``openai`` provider id's
model catalogs without colliding.
"""

from __future__ import annotations

from ..api import openai_responses
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..registry import Provider, create_provider
from ..types import Model, ModelCost

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

# gpt-5.1/gpt-5/gpt-5-mini/gpt-5-nano are always-reasoning models; "off" maps to
# None so build_params omits the `reasoning` field entirely rather than sending
# an effort level the model would reject.
_REASONING_THINKING_LEVEL_MAP = {
    "off": None,
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
}


def _model(
    model_id: str,
    name: str,
    context_window: int,
    max_tokens: int,
    cost: ModelCost,
    compat: dict[str, object] | None = None,
) -> Model:
    return Model(
        id=model_id,
        name=name,
        api="openai-responses",
        provider="",
        base_url="",
        reasoning=True,
        thinking_level_map=dict(_REASONING_THINKING_LEVEL_MAP),
        input=["text", "image"],
        cost=cost,
        context_window=context_window,
        max_tokens=max_tokens,
        compat=compat or {},
    )


OPENAI_RESPONSES_MODELS = [
    _model(
        "gpt-5.1",
        "GPT-5.1",
        context_window=400_000,
        max_tokens=128_000,
        cost=ModelCost(input=1.25, output=10.0, cache_read=0.125),
        # Generated OpenAI catalogs enable strict mode and grammar tools explicitly.
        compat={"supportsStrictMode": True, "supportsOpenAIGrammarTools": True},
    ),
    _model(
        "gpt-5-mini",
        "GPT-5 mini",
        context_window=400_000,
        max_tokens=128_000,
        cost=ModelCost(input=0.25, output=2.0, cache_read=0.025),
        compat={"supportsStrictMode": True, "supportsOpenAIGrammarTools": True},
    ),
    _model(
        "gpt-5-nano",
        "GPT-5 nano",
        context_window=400_000,
        max_tokens=128_000,
        cost=ModelCost(input=0.05, output=0.4, cache_read=0.005),
        compat={"supportsStrictMode": True, "supportsOpenAIGrammarTools": True},
    ),
]


def openai_responses_provider() -> Provider:
    """Build the built-in OpenAI Responses provider, authenticating via OPENAI_API_KEY.

    Registered under the ``openai-responses`` provider id (distinct from the
    Chat Completions ``openai`` provider in
    :func:`pi_ai.providers.openai_compatible.openai_provider`) so both can be
    added to the same :class:`~pi_ai.registry.Models` registry without
    colliding.
    """
    return create_provider(
        id="openai-responses",
        name="OpenAI (Responses)",
        auth=ProviderAuth(api_key=env_api_key_auth("OpenAI API key", [OPENAI_API_KEY_ENV])),
        api=openai_responses,
        models=OPENAI_RESPONSES_MODELS,
        base_url="https://api.openai.com/v1",
    )
