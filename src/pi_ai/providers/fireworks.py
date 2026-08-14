"""Fireworks provider factory.

Python port of `packages/ai/src/providers/fireworks.ts`. The model list comes
from the generated catalog shard `pi_ai/providers/data/fireworks.json`, the
Python equivalent of TypeScript's generated `providers/fireworks.models.ts`
(both produced by `packages/ai/scripts/generate-models.ts`).

Fireworks serves some models over the Anthropic Messages wire format and the
rest over OpenAI Chat Completions, so the provider dispatches on ``model.api``.
"""

from __future__ import annotations

from ..api import anthropic_messages, openai_completions
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

FIREWORKS_MODELS: list[Model] = load_models("fireworks")


def fireworks_provider() -> Provider:
    """Build the built-in Fireworks provider."""
    return create_provider(
        id="fireworks",
        name="Fireworks",
        auth=ProviderAuth(api_key=env_api_key_auth("Fireworks API key", ["FIREWORKS_API_KEY"])),
        api={
            "anthropic-messages": anthropic_messages,
            "openai-completions": openai_completions,
        },
        models=FIREWORKS_MODELS,
        base_url="https://api.fireworks.ai/inference",
    )
