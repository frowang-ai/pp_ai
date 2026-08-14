"""Mistral provider factory.

Python port of `packages/ai/src/providers/mistral.ts`. The model list comes from the
generated catalog shard `pi_ai/providers/data/mistral.json`, the Python
equivalent of TypeScript's generated `providers/mistral.models.ts` (both are
produced by `packages/ai/scripts/generate-models.ts`).

Model ids `mistral-small-2603`, `mistral-small-latest` and `mistral-medium-3.5`
are load-bearing: `mistral_conversations._uses_reasoning_effort` matches those
exact strings to route reasoning through Mistral's `reasoningEffort` field
instead of `promptMode: "reasoning"`.
"""

from __future__ import annotations

from ..api import mistral_conversations
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

MISTRAL_API_KEY_ENV = "MISTRAL_API_KEY"

MISTRAL_MODELS: list[Model] = load_models("mistral")


def mistral_provider() -> Provider:
    """Build the built-in Mistral provider, authenticating via MISTRAL_API_KEY."""
    return create_provider(
        id="mistral",
        name="Mistral",
        auth=ProviderAuth(
            api_key=env_api_key_auth("Mistral API key", [MISTRAL_API_KEY_ENV]),
        ),
        api=mistral_conversations,
        models=MISTRAL_MODELS,
        base_url="https://api.mistral.ai",
    )
