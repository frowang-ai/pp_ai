"""Cerebras provider factory.

Python port of `packages/ai/src/providers/cerebras.ts`. The model list comes from the
generated catalog shard `pi_ai/providers/data/cerebras.json`, which is the Python
equivalent of TypeScript's generated `providers/cerebras.models.ts` module (both
are produced by `packages/ai/scripts/generate-models.ts`).
"""

from __future__ import annotations

from ..api import openai_completions
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

CEREBRAS_MODELS: list[Model] = load_models("cerebras")


def cerebras_provider() -> Provider:
    """Build the built-in Cerebras provider."""
    return create_provider(
        id="cerebras",
        name="Cerebras",
        auth=ProviderAuth(api_key=env_api_key_auth("Cerebras API key", ["CEREBRAS_API_KEY"])),
        api=openai_completions,
        models=CEREBRAS_MODELS,
        base_url="https://api.cerebras.ai/v1",
    )
