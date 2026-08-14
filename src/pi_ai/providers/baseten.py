"""Baseten provider factory.

Python port of `packages/ai/src/providers/baseten.ts`. The model list comes from the
generated catalog shard `pi_ai/providers/data/baseten.json`, which is the Python
equivalent of TypeScript's generated `providers/baseten.models.ts` module (both
are produced by `packages/ai/scripts/generate-models.ts`).
"""

from __future__ import annotations

from ..api import openai_completions
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

BASETEN_MODELS: list[Model] = load_models("baseten")


def baseten_provider() -> Provider:
    """Build the built-in Baseten provider."""
    return create_provider(
        id="baseten",
        name="Baseten",
        auth=ProviderAuth(api_key=env_api_key_auth("Baseten API key", ["BASETEN_API_KEY"])),
        api=openai_completions,
        models=BASETEN_MODELS,
        base_url="https://inference.baseten.co/v1",
    )
