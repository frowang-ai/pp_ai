"""Moonshot AI provider factory.

Python port of `packages/ai/src/providers/moonshotai.ts`. The model list comes from the
generated catalog shard `pi_ai/providers/data/moonshotai.json`, which is the Python
equivalent of TypeScript's generated `providers/moonshotai.models.ts` module (both
are produced by `packages/ai/scripts/generate-models.ts`).
"""

from __future__ import annotations

from ..api import openai_completions
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

MOONSHOTAI_MODELS: list[Model] = load_models("moonshotai")


def moonshotai_provider() -> Provider:
    """Build the built-in Moonshot AI provider."""
    return create_provider(
        id="moonshotai",
        name="Moonshot AI",
        auth=ProviderAuth(api_key=env_api_key_auth("Moonshot AI API key", ["MOONSHOT_API_KEY"])),
        api=openai_completions,
        models=MOONSHOTAI_MODELS,
        base_url="https://api.moonshot.ai/v1",
    )
