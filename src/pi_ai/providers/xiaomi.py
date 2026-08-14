"""Xiaomi provider factory.

Python port of `packages/ai/src/providers/xiaomi.ts`. The model list comes from the
generated catalog shard `pi_ai/providers/data/xiaomi.json`, which is the Python
equivalent of TypeScript's generated `providers/xiaomi.models.ts` module (both
are produced by `packages/ai/scripts/generate-models.ts`).
"""

from __future__ import annotations

from ..api import openai_completions
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

XIAOMI_MODELS: list[Model] = load_models("xiaomi")


def xiaomi_provider() -> Provider:
    """Build the built-in Xiaomi provider."""
    return create_provider(
        id="xiaomi",
        name="Xiaomi",
        auth=ProviderAuth(api_key=env_api_key_auth("Xiaomi API key", ["XIAOMI_API_KEY"])),
        api=openai_completions,
        models=XIAOMI_MODELS,
        base_url="https://api.xiaomimimo.com/v1",
    )
