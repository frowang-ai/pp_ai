"""Ant Ling provider factory.

Python port of `packages/ai/src/providers/ant-ling.ts`. The model list comes from the
generated catalog shard `pi_ai/providers/data/ant-ling.json`, which is the Python
equivalent of TypeScript's generated `providers/ant-ling.models.ts` module (both
are produced by `packages/ai/scripts/generate-models.ts`).
"""

from __future__ import annotations

from ..api import openai_completions
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

ANT_LING_MODELS: list[Model] = load_models("ant-ling")


def ant_ling_provider() -> Provider:
    """Build the built-in Ant Ling provider."""
    return create_provider(
        id="ant-ling",
        name="Ant Ling",
        auth=ProviderAuth(api_key=env_api_key_auth("Ant Ling API key", ["ANT_LING_API_KEY"])),
        api=openai_completions,
        models=ANT_LING_MODELS,
        base_url="https://api.ant-ling.com/v1",
    )
