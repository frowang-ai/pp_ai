"""Z.AI provider factory.

Python port of `packages/ai/src/providers/zai.ts`. The model list comes from the
generated catalog shard `pi_ai/providers/data/zai.json`, which is the Python
equivalent of TypeScript's generated `providers/zai.models.ts` module (both
are produced by `packages/ai/scripts/generate-models.ts`).
"""

from __future__ import annotations

from ..api import openai_completions
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

ZAI_MODELS: list[Model] = load_models("zai")


def zai_provider() -> Provider:
    """Build the built-in Z.AI provider."""
    return create_provider(
        id="zai",
        name="Z.AI",
        auth=ProviderAuth(api_key=env_api_key_auth("Z.AI API key", ["ZAI_API_KEY"])),
        api=openai_completions,
        models=ZAI_MODELS,
        base_url="https://api.z.ai/api/coding/paas/v4",
    )
