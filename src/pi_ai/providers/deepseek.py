"""DeepSeek provider factory.

Python port of `packages/ai/src/providers/deepseek.ts`. The model list comes from the
generated catalog shard `pi_ai/providers/data/deepseek.json`, which is the Python
equivalent of TypeScript's generated `providers/deepseek.models.ts` module (both
are produced by `packages/ai/scripts/generate-models.ts`).
"""

from __future__ import annotations

from ..api import openai_completions
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

DEEPSEEK_MODELS: list[Model] = load_models("deepseek")


def deepseek_provider() -> Provider:
    """Build the built-in DeepSeek provider."""
    return create_provider(
        id="deepseek",
        name="DeepSeek",
        auth=ProviderAuth(api_key=env_api_key_auth("DeepSeek API key", ["DEEPSEEK_API_KEY"])),
        api=openai_completions,
        models=DEEPSEEK_MODELS,
        base_url="https://api.deepseek.com",
    )
