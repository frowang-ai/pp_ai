"""Together provider factory.

Python port of `packages/ai/src/providers/together.ts`. The model list comes from the
generated catalog shard `pi_ai/providers/data/together.json`, which is the Python
equivalent of TypeScript's generated `providers/together.models.ts` module (both
are produced by `packages/ai/scripts/generate-models.ts`).
"""

from __future__ import annotations

from ..api import openai_completions
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

TOGETHER_MODELS: list[Model] = load_models("together")


def together_provider() -> Provider:
    """Build the built-in Together provider."""
    return create_provider(
        id="together",
        name="Together",
        auth=ProviderAuth(api_key=env_api_key_auth("Together API key", ["TOGETHER_API_KEY"])),
        api=openai_completions,
        models=TOGETHER_MODELS,
        base_url="https://api.together.ai/v1",
    )
