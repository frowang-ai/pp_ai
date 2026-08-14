"""Xiaomi Token Plan AMS provider factory.

Python port of `packages/ai/src/providers/xiaomi-token-plan-ams.ts`. The model list comes from the
generated catalog shard `pi_ai/providers/data/xiaomi-token-plan-ams.json`, which is the Python
equivalent of TypeScript's generated `providers/xiaomi-token-plan-ams.models.ts` module (both
are produced by `packages/ai/scripts/generate-models.ts`).
"""

from __future__ import annotations

from ..api import openai_completions
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

XIAOMI_TOKEN_PLAN_AMS_MODELS: list[Model] = load_models("xiaomi-token-plan-ams")


def xiaomi_token_plan_ams_provider() -> Provider:
    """Build the built-in Xiaomi Token Plan AMS provider."""
    return create_provider(
        id="xiaomi-token-plan-ams",
        name="Xiaomi Token Plan AMS",
        auth=ProviderAuth(api_key=env_api_key_auth("Xiaomi Token Plan AMS API key", ["XIAOMI_TOKEN_PLAN_AMS_API_KEY"])),
        api=openai_completions,
        models=XIAOMI_TOKEN_PLAN_AMS_MODELS,
        base_url="https://token-plan-ams.xiaomimimo.com/v1",
    )
