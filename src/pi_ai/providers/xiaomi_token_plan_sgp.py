"""Xiaomi Token Plan SGP provider factory.

Python port of `packages/ai/src/providers/xiaomi-token-plan-sgp.ts`. The model list comes from the
generated catalog shard `pi_ai/providers/data/xiaomi-token-plan-sgp.json`, which is the Python
equivalent of TypeScript's generated `providers/xiaomi-token-plan-sgp.models.ts` module (both
are produced by `packages/ai/scripts/generate-models.ts`).
"""

from __future__ import annotations

from ..api import openai_completions
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

XIAOMI_TOKEN_PLAN_SGP_MODELS: list[Model] = load_models("xiaomi-token-plan-sgp")


def xiaomi_token_plan_sgp_provider() -> Provider:
    """Build the built-in Xiaomi Token Plan SGP provider."""
    return create_provider(
        id="xiaomi-token-plan-sgp",
        name="Xiaomi Token Plan SGP",
        auth=ProviderAuth(api_key=env_api_key_auth("Xiaomi Token Plan SGP API key", ["XIAOMI_TOKEN_PLAN_SGP_API_KEY"])),
        api=openai_completions,
        models=XIAOMI_TOKEN_PLAN_SGP_MODELS,
        base_url="https://token-plan-sgp.xiaomimimo.com/v1",
    )
