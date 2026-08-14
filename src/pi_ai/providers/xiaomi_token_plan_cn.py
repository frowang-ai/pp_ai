"""Xiaomi Token Plan CN provider factory.

Python port of `packages/ai/src/providers/xiaomi-token-plan-cn.ts`. The model list comes from the
generated catalog shard `pi_ai/providers/data/xiaomi-token-plan-cn.json`, which is the Python
equivalent of TypeScript's generated `providers/xiaomi-token-plan-cn.models.ts` module (both
are produced by `packages/ai/scripts/generate-models.ts`).
"""

from __future__ import annotations

from ..api import openai_completions
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

XIAOMI_TOKEN_PLAN_CN_MODELS: list[Model] = load_models("xiaomi-token-plan-cn")


def xiaomi_token_plan_cn_provider() -> Provider:
    """Build the built-in Xiaomi Token Plan CN provider."""
    return create_provider(
        id="xiaomi-token-plan-cn",
        name="Xiaomi Token Plan CN",
        auth=ProviderAuth(api_key=env_api_key_auth("Xiaomi Token Plan CN API key", ["XIAOMI_TOKEN_PLAN_CN_API_KEY"])),
        api=openai_completions,
        models=XIAOMI_TOKEN_PLAN_CN_MODELS,
        base_url="https://token-plan-cn.xiaomimimo.com/v1",
    )
