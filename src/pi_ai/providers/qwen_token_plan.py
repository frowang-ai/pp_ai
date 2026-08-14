"""Qwen Token Plan provider factory.

Python port of `packages/ai/src/providers/qwen-token-plan.ts`. The model list comes from the
generated catalog shard `pi_ai/providers/data/qwen-token-plan.json`, which is the Python
equivalent of TypeScript's generated `providers/qwen-token-plan.models.ts` module (both
are produced by `packages/ai/scripts/generate-models.ts`).
"""

from __future__ import annotations

from ..api import openai_completions
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

QWEN_TOKEN_PLAN_MODELS: list[Model] = load_models("qwen-token-plan")


def qwen_token_plan_provider() -> Provider:
    """Build the built-in Qwen Token Plan provider."""
    return create_provider(
        id="qwen-token-plan",
        name="Qwen Token Plan",
        auth=ProviderAuth(api_key=env_api_key_auth("Qwen Token Plan API key", ["QWEN_TOKEN_PLAN_API_KEY"])),
        api=openai_completions,
        models=QWEN_TOKEN_PLAN_MODELS,
        base_url="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
    )
