"""MiniMax CN provider factory.

Python port of `packages/ai/src/providers/minimax-cn.ts`. The model list comes from the
generated catalog shard `pi_ai/providers/data/minimax-cn.json`, which is the Python
equivalent of TypeScript's generated `providers/minimax-cn.models.ts` module (both
are produced by `packages/ai/scripts/generate-models.ts`).
"""

from __future__ import annotations

from ..api import anthropic_messages
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

MINIMAX_CN_MODELS: list[Model] = load_models("minimax-cn")


def minimax_cn_provider() -> Provider:
    """Build the built-in MiniMax CN provider."""
    return create_provider(
        id="minimax-cn",
        name="MiniMax CN",
        auth=ProviderAuth(api_key=env_api_key_auth("MiniMax CN API key", ["MINIMAX_CN_API_KEY"])),
        api=anthropic_messages,
        models=MINIMAX_CN_MODELS,
        base_url="https://api.minimaxi.com/anthropic",
    )
