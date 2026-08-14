"""MiniMax provider factory.

Python port of `packages/ai/src/providers/minimax.ts`. The model list comes from the
generated catalog shard `pi_ai/providers/data/minimax.json`, which is the Python
equivalent of TypeScript's generated `providers/minimax.models.ts` module (both
are produced by `packages/ai/scripts/generate-models.ts`).
"""

from __future__ import annotations

from ..api import anthropic_messages
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

MINIMAX_MODELS: list[Model] = load_models("minimax")


def minimax_provider() -> Provider:
    """Build the built-in MiniMax provider."""
    return create_provider(
        id="minimax",
        name="MiniMax",
        auth=ProviderAuth(api_key=env_api_key_auth("MiniMax API key", ["MINIMAX_API_KEY"])),
        api=anthropic_messages,
        models=MINIMAX_MODELS,
        base_url="https://api.minimax.io/anthropic",
    )
