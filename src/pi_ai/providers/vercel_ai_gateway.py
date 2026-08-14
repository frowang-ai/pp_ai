"""Vercel AI Gateway provider factory.

Python port of `packages/ai/src/providers/vercel-ai-gateway.ts`. The model list comes from the
generated catalog shard `pi_ai/providers/data/vercel-ai-gateway.json`, which is the Python
equivalent of TypeScript's generated `providers/vercel-ai-gateway.models.ts` module (both
are produced by `packages/ai/scripts/generate-models.ts`).
"""

from __future__ import annotations

from ..api import anthropic_messages
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

VERCEL_AI_GATEWAY_MODELS: list[Model] = load_models("vercel-ai-gateway")


def vercel_ai_gateway_provider() -> Provider:
    """Build the built-in Vercel AI Gateway provider."""
    return create_provider(
        id="vercel-ai-gateway",
        name="Vercel AI Gateway",
        auth=ProviderAuth(api_key=env_api_key_auth("Vercel AI Gateway API key", ["AI_GATEWAY_API_KEY"])),
        api=anthropic_messages,
        models=VERCEL_AI_GATEWAY_MODELS,
        base_url="https://ai-gateway.vercel.sh",
    )
