"""Cloudflare AI Gateway provider factory.

Python port of `packages/ai/src/providers/cloudflare-ai-gateway.ts`. The model
list comes from the generated catalog shard
`pi_ai/providers/data/cloudflare-ai-gateway.json`, the Python equivalent of
TypeScript's generated `providers/cloudflare-ai-gateway.models.ts` (both
produced by `packages/ai/scripts/generate-models.ts`).

The gateway proxies several upstream wire formats, so the provider dispatches
on ``model.api`` across three API modules.
"""

from __future__ import annotations

from ..api import anthropic_messages, openai_completions, openai_responses
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model
from .cloudflare_auth import cloudflare_ai_gateway_auth
from .cloudflare_stream import cloudflare_streams

CLOUDFLARE_AI_GATEWAY_MODELS: list[Model] = load_models("cloudflare-ai-gateway")


def cloudflare_ai_gateway_provider() -> Provider:
    """Build the built-in Cloudflare AI Gateway provider."""
    return create_provider(
        id="cloudflare-ai-gateway",
        name="Cloudflare AI Gateway",
        auth=ProviderAuth(api_key=cloudflare_ai_gateway_auth()),
        api={
            "anthropic-messages": cloudflare_streams(anthropic_messages),
            "openai-completions": cloudflare_streams(openai_completions),
            "openai-responses": cloudflare_streams(openai_responses),
        },
        models=CLOUDFLARE_AI_GATEWAY_MODELS,
    )
