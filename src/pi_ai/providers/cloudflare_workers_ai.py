"""Cloudflare Workers AI provider factory.

Python port of `packages/ai/src/providers/cloudflare-workers-ai.ts`. The model
list comes from the generated catalog shard
`pi_ai/providers/data/cloudflare-workers-ai.json`, the Python equivalent of
TypeScript's generated `providers/cloudflare-workers-ai.models.ts` (both
produced by `packages/ai/scripts/generate-models.ts`).
"""

from __future__ import annotations

from ..api import openai_completions
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model
from .cloudflare_auth import cloudflare_workers_ai_auth
from .cloudflare_stream import cloudflare_streams

CLOUDFLARE_WORKERS_AI_MODELS: list[Model] = load_models("cloudflare-workers-ai")


def cloudflare_workers_ai_provider() -> Provider:
    """Build the built-in Cloudflare Workers AI provider."""
    return create_provider(
        id="cloudflare-workers-ai",
        name="Cloudflare Workers AI",
        auth=ProviderAuth(api_key=cloudflare_workers_ai_auth()),
        api=cloudflare_streams(openai_completions),
        models=CLOUDFLARE_WORKERS_AI_MODELS,
    )
