"""Cloudflare Workers AI / AI Gateway endpoint templates.

Python port of `packages/ai/src/api/cloudflare.ts`. These are URL templates
with `{CLOUDFLARE_ACCOUNT_ID}`/`{CLOUDFLARE_GATEWAY_ID}` placeholders that
provider configuration substitutes at request time; no client logic lives
here.
"""

from __future__ import annotations

CLOUDFLARE_WORKERS_AI_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1"
"""Workers AI direct endpoint."""

CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL = (
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/compat"
)
"""AI Gateway Unified API. https://developers.cloudflare.com/ai-gateway/usage/unified-api/"""

CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL = (
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/openai"
)
"""AI Gateway -> OpenAI passthrough. Used until /compat supports /v1/responses."""

CLOUDFLARE_AI_GATEWAY_ANTHROPIC_BASE_URL = (
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/anthropic"
)
"""AI Gateway -> Anthropic passthrough."""
