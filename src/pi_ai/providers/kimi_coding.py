"""Kimi For Coding provider factory.

Python port of `packages/ai/src/providers/kimi-coding.ts`. The model list comes
from the generated catalog shard `pi_ai/providers/data/kimi-coding.json`, the
Python equivalent of TypeScript's generated `providers/kimi-coding.models.ts`
(both produced by `packages/ai/scripts/generate-models.ts`).
"""

from __future__ import annotations

from ..api import anthropic_messages
from ..auth.helpers import env_api_key_auth, lazy_oauth
from ..auth.oauth.load import load_kimi_coding_oauth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

KIMI_CODING_MODELS: list[Model] = load_models("kimi-coding")


def kimi_coding_provider() -> Provider:
    """Build the built-in Kimi For Coding provider."""
    return create_provider(
        id="kimi-coding",
        name="Kimi For Coding",
        auth=ProviderAuth(
            api_key=env_api_key_auth("Kimi API key", ["KIMI_API_KEY"]),
            oauth=lazy_oauth(
                "Kimi Code (subscription)",
                load_kimi_coding_oauth,
                is_subscription=True,
                login_label="Sign in with Kimi Code",
            ),
        ),
        api=anthropic_messages,
        models=KIMI_CODING_MODELS,
        base_url="https://api.kimi.com/coding",
    )
