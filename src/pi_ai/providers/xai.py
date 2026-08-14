"""xAI provider factory.

Python port of `packages/ai/src/providers/xai.ts`. The model list comes from
the generated catalog shard `pi_ai/providers/data/xai.json`, the Python
equivalent of TypeScript's generated `providers/xai.models.ts` (both produced
by `packages/ai/scripts/generate-models.ts`).

xAI exposes both Chat Completions and Responses endpoints, so the provider
dispatches on ``model.api``.
"""

from __future__ import annotations

from ..api import openai_completions, openai_responses
from ..auth.helpers import env_api_key_auth, lazy_oauth
from ..auth.oauth.load import load_xai_oauth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

XAI_MODELS: list[Model] = load_models("xai")


def xai_provider() -> Provider:
    """Build the built-in xAI provider."""
    return create_provider(
        id="xai",
        name="xAI",
        auth=ProviderAuth(
            api_key=env_api_key_auth("xAI API key", ["XAI_API_KEY"]),
            oauth=lazy_oauth(
                "xAI (Grok/X subscription)",
                load_xai_oauth,
                is_subscription=True,
                login_label="Sign in with SuperGrok or X Premium",
            ),
        ),
        api={
            "openai-completions": openai_completions,
            "openai-responses": openai_responses,
        },
        models=XAI_MODELS,
        base_url="https://api.x.ai/v1",
    )
