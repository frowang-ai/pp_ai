"""OpenRouter provider factory (chat completions).

Python port of `packages/ai/src/providers/openrouter.ts`. The model list comes
from the generated catalog shard `pi_ai/providers/data/openrouter.json`, the
Python equivalent of TypeScript's generated `providers/openrouter.models.ts`
(both produced by `packages/ai/scripts/generate-models.ts`).

OpenRouter speaks the OpenAI Chat Completions wire format, so it reuses
`openai_completions` (`api="openai-completions"`) exactly like the TypeScript
source does (`openAICompletionsApi()`).
"""

from __future__ import annotations

from ..api import openai_completions
from ..auth.helpers import env_api_key_auth, lazy_oauth
from ..auth.oauth.load import load_openrouter_oauth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"

OPENROUTER_MODELS: list[Model] = load_models("openrouter")


def openrouter_provider() -> Provider:
    """Build the built-in OpenRouter provider."""
    return create_provider(
        id="openrouter",
        name="OpenRouter",
        auth=ProviderAuth(
            api_key=env_api_key_auth("OpenRouter API key", [OPENROUTER_API_KEY_ENV]),
            oauth=lazy_oauth("OpenRouter OAuth", load_openrouter_oauth, login_label="Sign in with OpenRouter"),
        ),
        api=openai_completions,
        models=OPENROUTER_MODELS,
        base_url="https://openrouter.ai/api/v1",
    )
