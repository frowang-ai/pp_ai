"""OpenAI provider factory.

Python port of `packages/ai/src/providers/openai.ts`. The model list comes from
the generated catalog shard `pi_ai/providers/data/openai.json`, the Python
equivalent of TypeScript's generated `providers/openai.models.ts` (both
produced by `packages/ai/scripts/generate-models.ts`).

Like the TypeScript factory, this registers the ``openai`` provider id against
the Responses API. The older Chat Completions convenience factory still lives
in :mod:`pi_ai.providers.openai_compatible`.
"""

from __future__ import annotations

from ..api import openai_responses
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

OPENAI_MODELS: list[Model] = load_models("openai")


def openai_provider() -> Provider:
    """Build the built-in OpenAI provider, authenticating via OPENAI_API_KEY."""
    return create_provider(
        id="openai",
        name="OpenAI",
        auth=ProviderAuth(api_key=env_api_key_auth("OpenAI API key", [OPENAI_API_KEY_ENV])),
        api=openai_responses,
        models=OPENAI_MODELS,
        base_url="https://api.openai.com/v1",
    )
