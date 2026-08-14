"""Azure OpenAI provider factory.

Python port of `packages/ai/src/providers/azure-openai-responses.ts`. The model list comes from the
generated catalog shard `pi_ai/providers/data/azure-openai-responses.json`, which is the Python
equivalent of TypeScript's generated `providers/azure-openai-responses.models.ts` module (both
are produced by `packages/ai/scripts/generate-models.ts`).
"""

from __future__ import annotations

from ..api import azure_openai_responses
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

AZURE_OPENAI_RESPONSES_MODELS: list[Model] = load_models("azure-openai-responses")


def azure_openai_responses_provider() -> Provider:
    """Build the built-in Azure OpenAI provider."""
    return create_provider(
        id="azure-openai-responses",
        name="Azure OpenAI",
        auth=ProviderAuth(api_key=env_api_key_auth("Azure OpenAI API key", ["AZURE_OPENAI_API_KEY"])),
        api=azure_openai_responses,
        models=AZURE_OPENAI_RESPONSES_MODELS,
    )
