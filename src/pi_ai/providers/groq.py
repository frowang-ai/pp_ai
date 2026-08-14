"""Groq provider factory.

Python port of `packages/ai/src/providers/groq.ts`. The model list comes from the
generated catalog shard `pi_ai/providers/data/groq.json`, which is the Python
equivalent of TypeScript's generated `providers/groq.models.ts` module (both
are produced by `packages/ai/scripts/generate-models.ts`).
"""

from __future__ import annotations

from ..api import openai_completions
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

GROQ_MODELS: list[Model] = load_models("groq")


def groq_provider() -> Provider:
    """Build the built-in Groq provider."""
    return create_provider(
        id="groq",
        name="Groq",
        auth=ProviderAuth(api_key=env_api_key_auth("Groq API key", ["GROQ_API_KEY"])),
        api=openai_completions,
        models=GROQ_MODELS,
        base_url="https://api.groq.com/openai/v1",
    )
