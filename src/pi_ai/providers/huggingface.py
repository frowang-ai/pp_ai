"""Hugging Face provider factory.

Python port of `packages/ai/src/providers/huggingface.ts`. The model list comes from the
generated catalog shard `pi_ai/providers/data/huggingface.json`, which is the Python
equivalent of TypeScript's generated `providers/huggingface.models.ts` module (both
are produced by `packages/ai/scripts/generate-models.ts`).
"""

from __future__ import annotations

from ..api import openai_completions
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

HUGGINGFACE_MODELS: list[Model] = load_models("huggingface")


def huggingface_provider() -> Provider:
    """Build the built-in Hugging Face provider."""
    return create_provider(
        id="huggingface",
        name="Hugging Face",
        auth=ProviderAuth(api_key=env_api_key_auth("Hugging Face token", ["HF_TOKEN"])),
        api=openai_completions,
        models=HUGGINGFACE_MODELS,
        base_url="https://router.huggingface.co/v1",
    )
