"""NVIDIA provider factory.

Python port of `packages/ai/src/providers/nvidia.ts`. The model list comes from the
generated catalog shard `pi_ai/providers/data/nvidia.json`, which is the Python
equivalent of TypeScript's generated `providers/nvidia.models.ts` module (both
are produced by `packages/ai/scripts/generate-models.ts`).
"""

from __future__ import annotations

from ..api import openai_completions
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

NVIDIA_MODELS: list[Model] = load_models("nvidia")


def nvidia_provider() -> Provider:
    """Build the built-in NVIDIA provider."""
    return create_provider(
        id="nvidia",
        name="NVIDIA",
        auth=ProviderAuth(api_key=env_api_key_auth("NVIDIA API key", ["NVIDIA_API_KEY"])),
        api=openai_completions,
        models=NVIDIA_MODELS,
        base_url="https://integrate.api.nvidia.com/v1",
    )
