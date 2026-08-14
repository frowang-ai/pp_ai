"""Google provider factory.

Python port of `packages/ai/src/providers/google.ts`. The model list comes from the
generated catalog shard `pi_ai/providers/data/google.json`, the Python
equivalent of TypeScript's generated `providers/google.models.ts` (both are
produced by `packages/ai/scripts/generate-models.ts`).
"""

from __future__ import annotations

from ..api import google_generative_ai
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"

GOOGLE_MODELS: list[Model] = load_models("google")


def google_provider() -> Provider:
    """Build the built-in Google provider, authenticating via GEMINI_API_KEY."""
    return create_provider(
        id="google",
        name="Google",
        auth=ProviderAuth(
            api_key=env_api_key_auth("Gemini API key", [GEMINI_API_KEY_ENV]),
        ),
        api=google_generative_ai,
        models=GOOGLE_MODELS,
        base_url="https://generativelanguage.googleapis.com/v1beta",
    )
