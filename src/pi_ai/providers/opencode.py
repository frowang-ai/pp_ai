"""OpenCode Zen provider factory.

Python port of `packages/ai/src/providers/opencode.ts`. The model list comes
from the generated catalog shard `pi_ai/providers/data/opencode.json`, the
Python equivalent of TypeScript's generated `providers/opencode.models.ts`
(both produced by `packages/ai/scripts/generate-models.ts`).

OpenCode Zen fronts models from several vendors and keeps each vendor's native
wire format, so the provider dispatches on ``model.api`` and each model carries
its own base URL from the catalog.
"""

from __future__ import annotations

from ..api import anthropic_messages, google_generative_ai, openai_completions, openai_responses
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

OPENCODE_MODELS: list[Model] = load_models("opencode")


def opencode_provider() -> Provider:
    """Build the built-in OpenCode Zen provider."""
    return create_provider(
        id="opencode",
        name="OpenCode Zen",
        auth=ProviderAuth(api_key=env_api_key_auth("OpenCode API key", ["OPENCODE_API_KEY"])),
        api={
            "anthropic-messages": anthropic_messages,
            "google-generative-ai": google_generative_ai,
            "openai-completions": openai_completions,
            "openai-responses": openai_responses,
        },
        models=OPENCODE_MODELS,
    )
