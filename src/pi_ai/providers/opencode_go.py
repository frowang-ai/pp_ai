"""OpenCode Go provider factory.

Python port of `packages/ai/src/providers/opencode-go.ts`. The model list comes
from the generated catalog shard `pi_ai/providers/data/opencode-go.json`, the
Python equivalent of TypeScript's generated `providers/opencode-go.models.ts`
(both produced by `packages/ai/scripts/generate-models.ts`).

Like OpenCode Zen, each model keeps its vendor's native wire format and base
URL, so the provider dispatches on ``model.api``.
"""

from __future__ import annotations

from ..api import anthropic_messages, openai_completions, openai_responses
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

OPENCODE_GO_MODELS: list[Model] = load_models("opencode-go")


def opencode_go_provider() -> Provider:
    """Build the built-in OpenCode Go provider."""
    return create_provider(
        id="opencode-go",
        name="OpenCode Go",
        auth=ProviderAuth(api_key=env_api_key_auth("OpenCode API key", ["OPENCODE_API_KEY"])),
        api={
            "anthropic-messages": anthropic_messages,
            "openai-completions": openai_completions,
            "openai-responses": openai_responses,
        },
        models=OPENCODE_GO_MODELS,
    )
