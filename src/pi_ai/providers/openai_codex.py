"""OpenAI Codex provider factory.

Python port of `packages/ai/src/providers/openai-codex.ts`. The model list
comes from the generated catalog shard `pi_ai/providers/data/openai-codex.json`,
the Python equivalent of TypeScript's generated `providers/openai-codex.models.ts`
(both produced by `packages/ai/scripts/generate-models.ts`).

Two dependencies of the TypeScript factory are not ported, so this provider is
discovery-only:

- `packages/ai/src/apis/openai-codex-responses.ts` — see
  :mod:`pi_ai.api.openai_codex_responses`, which raises
  :class:`NotImplementedError` on stream.
- The ChatGPT OAuth flow (`loadOpenAICodexOAuth`) has no Python counterpart in
  `pi_ai.auth.oauth`, so only a placeholder API-key auth is wired. Codex does
  not accept a plain API key, so the placeholder never resolves; it exists
  because :class:`pi_ai.auth.types.ProviderAuth` requires an ``api_key`` entry.
"""

from __future__ import annotations

from ..api import openai_codex_responses
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

OPENAI_CODEX_MODELS: list[Model] = load_models("openai-codex")


def openai_codex_provider() -> Provider:
    """Build the built-in OpenAI Codex provider (models are discovery-only)."""
    return create_provider(
        id="openai-codex",
        name="OpenAI Codex",
        auth=ProviderAuth(api_key=env_api_key_auth("OpenAI Codex (ChatGPT Plus/Pro)", [])),
        api=openai_codex_responses,
        models=OPENAI_CODEX_MODELS,
        base_url="https://chatgpt.com/backend-api",
    )
