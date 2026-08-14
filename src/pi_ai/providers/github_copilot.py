"""GitHub Copilot provider factory.

Python port of `packages/ai/src/providers/github-copilot.ts`. The model list
comes from the generated catalog shard `pi_ai/providers/data/github-copilot.json`,
the Python equivalent of TypeScript's generated `providers/github-copilot.models.ts`
(both produced by `packages/ai/scripts/generate-models.ts`).

Copilot proxies three upstream wire formats, so the provider dispatches on
``model.api``.

TypeScript additionally passes `filterModels`, which narrows the catalog to the
model ids an OAuth credential actually grants. :class:`pi_ai.registry.Provider`
has no per-credential model filter hook, so that narrowing is not ported: the
full generated catalog is exposed and an ungranted model fails at request time.
:func:`filter_github_copilot_models` implements the same predicate for callers
that want to apply it themselves.
"""

from __future__ import annotations

from ..api import anthropic_messages, openai_completions, openai_responses
from ..auth.helpers import env_api_key_auth, lazy_oauth
from ..auth.oauth.load import load_github_copilot_oauth
from ..auth.types import Credential, ProviderAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

GITHUB_COPILOT_MODELS: list[Model] = load_models("github-copilot")


def filter_github_copilot_models(models: list[Model], credential: Credential | None) -> list[Model]:
    """Narrow ``models`` to the ids an OAuth credential grants.

    Port of the `filterModels` callback in `github-copilot.ts`. A non-OAuth
    credential, or one without a usable ``availableModelIds`` list, leaves the
    catalog untouched.
    """
    if credential is None or credential.type != "oauth":
        return models
    available_model_ids = credential.data.get("availableModelIds")
    if not isinstance(available_model_ids, list) or not all(isinstance(id, str) for id in available_model_ids):
        return models
    available = set(available_model_ids)
    return [model for model in models if model.id in available]


def github_copilot_provider() -> Provider:
    """Build the built-in GitHub Copilot provider."""
    return create_provider(
        id="github-copilot",
        name="GitHub Copilot",
        auth=ProviderAuth(
            api_key=env_api_key_auth("GitHub Copilot token", ["COPILOT_GITHUB_TOKEN"]),
            oauth=lazy_oauth("GitHub Copilot", load_github_copilot_oauth, is_subscription=True),
        ),
        api={
            "anthropic-messages": anthropic_messages,
            "openai-completions": openai_completions,
            "openai-responses": openai_responses,
        },
        models=GITHUB_COPILOT_MODELS,
        base_url="https://api.individual.githubcopilot.com",
        filter_models=filter_github_copilot_models,
    )
