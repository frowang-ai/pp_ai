"""Anthropic provider factory.

Python port of `packages/ai/src/providers/anthropic.ts`. The model list comes
from the generated catalog shard `pi_ai/providers/data/anthropic.json`, the
Python equivalent of TypeScript's generated `providers/anthropic.models.ts`
(both produced by `packages/ai/scripts/generate-models.ts`).

Anthropic's api-key resolution has one extra step over the standard env
lookup: `ANTHROPIC_AUTH_TOKEN` is a pre-formed bearer token and goes into the
`Authorization` header rather than the `x-api-key` slot.
"""

from __future__ import annotations

import inspect
import os

from ..api import anthropic_messages
from ..auth.helpers import lazy_oauth
from ..auth.oauth.load import load_anthropic_oauth
from ..auth.types import ApiKeyAuth, AuthResult, Credential, EnvLookup, ProviderAuth, ResolvedAuth
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
ANTHROPIC_AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"
ANTHROPIC_OAUTH_TOKEN_ENV = "ANTHROPIC_OAUTH_TOKEN"

ANTHROPIC_MODELS: list[Model] = load_models("anthropic")


async def _read_env(env: EnvLookup | None, name: str) -> str | None:
    lookup = env if env is not None else os.environ.get
    value = lookup(name)
    if inspect.isawaitable(value):
        value = await value
    return value or None


async def _resolve_anthropic_auth(
    credential: Credential | None = None,
    env: EnvLookup | None = None,
) -> AuthResult | None:
    if credential is not None and credential.key:
        return AuthResult(
            auth=ResolvedAuth(api_key=credential.key),
            source="stored credential",
            env=dict(credential.env),
        )

    auth_token = await _read_env(env, ANTHROPIC_AUTH_TOKEN_ENV)
    if auth_token:
        return AuthResult(
            auth=ResolvedAuth(headers={"Authorization": f"Bearer {auth_token}"}),
            source=ANTHROPIC_AUTH_TOKEN_ENV,
        )

    for env_var in (ANTHROPIC_OAUTH_TOKEN_ENV, ANTHROPIC_API_KEY_ENV):
        api_key = await _read_env(env, env_var)
        if api_key:
            return AuthResult(auth=ResolvedAuth(api_key=api_key), source=env_var)
    return None


def anthropic_api_key_auth() -> ApiKeyAuth:
    """Anthropic api-key auth, including the `ANTHROPIC_AUTH_TOKEN` bearer path."""
    return ApiKeyAuth(
        name="Anthropic API key",
        env_vars=(ANTHROPIC_OAUTH_TOKEN_ENV, ANTHROPIC_API_KEY_ENV),
        resolve=_resolve_anthropic_auth,
    )


def anthropic_provider() -> Provider:
    """Build the built-in Anthropic provider."""
    return create_provider(
        id="anthropic",
        name="Anthropic",
        auth=ProviderAuth(
            api_key=anthropic_api_key_auth(),
            oauth=lazy_oauth("Anthropic (Claude Pro/Max)", load_anthropic_oauth, is_subscription=True),
        ),
        api=anthropic_messages,
        models=ANTHROPIC_MODELS,
        base_url="https://api.anthropic.com",
    )
