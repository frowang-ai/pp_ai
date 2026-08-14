"""Amazon Bedrock provider factory.

Python port of `packages/ai/src/providers/amazon-bedrock.ts`. The model list
comes from the generated catalog shard `pi_ai/providers/data/amazon-bedrock.json`,
the Python equivalent of TypeScript's generated `providers/amazon-bedrock.models.ts`
(both produced by `packages/ai/scripts/generate-models.ts`).

Bedrock accepts a bearer token or the AWS SDK's default credential chain. The
resolver below detects ambient AWS credentials without copying them into pi's
credential store, exactly like the TypeScript `resolve`, and `login` asks which
of the two to store.

One piece of the TypeScript factory is not ported: the
`bedrock-converse-stream` API itself; see
:mod:`pi_ai.api.bedrock_converse_stream`. Models are listed for discovery,
and streaming raises :class:`NotImplementedError`.
"""

from __future__ import annotations

import inspect
import os

from ..api import bedrock_converse_stream
from ..auth.types import (
    ApiKeyAuth,
    AuthEvent,
    AuthInfoLink,
    AuthInteraction,
    AuthPrompt,
    AuthResult,
    Credential,
    EnvLookup,
    ProviderAuth,
    ResolvedAuth,
)
from ..model_catalog import load_models
from ..registry import Provider, create_provider
from ..types import Model

AMAZON_BEDROCK_MODELS: list[Model] = load_models("amazon-bedrock")


async def _read_env(env: EnvLookup | None, name: str) -> str | None:
    lookup = env if env is not None else os.environ.get
    value = lookup(name)
    if inspect.isawaitable(value):
        value = await value
    return value or None


async def _resolve_bedrock_auth(
    credential: Credential | None = None,
    env: EnvLookup | None = None,
) -> AuthResult | None:
    if credential is not None and credential.key:
        return AuthResult(
            auth=ResolvedAuth(api_key=credential.key),
            source="stored credential",
            env=dict(credential.env),
        )
    if await _read_env(env, "AWS_BEARER_TOKEN_BEDROCK"):
        return AuthResult(auth=ResolvedAuth(), source="AWS_BEARER_TOKEN_BEDROCK")

    stored_profile = credential.env.get("AWS_PROFILE") if credential is not None else None
    if stored_profile or await _read_env(env, "AWS_PROFILE"):
        return AuthResult(
            auth=ResolvedAuth(),
            source="stored credential" if stored_profile else "AWS_PROFILE",
            env=dict(credential.env) if credential is not None else {},
        )
    if await _read_env(env, "AWS_ACCESS_KEY_ID") and await _read_env(env, "AWS_SECRET_ACCESS_KEY"):
        return AuthResult(auth=ResolvedAuth(), source="AWS access keys")
    for name in ("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "AWS_CONTAINER_CREDENTIALS_FULL_URI"):
        if await _read_env(env, name):
            return AuthResult(auth=ResolvedAuth(), source="ECS task role")
    if await _read_env(env, "AWS_WEB_IDENTITY_TOKEN_FILE"):
        return AuthResult(auth=ResolvedAuth(), source="web identity token")
    return None


async def _bedrock_login(interaction: AuthInteraction) -> Credential:
    interaction.signal.throw_if_aborted()
    method = await interaction.prompt(
        AuthPrompt(
            type="select",
            message="Select Amazon Bedrock authentication method:",
            options=(
                {"id": "bearer-token", "label": "Bearer token"},
                {"id": "aws-profile", "label": "AWS profile"},
                {"id": "credential-chain", "label": "Existing AWS credential chain"},
            ),
        )
    )
    interaction.signal.throw_if_aborted()
    if method == "bearer-token":
        key = await interaction.prompt(AuthPrompt(type="secret", message="Enter Amazon Bedrock bearer token"))
        return Credential(type="api_key", key=key)

    interaction.notify(
        AuthEvent(
            type="info",
            message="Amazon Bedrock supports AWS profiles, IAM credentials, and role-based credentials.",
            links=(
                AuthInfoLink(
                    label="AWS credential provider chain",
                    url="https://docs.aws.amazon.com/sdkref/latest/guide/standardized-credentials.html",
                ),
            ),
        )
    )
    if method == "aws-profile":
        profile = await interaction.prompt(AuthPrompt(type="text", message="Enter AWS profile name"))
        return Credential(type="api_key", env={"AWS_PROFILE": profile})
    if method != "credential-chain":
        raise ValueError(f"Unknown Amazon Bedrock auth method: {method}")
    await interaction.prompt(AuthPrompt(type="text", message="Configure AWS credentials, then press Enter to continue"))
    return Credential(type="api_key")


def bedrock_auth() -> ApiKeyAuth:
    """AWS credentials or a Bedrock bearer token."""
    return ApiKeyAuth(name="AWS credentials or bearer token", resolve=_resolve_bedrock_auth, login=_bedrock_login)


def amazon_bedrock_provider() -> Provider:
    """Build the built-in Amazon Bedrock provider."""
    return create_provider(
        id="amazon-bedrock",
        name="Amazon Bedrock",
        auth=ProviderAuth(api_key=bedrock_auth()),
        api=bedrock_converse_stream,
        models=AMAZON_BEDROCK_MODELS,
    )
