"""Cloudflare API-key auth for Workers AI and AI Gateway.

Python port of `packages/ai/src/providers/cloudflare-auth.ts`.

Both Cloudflare endpoints need more than an API key: Workers AI needs the
account id, and AI Gateway also needs the gateway id. Resolution merges per
field, so a stored credential that carries only the API key still picks up the
account/gateway id from the environment.

The TypeScript `login` prompts are not ported:
:class:`pi_ai.auth.types.ApiKeyAuth` has no `login` hook, so the values come
from the environment or from a stored credential's ``env``.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass

from ..auth.types import ApiKeyAuth, AuthResult, Credential, EnvLookup, ResolvedAuth

CLOUDFLARE_API_KEY = "CLOUDFLARE_API_KEY"
CLOUDFLARE_ACCOUNT_ID = "CLOUDFLARE_ACCOUNT_ID"
CLOUDFLARE_GATEWAY_ID = "CLOUDFLARE_GATEWAY_ID"


@dataclass
class _ResolvedCloudflareEnv:
    api_key: str
    env: dict[str, str]
    source: str


async def _resolve_value(
    name: str,
    credential: Credential | None,
    env: EnvLookup | None,
) -> str | None:
    # Per-field merge: prefer the credential value, fall back to ambient env.
    if credential is not None:
        from_credential = credential.key if name == CLOUDFLARE_API_KEY else credential.env.get(name)
        if from_credential is not None:
            return from_credential
    lookup = env if env is not None else os.environ.get
    value = lookup(name)
    if inspect.isawaitable(value):
        value = await value
    return value or None


async def _resolve_cloudflare_env(
    kind: str,
    credential: Credential | None,
    env: EnvLookup | None,
) -> _ResolvedCloudflareEnv | None:
    api_key = await _resolve_value(CLOUDFLARE_API_KEY, credential, env)
    account_id = await _resolve_value(CLOUDFLARE_ACCOUNT_ID, credential, env)
    gateway_id = await _resolve_value(CLOUDFLARE_GATEWAY_ID, credential, env) if kind == "ai-gateway" else None

    if not api_key or not account_id or (kind == "ai-gateway" and not gateway_id):
        return None

    resolved_env = {CLOUDFLARE_ACCOUNT_ID: account_id}
    if gateway_id:
        resolved_env[CLOUDFLARE_GATEWAY_ID] = gateway_id
    return _ResolvedCloudflareEnv(
        api_key=api_key,
        env=resolved_env,
        source="stored credential" if credential is not None else CLOUDFLARE_API_KEY,
    )


async def _resolve_workers_ai(
    credential: Credential | None = None,
    env: EnvLookup | None = None,
) -> AuthResult | None:
    resolved = await _resolve_cloudflare_env("workers-ai", credential, env)
    if resolved is None:
        return None
    return AuthResult(auth=ResolvedAuth(api_key=resolved.api_key), env=resolved.env, source=resolved.source)


async def _resolve_ai_gateway(
    credential: Credential | None = None,
    env: EnvLookup | None = None,
) -> AuthResult | None:
    resolved = await _resolve_cloudflare_env("ai-gateway", credential, env)
    if resolved is None:
        return None
    # AI Gateway authenticates the gateway itself with `cf-aig-authorization`;
    # the upstream provider key travels separately, so `Authorization` and
    # `x-api-key` are nulled unless the caller supplies one (BYOK).
    return AuthResult(
        auth=ResolvedAuth(
            headers={
                "cf-aig-authorization": f"Bearer {resolved.api_key}",
                "Authorization": None,
                "x-api-key": None,
            }
        ),
        env=resolved.env,
        source=resolved.source,
    )


def cloudflare_workers_ai_auth() -> ApiKeyAuth:
    return ApiKeyAuth(name="Cloudflare API key", resolve=_resolve_workers_ai)


def cloudflare_ai_gateway_auth() -> ApiKeyAuth:
    return ApiKeyAuth(name="Cloudflare API key", resolve=_resolve_ai_gateway)
