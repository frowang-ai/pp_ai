"""Standard auth helpers.

Python port of `packages/ai/src/auth/helpers.ts`.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Awaitable, Callable

from .types import (
    ApiKeyAuth,
    AuthInteraction,
    AuthPrompt,
    AuthResult,
    Credential,
    EnvLookup,
    OAuthAuth,
    ResolvedAuth,
)


def env_api_key_auth(name: str, env_vars: list[str] | tuple[str, ...]) -> ApiKeyAuth:
    """Standard api-key auth: a stored credential wins, else the first set env var.

    The `login` prompts for the key itself, so a provider that only needs one
    secret does not have to write its own flow.
    """

    async def login(interaction: AuthInteraction) -> Credential:
        interaction.signal.throw_if_aborted()
        key = await interaction.prompt(AuthPrompt(type="secret", message=f"Enter {name}"))
        interaction.signal.throw_if_aborted()
        return Credential(type="api_key", key=key)

    return ApiKeyAuth(name=name, env_vars=tuple(env_vars), login=login)


async def resolve_api_key_auth(
    auth: ApiKeyAuth,
    credential: Credential | None = None,
    env: EnvLookup | None = None,
) -> AuthResult | None:
    """Resolve an api key from a stored credential, then from the environment."""
    if auth.resolve is not None:
        result = auth.resolve(credential=credential, env=env)
        if inspect.isawaitable(result):
            result = await result
        return result

    if credential is not None and credential.key:
        return AuthResult(
            auth=ResolvedAuth(api_key=credential.key),
            source="stored credential",
            env=dict(credential.env),
        )

    lookup = env if env is not None else os.environ.get
    for env_var in auth.env_vars:
        value = lookup(env_var)
        if inspect.isawaitable(value):
            value = await value
        if value:
            return AuthResult(auth=ResolvedAuth(api_key=value), source=env_var)

    return None


def lazy_oauth(
    name: str,
    load: Callable[[], Awaitable[OAuthAuth]],
    *,
    is_subscription: bool = False,
    login_label: str | None = None,
) -> OAuthAuth:
    """Wrap a lazily loaded :class:`OAuthAuth` so callers advertise OAuth support
    without eagerly constructing the flow (which may open sockets, read env
    vars for host overrides, etc.). `load` runs at most once; the result is
    cached and reused for every subsequent `login`/`refresh`/`to_auth` call.
    """
    loaded: OAuthAuth | None = None
    loading: asyncio.Task[OAuthAuth] | None = None

    async def get_loaded() -> OAuthAuth:
        nonlocal loaded, loading
        if loaded is not None:
            return loaded
        if loading is None:
            loading = asyncio.ensure_future(load())
        loaded = await loading
        return loaded

    async def login(interaction):
        flow = await get_loaded()
        return await flow.login(interaction)

    async def refresh(credential, signal):
        flow = await get_loaded()
        return await flow.refresh(credential, signal)

    async def to_auth(credential):
        flow = await get_loaded()
        return await flow.to_auth(credential)

    return OAuthAuth(
        name=name,
        is_subscription=is_subscription,
        login_label=login_label,
        login=login,
        refresh=refresh,
        to_auth=to_auth,
    )
