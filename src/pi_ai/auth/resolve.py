"""Auth resolution shared by provider auth lookups.

Python port of `packages/ai/src/auth/resolve.ts`. A stored credential owns the
provider: ambient/env lookup is consulted only when nothing is stored. No
silent env fallback after a failed refresh or for a credential type without a
matching handler.

Deviations from the TypeScript source:

- `ModelsError`/`ModelsErrorCode` already exist in :mod:`pi_ai.models`; this
  module reuses that class instead of defining a second one, so callers only
  ever catch one exception type.
- TypeScript's `CredentialStore.modify()` gives the refresh a per-provider
  storage lock enforced by the store implementation itself. The already-ported
  :class:`pi_ai.auth.types.CredentialStore` only has `get`/`set`/`delete`
  (no atomic read-modify-write), so the double-checked-locking pattern here
  uses a module-level `asyncio.Lock` per `(store, provider_id)` pair instead.
  This serializes refreshes within one Python process, which is what the
  in-memory store and every current caller needs; a store shared across
  processes would need its own locking, same as the TypeScript note about
  cross-process locks being store-specific.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace

from ..models import ModelsError
from ..utils.abort import AbortSignal, operation_signal, race_with_abort_signal
from .helpers import resolve_api_key_auth
from .types import AuthResult, Credential, CredentialStore, EnvLookup, OAuthAuth, ProviderAuth

DEFAULT_OAUTH_MINIMUM_VALIDITY_MS = 5 * 60 * 1000
DEFAULT_OAUTH_REFRESH_TIMEOUT_S = 15.0

# Keyed by (id(credential_store), provider_id) so unrelated stores/providers
# never contend for the same lock.
_refresh_locks: dict[tuple[int, str], asyncio.Lock] = {}


def _refresh_lock(credentials: CredentialStore, provider_id: str) -> asyncio.Lock:
    key = (id(credentials), provider_id)
    lock = _refresh_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _refresh_locks[key] = lock
    return lock


async def resolve_provider_auth(
    provider_id: str,
    auth: ProviderAuth,
    credentials: CredentialStore,
    env: EnvLookup | None = None,
    *,
    api_key_override: str | None = None,
    env_override: dict[str, str] | None = None,
    min_oauth_validity_ms: int | None = None,
    signal: AbortSignal | None = None,
) -> AuthResult | None:
    """Resolve auth for one provider.

    Precedence: an explicit ``api_key_override`` wins outright (mirrors a
    per-request override in TypeScript); otherwise a stored credential wins,
    falling back to ambient/env resolution only when nothing is stored.
    ``env_override`` is the per-request env map: it is attached to the
    credential handed to the provider's resolver, exactly like TypeScript's
    `{ ...stored, env: { ...stored.env, ...overrides.env } }`.
    """
    op_signal = operation_signal(signal)
    return await race_with_abort_signal(
        _resolve_provider_auth(
            provider_id,
            auth,
            credentials,
            env,
            api_key_override,
            env_override,
            min_oauth_validity_ms,
            op_signal,
        ),
        op_signal,
    )


async def _resolve_provider_auth(
    provider_id: str,
    auth: ProviderAuth,
    credentials: CredentialStore,
    env: EnvLookup | None,
    api_key_override: str | None,
    env_override: dict[str, str] | None,
    min_oauth_validity_ms: int | None,
    signal: AbortSignal,
) -> AuthResult | None:
    signal.throw_if_aborted()

    if api_key_override is not None and auth.api_key is not None:
        credential = Credential(type="api_key", key=api_key_override, env=dict(env_override or {}))
        return await _resolve_api_key(auth, provider_id, credential, env)

    stored = await _read_credential(credentials, provider_id)
    if stored is not None:
        if stored.type == "oauth" and auth.oauth is not None:
            return await _resolve_stored_oauth(
                credentials, provider_id, auth.oauth, stored, signal, min_oauth_validity_ms
            )
        if stored.type == "api_key" and auth.api_key is not None:
            if env_override:
                stored = replace(stored, env={**stored.env, **env_override})
            return await _resolve_api_key(auth, provider_id, stored, env)
        return None

    # Ambient (env vars, and whatever else a custom `resolve` implements).
    return await _resolve_api_key(auth, provider_id, None, env) if auth.api_key is not None else None


def _expires_soon(credential: Credential, minimum_validity_ms: float) -> bool:
    expires = credential.expires if credential.expires is not None else 0.0
    return time.time() * 1000 + minimum_validity_ms >= expires


async def _resolve_stored_oauth(
    credentials: CredentialStore,
    provider_id: str,
    oauth: OAuthAuth,
    stored: Credential,
    signal: AbortSignal,
    min_oauth_validity_ms: int | None,
) -> AuthResult | None:
    """OAuth resolution with double-checked locking.

    Tokens with less than five minutes remaining lock, re-check expiry under
    the lock, refresh once, and persist the rotated credential before release.
    """
    minimum_validity_ms = max(DEFAULT_OAUTH_MINIMUM_VALIDITY_MS, min_oauth_validity_ms or 0)
    credential = stored

    if _expires_soon(credential, minimum_validity_ms):
        lock = _refresh_lock(credentials, provider_id)
        async with lock:
            # Authoritative check under the lock: another request may have
            # already refreshed while we were waiting for it.
            current = await _read_credential(credentials, provider_id)
            if current is None or current.type != "oauth":
                return None  # logged out meanwhile
            if _expires_soon(current, minimum_validity_ms):
                try:
                    refresh_signal = operation_signal(signal)
                    refreshed = await asyncio.wait_for(
                        oauth.refresh(current, refresh_signal), timeout=DEFAULT_OAUTH_REFRESH_TIMEOUT_S
                    )
                except Exception as error:
                    raise ModelsError("oauth", f"OAuth refresh failed for {provider_id}", error) from error
                # A refresh that finishes after the caller gave up must not
                # overwrite the stored credential: TypeScript persists inside
                # `credentials.modify(..., { signal })`, which is skipped once
                # the signal aborts.
                signal.throw_if_aborted()
                await credentials.set(provider_id, refreshed)
                credential = refreshed
            else:
                credential = current

            if min_oauth_validity_ms is not None and _expires_soon(credential, minimum_validity_ms):
                raise ModelsError("oauth", f"OAuth refresh returned a token that expires too soon for {provider_id}")

    try:
        resolved = await oauth.to_auth(credential)
    except Exception as error:
        raise ModelsError("oauth", f"OAuth auth derivation failed for {provider_id}", error) from error
    return AuthResult(auth=resolved, source="OAuth")


async def _resolve_api_key(
    auth: ProviderAuth,
    provider_id: str,
    credential: Credential | None,
    env: EnvLookup | None,
) -> AuthResult | None:
    try:
        return await resolve_api_key_auth(auth.api_key, credential, env)
    except Exception as error:
        raise ModelsError("auth", f"API key auth failed for provider {provider_id}: {error}") from error


async def _read_credential(credentials: CredentialStore, provider_id: str) -> Credential | None:
    try:
        return await credentials.get(provider_id)
    except Exception as error:
        raise ModelsError("auth", f"Credential store read failed for {provider_id}", error) from error
