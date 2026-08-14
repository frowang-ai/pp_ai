"""Provider authentication types.

Python port of `packages/ai/src/auth/types.ts` (plus the credential-store and
interaction shapes from `packages/ai/src/auth/context.ts` /
`credential-store.ts`). The TypeScript version keeps API-key and OAuth
credentials as a tagged union (`ApiKeyCredential | OAuthCredential`); this port
keeps the single already-in-use `Credential` dataclass and extends it with the
OAuth fields (`access`, `refresh`, `expires`) rather than splitting it, so
every existing API-key call site keeps working unchanged.

**The legacy extension OAuth callback surface is not ported.**
`packages/ai/src/compat/extension-oauth-types.ts` and the type-only re-export
entry point `packages/ai/src/oauth.ts` declare `OAuthLoginCallbacks` and its
`OAuthPrompt`/`OAuthAuthInfo`/`OAuthDeviceCodeInfo`/`OAuthSelectPrompt`
payloads. Upstream keeps them solely so coding-agent extensions written
against the old multi-callback API keep compiling; the flows themselves use
:class:`AuthInteraction` below, whose single `prompt()`/`notify()` pair covers
the same cases. This port has no extensions that declare OAuth providers, so
porting a compatibility shim for an API nothing here calls would add a second
way to express the same thing. `AuthInteraction` is the surface to use.

`packages/ai/src/bun-oauth.ts` is likewise not ported: it statically registers
the bundled OAuth flows for the standalone Bun binary, which has no Python
counterpart. :mod:`pi_ai.auth.oauth.load` imports the flow modules directly,
so they are always available.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from ..utils.abort import AbortSignal

AuthType = Literal["api_key", "oauth"]


@dataclass
class Credential:
    """A stored provider credential.

    ``access``/``refresh``/``expires`` are populated only for ``type="oauth"``
    credentials (mirrors TypeScript's `OAuthCredential`); ``expires`` is a
    millisecond epoch timestamp (`Date.now()`-compatible, see
    :func:`pi_ai.types.now_ms`). ``data`` carries provider-specific extras that
    don't have a dedicated field, e.g. GitHub Copilot's `enterpriseUrl` /
    `availableModelIds`.
    """

    type: AuthType = "api_key"
    key: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    access: str | None = None
    refresh: str | None = None
    expires: float | None = None


@dataclass
class CredentialInfo:
    """Non-secret credential metadata for account/status enumeration."""

    provider_id: str
    type: AuthType = "api_key"


@dataclass
class ResolvedAuth:
    """The auth material a provider request needs.

    Mirrors TypeScript `ModelAuth`. ``base_url`` is set only by auth methods
    whose endpoint depends on the resolved credential (GitHub Copilot's
    per-token proxy host); most auth methods leave it ``None`` and the
    provider's configured base URL applies.
    """

    api_key: str | None = None
    headers: dict[str, str | None] = field(default_factory=dict)
    """Extra request headers. A ``None`` value *removes* the header, matching
    TypeScript's `ModelAuth.headers` (Cloudflare AI Gateway sets
    ``Authorization: null`` so no upstream bearer token is sent)."""
    base_url: str | None = None


@dataclass
class AuthResult:
    """Resolved auth plus a human-readable source label for status UI."""

    auth: ResolvedAuth
    source: str
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class AuthCheck:
    """Whether a provider has complete auth configuration."""

    configured: bool
    source: str | None = None
    type: AuthType | None = None


@dataclass
class ApiKeyAuth:
    """API-key auth for one provider."""

    name: str
    env_vars: tuple[str, ...] = ()
    resolve: Callable[..., Any] | None = None
    """Custom resolver. Defaults to :func:`env_api_key_resolve`."""
    login: Callable[[AuthInteraction], Awaitable[Credential]] | None = None
    """Interactive login producing an ``api_key`` credential.

    Providers whose credential is not a single secret use this to ask for what
    they actually need (Bedrock: bearer token vs AWS profile; Vertex: API key
    vs Application Default Credentials), which is why it can return a
    credential carrying only ``env``. Ambient-only providers omit it.
    """


@dataclass
class AuthOperationOptions:
    """Optional cancellation for public auth and credential operations."""

    signal: AbortSignal | None = None


@dataclass
class AuthPrompt:
    """Prompt shown to the user during login.

    One dataclass covers every TypeScript `AuthPrompt` variant (`text`,
    `secret`, `select`, `manual_code`); unused fields stay at their default.
    ``signal`` lets a flow cancel a pending prompt when an out-of-band event
    resolves the step, e.g. a `manual_code` prompt raced against a callback
    server, aborted when the callback wins.
    """

    type: Literal["text", "secret", "select", "manual_code"]
    message: str
    placeholder: str | None = None
    options: tuple[dict[str, str], ...] = ()
    """For ``type="select"``: ``{"id": ..., "label": ..., "description": ...}`` entries."""
    signal: AbortSignal | None = None


@dataclass
class AuthInfoLink:
    url: str
    label: str | None = None


@dataclass
class AuthEvent:
    """One dataclass covers every TypeScript `AuthEvent` variant.

    ``type`` selects which fields apply: ``info`` (``message``, ``links``),
    ``auth_url`` (``url``, ``instructions``), ``device_code`` (``user_code``,
    ``verification_uri``, ``interval_seconds``, ``expires_in_seconds``), or
    ``progress`` (``message``).
    """

    type: Literal["info", "auth_url", "device_code", "progress"]
    message: str | None = None
    links: tuple[AuthInfoLink, ...] = ()
    url: str | None = None
    instructions: str | None = None
    user_code: str | None = None
    verification_uri: str | None = None
    interval_seconds: float | None = None
    expires_in_seconds: float | None = None


class AuthInteraction:
    """Login interaction callbacks serving both api-key and OAuth flows.

    ``prompt()`` returns the entered/selected string (``select`` returns the
    option id) and should raise on cancel/abort. ``signal`` aborts the whole
    login flow. Concrete flows receive an instance with ``signal`` always set
    (TypeScript's `ProviderAuthInteraction`).
    """

    signal: AbortSignal

    async def prompt(self, prompt: AuthPrompt) -> str:
        raise NotImplementedError

    def notify(self, event: AuthEvent) -> None:
        raise NotImplementedError


@dataclass
class OAuthAuth:
    """OAuth auth for one provider.

    The ``refresh``/``to_auth`` split lets callers own the locked refresh
    pattern: ``refresh`` produces a credential, ``to_auth`` derives request
    auth from whatever credential ends up stored.
    """

    name: str
    login: Callable[[AuthInteraction], Awaitable[Credential]]
    refresh: Callable[[Credential, AbortSignal], Awaitable[Credential]]
    to_auth: Callable[[Credential], Awaitable[ResolvedAuth]]
    is_subscription: bool = False
    """Whether access through this auth method is backed by a provider subscription."""
    login_label: str | None = None
    """Selector label for the OAuth login option, e.g. "Sign in with SuperGrok or X Premium"."""


@dataclass
class ProviderAuth:
    """Auth methods a provider supports; both are optional.

    TypeScript declares `apiKey?: ApiKeyAuth` (`auth/types.ts:238`). Making
    `api_key` required here forced every OAuth-only provider to carry a
    fabricated API-key method that upstream would never create -- an
    unusable "enter API key" login offered for providers that only support
    OAuth. `composeApiKeyAuth` returns `undefined` in exactly that case.
    """

    api_key: ApiKeyAuth | None = None
    oauth: OAuthAuth | None = None


class CredentialStore:
    """Stores provider credentials. The default implementation is in-memory."""

    async def get(self, provider_id: str) -> Credential | None:
        raise NotImplementedError

    async def set(self, provider_id: str, credential: Credential) -> None:
        raise NotImplementedError

    async def delete(self, provider_id: str) -> None:
        raise NotImplementedError

    async def list(self) -> list[CredentialInfo]:
        raise NotImplementedError


class InMemoryCredentialStore(CredentialStore):
    """Python port of `packages/ai/src/auth/credential-store.ts`."""

    def __init__(self, credentials: dict[str, Credential] | None = None) -> None:
        self._credentials: dict[str, Credential] = dict(credentials or {})

    async def get(self, provider_id: str) -> Credential | None:
        return self._credentials.get(provider_id)

    async def set(self, provider_id: str, credential: Credential) -> None:
        self._credentials[provider_id] = credential

    async def delete(self, provider_id: str) -> None:
        self._credentials.pop(provider_id, None)

    async def list(self) -> list[CredentialInfo]:
        return [
            CredentialInfo(provider_id=provider_id, type=credential.type)
            for provider_id, credential in self._credentials.items()
        ]


EnvLookup = Callable[[str], str | Awaitable[str | None] | None]
