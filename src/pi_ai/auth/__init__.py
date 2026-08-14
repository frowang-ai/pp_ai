"""Provider authentication: types, resolution, and OAuth flows.

Python port of `packages/ai/src/auth/`.
"""

from __future__ import annotations

from .helpers import env_api_key_auth, lazy_oauth, resolve_api_key_auth
from .resolve import resolve_provider_auth
from .types import (
    ApiKeyAuth,
    AuthCheck,
    AuthEvent,
    AuthInfoLink,
    AuthInteraction,
    AuthOperationOptions,
    AuthPrompt,
    AuthResult,
    AuthType,
    Credential,
    CredentialInfo,
    CredentialStore,
    EnvLookup,
    InMemoryCredentialStore,
    OAuthAuth,
    ProviderAuth,
    ResolvedAuth,
)

__all__ = [
    "ApiKeyAuth",
    "AuthCheck",
    "AuthEvent",
    "AuthInfoLink",
    "AuthInteraction",
    "AuthOperationOptions",
    "AuthPrompt",
    "AuthResult",
    "AuthType",
    "Credential",
    "CredentialInfo",
    "CredentialStore",
    "EnvLookup",
    "InMemoryCredentialStore",
    "OAuthAuth",
    "ProviderAuth",
    "ResolvedAuth",
    "env_api_key_auth",
    "lazy_oauth",
    "resolve_api_key_auth",
    "resolve_provider_auth",
]
