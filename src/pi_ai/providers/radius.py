"""Radius gateway provider factory.

Python port of `packages/ai/src/providers/radius.ts`.

Radius is the one built-in provider with no generated catalog entry: its models
come from the gateway's `/v1/config` endpoint, cached on the OAuth credential.
A freshly constructed provider therefore starts with whatever the credential
carries (usually nothing) and grows a catalog once
:func:`refresh_radius_models` runs.

The TypeScript factory drives that refresh through `ModelsStore`
(`context.publish`, persisted catalogs, `allowNetwork` gating). That store is
not ported, so the refresh here is an explicit call that returns the models and
updates the provider in place; persistence is the caller's decision.
"""

from __future__ import annotations

from ..api import pi_messages
from ..auth.helpers import env_api_key_auth, lazy_oauth
from ..auth.oauth.load import load_radius_oauth
from ..auth.types import Credential, ProviderAuth
from ..registry import Provider
from ..types import Model
from ..utils.abort import AbortSignal
from .radius_config import (
    DEFAULT_RADIUS_GATEWAY,
    get_radius_models,
    get_radius_models_from_config,
    load_radius_gateway_config,
    normalize_radius_gateway_url,
)


def radius_provider(
    id: str = "radius",
    name: str = "Radius",
    gateway: str = DEFAULT_RADIUS_GATEWAY,
    credential: Credential | None = None,
) -> Provider:
    """Build a Radius gateway provider.

    ``credential`` seeds the catalog from a stored OAuth credential's cached
    gateway config, mirroring TypeScript's `getRadiusModels(id, credential)`.
    """
    provider = Provider(
        id=id,
        name=name,
        auth=ProviderAuth(
            api_key=env_api_key_auth("Radius API key", ["RADIUS_API_KEY"]),
            oauth=lazy_oauth(name, lambda: load_radius_oauth(name, normalize_radius_gateway_url(gateway))),
        ),
        api=pi_messages,
        models=get_radius_models(id, credential),
    )
    return provider


async def refresh_radius_models(
    provider: Provider,
    gateway: str = DEFAULT_RADIUS_GATEWAY,
    credential: Credential | None = None,
    signal: AbortSignal | None = None,
) -> list[Model]:
    """Fetch the gateway catalog and replace ``provider.models`` with it."""
    api_key = credential.access if credential is not None and credential.type == "oauth" else None
    if api_key is None and credential is not None:
        api_key = credential.key
    config = await load_radius_gateway_config(normalize_radius_gateway_url(gateway), api_key, signal)
    provider.models = get_radius_models_from_config(provider.id, config)
    return provider.get_models()
