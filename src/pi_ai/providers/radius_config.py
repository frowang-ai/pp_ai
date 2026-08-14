"""Radius gateway configuration and catalog helpers.

Python port of `packages/ai/src/providers/radius-config.ts`.

Radius has no static catalog: the gateway publishes its own base URL and model
list at `GET /v1/config`, and the OAuth credential caches the last response so
the provider has models before any network call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import httpx

from ..auth.types import Credential
from ..types import Model, ModelCost, ModelCostTier
from ..utils.abort import AbortSignal

DEFAULT_RADIUS_GATEWAY = "https://radius.pi.dev"
REQUEST_TIMEOUT_S = 30.0


@dataclass
class RadiusGatewayModel:
    """One model entry as published by a Radius gateway."""

    id: str
    name: str
    reasoning: bool
    input: list[str]
    cost: ModelCost
    context_window: int
    max_tokens: int
    thinking_level_map: dict[str, str | None] | None = None


@dataclass
class RadiusGatewayConfig:
    """A gateway's published endpoint plus its model catalog."""

    base_url: str
    models: list[RadiusGatewayModel]


def _cost_from_json(data: dict[str, Any]) -> ModelCost:
    tiers = [
        ModelCostTier(
            input_tokens_above=int(tier.get("inputTokensAbove", 0)),
            input=float(tier.get("input", 0)),
            output=float(tier.get("output", 0)),
            cache_read=float(tier.get("cacheRead", 0)),
            cache_write=float(tier.get("cacheWrite", 0)),
        )
        for tier in data.get("tiers", [])
        if isinstance(tier, dict)
    ]
    return ModelCost(
        input=float(data.get("input", 0)),
        output=float(data.get("output", 0)),
        cache_read=float(data.get("cacheRead", 0)),
        cache_write=float(data.get("cacheWrite", 0)),
        tiers=tiers,
    )


def _radius_gateway_model(value: object) -> RadiusGatewayModel | None:
    """Port of `isRadiusGatewayModel`, returning the parsed model instead of a type guard."""
    if not isinstance(value, dict):
        return None
    if not isinstance(value.get("id"), str) or not isinstance(value.get("name"), str):
        return None
    if not isinstance(value.get("reasoning"), bool) or not isinstance(value.get("input"), list):
        return None
    cost = value.get("cost")
    if not isinstance(cost, dict):
        return None
    if not isinstance(value.get("contextWindow"), int) or not isinstance(value.get("maxTokens"), int):
        return None
    thinking_level_map = value.get("thinkingLevelMap")
    return RadiusGatewayModel(
        id=value["id"],
        name=value["name"],
        reasoning=value["reasoning"],
        input=[modality for modality in value["input"] if isinstance(modality, str)],
        cost=_cost_from_json(cost),
        context_window=value["contextWindow"],
        max_tokens=value["maxTokens"],
        thinking_level_map=dict(thinking_level_map) if isinstance(thinking_level_map, dict) else None,
    )


def sanitize_radius_gateway_config(config: object) -> RadiusGatewayConfig | None:
    """Port of `sanitizeRadiusGatewayConfig`: drop anything that isn't a usable config."""
    if not isinstance(config, dict):
        return None
    base_url = config.get("baseUrl")
    models = config.get("models")
    if not isinstance(base_url, str) or not isinstance(models, list):
        return None
    parsed = [_radius_gateway_model(model) for model in models]
    return RadiusGatewayConfig(base_url=base_url, models=[model for model in parsed if model is not None])


def normalize_radius_gateway_url(value: str) -> str:
    """Add a scheme when missing and strip trailing slashes."""
    with_scheme = value if re.match(r"^https?://", value, re.IGNORECASE) else f"https://{value}"
    return re.sub(r"/+$", "", with_scheme)


def get_radius_credential_config(credential: Credential | None) -> RadiusGatewayConfig | None:
    """The gateway config cached on an OAuth credential, if any."""
    if credential is None:
        return None
    return sanitize_radius_gateway_config(credential.data.get("gatewayConfig"))


def get_radius_models_from_config(provider_id: str, config: RadiusGatewayConfig) -> list[Model]:
    """Turn a gateway config into `pi-messages` models owned by ``provider_id``."""
    return [
        Model(
            id=model.id,
            name=model.name,
            api="pi-messages",
            provider=provider_id,
            base_url=config.base_url,
            reasoning=model.reasoning,
            thinking_level_map=dict(model.thinking_level_map or {}),
            input=list(model.input),
            cost=model.cost,
            context_window=model.context_window,
            max_tokens=model.max_tokens,
        )
        for model in config.models
    ]


def get_radius_models(provider_id: str, credential: Credential | None) -> list[Model]:
    """Models from a credential's cached gateway config, or an empty list."""
    config = get_radius_credential_config(credential)
    return get_radius_models_from_config(provider_id, config) if config else []


def _truncate_http_body(body: str) -> str:
    trimmed = body.strip()
    return f"{trimmed[:512]}…" if len(trimmed) > 512 else trimmed


async def load_radius_gateway_config(
    gateway: str,
    api_key: str | None = None,
    signal: AbortSignal | None = None,
) -> RadiusGatewayConfig:
    """Fetch and validate `GET {gateway}/v1/config`."""
    headers = {"accept": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    if signal is not None:
        signal.throw_if_aborted()
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
        response = await client.get(f"{normalize_radius_gateway_url(gateway)}/v1/config", headers=headers)
    if response.status_code >= 400:
        raise RuntimeError(
            f"Could not load Radius config from {gateway}: {response.status_code}: {_truncate_http_body(response.text)}"
        )
    config = sanitize_radius_gateway_config(response.json())
    if config is None:
        raise RuntimeError(f"Invalid Radius config from {gateway}")
    return config
