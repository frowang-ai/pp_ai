"""Generated model catalog loading.

Python port of `packages/ai/src/model-catalog.ts` (`flattenModelCatalog` /
`ModelCatalog`) and of the JSON-shard loading that TypeScript performs through
its generated `providers/<provider>.models.ts` files.

TypeScript builds one `<provider>.models.ts` module per provider that imports
`./data/<provider>.json` and flattens its per-API groups into a single record.
The data directory is generated at build time and gitignored there. This port
has no code-generation step at import time, so the same JSON shards are
committed under `pi_ai/providers/data/` and read here at runtime:
:func:`load_model_catalog` reads one shard and returns the flattened
``{model id: Model}`` mapping that `ModelCatalog` describes.

JSON keys keep their exact TypeScript spelling (``baseUrl``, ``contextWindow``,
``thinkingLevelMap``, ``cost.cacheRead``, ...) because those files are written
by the generator port in `packages/ai/scripts/generate-models.ts`; the
conversion to snake_case happens here, when the JSON becomes a
:class:`~pi_ai.types.Model`. `packages/ai/src/providers/data-json.d.ts` exists
only to tell the TypeScript compiler that a `*.json` import has a value; the
Python side reads the same files with :mod:`json` and needs no such shim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any

from .types import Model, ModelCost, ModelCostTier

DATA_DIR = Path(__file__).resolve().parent / "providers" / "data"
"""Directory holding the committed per-provider JSON shards."""

MODEL_DATA_MANIFEST_FILE = ".manifest.json"
MODEL_DATA_SCHEMA_VERSION = 3

ModelGroups = dict[str, dict[str, dict[str, Any]]]
"""A provider shard: ``{api: {model id: model JSON}}``."""

ModelCatalog = dict[str, Model]
"""The flattened form of a provider shard, keyed by model id."""


@dataclass
class ModelDataManifest:
    """Port of `ModelDataManifest` in `packages/ai/scripts/model-data.ts`."""

    schema_version: int
    generated_at: str
    structure_hash: str
    files: dict[str, str]


def _cost_from_data(data: dict[str, Any] | None) -> ModelCost:
    data = data or {}
    tiers = [
        ModelCostTier(
            input_tokens_above=int(tier.get("inputTokensAbove", 0)),
            input=float(tier.get("input", 0)),
            output=float(tier.get("output", 0)),
            cache_read=float(tier.get("cacheRead", 0)),
            cache_write=float(tier.get("cacheWrite", 0)),
        )
        for tier in data.get("tiers", [])
    ]
    return ModelCost(
        input=float(data.get("input", 0)),
        output=float(data.get("output", 0)),
        cache_read=float(data.get("cacheRead", 0)),
        cache_write=float(data.get("cacheWrite", 0)),
        tiers=tiers,
    )


def model_from_data(data: dict[str, Any]) -> Model:
    """Build a :class:`~pi_ai.types.Model` from one generated JSON entry."""
    return Model(
        id=str(data["id"]),
        name=str(data.get("name") or data["id"]),
        api=str(data.get("api", "openai-completions")),
        provider=str(data.get("provider", "")),
        base_url=str(data.get("baseUrl", "")),
        reasoning=bool(data.get("reasoning", False)),
        thinking_level_map=dict(data.get("thinkingLevelMap") or {}),
        input=list(data.get("input") or ["text"]),
        cost=_cost_from_data(data.get("cost")),
        context_window=int(data.get("contextWindow", 0)),
        max_tokens=int(data.get("maxTokens", 0)),
        headers=dict(data.get("headers") or {}),
        compat=dict(data.get("compat") or {}),
    )


def flatten_model_catalog(provider: str, groups: ModelGroups) -> ModelCatalog:
    """Flatten a provider's per-API groups into one ``{model id: Model}`` mapping.

    The ``provider`` argument is unused at runtime (TypeScript only needs it to
    carry the provider id into the derived type), but it is kept so the two
    signatures line up and so callers read the same way.
    """
    catalog: ModelCatalog = {}
    for models in groups.values():
        for model_id, data in models.items():
            catalog[model_id] = model_from_data(data)
    return catalog


@cache
def _read_shard(path: str) -> ModelGroups:
    with open(path, encoding="utf-8") as handle:
        groups = json.load(handle)
    if not isinstance(groups, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return groups


def load_model_groups(provider_id: str, data_dir: Path | None = None) -> ModelGroups:
    """Read one provider's raw JSON shard, grouped by API."""
    path = (data_dir or DATA_DIR) / f"{provider_id}.json"
    if not path.is_file():
        return {}
    return _read_shard(str(path))


def load_model_catalog(provider_id: str, data_dir: Path | None = None) -> ModelCatalog:
    """Load and flatten one provider's generated catalog.

    Returns an empty mapping when the provider has no generated shard, so a
    provider factory keeps working against a partially hydrated data
    directory instead of failing at import time.
    """
    return flatten_model_catalog(provider_id, load_model_groups(provider_id, data_dir))


def load_models(provider_id: str, data_dir: Path | None = None) -> list[Model]:
    """The generated catalog as the list a provider factory passes to `create_provider`."""
    return list(load_model_catalog(provider_id, data_dir).values())


def get_model_data_provider_ids(data_dir: Path | None = None) -> list[str]:
    """Every provider id that has a generated shard, sorted."""
    directory = data_dir or DATA_DIR
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.json") if path.name != MODEL_DATA_MANIFEST_FILE)


def read_model_data_manifest(data_dir: Path | None = None) -> ModelDataManifest | None:
    """Read `.manifest.json`, or ``None`` when the data directory is not hydrated."""
    path = (data_dir or DATA_DIR) / MODEL_DATA_MANIFEST_FILE
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return None
    return ModelDataManifest(
        schema_version=int(data.get("schemaVersion", 0)),
        generated_at=str(data.get("generatedAt", "")),
        structure_hash=str(data.get("structureHash", "")),
        files=dict(data.get("files") or {}),
    )


def get_model_data_generated_at(data_dir: Path | None = None) -> int | None:
    """The catalog generation timestamp in milliseconds, or ``None`` if unknown."""
    manifest = read_model_data_manifest(data_dir)
    if manifest is None or not manifest.generated_at:
        return None
    try:
        parsed = datetime.fromisoformat(manifest.generated_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return int(parsed.timestamp() * 1000)
