"""The generated image-model catalog.

Python port of `packages/ai/src/image-models.ts` and its data source
`packages/ai/src/image-models.generated.ts`.

TypeScript generates a `IMAGE_MODELS` object literal into a checked-in
`.generated.ts` module and builds a two-level `Map` from it at import time.
This port keeps the same two-level shape but reads it from committed JSON under
`pi_ai/providers/data/images/`, one file per provider, written by
`packages/pi-ai/scripts/generate_image_models.py` — the same arrangement the
chat catalog uses (see :mod:`pi_ai.model_catalog`).

The TypeScript signatures are generic over the provider and model id so that
`getImageModel("openrouter", "…")` returns a model typed with its exact api.
Python has no equivalent of indexing a literal type, so the functions here take
plain strings and return :class:`~pi_ai.types.ImagesModel`; `KnownImagesProvider`
is likewise just `str` in :mod:`pi_ai.types`.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from .types import ImagesModel, ModelCost

IMAGES_DATA_DIR = Path(__file__).resolve().parent / "providers" / "data" / "images"
"""Directory holding the committed per-provider image-model shards."""

ImagesCatalog = dict[str, ImagesModel]
"""One provider's image models, keyed by model id."""


def images_model_from_data(data: dict[str, Any]) -> ImagesModel:
    """Build an :class:`~pi_ai.types.ImagesModel` from one generated JSON entry."""
    cost = data.get("cost") or {}
    return ImagesModel(
        id=str(data["id"]),
        name=str(data.get("name") or data["id"]),
        api=str(data.get("api", "openrouter-images")),
        provider=str(data.get("provider", "")),
        base_url=str(data.get("baseUrl", "")),
        input=list(data.get("input") or ["text"]),
        output=list(data.get("output") or ["image"]),
        cost=ModelCost(
            input=float(cost.get("input", 0)),
            output=float(cost.get("output", 0)),
            cache_read=float(cost.get("cacheRead", 0)),
            cache_write=float(cost.get("cacheWrite", 0)),
        ),
        headers=dict(data.get("headers") or {}),
    )


@cache
def _read_images_shard(path: str) -> dict[str, dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        catalog = json.load(handle)
    if not isinstance(catalog, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return catalog


def load_image_catalog(provider: str, data_dir: Path | None = None) -> ImagesCatalog:
    """Load one provider's generated image catalog, keyed by model id.

    Returns an empty mapping for an unknown provider, so a provider factory
    keeps working against a partially hydrated data directory instead of
    failing at import time.
    """
    path = (data_dir or IMAGES_DATA_DIR) / f"{provider}.json"
    if not path.is_file():
        return {}
    return {model_id: images_model_from_data(data) for model_id, data in _read_images_shard(str(path)).items()}


def get_image_providers(data_dir: Path | None = None) -> list[str]:
    """Every image provider id that has a generated shard, sorted."""
    directory = data_dir or IMAGES_DATA_DIR
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.json"))


def get_image_models(provider: str, data_dir: Path | None = None) -> list[ImagesModel]:
    """Every generated image model of one provider."""
    return list(load_image_catalog(provider, data_dir).values())


def get_image_model(provider: str, model_id: str, data_dir: Path | None = None) -> ImagesModel | None:
    """One generated image model, or ``None`` when the provider or id is unknown.

    TypeScript's `getImageModel` is typed as always returning a model and
    returns `undefined` at runtime for an unknown id; this returns ``None``
    explicitly so callers have to handle it.
    """
    return load_image_catalog(provider, data_dir).get(model_id)
