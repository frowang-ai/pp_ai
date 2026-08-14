#!/usr/bin/env python3
"""Regenerate the built-in image-model catalog from the OpenRouter API.

Python port of `packages/ai/scripts/generate-image-models.ts`.

The TypeScript version emits a `src/image-models.generated.ts` module holding
one `IMAGE_MODELS` object literal. Emitting Python source would be the literal
translation, but this port already ships the chat catalog as committed JSON
shards read at runtime (`packages/ai/scripts/generate-models.ts` ->
`generate_models.py`), so the image catalog uses the same mechanism: one JSON
file per provider under `pi_ai/providers/data/images/`, read back by
:mod:`pi_ai.image_models`. The fetch, filtering and cost maths below are a
faithful port; only the output format differs.

Usage::

    uv run python packages/pi-ai/scripts/generate_image_models.py [--strict]

``--strict`` turns an empty or unusable OpenRouter response into a failure
instead of writing an empty catalog, exactly as upstream.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PACKAGE_ROOT / "src" / "pi_ai" / "providers" / "data" / "images"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_IMAGE_MODELS_URL = f"{OPENROUTER_BASE_URL}/models?output_modalities=image"

ImagesModelData = dict[str, Any]

_MODALITIES = ("text", "image")


def _unique_modalities(values: object) -> list[str]:
    """Keep only ``text``/``image``, de-duplicated, in first-seen order."""
    if not isinstance(values, list):
        return []
    seen: list[str] = []
    for value in values:
        if value in _MODALITIES and value not in seen:
            seen.append(str(value))
    return seen


def _price_per_million(value: object) -> float:
    """OpenRouter prices per token; the catalog stores cost per million tokens."""
    try:
        return float(value or 0) * 1_000_000
    except (TypeError, ValueError):
        return 0.0


def parse_openrouter_image_models(payload: object, strict: bool) -> list[ImagesModelData]:
    """Port of `parseOpenRouterImageModels`."""
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or not data:
        if strict:
            raise ValueError("OpenRouter API returned a missing or empty image model list")
        return []

    models: list[ImagesModelData] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        architecture = entry.get("architecture") or {}
        input_modalities = _unique_modalities(architecture.get("input_modalities"))
        output_modalities = _unique_modalities(architecture.get("output_modalities"))

        if "image" not in output_modalities:
            continue
        if not input_modalities:
            input_modalities.append("text")

        pricing = entry.get("pricing") or {}
        models.append(
            {
                "id": entry.get("id"),
                "name": entry.get("name"),
                "api": "openrouter-images",
                "provider": "openrouter",
                "baseUrl": OPENROUTER_BASE_URL,
                "input": input_modalities,
                "output": output_modalities,
                "cost": {
                    "input": _price_per_million(pricing.get("prompt")),
                    "output": _price_per_million(pricing.get("completion")),
                    "cacheRead": _price_per_million(pricing.get("input_cache_read")),
                    "cacheWrite": _price_per_million(pricing.get("input_cache_write")),
                },
            }
        )

    if strict and not models:
        raise ValueError("OpenRouter API returned no usable image models")
    return models


def fetch_openrouter_image_models(strict: bool) -> list[ImagesModelData]:
    """Port of `fetchOpenRouterImageModels`."""
    try:
        print("Fetching image models from OpenRouter API...")
        request = urllib.request.Request(
            OPENROUTER_IMAGE_MODELS_URL,
            headers={"User-Agent": "pi-ai-generate-image-models/1.0"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 200:
                raise RuntimeError(f"OpenRouter API returned {response.status}")
            payload = json.loads(response.read().decode("utf-8"))
        models = parse_openrouter_image_models(payload, strict)
        print(f"Fetched {len(models)} image models from OpenRouter")
        return models
    except Exception as error:
        print(f"Failed to fetch OpenRouter image models: {error}", file=sys.stderr)
        if strict:
            raise
        return []


def _normalize_numbers(value: Any) -> Any:
    """Serialize integral floats as integers, matching `JSON.stringify` output.

    Same helper as in `generate_models.py`: without it Python writes `0.0`
    where the TypeScript catalog has `0`.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: _normalize_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_numbers(item) for item in value]
    return value


def serialize_image_models(models: list[ImagesModelData]) -> str:
    """One provider shard: ``{model id: model}``, sorted by id.

    Replaces `generateImageModelsFile`, which serializes the same models as a
    TypeScript `IMAGE_MODELS` object literal.
    """
    catalog = {str(model["id"]): model for model in sorted(models, key=lambda model: str(model["id"]))}
    return json.dumps(_normalize_numbers(catalog), indent=2, ensure_ascii=False) + "\n"


def write_image_models(models: list[ImagesModelData], data_dir: Path | None = None) -> Path:
    directory = data_dir or DATA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "openrouter.json"
    path.write_text(serialize_image_models(models), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate the built-in image-model catalog.")
    parser.add_argument("--strict", action="store_true", help="fail instead of writing an empty catalog")
    args = parser.parse_args(argv)

    models = fetch_openrouter_image_models(args.strict)
    path = write_image_models(models)
    print(f"Generated {path} ({len(models)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
