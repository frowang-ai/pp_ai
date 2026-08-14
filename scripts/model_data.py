"""Generated model-data manifest and validation.

Python port of `packages/ai/scripts/model-data.ts` and its CLI entry point
`packages/ai/scripts/check-model-data.ts`.

One structural difference: TypeScript derives the set of generated providers
from the imports in `src/models.generated.ts`, because its shards are
`<provider>.models.ts` modules generated next to a gitignored data directory.
This port has no generated modules — the JSON shards under
`pi_ai/providers/data/` are the committed source of truth — so the provider
set comes from the shard filenames instead.

Run this module directly to validate the committed data directory, the
equivalent of `check-model-data.ts`.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

MODEL_DATA_SCHEMA_VERSION = 3
MODEL_DATA_MANIFEST_FILE = ".manifest.json"

ModelDataStructure = dict[str, dict[str, str]]
"""``{provider id: {model id: api}}`` for every generated shard."""

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PACKAGE_ROOT / "src" / "pi_ai" / "providers" / "data"


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sorted_record(entries: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    return dict(sorted(entries, key=lambda entry: entry[0]))


def describe_set_difference(expected: Iterable[str], actual: Iterable[str]) -> str:
    expected_list = list(expected)
    actual_list = list(actual)
    expected_set = set(expected_list)
    actual_set = set(actual_list)
    missing = [value for value in expected_list if value not in actual_set]
    extra = [value for value in actual_list if value not in expected_set]
    parts = []
    if missing:
        parts.append(f"missing: {', '.join(missing)}")
    if extra:
        parts.append(f"extra: {', '.join(extra)}")
    return "; ".join(parts)


def assert_exact_model_ids(label: str, expected: Iterable[str], actual: Iterable[str]) -> None:
    expected_ids = sorted(set(expected))
    actual_ids = sorted(set(actual))
    if expected_ids == actual_ids:
        return
    raise ValueError(f"{label} model IDs do not match ({describe_set_difference(expected_ids, actual_ids)})")


def read_model_data_provider_ids(data_dir: Path | None = None) -> list[str]:
    """Every provider that has a generated shard, sorted."""
    directory = data_dir or DATA_DIR
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.json") if path.name != MODEL_DATA_MANIFEST_FILE)


def read_provider_structure(path: Path, provider_id: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as handle:
        groups = json.load(handle)
    if not isinstance(groups, dict):
        raise ValueError(f"{provider_id}.json must contain a JSON object")

    models: dict[str, str] = {}
    for api, value in groups.items():
        if not isinstance(value, dict):
            raise ValueError(f"{path} API group {api!r} must be an object")
        for model_id in value:
            if model_id in models:
                raise ValueError(f"{path} contains model {model_id} in more than one API group")
            models[model_id] = api
    if not models:
        raise ValueError(f"{path} contains no generated model data")
    return sorted_record(models.items())


def read_model_data_structure(data_dir: Path | None = None) -> ModelDataStructure:
    directory = data_dir or DATA_DIR
    return sorted_record(
        (provider_id, read_provider_structure(directory / f"{provider_id}.json", provider_id))
        for provider_id in read_model_data_provider_ids(directory)
    )


def model_data_structure_hash(structure: ModelDataStructure) -> str:
    normalized = sorted_record(
        (provider_id, sorted_record(models.items())) for provider_id, models in structure.items()
    )
    return sha256(json.dumps(normalized, separators=(",", ":"), ensure_ascii=False))


def create_model_data_manifest(
    structure: ModelDataStructure,
    file_contents: Mapping[str, str],
    generated_at: str,
) -> dict[str, Any]:
    return {
        "schemaVersion": MODEL_DATA_SCHEMA_VERSION,
        "generatedAt": generated_at,
        "structureHash": model_data_structure_hash(structure),
        "files": sorted_record((file, sha256(content)) for file, content in file_contents.items()),
    }


def _validate_model_value(
    value: Any,
    provider_id: str,
    model_id: str,
    expected_api: str,
    errors: list[str],
) -> None:
    label = f"{provider_id}/{model_id}"
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return
    if value.get("id") != model_id:
        errors.append(f"{label} has id {value.get('id')!r}, expected {model_id!r}")
    if value.get("provider") != provider_id:
        errors.append(f"{label} has provider {value.get('provider')!r}, expected {provider_id!r}")
    if value.get("api") != expected_api:
        errors.append(f"{label} has api {value.get('api')!r}, expected {expected_api!r}")
    name = value.get("name")
    if not isinstance(name, str) or not name:
        errors.append(f"{label} has no model name")
    if not isinstance(value.get("baseUrl"), str):
        errors.append(f"{label} has no baseUrl string")
    if not isinstance(value.get("reasoning"), bool):
        errors.append(f"{label} has no reasoning boolean")
    modalities = value.get("input")
    if (
        not isinstance(modalities, list)
        or not modalities
        or any(entry not in ("text", "image") for entry in modalities)
    ):
        errors.append(f"{label} has invalid input modalities")
    for field, message in (("contextWindow", "contextWindow"), ("maxTokens", "maxTokens")):
        number = value.get(field)
        if not isinstance(number, int | float) or isinstance(number, bool) or number <= 0:
            errors.append(f"{label} has invalid {message}")
    cost = value.get("cost")
    if not isinstance(cost, dict):
        errors.append(f"{label} has invalid cost metadata")
    else:
        for field in ("input", "output", "cacheRead", "cacheWrite"):
            rate = cost.get(field)
            if not isinstance(rate, int | float) or isinstance(rate, bool):
                errors.append(f"{label} has invalid cost.{field}")


def _is_valid_generated_at(value: Any) -> bool:
    """Mirrors TypeScript's `Number.isNaN(Date.parse(generatedAt))` guard: the
    stamp must be a string that actually parses as a date, not just a string."""
    if not isinstance(value, str):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def validate_model_data_directory(structure: ModelDataStructure, data_dir: Path) -> None:
    if not data_dir.is_dir():
        raise ValueError(f"Generated model data directory does not exist: {data_dir}")

    errors: list[str] = []
    expected_files = sorted(f"{provider_id}.json" for provider_id in structure)
    actual_files = sorted(path.name for path in data_dir.glob("*.json") if path.name != MODEL_DATA_MANIFEST_FILE)
    if expected_files != actual_files:
        errors.append(
            "provider data files do not match the generated catalog "
            f"({describe_set_difference(expected_files, actual_files)})"
        )

    manifest_path = data_dir / MODEL_DATA_MANIFEST_FILE
    manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = loaded if isinstance(loaded, dict) else None
        except ValueError as error:
            errors.append(f"model data manifest is not valid JSON: {error}")
    if manifest is None:
        errors.append("model data manifest is missing")
    else:
        if manifest.get("schemaVersion") != MODEL_DATA_SCHEMA_VERSION:
            errors.append(
                f"model data schema is {manifest.get('schemaVersion')!r}, expected {MODEL_DATA_SCHEMA_VERSION}"
            )
        if not _is_valid_generated_at(manifest.get("generatedAt")):
            errors.append("model data manifest has an invalid generation timestamp")
        if manifest.get("structureHash") != model_data_structure_hash(structure):
            errors.append("model data generation stamp does not match the generated catalog")

    manifest_files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(manifest_files, dict):
        manifest_files = None
        errors.append("model data manifest has no file hashes")
    elif expected_files != sorted(manifest_files):
        errors.append(
            "manifest file hashes do not match provider data files "
            f"({describe_set_difference(expected_files, sorted(manifest_files))})"
        )

    for provider_id, expected_models in structure.items():
        filename = f"{provider_id}.json"
        path = data_dir / filename
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        if manifest_files is not None and manifest_files.get(filename) != sha256(content):
            errors.append(f"{filename} does not match its manifest hash")
        groups = json.loads(content)
        if not isinstance(groups, dict):
            errors.append(f"{filename} must contain a JSON object")
            continue

        actual_models: dict[str, str] = {}
        for api, value in groups.items():
            if not isinstance(value, dict):
                errors.append(f"{filename} API group {api!r} must be an object")
                continue
            for model_id, model in value.items():
                if model_id in actual_models:
                    errors.append(f"{provider_id}/{model_id} appears in more than one API group")
                    continue
                actual_models[model_id] = api
                _validate_model_value(model, provider_id, model_id, api, errors)

        if sorted(expected_models) != sorted(actual_models):
            errors.append(
                f"{filename} model IDs do not match the generated catalog "
                f"({describe_set_difference(sorted(expected_models), sorted(actual_models))})"
            )
        for model_id, expected_api in expected_models.items():
            actual_api = actual_models.get(model_id)
            if actual_api is not None and actual_api != expected_api:
                errors.append(
                    f"{provider_id}/{model_id} is grouped under API {actual_api!r}, expected {expected_api!r}"
                )

    if errors:
        visible = errors[:30]
        suffix = f"\n  ... and {len(errors) - len(visible)} more" if len(errors) > len(visible) else ""
        listed = "\n".join(f"  - {error}" for error in visible)
        raise ValueError(f"Invalid generated model data:\n{listed}{suffix}")


def validate_generated_model_data(data_dir: Path | None = None) -> None:
    directory = data_dir or DATA_DIR
    validate_model_data_directory(read_model_data_structure(directory), directory)


def main() -> int:
    try:
        validate_generated_model_data()
    except (ValueError, OSError) as error:
        print(error, file=sys.stderr)
        print("\nModel data is missing or stale. Run `python scripts/generate_models.py`.", file=sys.stderr)
        return 1
    print("Generated model data is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
