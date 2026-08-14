"""Python port of `packages/ai/test/model-data-validation.test.ts`.

No direct equivalent: "rejects missing provider shards imported by the aggregator".
TypeScript derives the generated-provider set from the `import` statements in
`src/models.generated.ts` and cross-checks it against the shard files, so a
shard imported but not present is an error. The port has no generated modules
at all (`packages/pi-ai/scripts/model_data.py` documents this): the JSON shards
under `pi_ai/providers/data/` *are* the source of truth, and the provider set
comes from the shard filenames, so there is no aggregator to disagree with. The
last test in this file pins that replacement behavior.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from model_data import (
    MODEL_DATA_MANIFEST_FILE,
    MODEL_DATA_SCHEMA_VERSION,
    ModelDataStructure,
    assert_exact_model_ids,
    create_model_data_manifest,
    read_model_data_structure,
    validate_model_data_directory,
)

GENERATED_AT = "2026-07-23T10:00:00.000Z"

STRUCTURE: ModelDataStructure = {"test-provider": {"model-a": "openai-completions"}}


def model_a_values() -> dict[str, object]:
    return {
        "model-a": {
            "id": "model-a",
            "name": "Model A",
            "api": "openai-completions",
            "provider": "test-provider",
            "baseUrl": "https://example.test/v1",
            "reasoning": False,
            "input": ["text"],
            "cost": {"input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 1000,
            "maxTokens": 100,
        }
    }


def write_fixture_data(
    data_dir: Path,
    structure: ModelDataStructure,
    values: dict[str, object],
    manifest_schema_version: int = MODEL_DATA_SCHEMA_VERSION,
    api_group: str = "openai-completions",
) -> None:
    filename = "test-provider.json"
    content = json.dumps({api_group: values}) + "\n"
    (data_dir / filename).write_text(content, encoding="utf-8")
    manifest = create_model_data_manifest(structure, {filename: content}, GENERATED_AT)
    manifest["schemaVersion"] = manifest_schema_version
    (data_dir / MODEL_DATA_MANIFEST_FILE).write_text(json.dumps(manifest) + "\n", encoding="utf-8")


@pytest.fixture
def fixture(tmp_path: Path) -> tuple[Path, ModelDataStructure, dict[str, object]]:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    values = model_a_values()
    write_fixture_data(data_dir, STRUCTURE, values)
    return data_dir, STRUCTURE, values


def read_manifest(data_dir: Path) -> dict[str, object]:
    return json.loads((data_dir / MODEL_DATA_MANIFEST_FILE).read_text(encoding="utf-8"))


def write_manifest(data_dir: Path, manifest: dict[str, object]) -> None:
    (data_dir / MODEL_DATA_MANIFEST_FILE).write_text(json.dumps(manifest) + "\n", encoding="utf-8")


def test_rejects_a_missing_upstream_model_from_an_exact_generated_allowlist() -> None:
    with pytest.raises(ValueError, match=r"qwen-token-plan-individual model IDs do not match \(missing: model-b\)"):
        assert_exact_model_ids("qwen-token-plan-individual", ["model-a", "model-b"], ["model-a"])


def test_rejects_an_unexpected_model_from_an_exact_generated_allowlist() -> None:
    with pytest.raises(ValueError, match=r"test-provider model IDs do not match \(extra: model-b\)"):
        assert_exact_model_ids("test-provider", ["model-a"], ["model-a", "model-b"])


def test_reads_and_validates_api_grouped_model_data(
    fixture: tuple[Path, ModelDataStructure, dict[str, object]],
) -> None:
    data_dir, structure, _values = fixture
    assert read_model_data_structure(data_dir) == structure
    validate_model_data_directory(structure, data_dir)


def test_rejects_a_missing_model_data_directory(
    fixture: tuple[Path, ModelDataStructure, dict[str, object]],
) -> None:
    data_dir, structure, _values = fixture
    for child in data_dir.iterdir():
        child.unlink()
    data_dir.rmdir()
    with pytest.raises(ValueError, match="does not exist"):
        validate_model_data_directory(structure, data_dir)


@pytest.mark.parametrize(
    ("field", "value", "expected_message"),
    [
        ("id", "wrong-id", "has id"),
        ("provider", "wrong-provider", "has provider"),
        ("api", "anthropic-messages", "has api"),
    ],
)
def test_rejects_a_wrong_model_field(
    fixture: tuple[Path, ModelDataStructure, dict[str, object]],
    field: str,
    value: str,
    expected_message: str,
) -> None:
    data_dir, structure, values = fixture
    model = values["model-a"]
    assert isinstance(model, dict)
    model[field] = value
    write_fixture_data(data_dir, structure, values)
    with pytest.raises(ValueError, match=expected_message):
        validate_model_data_directory(structure, data_dir)


def test_rejects_a_model_in_the_wrong_api_group(
    fixture: tuple[Path, ModelDataStructure, dict[str, object]],
) -> None:
    data_dir, structure, values = fixture
    write_fixture_data(data_dir, structure, values, MODEL_DATA_SCHEMA_VERSION, "anthropic-messages")
    with pytest.raises(ValueError, match="grouped under API"):
        validate_model_data_directory(structure, data_dir)


def test_rejects_duplicate_model_ids_across_api_groups(
    fixture: tuple[Path, ModelDataStructure, dict[str, object]],
) -> None:
    data_dir, structure, values = fixture
    filename = "test-provider.json"
    content = json.dumps({"openai-completions": values, "anthropic-messages": values}) + "\n"
    (data_dir / filename).write_text(content, encoding="utf-8")
    write_manifest(data_dir, create_model_data_manifest(structure, {filename: content}, GENERATED_AT))
    with pytest.raises(ValueError, match="more than one API group"):
        validate_model_data_directory(structure, data_dir)


def test_rejects_missing_model_ids_and_stale_file_hashes(
    fixture: tuple[Path, ModelDataStructure, dict[str, object]],
) -> None:
    data_dir, structure, _values = fixture
    (data_dir / "test-provider.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"manifest hash|model IDs"):
        validate_model_data_directory(structure, data_dir)


def test_rejects_incompatible_schema_and_generation_stamps(
    fixture: tuple[Path, ModelDataStructure, dict[str, object]],
) -> None:
    data_dir, structure, values = fixture
    write_fixture_data(data_dir, structure, values, MODEL_DATA_SCHEMA_VERSION + 1)
    with pytest.raises(ValueError, match="model data schema"):
        validate_model_data_directory(structure, data_dir)

    manifest = read_manifest(data_dir)
    manifest["structureHash"] = "stale"
    write_manifest(data_dir, manifest)
    with pytest.raises(ValueError, match="generation stamp"):
        validate_model_data_directory(structure, data_dir)


def test_rejects_an_invalid_generation_timestamp(
    fixture: tuple[Path, ModelDataStructure, dict[str, object]],
) -> None:
    data_dir, structure, _values = fixture
    manifest = read_manifest(data_dir)
    manifest["generatedAt"] = "invalid"
    write_manifest(data_dir, manifest)
    with pytest.raises(ValueError, match="generation timestamp"):
        validate_model_data_directory(structure, data_dir)


def test_derives_the_provider_set_from_the_shards_instead_of_an_aggregator(
    fixture: tuple[Path, ModelDataStructure, dict[str, object]],
) -> None:
    # TS counterpart: "rejects missing provider shards imported by the aggregator".
    # TypeScript reads the provider set from the `import` statements in
    # `src/models.generated.ts`, so an import without a matching shard file
    # throws "aggregator and provider shards do not match". The port has no
    # generated aggregator module: `read_model_data_structure` globs the shard
    # files themselves, so the two can never disagree. Pinning that instead:
    # the returned structure is exactly what is on disk, and a shard that only
    # an aggregator would have mentioned simply does not appear.
    data_dir, structure, _values = fixture
    assert read_model_data_structure(data_dir) == structure

    (data_dir / "missing.json").write_text(
        json.dumps({"openai-completions": {"model-z": {"id": "model-z"}}}) + "\n", encoding="utf-8"
    )
    assert read_model_data_structure(data_dir) == {
        **structure,
        "missing": {"model-z": "openai-completions"},
    }
