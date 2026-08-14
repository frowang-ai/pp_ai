"""Tests for the image-model catalog generator.

`packages/pi-ai/scripts/generate_image_models.py` is the port of
`packages/ai/scripts/generate-image-models.ts`. Only its parsing and
serialization are tested; the fetch is never called, so no network is used.
"""

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import generate_image_models as gen  # noqa: E402
from pi_ai.image_models import get_image_model, load_image_catalog  # noqa: E402


def entry(**overrides):
    base = {
        "id": "vendor/model",
        "name": "Vendor: Model",
        "architecture": {"input_modalities": ["text"], "output_modalities": ["image"]},
        "pricing": {},
    }
    base.update(overrides)
    return base


def payload(*entries):
    return {"data": list(entries)}


def test_an_image_model_is_parsed_into_a_catalog_entry():
    [model] = gen.parse_openrouter_image_models(payload(entry()), False)
    assert model == {
        "id": "vendor/model",
        "name": "Vendor: Model",
        "api": "openrouter-images",
        "provider": "openrouter",
        "baseUrl": "https://openrouter.ai/api/v1",
        "input": ["text"],
        "output": ["image"],
        "cost": {"input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0},
    }


def test_models_that_cannot_output_images_are_dropped():
    text_only = entry(id="vendor/text", architecture={"input_modalities": ["text"], "output_modalities": ["text"]})
    assert gen.parse_openrouter_image_models(payload(text_only), False) == []


def test_a_model_with_no_usable_input_modality_defaults_to_text():
    [model] = gen.parse_openrouter_image_models(
        payload(entry(architecture={"input_modalities": ["audio"], "output_modalities": ["image"]})), False
    )
    assert model["input"] == ["text"]


def test_modalities_are_filtered_and_deduplicated_in_order():
    [model] = gen.parse_openrouter_image_models(
        payload(
            entry(
                architecture={
                    "input_modalities": ["image", "text", "image", "video"],
                    "output_modalities": ["image", "text", "image"],
                }
            )
        ),
        False,
    )
    assert model["input"] == ["image", "text"]
    assert model["output"] == ["image", "text"]


def test_prices_are_converted_to_cost_per_million_tokens():
    [model] = gen.parse_openrouter_image_models(
        payload(
            entry(
                pricing={
                    "prompt": "0.000002",
                    "completion": "0.00001",
                    "input_cache_read": "0.0000005",
                    "input_cache_write": "0.0000025",
                }
            )
        ),
        False,
    )
    assert model["cost"] == {"input": 2.0, "output": 10.0, "cacheRead": 0.5, "cacheWrite": 2.5}


def test_unparseable_prices_fall_back_to_zero():
    [model] = gen.parse_openrouter_image_models(payload(entry(pricing={"prompt": "free", "completion": None})), False)
    assert model["cost"]["input"] == 0.0
    assert model["cost"]["output"] == 0.0


def test_an_empty_payload_yields_an_empty_catalog():
    assert gen.parse_openrouter_image_models({"data": []}, False) == []
    assert gen.parse_openrouter_image_models({}, False) == []
    assert gen.parse_openrouter_image_models(None, False) == []


def test_strict_mode_rejects_a_missing_model_list():
    with pytest.raises(ValueError, match="missing or empty image model list"):
        gen.parse_openrouter_image_models({"data": []}, True)


def test_strict_mode_rejects_a_list_with_no_image_models():
    text_only = entry(architecture={"input_modalities": ["text"], "output_modalities": ["text"]})
    with pytest.raises(ValueError, match="no usable image models"):
        gen.parse_openrouter_image_models(payload(text_only), True)


def test_the_shard_is_sorted_by_model_id():
    models = gen.parse_openrouter_image_models(payload(entry(id="b/two"), entry(id="a/one")), False)
    assert list(json.loads(gen.serialize_image_models(models))) == ["a/one", "b/two"]


def test_integral_costs_serialize_as_integers():
    models = gen.parse_openrouter_image_models(payload(entry(pricing={"prompt": "0.000002"})), False)
    assert '"input": 2,' in gen.serialize_image_models(models)


def test_a_written_shard_loads_back_through_the_runtime_catalog(tmp_path):
    models = gen.parse_openrouter_image_models(payload(entry(pricing={"prompt": "0.000002"})), False)
    gen.write_image_models(models, tmp_path)

    assert (tmp_path / "openrouter.json").is_file()
    catalog = load_image_catalog("openrouter", tmp_path)
    assert list(catalog) == ["vendor/model"]
    assert get_image_model("openrouter", "vendor/model", tmp_path).cost.input == 2.0


def test_the_committed_shard_matches_what_the_generator_would_write():
    committed = json.loads((gen.DATA_DIR / "openrouter.json").read_text(encoding="utf-8"))
    rewritten = json.loads(gen.serialize_image_models(list(committed.values())))
    assert rewritten == committed
