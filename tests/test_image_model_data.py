"""Python port of `packages/ai/test/image-model-data.test.ts`."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_image_models import parse_openrouter_image_models  # noqa: E402

VALID_IMAGE_MODEL = {
    "id": "example/image-model",
    "name": "Example Image Model",
    "architecture": {
        "input_modalities": ["text", "image"],
        "output_modalities": ["image"],
    },
    "pricing": {
        "prompt": "0.000001",
        "completion": "0.000002",
    },
}


@pytest.mark.parametrize("payload", [{}, {"data": []}, {"data": "invalid"}])
def test_rejects_a_missing_or_empty_strict_catalog(payload: object):
    with pytest.raises(ValueError, match="missing or empty image model list"):
        parse_openrouter_image_models(payload, True)


def test_rejects_a_strict_catalog_with_no_usable_image_models():
    with pytest.raises(ValueError, match="no usable image models"):
        parse_openrouter_image_models(
            {
                "data": [
                    {
                        **VALID_IMAGE_MODEL,
                        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                    }
                ]
            },
            True,
        )


def test_parses_a_non_empty_image_model_catalog():
    [model] = parse_openrouter_image_models({"data": [VALID_IMAGE_MODEL]}, True)
    assert model["id"] == "example/image-model"
    assert model["input"] == ["text", "image"]
    assert model["output"] == ["image"]
