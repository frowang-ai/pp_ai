"""Python port of `packages/ai/test/openrouter-cache-control-models.test.ts`."""

from __future__ import annotations

import pytest
from pi_ai.providers.all import get_builtin_model

OPENROUTER_ANTHROPIC_LATEST_MODEL_IDS = [
    "~anthropic/claude-fable-latest",
    "~anthropic/claude-haiku-latest",
    "~anthropic/claude-opus-latest",
    "~anthropic/claude-sonnet-latest",
]


@pytest.mark.parametrize("model_id", OPENROUTER_ANTHROPIC_LATEST_MODEL_IDS)
def test_enables_cache_control_for_openrouter_anthropic_latest(model_id: str) -> None:
    model = get_builtin_model("openrouter", model_id)
    assert model is not None
    assert model.compat.get("cacheControlFormat") == "anthropic"
