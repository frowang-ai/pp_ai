"""Python port of `packages/ai/test/xiaomi-models.test.ts`."""

from __future__ import annotations

import pytest
from pi_ai.providers.all import get_builtin_model, get_builtin_models


@pytest.mark.parametrize("model_id", ["mimo-v2-flash", "mimo-v2-omni"])
def test_keeps_api_billing_models_on_the_api_billing_provider(model_id: str) -> None:
    assert get_builtin_model("xiaomi", model_id) is not None


@pytest.mark.parametrize(
    "provider",
    ["xiaomi-token-plan-cn", "xiaomi-token-plan-ams", "xiaomi-token-plan-sgp"],
)
def test_omits_api_billing_only_models_from_token_plan_providers(provider: str) -> None:
    model_ids = [model.id for model in get_builtin_models(provider)]
    assert "mimo-v2-flash" not in model_ids
    assert "mimo-v2-omni" not in model_ids
