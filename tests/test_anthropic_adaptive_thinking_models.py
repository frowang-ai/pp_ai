"""Python port of `packages/ai/test/anthropic-adaptive-thinking-models.test.ts`."""

from __future__ import annotations

import re

from pi_ai.providers.all import get_builtin_models, get_builtin_providers
from pi_ai.types import Model

EXPECTED_CURRENT_ADAPTIVE_THINKING_MODELS = [
    "anthropic/claude-fable-5",
    "anthropic/claude-opus-4-8",
    "anthropic/claude-opus-5",
    "anthropic/claude-sonnet-5",
    "cloudflare-ai-gateway/claude-fable-5",
    "kimi-coding/kimi-for-coding",
    "kimi-coding/k3",
    "kimi-coding/kimi-for-coding-highspeed",
    "opencode/claude-opus-4-8",
    "opencode/claude-opus-5",
    "vercel-ai-gateway/anthropic/claude-opus-4.8",
    "vercel-ai-gateway/anthropic/claude-opus-5",
    "vercel-ai-gateway/anthropic/claude-sonnet-5",
]

ADAPTIVE_MODEL_ID_PATTERN = re.compile(r"(opus[-.](4[-.][678]|5)|sonnet[-.]4[-.]6|sonnet[-.]5|fable[-.]5|kimi-coding/)")


def _get_all_models() -> list[Model]:
    return [model for provider in get_builtin_providers() for model in get_builtin_models(provider)]


def test_marks_built_in_anthropic_messages_models_that_use_adaptive_thinking():
    flagged_models = sorted(
        f"{model.provider}/{model.id}"
        for model in _get_all_models()
        if model.api == "anthropic-messages" and model.compat.get("forceAdaptiveThinking") is True
    )

    assert set(EXPECTED_CURRENT_ADAPTIVE_THINKING_MODELS) <= set(flagged_models)
    assert flagged_models == [model_id for model_id in flagged_models if ADAPTIVE_MODEL_ID_PATTERN.search(model_id)]
