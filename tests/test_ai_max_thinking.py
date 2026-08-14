"""Python port of `packages/ai/test/max-thinking.test.ts`.

The final TypeScript case ("sends max to the Codex Responses API") drives
`openai-codex-responses`, which this port does not implement (it needs the Codex
OAuth/WebSocket transport; see the package README). Its `stream_simple` raises
`NotImplementedError`, so there is no payload to capture and the case is left
out. The catalog metadata it depends on is still pinned by the
`openai-codex/gpt-5.6-*` case below.
"""

from __future__ import annotations

import pytest
from pi_ai.models import clamp_thinking_level, get_supported_thinking_levels
from pi_ai.providers.all import get_builtin_model
from pi_ai.types import Model, ModelCost


def test_is_opt_in_for_ordinary_reasoning_models():
    model = Model(
        id="ordinary-reasoning",
        name="Ordinary Reasoning",
        api="openai-completions",
        provider="test",
        base_url="https://example.com/v1",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128000,
        max_tokens=4096,
    )

    assert get_supported_thinking_levels(model) == ["off", "minimal", "low", "medium", "high"]
    assert clamp_thinking_level(model, "max") == "high"


@pytest.mark.parametrize("model_id", ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"])
def test_exposes_xhigh_and_max_for_openai_codex(model_id: str):
    model = get_builtin_model("openai-codex", model_id)
    assert model is not None
    assert model.thinking_level_map["xhigh"] == "xhigh"
    assert model.thinking_level_map["max"] == "max"
    assert get_supported_thinking_levels(model) == [
        "off",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]


def test_supports_a_hole_between_high_and_max():
    model = Model(
        id="high-and-max",
        name="High and Max",
        api="openai-completions",
        provider="test",
        base_url="https://example.com/v1",
        reasoning=True,
        thinking_level_map={"xhigh": None, "max": "max"},
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128000,
        max_tokens=4096,
    )

    assert get_supported_thinking_levels(model) == ["off", "minimal", "low", "medium", "high", "max"]
    assert clamp_thinking_level(model, "xhigh") == "max"


@pytest.mark.skip(
    reason=(
        "TS case 'sends max to the Codex Responses API' (max-thinking.test.ts) drives "
        "streamSimple from src/api/openai-codex-responses.ts with reasoning: 'max' and an "
        "onPayload callback that captures the request body before throwing to abort the "
        "stream; it then asserts payload toMatchObject({ reasoning: { effort: 'max', "
        "summary: 'auto' } }). This port's openai_codex_responses.stream_simple raises "
        "NotImplementedError unconditionally (documented omission - see module docstring), "
        "so there is no payload to capture and the assertion cannot be reproduced."
    )
)
def test_sends_max_to_the_codex_responses_api():
    pass
