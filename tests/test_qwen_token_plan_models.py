"""Python port of `packages/ai/test/qwen-token-plan-models.test.ts`.

TypeScript mocks the `openai` SDK to keep `streamSimple` offline. This port
passes an `httpx.MockTransport`-backed client instead; both stop the request at
the same point, after `on_payload` has observed the built params.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pi_ai.compat import register_builtin_api_providers, stream_simple
from pi_ai.env_api_keys import find_env_keys
from pi_ai.providers.all import get_builtin_models
from pi_ai.types import Context, Model, SimpleStreamOptions, UserMessage, now_ms

register_builtin_api_providers()

TEXT_MODELS = [
    "MiniMax-M2.5",
    "deepseek-v3.2",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "glm-5",
    "glm-5.1",
    "glm-5.2",
    "kimi-k2.5",
    "kimi-k2.6",
    "kimi-k2.7-code",
    "qwen3.6-flash",
    "qwen3.6-plus",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.8-max",
]

INDIVIDUAL_TEXT_MODELS = [
    "deepseek-v4-flash-0731",
    "deepseek-v4-pro",
    "glm-5.2",
    "qwen3.6-flash",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.8-max",
]

IMAGE_MODELS = ["qwen-image-2.0", "qwen-image-2.0-pro", "wan2.7-image", "wan2.7-image-pro"]

QWEN_THINKING_MODELS = [
    "deepseek-v3.2",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "glm-5",
    "glm-5.1",
    "glm-5.2",
    "kimi-k2.5",
    "kimi-k2.6",
    "kimi-k2.7-code",
    "qwen3.6-flash",
    "qwen3.6-plus",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.8-max",
]

QWEN_THINKING_MODEL_CASES = [
    (provider, model_id) for provider in ("qwen-token-plan", "qwen-token-plan-cn") for model_id in QWEN_THINKING_MODELS
] + [("qwen-token-plan-individual", model_id) for model_id in INDIVIDUAL_TEXT_MODELS]

QWEN_REASONING_EFFORT_MODELS = ["deepseek-v4-flash", "deepseek-v4-pro", "glm-5", "glm-5.1", "glm-5.2"]

QWEN_REASONING_EFFORT_MODEL_CASES = [
    (provider, model_id)
    for provider in ("qwen-token-plan", "qwen-token-plan-cn")
    for model_id in QWEN_REASONING_EFFORT_MODELS
] + [("qwen-token-plan-individual", model_id) for model_id in ("deepseek-v4-flash-0731", "deepseek-v4-pro", "glm-5.2")]

ALL_PROVIDERS = ["qwen-token-plan", "qwen-token-plan-cn", "qwen-token-plan-individual"]

_CHUNK: dict[str, Any] = {
    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    "usage": {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
}


def find_model(provider: str, model_id: str) -> Model:
    model = next((m for m in get_builtin_models(provider) if m.id == model_id), None)
    assert model is not None, f"Missing model: {provider}/{model_id}"
    return model


async def capture_payload(model: Model, reasoning: str) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    body = f"data: {json.dumps(_CHUNK)}\n\ndata: [DONE]\n\n"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def on_payload(params: dict[str, Any], _model: Model) -> None:
        captured["payload"] = params

    await stream_simple(
        model,
        Context(messages=[UserMessage(content="Hi", timestamp=now_ms())]),
        SimpleStreamOptions(api_key="test", reasoning=reasoning, on_payload=on_payload),
        client=client,
    ).result()

    assert "payload" in captured, "on_payload was never called"
    return captured["payload"]


def test_exposes_exactly_the_documented_individual_text_models() -> None:
    model_ids = sorted(model.id for model in get_builtin_models("qwen-token-plan-individual"))
    assert model_ids == sorted(INDIVIDUAL_TEXT_MODELS)


def test_reuses_the_international_token_plan_environment_variable() -> None:
    assert find_env_keys("qwen-token-plan-individual", {"QWEN_TOKEN_PLAN_API_KEY": "test"}) == [
        "QWEN_TOKEN_PLAN_API_KEY"
    ]


@pytest.mark.parametrize("provider", ["qwen-token-plan", "qwen-token-plan-cn"])
def test_exposes_all_text_models(provider: str) -> None:
    model_ids = [model.id for model in get_builtin_models(provider)]
    for expected in TEXT_MODELS:
        assert expected in model_ids, f"{provider} should include {expected}"


@pytest.mark.parametrize("provider", ["qwen-token-plan", "qwen-token-plan-cn"])
def test_omits_image_models(provider: str) -> None:
    model_ids = [model.id for model in get_builtin_models(provider)]
    for excluded in IMAGE_MODELS:
        assert excluded not in model_ids, f"{provider} should not include {excluded}"


# docs: https://modelstudio.console.alibabacloud.com/ap-southeast-1?tab=api&commonbuy=1#/api/?type=model&url=3016807
@pytest.mark.parametrize(("provider", "model_id"), QWEN_THINKING_MODEL_CASES)
async def test_sends_qwen_thinking_fields(provider: str, model_id: str) -> None:
    payload = await capture_payload(find_model(provider, model_id), "high")

    assert payload["enable_thinking"] is True
    assert "thinking" not in payload


@pytest.mark.parametrize(("provider", "model_id"), QWEN_REASONING_EFFORT_MODEL_CASES)
def test_exposes_qwen_reasoning_effort_levels(provider: str, model_id: str) -> None:
    model = find_model(provider, model_id)

    for key, value in {
        "minimal": None,
        "low": None,
        "medium": None,
        "high": "high",
        "xhigh": None,
        "max": "max",
    }.items():
        assert model.thinking_level_map.get(key) == value, key


@pytest.mark.parametrize("provider", ALL_PROVIDERS)
def test_exposes_qwen38_reasoning_effort_levels(provider: str) -> None:
    model = find_model(provider, "qwen3.8-max")

    for key, value in {
        "minimal": None,
        "low": "low",
        "medium": "medium",
        "high": None,
        "xhigh": "xhigh",
        "max": None,
    }.items():
        assert model.thinking_level_map.get(key) == value, key


@pytest.mark.parametrize("provider", ALL_PROVIDERS)
def test_omits_retired_qwen38_max_preview(provider: str) -> None:
    model_ids = [model.id for model in get_builtin_models(provider)]
    assert "qwen3.8-max-preview" not in model_ids


@pytest.mark.parametrize(("provider", "model_id"), QWEN_REASONING_EFFORT_MODEL_CASES)
async def test_sends_qwen_reasoning_effort(provider: str, model_id: str) -> None:
    payload = await capture_payload(find_model(provider, model_id), "high")

    assert payload["reasoning_effort"] == "high"


@pytest.mark.parametrize("provider", ALL_PROVIDERS)
async def test_sends_qwen38_max_reasoning_effort(provider: str) -> None:
    payload = await capture_payload(find_model(provider, "qwen3.8-max"), "xhigh")

    assert payload["enable_thinking"] is True
    assert payload["reasoning_effort"] == "xhigh"
    assert "thinking" not in payload
