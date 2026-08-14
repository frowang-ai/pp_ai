"""Python port of `packages/ai/test/baseten-models.test.ts`."""

from __future__ import annotations

from typing import Any

import pytest
from pi_ai.compat import stream_simple
from pi_ai.env_api_keys import find_env_keys, get_env_api_key
from pi_ai.models import get_supported_thinking_levels
from pi_ai.providers.all import get_builtin_model
from pi_ai.types import Context, Model, SimpleStreamOptions, UserMessage


class PayloadCaptured(Exception):
    def __init__(self) -> None:
        super().__init__("payload captured")


async def capture_payload(model: Model, options: SimpleStreamOptions) -> dict[str, Any]:
    captured: dict[str, Any] | None = None

    def on_payload(payload: dict[str, Any], _model: Model) -> None:
        nonlocal captured
        captured = payload
        raise PayloadCaptured()

    options.on_payload = on_payload
    await stream_simple(model, Context(messages=[UserMessage(content="test")]), options).result()
    assert captured is not None
    return captured


def test_registers_glm_5_2_as_the_default_openai_compatible_reasoning_model():
    model = get_builtin_model("baseten", "zai-org/GLM-5.2")

    assert model.api == "openai-completions"
    assert model.provider == "baseten"
    assert model.base_url == "https://inference.baseten.co/v1"
    assert model.reasoning is True
    assert model.thinking_level_map == {
        "off": "none",
        "minimal": None,
        "low": None,
        "medium": None,
        "high": "high",
        "xhigh": None,
        "max": "max",
    }
    assert model.input == ["text"]
    assert model.context_window == 1048576
    assert model.max_tokens == 262144
    assert model.cost.input == 1.4
    assert model.cost.output == 4.4
    assert model.cost.cache_read == 0.3
    assert model.cost.cache_write == 0
    for key, value in {
        "supportsStore": False,
        "supportsDeveloperRole": False,
        "supportsReasoningEffort": True,
        "supportsUsageInStreaming": True,
        "maxTokensField": "max_tokens",
        "supportsStrictMode": True,
        "supportsLongCacheRetention": False,
        "thinkingFormat": "baseten",
        "chatTemplateArgs": {"enable_thinking": {"$var": "thinking.enabled"}},
    }.items():
        assert model.compat[key] == value


async def test_models_kimi_k2_6_reasoning_as_an_explicit_off_on_toggle():
    model = get_builtin_model("baseten", "moonshotai/Kimi-K2.6")

    assert model.thinking_level_map == {
        "off": "off",
        "minimal": None,
        "low": None,
        "medium": None,
        "high": "high",
        "xhigh": None,
        "max": None,
    }
    assert model.compat["supportsReasoningEffort"] is False
    assert model.compat["thinkingFormat"] == "baseten"
    assert model.compat["chatTemplateArgs"] == {"enable_thinking": {"$var": "thinking.enabled"}}
    assert get_supported_thinking_levels(model) == ["off", "high"]

    payload = await capture_payload(model, SimpleStreamOptions(api_key="test-baseten-key", reasoning="high"))
    assert payload["chat_template_args"] == {"enable_thinking": True}
    assert "reasoning_effort" not in payload


async def test_sends_baseten_chat_template_args_with_reasoning_effort():
    model = get_builtin_model("baseten", "zai-org/GLM-5.2")
    payload = await capture_payload(model, SimpleStreamOptions(api_key="test-baseten-key", reasoning="high"))
    assert payload["chat_template_args"] == {"enable_thinking": True}
    assert payload["reasoning_effort"] == "high"


async def test_disables_baseten_opt_in_reasoning_when_thinking_is_off():
    model = get_builtin_model("baseten", "zai-org/GLM-5.2")
    payload = await capture_payload(model, SimpleStreamOptions(api_key="test-baseten-key"))
    assert payload["chat_template_args"] == {"enable_thinking": False}
    assert payload["reasoning_effort"] == "none"


def test_resolves_baseten_api_key_from_the_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BASETEN_API_KEY", "test-baseten-key")

    assert find_env_keys("baseten") == ["BASETEN_API_KEY"]
    assert get_env_api_key("baseten") == "test-baseten-key"
