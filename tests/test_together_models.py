"""Python port of `packages/ai/test/together-models.test.ts`."""

from __future__ import annotations

import pytest
from pi_ai.env_api_keys import find_env_keys, get_env_api_key
from pi_ai.providers.all import get_builtin_model
from pi_ai.types import ModelCost


@pytest.fixture(autouse=True)
def _clean_together_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)


def test_registers_the_default_kimi_k26_model_via_openai_compatible_chat_completions() -> None:
    model = get_builtin_model("together", "moonshotai/Kimi-K2.6")

    assert model is not None
    assert model.api == "openai-completions"
    assert model.provider == "together"
    assert model.base_url == "https://api.together.ai/v1"
    assert model.reasoning is True
    assert model.thinking_level_map == {"minimal": None, "low": None, "medium": None}
    assert model.input == ["text", "image"]
    assert model.context_window == 262144
    assert model.max_tokens == 131000
    assert model.cost == ModelCost(input=1.2, output=4.5, cache_read=0.2, cache_write=0.0)
    assert model.compat == {
        "supportsStore": False,
        "supportsDeveloperRole": False,
        "supportsReasoningEffort": False,
        "maxTokensField": "max_tokens",
        "thinkingFormat": "together",
        "supportsStrictMode": False,
        "supportsLongCacheRetention": False,
    }


def test_models_together_reasoning_controls_from_the_together_api_surface() -> None:
    gpt_oss = get_builtin_model("together", "openai/gpt-oss-120b")
    assert gpt_oss is not None
    assert gpt_oss.thinking_level_map == {
        "off": None,
        "minimal": None,
        "low": "low",
        "medium": "medium",
        "high": "high",
        "max": None,
        "xhigh": None,
    }
    assert gpt_oss.compat["supportsReasoningEffort"] is True
    assert gpt_oss.compat["thinkingFormat"] == "openai"

    deep_seek_v4 = get_builtin_model("together", "deepseek-ai/DeepSeek-V4-Pro")
    assert deep_seek_v4 is not None
    assert deep_seek_v4.thinking_level_map == {
        "minimal": None,
        "low": None,
        "medium": None,
        "high": "high",
        "xhigh": None,
    }
    assert deep_seek_v4.compat["supportsReasoningEffort"] is True
    assert deep_seek_v4.compat["thinkingFormat"] == "together"

    minimax = get_builtin_model("together", "MiniMaxAI/MiniMax-M2.7")
    assert minimax is not None
    assert minimax.thinking_level_map == {"off": None, "minimal": None, "low": None, "medium": None}
    assert minimax.compat.get("thinkingFormat") is None
    assert minimax.compat["supportsReasoningEffort"] is False


def test_resolves_together_api_key_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOGETHER_API_KEY", "test-together-key")

    assert find_env_keys("together") == ["TOGETHER_API_KEY"]
    assert get_env_api_key("together") == "test-together-key"
