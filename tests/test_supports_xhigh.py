"""Python port of `packages/ai/test/supports-xhigh.test.ts`."""

from __future__ import annotations

import pytest
from pi_ai.models import get_supported_thinking_levels
from pi_ai.providers.all import get_builtin_model


def levels(provider: str, model_id: str) -> list[str]:
    model = get_builtin_model(provider, model_id)
    assert model is not None
    return list(get_supported_thinking_levels(model))


def test_includes_max_but_not_xhigh_for_anthropic_opus_4_6() -> None:
    got = levels("anthropic", "claude-opus-4-6")
    assert "max" in got
    assert "xhigh" not in got


def test_includes_xhigh_and_max_for_anthropic_opus_4_8() -> None:
    got = levels("anthropic", "claude-opus-4-8")
    assert "xhigh" in got
    assert "max" in got


def test_includes_xhigh_and_max_for_anthropic_opus_5() -> None:
    got = levels("anthropic", "claude-opus-5")
    assert "xhigh" in got
    assert "max" in got


def test_includes_max_but_not_xhigh_for_anthropic_sonnet_4_6() -> None:
    got = levels("anthropic", "claude-sonnet-4-6")
    assert "max" in got
    assert "xhigh" not in got


def test_includes_xhigh_and_max_for_anthropic_sonnet_5() -> None:
    got = levels("anthropic", "claude-sonnet-5")
    assert "xhigh" in got
    assert "max" in got


def test_includes_xhigh_and_max_but_not_off_for_anthropic_claude_fable_5() -> None:
    got = levels("anthropic", "claude-fable-5")
    assert "xhigh" in got
    assert "max" in got
    assert "off" not in got


def test_does_not_include_xhigh_or_max_for_claude_sonnet_4_5() -> None:
    got = levels("anthropic", "claude-sonnet-4-5")
    assert "xhigh" not in got
    assert "max" not in got


@pytest.mark.parametrize("model_id", ["gpt-5.4", "gpt-5.5", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"])
def test_includes_xhigh_for_openai_codex_models(model_id: str) -> None:
    assert "xhigh" in levels("openai-codex", model_id)


@pytest.mark.parametrize("model_id", ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"])
def test_includes_xhigh_and_max_for_openai_models(model_id: str) -> None:
    assert levels("openai", model_id) == ["off", "low", "medium", "high", "xhigh", "max"]


def test_includes_only_medium_high_xhigh_for_openai_gpt_5_5_pro() -> None:
    assert levels("openai", "gpt-5.5-pro") == ["medium", "high", "xhigh"]


def test_includes_only_medium_high_xhigh_for_openrouter_gpt_5_5_pro() -> None:
    assert levels("openrouter", "openai/gpt-5.5-pro") == ["medium", "high", "xhigh"]


def test_includes_only_high_max_plus_off_for_deepseek_v4_flash_on_deepseek() -> None:
    assert levels("deepseek", "deepseek-v4-flash") == ["off", "high", "max"]


def test_includes_only_high_max_plus_off_for_deepseek_v4_flash_on_opencode_go() -> None:
    assert levels("opencode-go", "deepseek-v4-flash") == ["off", "high", "max"]


def test_includes_only_high_plus_off_for_opencode_go_kimi_k2_6() -> None:
    assert levels("opencode-go", "kimi-k2.6") == ["off", "high"]


@pytest.mark.parametrize("provider", ["moonshotai", "moonshotai-cn"])
def test_excludes_thinking_off_for_moonshot_kimi_k2_7_code_models(provider: str) -> None:
    assert levels(provider, "kimi-k2.7-code") == ["minimal", "low", "medium", "high"]


@pytest.mark.parametrize("provider", ["moonshotai", "moonshotai-cn"])
def test_uses_the_verified_effort_options_for_kimi_k3(provider: str) -> None:
    assert levels(provider, "kimi-k3") == ["low", "high", "max"]


def test_includes_only_low_high_max_for_kimi_coding_k3() -> None:
    assert levels("kimi-coding", "k3") == ["low", "high", "max"]


def test_includes_only_high_for_opencode_grok_build() -> None:
    assert levels("opencode", "grok-build-0.1") == ["high"]


def test_includes_only_high_xhigh_plus_off_for_deepseek_v4_flash_on_openrouter() -> None:
    assert levels("openrouter", "deepseek/deepseek-v4-flash") == ["off", "high", "xhigh"]


def test_includes_max_but_not_xhigh_for_openrouter_opus_4_6() -> None:
    got = levels("openrouter", "anthropic/claude-opus-4.6")
    assert "max" in got
    assert "xhigh" not in got


def test_includes_xhigh_and_max_for_bedrock_claude_opus_5() -> None:
    got = levels("amazon-bedrock", "global.anthropic.claude-opus-5")
    assert "xhigh" in got
    assert "max" in got


def test_includes_xhigh_and_max_but_not_off_for_bedrock_claude_fable_5() -> None:
    got = levels("amazon-bedrock", "global.anthropic.claude-fable-5")
    assert "xhigh" in got
    assert "max" in got
    assert "off" not in got
