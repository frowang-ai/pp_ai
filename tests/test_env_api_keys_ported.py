"""Python port of `packages/ai/test/env-api-keys.test.ts`.

Named `_ported` because `tests/test_env_api_keys.py` already exists: it is the
port's own unit test for `pi_ai/env_api_keys.py`, not a port of this file.
"""

from __future__ import annotations

import pytest
from pi_ai.env_api_keys import find_env_keys, get_env_api_key

MANAGED_ENV_VARS = (
    "COPILOT_GITHUB_TOKEN",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "ZAI_CODING_CN_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_OAUTH_TOKEN",
    "ANTHROPIC_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in MANAGED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_does_not_treat_generic_github_tokens_as_github_copilot_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GH_TOKEN", "gh-token")
    monkeypatch.setenv("GITHUB_TOKEN", "github-token")

    assert find_env_keys("github-copilot") is None
    assert get_env_api_key("github-copilot") is None


def test_resolves_github_copilot_credentials_from_copilot_github_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "copilot-token")
    monkeypatch.setenv("GH_TOKEN", "gh-token")
    monkeypatch.setenv("GITHUB_TOKEN", "github-token")

    assert find_env_keys("github-copilot") == ["COPILOT_GITHUB_TOKEN"]
    assert get_env_api_key("github-copilot") == "copilot-token"


def test_resolves_zai_china_coding_plan_credentials_from_zai_coding_cn_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZAI_CODING_CN_API_KEY", "zai-coding-cn-token")

    assert find_env_keys("zai-coding-cn") == ["ZAI_CODING_CN_API_KEY"]
    assert get_env_api_key("zai-coding-cn") == "zai-coding-cn-token"


def test_reports_anthropic_auth_token_but_preserves_oauth_token_api_key_lookup(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "auth-token")
    monkeypatch.setenv("ANTHROPIC_OAUTH_TOKEN", "oauth-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "api-key")

    assert find_env_keys("anthropic") == ["ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"]
    assert get_env_api_key("anthropic") == "oauth-token"


def test_does_not_return_anthropic_auth_token_as_an_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "auth-token")

    assert find_env_keys("anthropic") == ["ANTHROPIC_AUTH_TOKEN"]
    assert get_env_api_key("anthropic") is None


def test_preserves_anthropic_oauth_token_as_an_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_OAUTH_TOKEN", "oauth-token")

    assert find_env_keys("anthropic") == ["ANTHROPIC_OAUTH_TOKEN"]
    assert get_env_api_key("anthropic") == "oauth-token"


def test_falls_back_to_anthropic_api_key_for_api_key_lookup(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "api-key")

    assert get_env_api_key("anthropic") == "api-key"
