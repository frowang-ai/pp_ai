"""Tests for `env_api_keys.py` (well-known provider env-var lookup) and
`api/cloudflare.py` (endpoint URL templates). Never touches the real home
directory or real host environment: every relevant variable is scrubbed via
`monkeypatch.delenv`, and Vertex ADC lookups are pointed at `tmp_path`.
"""

from __future__ import annotations

import pi_ai.env_api_keys as env_api_keys
import pytest
from pi_ai.api import cloudflare

ALL_ENV_VARS = (
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_OAUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "COPILOT_GITHUB_TOKEN",
    "OPENAI_API_KEY",
    "XAI_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLOUD_API_KEY",
    "GOOGLE_CLOUD_PROJECT",
    "GCLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "AWS_PROFILE",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ALL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # `_has_vertex_adc_credentials` caches its ADC-file-exists check process-wide
    # the first time it runs without an explicit `GOOGLE_APPLICATION_CREDENTIALS`;
    # reset it so tests don't leak state into each other.
    monkeypatch.setattr(env_api_keys, "_cached_vertex_adc_credentials_exists", None)


# --------------------------------------------------------------------------
# find_env_keys
# --------------------------------------------------------------------------


def test_find_env_keys_unknown_provider_returns_none():
    assert env_api_keys.find_env_keys("totally-unknown-provider", {}) is None


def test_find_env_keys_single_var_provider_not_set():
    assert env_api_keys.find_env_keys("openai", {}) is None


def test_find_env_keys_single_var_provider_set():
    assert env_api_keys.find_env_keys("openai", {"OPENAI_API_KEY": "sk-1"}) == ["OPENAI_API_KEY"]


def test_find_env_keys_github_copilot_uses_dedicated_var():
    assert env_api_keys.find_env_keys("github-copilot", {}) is None
    assert env_api_keys.find_env_keys("github-copilot", {"COPILOT_GITHUB_TOKEN": "ghu_x"}) == ["COPILOT_GITHUB_TOKEN"]


def test_find_env_keys_anthropic_reports_all_configured_vars_in_priority_order():
    env = {"ANTHROPIC_AUTH_TOKEN": "auth", "ANTHROPIC_API_KEY": "key"}
    assert env_api_keys.find_env_keys("anthropic", env) == ["ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"]


def test_find_env_keys_anthropic_none_configured():
    assert env_api_keys.find_env_keys("anthropic", {}) is None


# --------------------------------------------------------------------------
# get_env_api_key: standard providers
# --------------------------------------------------------------------------


def test_get_env_api_key_unknown_provider_returns_none():
    assert env_api_keys.get_env_api_key("totally-unknown-provider", {}) is None


def test_get_env_api_key_standard_provider():
    assert env_api_keys.get_env_api_key("xai", {"XAI_API_KEY": "xk-1"}) == "xk-1"
    assert env_api_keys.get_env_api_key("xai", {}) is None


def test_get_env_api_key_anthropic_prefers_api_key_over_auth_token():
    env = {"ANTHROPIC_AUTH_TOKEN": "auth-tok", "ANTHROPIC_API_KEY": "api-key-tok"}
    assert env_api_keys.get_env_api_key("anthropic", env) == "api-key-tok"


def test_get_env_api_key_anthropic_skips_auth_token_only_setup():
    # ANTHROPIC_AUTH_TOKEN is reported by find_env_keys (for status/env discovery)
    # but must never be selected by get_env_api_key: requests need it passed as
    # `Authorization: Bearer`, not as a bare api key.
    env = {"ANTHROPIC_AUTH_TOKEN": "auth-tok"}
    assert env_api_keys.find_env_keys("anthropic", env) == ["ANTHROPIC_AUTH_TOKEN"]
    assert env_api_keys.get_env_api_key("anthropic", env) is None


def test_get_env_api_key_anthropic_falls_back_to_oauth_token():
    env = {"ANTHROPIC_OAUTH_TOKEN": "oauth-tok"}
    assert env_api_keys.get_env_api_key("anthropic", env) == "oauth-tok"


# --------------------------------------------------------------------------
# get_env_api_key: google-vertex (ADC three-way AND)
# --------------------------------------------------------------------------


def test_get_env_api_key_google_vertex_explicit_key_wins_without_adc():
    env = {"GOOGLE_CLOUD_API_KEY": "gk-1"}
    assert env_api_keys.get_env_api_key("google-vertex", env) == "gk-1"


def test_get_env_api_key_google_vertex_adc_requires_project_and_location(tmp_path):
    creds_file = tmp_path / "adc.json"
    creds_file.write_text("{}")
    env = {"GOOGLE_APPLICATION_CREDENTIALS": str(creds_file), "GOOGLE_CLOUD_PROJECT": "proj-1"}
    # Missing GOOGLE_CLOUD_LOCATION: not authenticated yet.
    assert env_api_keys.get_env_api_key("google-vertex", env) is None

    env["GOOGLE_CLOUD_LOCATION"] = "us-central1"
    assert env_api_keys.get_env_api_key("google-vertex", env) == "<authenticated>"


def test_get_env_api_key_google_vertex_adc_file_missing_is_not_authenticated(tmp_path):
    missing = tmp_path / "does-not-exist.json"
    env = {
        "GOOGLE_APPLICATION_CREDENTIALS": str(missing),
        "GOOGLE_CLOUD_PROJECT": "proj-1",
        "GOOGLE_CLOUD_LOCATION": "us-central1",
    }
    assert env_api_keys.get_env_api_key("google-vertex", env) is None


def test_get_env_api_key_google_vertex_accepts_gcloud_project_alias(tmp_path):
    creds_file = tmp_path / "adc.json"
    creds_file.write_text("{}")
    env = {
        "GOOGLE_APPLICATION_CREDENTIALS": str(creds_file),
        "GCLOUD_PROJECT": "proj-2",
        "GOOGLE_CLOUD_LOCATION": "us-central1",
    }
    assert env_api_keys.get_env_api_key("google-vertex", env) == "<authenticated>"


def test_get_env_api_key_google_vertex_falls_back_to_home_adc_path(tmp_path, monkeypatch):
    # No explicit GOOGLE_APPLICATION_CREDENTIALS: falls back to the well-known
    # gcloud ADC path under the user's home directory. Point `Path.home()` at
    # `tmp_path` so this never touches the real home directory.
    monkeypatch.setattr(env_api_keys.Path, "home", classmethod(lambda cls: tmp_path))
    adc_dir = tmp_path / ".config" / "gcloud"
    adc_dir.mkdir(parents=True)
    (adc_dir / "application_default_credentials.json").write_text("{}")

    env = {"GOOGLE_CLOUD_PROJECT": "proj-3", "GOOGLE_CLOUD_LOCATION": "us-central1"}
    assert env_api_keys.get_env_api_key("google-vertex", env) == "<authenticated>"


def test_get_env_api_key_google_vertex_home_adc_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(env_api_keys.Path, "home", classmethod(lambda cls: tmp_path))
    env = {"GOOGLE_CLOUD_PROJECT": "proj-3", "GOOGLE_CLOUD_LOCATION": "us-central1"}
    assert env_api_keys.get_env_api_key("google-vertex", env) is None


# --------------------------------------------------------------------------
# get_env_api_key: amazon-bedrock (ambient credential OR)
# --------------------------------------------------------------------------


def test_get_env_api_key_amazon_bedrock_none_configured():
    assert env_api_keys.get_env_api_key("amazon-bedrock", {}) is None


def test_get_env_api_key_amazon_bedrock_profile_authenticates():
    assert env_api_keys.get_env_api_key("amazon-bedrock", {"AWS_PROFILE": "default"}) == "<authenticated>"


def test_get_env_api_key_amazon_bedrock_requires_both_access_key_parts():
    assert env_api_keys.get_env_api_key("amazon-bedrock", {"AWS_ACCESS_KEY_ID": "AKIA"}) is None
    env = {"AWS_ACCESS_KEY_ID": "AKIA", "AWS_SECRET_ACCESS_KEY": "secret"}
    assert env_api_keys.get_env_api_key("amazon-bedrock", env) == "<authenticated>"


@pytest.mark.parametrize(
    "var",
    [
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    ],
)
def test_get_env_api_key_amazon_bedrock_ambient_credential_sources(var):
    assert env_api_keys.get_env_api_key("amazon-bedrock", {var: "some-value"}) == "<authenticated>"


# --------------------------------------------------------------------------
# api/cloudflare.py
# --------------------------------------------------------------------------


def test_cloudflare_workers_ai_base_url_has_account_placeholder():
    assert cloudflare.CLOUDFLARE_WORKERS_AI_BASE_URL == (
        "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1"
    )


def test_cloudflare_ai_gateway_urls_have_account_and_gateway_placeholders():
    for url, suffix in (
        (cloudflare.CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL, "compat"),
        (cloudflare.CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL, "openai"),
        (cloudflare.CLOUDFLARE_AI_GATEWAY_ANTHROPIC_BASE_URL, "anthropic"),
    ):
        assert url.startswith("https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/")
        assert url.endswith(suffix)


def test_cloudflare_urls_are_formattable_with_real_ids():
    formatted = cloudflare.CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL.format(
        CLOUDFLARE_ACCOUNT_ID="acct123", CLOUDFLARE_GATEWAY_ID="gw456"
    )
    assert formatted == "https://gateway.ai.cloudflare.com/v1/acct123/gw456/compat"
