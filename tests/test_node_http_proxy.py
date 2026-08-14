"""Tests for `pi_ai.utils.node_http_proxy`.

Python port of the proxy behaviour in `packages/ai/src/utils/node-http-proxy.ts`,
including the Python port of `packages/ai/test/node-http-proxy.test.ts`.
No network is used: only the environment resolution is exercised.
"""

import pytest
from pi_ai.utils.node_http_proxy import (
    UNSUPPORTED_PROXY_PROTOCOL_MESSAGE,
    get_proxy_for_url,
    resolve_http_proxy_url_for_target,
)

PROXY_VARS = (
    "http_proxy",
    "HTTP_PROXY",
    "https_proxy",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
    "no_proxy",
    "NO_PROXY",
)


@pytest.fixture(autouse=True)
def clean_proxy_env(monkeypatch):
    for name in PROXY_VARS:
        monkeypatch.delenv(name, raising=False)


def test_no_proxy_env_means_no_proxy():
    assert get_proxy_for_url("https://api.openai.com/v1") == ""
    assert resolve_http_proxy_url_for_target("https://api.openai.com/v1") is None


def test_the_proxy_is_selected_by_target_scheme(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://proxy.invalid:3128")
    monkeypatch.setenv("https_proxy", "http://secure-proxy.invalid:3129")
    assert get_proxy_for_url("http://example.invalid") == "http://proxy.invalid:3128"
    assert get_proxy_for_url("https://example.invalid") == "http://secure-proxy.invalid:3129"


def test_all_proxy_is_the_fallback(monkeypatch):
    monkeypatch.setenv("all_proxy", "http://proxy.invalid:3128")
    assert get_proxy_for_url("https://example.invalid") == "http://proxy.invalid:3128"
    monkeypatch.setenv("https_proxy", "http://specific.invalid:3128")
    assert get_proxy_for_url("https://example.invalid") == "http://specific.invalid:3128"


def test_uppercase_env_names_are_accepted(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    assert get_proxy_for_url("https://example.invalid") == "http://proxy.invalid:3128"


def test_a_scheme_less_proxy_value_inherits_the_target_scheme(monkeypatch):
    monkeypatch.setenv("http_proxy", "proxy.invalid:3128")
    assert get_proxy_for_url("http://example.invalid") == "http://proxy.invalid:3128"


def test_a_provider_scoped_env_overrides_the_process_env(monkeypatch):
    monkeypatch.setenv("https_proxy", "http://process.invalid:3128")
    assert get_proxy_for_url("https://example.invalid", {"https_proxy": "http://scoped.invalid:3128"}) == (
        "http://scoped.invalid:3128"
    )


def test_no_proxy_star_disables_proxying(monkeypatch):
    monkeypatch.setenv("https_proxy", "http://proxy.invalid:3128")
    monkeypatch.setenv("no_proxy", "*")
    assert get_proxy_for_url("https://example.invalid") == ""


def test_no_proxy_matches_an_exact_host(monkeypatch):
    monkeypatch.setenv("https_proxy", "http://proxy.invalid:3128")
    monkeypatch.setenv("no_proxy", "example.invalid")
    assert get_proxy_for_url("https://example.invalid") == ""
    assert get_proxy_for_url("https://api.example.invalid") == "http://proxy.invalid:3128"


def test_no_proxy_suffix_entries_match_subdomains(monkeypatch):
    monkeypatch.setenv("https_proxy", "http://proxy.invalid:3128")
    monkeypatch.setenv("no_proxy", ".example.invalid")
    assert get_proxy_for_url("https://api.example.invalid") == ""

    monkeypatch.setenv("no_proxy", "*.example.invalid")
    assert get_proxy_for_url("https://api.example.invalid") == ""


def test_no_proxy_entries_may_pin_a_port(monkeypatch):
    monkeypatch.setenv("https_proxy", "http://proxy.invalid:3128")
    monkeypatch.setenv("no_proxy", "example.invalid:8443")
    assert get_proxy_for_url("https://example.invalid:8443") == ""
    # The default HTTPS port 443 does not match the pinned 8443.
    assert get_proxy_for_url("https://example.invalid") == "http://proxy.invalid:3128"


def test_no_proxy_accepts_comma_and_whitespace_separators(monkeypatch):
    monkeypatch.setenv("https_proxy", "http://proxy.invalid:3128")
    monkeypatch.setenv("no_proxy", "first.invalid, second.invalid\tthird.invalid")
    for host in ("first.invalid", "second.invalid", "third.invalid"):
        assert get_proxy_for_url(f"https://{host}") == ""
    assert get_proxy_for_url("https://fourth.invalid") == "http://proxy.invalid:3128"


def test_no_proxy_matching_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("https_proxy", "http://proxy.invalid:3128")
    monkeypatch.setenv("no_proxy", "EXAMPLE.INVALID")
    assert get_proxy_for_url("https://example.invalid") == ""


def test_a_non_url_target_is_never_proxied(monkeypatch):
    monkeypatch.setenv("all_proxy", "http://proxy.invalid:3128")
    assert get_proxy_for_url("not-a-url") == ""


def test_socks_proxies_are_rejected(monkeypatch):
    monkeypatch.setenv("https_proxy", "socks5://proxy.invalid:1080")
    with pytest.raises(ValueError) as error:
        resolve_http_proxy_url_for_target("https://example.invalid")
    assert UNSUPPORTED_PROXY_PROTOCOL_MESSAGE in str(error.value)


def test_pac_proxies_are_rejected(monkeypatch):
    monkeypatch.setenv("https_proxy", "pac+https://proxy.invalid/proxy.pac")
    with pytest.raises(ValueError) as error:
        resolve_http_proxy_url_for_target("https://example.invalid")
    assert UNSUPPORTED_PROXY_PROTOCOL_MESSAGE in str(error.value)


def test_a_hostless_proxy_url_is_rejected(monkeypatch):
    monkeypatch.setenv("https_proxy", "http://")
    with pytest.raises(ValueError, match="missing scheme or host"):
        resolve_http_proxy_url_for_target("https://example.invalid")


def test_an_http_proxy_url_is_returned_unchanged(monkeypatch):
    monkeypatch.setenv("https_proxy", "http://user:secret@proxy.invalid:3128")
    assert resolve_http_proxy_url_for_target("https://example.invalid") == "http://user:secret@proxy.invalid:3128"


# --------------------------------------------------------------------------
# Ported from `packages/ai/test/node-http-proxy.test.ts`
#
# TypeScript's `resolveHttpProxyUrlForTarget` returns a WHATWG `URL`, so its
# assertions read `.toString()` and see the URL-normalized trailing slash
# ("http://proxy.example:8080/"). The port returns the raw proxy string,
# because httpx takes a string proxy; the assertions compare that instead.
# --------------------------------------------------------------------------


def test_ts_respects_no_proxy_exclusions(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("NO_PROXY", "bedrock-runtime.us-east-1.amazonaws.com")
    assert resolve_http_proxy_url_for_target("https://bedrock-runtime.us-east-1.amazonaws.com") is None


def test_ts_resolves_http_and_https_proxy_urls(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    assert (
        resolve_http_proxy_url_for_target("https://bedrock-runtime.us-east-1.amazonaws.com")
        == "http://proxy.example:8080"
    )


def test_ts_prefers_scoped_proxy_env_aliases_before_process_env_aliases(monkeypatch):
    monkeypatch.setenv("https_proxy", "http://process-proxy.example:8080")
    assert (
        resolve_http_proxy_url_for_target(
            "https://bedrock-runtime.us-east-1.amazonaws.com",
            {"HTTPS_PROXY": "http://scoped-proxy.example:8080"},
        )
        == "http://scoped-proxy.example:8080"
    )


def test_ts_rejects_socks_and_pac_proxy_urls_explicitly(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "socks5://proxy.example:1080")
    with pytest.raises(ValueError) as error:
        resolve_http_proxy_url_for_target("https://bedrock-runtime.us-east-1.amazonaws.com")
    assert UNSUPPORTED_PROXY_PROTOCOL_MESSAGE in str(error.value)
