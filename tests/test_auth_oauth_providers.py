"""Per-provider OAuth flow tests: request construction (URL, method, headers,
body) against `httpx.MockTransport`, plus each flow's `refresh`/`to_auth`
behavior. No test in this file reaches the network.

Flows that poll or back off take an injected `VirtualClock`, so their waits are
instant and their deadline is never judged against real elapsed time.
"""

from __future__ import annotations

import json
import time

import httpx
import pytest
from pi_ai.auth.oauth import anthropic, github_copilot, kimi_coding, openrouter, radius, xai
from pi_ai.auth.oauth.device_code import DeviceCodeClock
from pi_ai.auth.types import Credential
from pi_ai.utils.abort import AbortController


def make_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def now_ms() -> float:
    return time.time() * 1000


class VirtualClock:
    """The single time source a device-code flow reads; its waits are instant."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def as_device_code_clock(self) -> DeviceCodeClock:
        return DeviceCodeClock(monotonic=self.monotonic, sleep=self.sleep)


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------


async def test_anthropic_exchange_authorization_code_request():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"access_token": "acc", "refresh_token": "ref", "expires_in": 3600})

    async with make_client(handler) as client:
        credential = await anthropic.exchange_authorization_code(
            "code123", "state456", "verifier789", "http://localhost:9999/callback", client
        )

    assert captured["method"] == "POST"
    assert captured["url"] == anthropic.TOKEN_URL
    assert captured["body"] == {
        "grant_type": "authorization_code",
        "client_id": anthropic.CLIENT_ID,
        "code": "code123",
        "state": "state456",
        "redirect_uri": "http://localhost:9999/callback",
        "code_verifier": "verifier789",
    }
    assert captured["headers"]["content-type"] == "application/json"
    assert credential.access == "acc"
    assert credential.refresh == "ref"
    # Refresh skew subtracts 5 minutes from the reported expiry.
    assert credential.expires < now_ms() + 3600 * 1000
    assert credential.expires > now_ms() + 3000 * 1000


async def test_anthropic_refresh_token_request():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"access_token": "new-acc", "refresh_token": "new-ref", "expires_in": 3600})

    async with make_client(handler) as client:
        credential = await anthropic.refresh_anthropic_token("old-ref", client)

    assert captured["body"] == {
        "grant_type": "refresh_token",
        "client_id": anthropic.CLIENT_ID,
        "refresh_token": "old-ref",
    }
    assert credential.access == "new-acc"


async def test_anthropic_build_authorize_url_contains_pkce_challenge_and_state():
    url = anthropic.build_authorize_url("the-challenge", "the-state", "http://localhost:1234/cb")
    assert url.startswith(anthropic.AUTHORIZE_URL)
    assert "code_challenge=the-challenge" in url
    assert "state=the-state" in url
    assert "code_challenge_method=S256" in url


async def test_anthropic_to_auth_returns_access_token_as_api_key():
    credential = Credential(type="oauth", access="my-access-token")
    resolved = await anthropic.to_auth(credential)
    assert resolved.api_key == "my-access-token"


async def test_anthropic_refresh_wrapper_accepts_injectable_client():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"access_token": "acc2", "refresh_token": "ref2", "expires_in": 3600})

    credential = Credential(type="oauth", access="a", refresh="r")
    async with make_client(handler) as client:
        result = await anthropic.refresh(credential, AbortController().signal, client=client)
    assert result.access == "acc2"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid_grant"})

    async with make_client(handler) as client:
        with pytest.raises(RuntimeError, match="HTTP request failed"):
            await anthropic.exchange_authorization_code("code", "state", "verifier", client=client)


@pytest.mark.parametrize(
    "raw_input,expected_code,expected_state",
    [
        ("", None, None),
        ("bare-code-only", "bare-code-only", None),
        ("code#state123", "code", "state123"),
        ("code=urlcode&state=urlstate", "urlcode", "urlstate"),
        ("https://example.com/cb?code=urlcode2&state=urlstate2", "urlcode2", "urlstate2"),
    ],
)
def test_anthropic_parse_authorization_input_variants(raw_input, expected_code, expected_state):
    assert anthropic._parse_authorization_input(raw_input) == (expected_code, expected_state)


# --------------------------------------------------------------------------
# GitHub Copilot
# --------------------------------------------------------------------------


async def test_github_copilot_start_device_flow_request():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["content_type"] = request.headers.get("content-type")
        return httpx.Response(
            200,
            json={
                "device_code": "devcode",
                "user_code": "USER-CODE",
                "verification_uri": "https://github.com/login/device",
                "interval": 5,
                "expires_in": 900,
            },
        )

    async with make_client(handler) as client:
        device = await github_copilot.start_device_flow("github.com", client)

    assert captured["url"] == "https://github.com/login/device/code"
    assert "application/x-www-form-urlencoded" in captured["content_type"]
    assert device["device_code"] == "devcode"
    assert device["user_code"] == "USER-CODE"


async def test_github_copilot_start_device_flow_rejects_untrusted_verification_uri():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "device_code": "devcode",
                "user_code": "USER-CODE",
                "verification_uri": "ftp://evil.example/device",
                "interval": 5,
                "expires_in": 900,
            },
        )

    async with make_client(handler) as client:
        with pytest.raises(RuntimeError, match="Untrusted verification_uri"):
            await github_copilot.start_device_flow("github.com", client)


async def test_github_copilot_refresh_access_token_sends_bearer_header():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json={"token": "copilot-token", "expires_at": int(time.time()) + 1800})

    async with make_client(handler) as client:
        credential = await github_copilot.refresh_github_copilot_access_token("gh-access-token", None, client)

    assert captured["url"] == "https://api.github.com/copilot_internal/v2/token"
    assert captured["authorization"] == "Bearer gh-access-token"
    assert credential.access == "copilot-token"
    assert credential.refresh == "gh-access-token"


async def test_github_copilot_get_base_url_from_token_proxy_ep():
    token = "tid=abc;exp=123;proxy-ep=proxy.individual.githubcopilot.com;foo=bar"
    assert github_copilot.get_base_url_from_token(token) == "https://api.individual.githubcopilot.com"


async def test_github_copilot_get_base_url_falls_back_to_enterprise_domain():
    assert github_copilot.get_github_copilot_base_url(None, "acme.ghe.com") == "https://copilot-api.acme.ghe.com"
    assert github_copilot.get_github_copilot_base_url(None, None) == "https://api.individual.githubcopilot.com"


async def test_github_copilot_normalize_domain():
    assert github_copilot.normalize_domain("acme.ghe.com") == "acme.ghe.com"
    assert github_copilot.normalize_domain("https://acme.ghe.com/") == "acme.ghe.com"
    assert github_copilot.normalize_domain("  ") is None


async def test_github_copilot_fetch_available_model_ids_sends_bearer_and_parses_picker_ids():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "gpt-4", "model_picker_enabled": True, "policy": {"state": "enabled"}},
                    {"id": "gpt-3", "model_picker_enabled": False, "policy": {"state": "disabled"}},
                ]
            },
        )

    async with make_client(handler) as client:
        model_ids = await github_copilot.fetch_available_github_copilot_model_ids("copilot-tok", None, client)

    assert captured["authorization"] == "Bearer copilot-tok"
    assert model_ids == ["gpt-4"]


async def test_github_copilot_enable_model_sends_bearer_and_policy_body():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={})

    async with make_client(handler) as client:
        enabled = await github_copilot.enable_github_copilot_model("copilot-tok", "claude-x", None, client)

    assert enabled is True
    assert captured["url"] == "https://api.individual.githubcopilot.com/models/claude-x/policy"
    assert captured["authorization"] == "Bearer copilot-tok"
    assert captured["body"] == {"state": "enabled"}


async def test_github_copilot_to_auth_derives_base_url_from_token():
    credential = Credential(type="oauth", access="tid=x;proxy-ep=proxy.business.githubcopilot.com;exp=1")
    resolved = await github_copilot.to_auth(credential)
    assert resolved.api_key == credential.access
    assert resolved.base_url == "https://api.business.githubcopilot.com"


async def test_github_copilot_refresh_wrapper_accepts_injectable_client_no_network():
    def handler(request: httpx.Request) -> httpx.Response:
        if "policy" in str(request.url):
            return httpx.Response(200, json={})
        if "models" in str(request.url):
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, json={"token": "new-copilot-token", "expires_at": int(time.time()) + 1800})

    credential = Credential(type="oauth", access="old", refresh="gh-token")
    async with make_client(handler) as client:
        result = await github_copilot.refresh(credential, AbortController().signal, client=client)
    assert result.access == "new-copilot-token"


# --------------------------------------------------------------------------
# OpenRouter
# --------------------------------------------------------------------------


async def test_openrouter_exchange_authorization_code_request():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"key": "sk-or-v1-abc"})

    async with make_client(handler) as client:
        credential = await openrouter.exchange_authorization_code("code1", "verifier1", client)

    assert captured["url"] == openrouter.TOKEN_URL
    assert captured["body"] == {"code": "code1", "code_verifier": "verifier1", "code_challenge_method": "S256"}
    assert credential.access == "sk-or-v1-abc"
    assert credential.refresh == ""


async def test_openrouter_refresh_is_a_no_op():
    credential = Credential(type="oauth", access="sk-or-v1-abc")
    result = await openrouter.refresh(credential, AbortController().signal)
    assert result is credential


async def test_openrouter_exchange_failure_surfaces_error_detail():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error_description": "bad code"})

    async with make_client(handler) as client:
        with pytest.raises(RuntimeError, match="bad code"):
            await openrouter.exchange_authorization_code("bad", "verifier", client)


async def test_openrouter_build_authorize_url():
    url = openrouter.build_authorize_url("http://localhost:1234/cb", "chal123")
    assert url.startswith(openrouter.AUTHORIZE_URL)
    assert "callback_url=http" in url
    assert "code_challenge=chal123" in url


async def test_openrouter_exchange_failure_detail_from_message_field():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"message": "denied by policy"})

    async with make_client(handler) as client:
        with pytest.raises(RuntimeError, match="denied by policy"):
            await openrouter.exchange_authorization_code("bad", "verifier", client)


async def test_openrouter_exchange_failure_detail_from_error_string():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid_grant"})

    async with make_client(handler) as client:
        with pytest.raises(RuntimeError, match="invalid_grant"):
            await openrouter.exchange_authorization_code("bad", "verifier", client)


async def test_openrouter_exchange_failure_no_detail_when_body_unparseable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, content=b"not json")

    async with make_client(handler) as client:
        with pytest.raises(RuntimeError, match=r"HTTP 400\)$"):
            await openrouter.exchange_authorization_code("bad", "verifier", client)


async def test_openrouter_exchange_missing_key_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"not_a_key": "value"})

    async with make_client(handler) as client:
        with pytest.raises(RuntimeError, match='carries no "key"'):
            await openrouter.exchange_authorization_code("code", "verifier", client)


def test_openrouter_error_detail_from_nested_error_message():
    assert openrouter._error_detail({"error": {"message": "nested detail"}}) == "nested detail"


def test_openrouter_error_detail_none_when_no_recognized_field():
    assert openrouter._error_detail({"unrelated": "x"}) is None


# --------------------------------------------------------------------------
# xAI
# --------------------------------------------------------------------------


async def test_xai_request_device_code():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["content_type"] = request.headers.get("content-type")
        return httpx.Response(
            200,
            json={
                "device_code": "devcode",
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://auth.x.ai/activate",
                "interval": 5,
                "expires_in": 600,
            },
        )

    async with make_client(handler) as client:
        device = await xai.request_device_code(client)

    assert captured["url"] == xai.DEVICE_CODE_URL
    assert "application/x-www-form-urlencoded" in captured["content_type"]
    assert device["device_code"] == "devcode"
    assert device["verification_uri"] == "https://auth.x.ai/activate"


async def test_xai_request_device_code_rejects_non_https_verification_uri():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "device_code": "devcode",
                "user_code": "ABCD-EFGH",
                "verification_uri": "http://auth.x.ai/activate",
                "interval": 5,
                "expires_in": 600,
            },
        )

    async with make_client(handler) as client:
        with pytest.raises(RuntimeError, match="Untrusted verification URI"):
            await xai.request_device_code(client)


async def test_xai_poll_for_tokens_pending_then_complete():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(400, json={"error": "authorization_pending"})
        return httpx.Response(200, json={"access_token": "acc", "refresh_token": "ref", "expires_in": 3600})

    device = {"device_code": "devcode", "interval_seconds": 0.01, "expires_in_seconds": 5}
    async with make_client(handler) as client:
        credential = await xai.poll_for_tokens(
            device, AbortController().signal, client, VirtualClock().as_device_code_clock()
        )

    assert credential.access == "acc"
    assert credential.refresh == "ref"


async def test_xai_refresh_token_request():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(200, json={"access_token": "new-acc", "refresh_token": "new-ref", "expires_in": 3600})

    credential = Credential(type="oauth", access="old", refresh="old-ref")
    async with make_client(handler) as client:
        result = await xai.refresh(credential, AbortController().signal, client=client)
    assert result.access == "new-acc"


async def test_xai_to_auth():
    credential = Credential(type="oauth", access="xai-token")
    resolved = await xai.to_auth(credential)
    assert resolved.api_key == "xai-token"


# --------------------------------------------------------------------------
# Kimi Coding
# --------------------------------------------------------------------------


async def test_kimi_coding_start_device_authorization_request():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "device_code": "devcode",
                "user_code": "USER-CODE",
                "verification_uri": "https://auth.kimi.com/device",
                "verification_uri_complete": "https://auth.kimi.com/device?code=USER-CODE",
                "interval": 5,
                "expires_in": 900,
            },
        )

    async with make_client(handler) as client:
        device = await kimi_coding.start_device_authorization(kimi_coding.DEFAULT_OAUTH_HOST, client)

    assert captured["url"] == "https://auth.kimi.com/api/oauth/device_authorization"
    assert device["device_code"] == "devcode"


async def test_kimi_coding_poll_for_token_authorization_pending_then_complete():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json={"error": "authorization_pending"})
        return httpx.Response(200, json={"access_token": "acc", "refresh_token": "ref", "expires_in": 3600})

    device = {"device_code": "devcode", "interval_seconds": 0.01, "expires_in_seconds": 5}
    async with make_client(handler) as client:
        credential = await kimi_coding.poll_for_token(
            kimi_coding.DEFAULT_OAUTH_HOST,
            device,
            AbortController().signal,
            client,
            VirtualClock().as_device_code_clock(),
        )

    assert credential.access == "acc"


async def test_kimi_coding_refresh_retries_on_5xx_then_succeeds():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(500, json={"error": "server_error"})
        return httpx.Response(200, json={"access_token": "acc", "refresh_token": "ref", "expires_in": 3600})

    async with make_client(handler) as client:
        credential = await kimi_coding.refresh_token(
            kimi_coding.DEFAULT_OAUTH_HOST,
            "old-ref",
            AbortController().signal,
            client,
            VirtualClock().as_device_code_clock(),
        )

    assert calls == 3
    assert credential.access == "acc"


async def test_kimi_coding_refresh_unauthorized_raises_immediately():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": "invalid_grant"})

    async with make_client(handler) as client:
        with pytest.raises(RuntimeError, match="unauthorized"):
            await kimi_coding.refresh_token(kimi_coding.DEFAULT_OAUTH_HOST, "bad-ref", AbortController().signal, client)

    assert calls == 1


async def test_kimi_coding_to_auth_returns_headers_only_no_api_key():
    credential = Credential(type="oauth", access="kimi-access")
    resolved = await kimi_coding.to_auth(credential)
    assert resolved.headers["Authorization"] == "Bearer kimi-access"
    assert resolved.api_key is None


async def test_kimi_coding_refresh_wrapper_accepts_injectable_client_no_network():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"access_token": "new-acc", "refresh_token": "new-ref", "expires_in": 3600})

    credential = Credential(type="oauth", access="old", refresh="old-ref")
    async with make_client(handler) as client:
        result = await kimi_coding.refresh(credential, AbortController().signal, client=client)
    assert result.access == "new-acc"


# --------------------------------------------------------------------------
# Radius
# --------------------------------------------------------------------------


async def test_radius_load_oauth_discovery_request():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://gateway.example/v1/oauth"
        return httpx.Response(200, json={"authorizationEndpoint": "https://gateway.example/oauth/authorize"})

    async with make_client(handler) as client:
        discovery = await radius.load_radius_oauth_discovery("https://gateway.example", client)

    assert discovery["authorization_endpoint"] == "https://gateway.example/oauth/authorize"


async def test_radius_request_oauth_token_request():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["content_type"] = request.headers.get("content-type")
        return httpx.Response(
            200, json={"access_token": "acc", "refresh_token": "ref", "expires_in": 3600, "scope": "gateway"}
        )

    async with make_client(handler) as client:
        credential = await radius.request_oauth_token(
            "https://gateway.example",
            {"grant_type": "authorization_code", "code": "c1"},
            client,
        )

    assert captured["url"] == "https://gateway.example/v1/oauth/token"
    assert "application/x-www-form-urlencoded" in captured["content_type"]
    assert credential.access == "acc"
    assert credential.data == {"scope": "gateway"}


async def test_radius_request_device_authorization_request():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://gateway.example/v1/oauth/device"
        return httpx.Response(
            200,
            json={
                "device_code": "devcode",
                "user_code": "USER-CODE",
                "verification_uri": "https://gateway.example/device",
                "expires_in": 900,
                "interval": 5,
            },
        )

    async with make_client(handler) as client:
        device = await radius.request_device_authorization("https://gateway.example", client)

    assert device["device_code"] == "devcode"


async def test_radius_oauth_response_error_maps_authorization_pending():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "authorization_pending"})

    async with make_client(handler) as client:
        with pytest.raises(radius.OAuthResponseError) as exc_info:
            await radius.request_oauth_token("https://gateway.example", {"grant_type": "x"}, client)

    assert exc_info.value.oauth_error == "authorization_pending"
    assert exc_info.value.status == 400


async def test_radius_to_auth_and_refresh():
    credential = Credential(type="oauth", access="radius-access")
    oauth = radius.create_radius_oauth("Radius", "https://gateway.example")
    resolved = await oauth.to_auth(credential)
    assert resolved.api_key == "radius-access"


async def test_radius_refresh_wrapper_accepts_injectable_client_no_network():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"access_token": "acc2", "refresh_token": "ref2", "expires_in": 3600, "scope": None}
        )

    credential = Credential(type="oauth", access="old", refresh="old-ref")
    oauth = radius.create_radius_oauth("Radius", "https://gateway.example")
    async with make_client(handler) as client:
        result = await oauth.refresh(credential, AbortController().signal, client=client)
    assert result.access == "acc2"


async def test_radius_normalize_gateway_url_strips_trailing_slash():
    assert radius.normalize_radius_gateway_url("https://gateway.example/") == "https://gateway.example"
    assert radius.normalize_radius_gateway_url("https://gateway.example") == "https://gateway.example"
