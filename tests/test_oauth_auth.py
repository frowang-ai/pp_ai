"""Python port of `packages/ai/test/oauth-auth.test.ts`.

The final TypeScript block goes through `createModels({ credentials })` +
`setProvider` + `getAuth`; the port spells the same chain as
`Models(credential_store=...)` + `add(provider)` + `get_auth(...)`.
"""

from __future__ import annotations

import time

import httpx
import pi_ai.auth.oauth.load as oauth_load
import pytest
from pi_ai.auth.oauth.anthropic import build_anthropic_oauth
from pi_ai.auth.oauth.github_copilot import build_github_copilot_oauth
from pi_ai.auth.oauth.kimi_coding import build_kimi_coding_oauth
from pi_ai.auth.oauth.openrouter import build_openrouter_oauth
from pi_ai.auth.oauth.xai import build_xai_oauth
from pi_ai.auth.types import Credential, InMemoryCredentialStore
from pi_ai.providers.anthropic import anthropic_provider
from pi_ai.providers.github_copilot import github_copilot_provider
from pi_ai.registry import Models
from pi_ai.utils.abort import AbortController

NEVER_ABORTED_SIGNAL = AbortController().signal
MAX_SAFE_INTEGER = 9007199254740991

_REAL_ASYNC_CLIENT = httpx.AsyncClient


def stub_fetch(monkeypatch: pytest.MonkeyPatch, handler) -> list[str]:
    urls: list[str] = []

    def recording_handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return handler(request)

    def factory(**kwargs: object) -> httpx.AsyncClient:
        return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(recording_handler))

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return urls


def test_keeps_the_extension_oauth_barrel_free_of_builtin_flow_implementations() -> None:
    # TS asserts `src/oauth.ts` (a type-only extension-compat barrel) exports no
    # `loginAnthropic`/`anthropicOAuth`. The port has no separate compat barrel;
    # the closest equivalent is `pi_ai.auth.oauth`'s public surface, which must
    # likewise expose only lazy loaders and shared primitives.
    import pi_ai.auth.oauth as oauth_barrel

    assert not hasattr(oauth_barrel, "login_anthropic")
    assert not hasattr(oauth_barrel, "anthropic_oauth")
    assert "login_anthropic" not in oauth_barrel.__all__
    assert "anthropic_oauth" not in oauth_barrel.__all__


def test_identifies_only_subscription_backed_oauth_flows_as_subscriptions() -> None:
    # TS also checks `openaiCodexOAuth`; the openai-codex flow is a documented
    # omission of this port (see README: openai-codex-responses needs its
    # OAuth/WebSocket transport), so there is no Python object to assert on.
    for oauth in [
        build_anthropic_oauth(),
        build_github_copilot_oauth(),
        build_kimi_coding_oauth(),
        build_xai_oauth(),
    ]:
        assert oauth.is_subscription is True
    assert build_openrouter_oauth().is_subscription is not True


def test_openai_codex_to_auth_is_not_ported() -> None:
    # TS counterpart: "openai-codex toAuth derives the api key from the access
    # token". The ChatGPT OAuth flow has no Python module (the api it exists for,
    # `openai-codex-responses`, is itself unported), so there is no `to_auth` to
    # call. Pinning the documented absence so this stays visible: the codex
    # provider is discovery-only and carries no OAuth entry at all.
    from pi_ai.providers.openai_codex import openai_codex_provider

    assert not hasattr(oauth_load, "load_openai_codex_oauth")
    provider = openai_codex_provider()
    assert provider.auth is not None
    assert provider.auth.oauth is None


async def test_anthropic_to_auth_derives_the_api_key_from_the_access_token() -> None:
    auth = await build_anthropic_oauth().to_auth(Credential(type="oauth", access="token", refresh="r", expires=0))
    assert auth.api_key == "token"
    assert auth.base_url is None
    assert auth.headers == {}


async def test_openrouter_derives_the_api_key_and_keeps_the_permanent_credential_on_refresh() -> None:
    credential = Credential(type="oauth", access="token", refresh="", expires=MAX_SAFE_INTEGER)
    oauth = build_openrouter_oauth()
    assert (await oauth.to_auth(credential)).api_key == "token"
    assert await oauth.refresh(credential, NEVER_ABORTED_SIGNAL) is credential


async def test_xai_to_auth_derives_the_api_key_from_the_access_token() -> None:
    auth = await build_xai_oauth().to_auth(Credential(type="oauth", access="token", refresh="r", expires=0))
    assert auth.api_key == "token"
    assert auth.base_url is None
    assert auth.headers == {}


async def test_github_copilot_to_auth_derives_base_url_from_the_token_proxy_endpoint() -> None:
    access = "tid=abc;exp=123;proxy-ep=proxy.enterprise.example;rest"
    auth = await build_github_copilot_oauth().to_auth(Credential(type="oauth", access=access, refresh="r", expires=0))
    assert auth.api_key == access
    assert auth.base_url == "https://api.enterprise.example"


async def test_github_copilot_to_auth_falls_back_to_the_enterprise_domain_then_the_individual_endpoint() -> None:
    oauth = build_github_copilot_oauth()
    # TS carries `enterpriseUrl` flat on the credential (its `OAuthCredentials`
    # has an index signature); the port nests provider extras under
    # `Credential.data` with the snake_case key the copilot flow writes.
    enterprise = await oauth.to_auth(
        Credential(
            type="oauth",
            access="no-proxy-ep",
            refresh="r",
            expires=0,
            data={"enterprise_url": "https://company.ghe.com"},
        )
    )
    assert enterprise.base_url == "https://copilot-api.company.ghe.com"

    individual = await oauth.to_auth(Credential(type="oauth", access="no-proxy-ep", refresh="r", expires=0))
    assert individual.base_url == "https://api.individual.githubcopilot.com"


async def test_anthropic_refresh_exchanges_the_refresh_token_and_returns_a_typed_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600}
        )

    stub_fetch(monkeypatch, handler)

    refreshed = await build_anthropic_oauth().refresh(
        Credential(type="oauth", access="old", refresh="old-r", expires=0), NEVER_ABORTED_SIGNAL
    )
    assert refreshed.type == "oauth"
    assert refreshed.access == "new-access"
    assert refreshed.refresh == "new-refresh"
    assert refreshed.expires > time.time() * 1000


async def test_github_copilot_refresh_preserves_the_enterprise_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/models"):
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, json={"token": "new-token", "expires_at": 9999999999})

    fetched_urls = stub_fetch(monkeypatch, handler)

    refreshed = await build_github_copilot_oauth().refresh(
        Credential(
            type="oauth",
            access="old",
            refresh="gh-token",
            expires=0,
            data={"enterprise_url": "company.ghe.com"},
        ),
        NEVER_ABORTED_SIGNAL,
    )
    assert refreshed.access == "new-token"
    assert refreshed.data["enterprise_url"] == "company.ghe.com"
    assert "api.company.ghe.com" in fetched_urls[0]


async def test_resolves_stored_anthropic_oauth_credentials_via_the_lazy_flow_import() -> None:
    credentials = InMemoryCredentialStore()
    await credentials.set(
        "anthropic",
        Credential(
            type="oauth",
            access="oauth-access-token",
            refresh="r",
            # Keep this beyond the resolver's refresh window.
            expires=time.time() * 1000 + 10 * 60_000,
        ),
    )
    models = Models(credential_store=credentials)
    models.add(anthropic_provider())

    model = models.get_models("anthropic")[0]
    result = await models.get_auth(model.provider)
    assert result is not None
    assert result.auth.api_key == "oauth-access-token"
    assert result.source == "OAuth"


async def test_resolves_stored_github_copilot_oauth_credentials_including_per_credential_base_url() -> None:
    access = "tid=abc;exp=123;proxy-ep=proxy.business.githubcopilot.com;rest"
    credentials = InMemoryCredentialStore()
    await credentials.set(
        "github-copilot",
        Credential(type="oauth", access=access, refresh="r", expires=time.time() * 1000 + 10 * 60_000),
    )
    models = Models(credential_store=credentials)
    models.add(github_copilot_provider())

    model = models.get_models("github-copilot")[0]
    result = await models.get_auth(model.provider)
    assert result is not None
    assert result.auth.api_key == access
    assert result.auth.base_url == "https://api.business.githubcopilot.com"
