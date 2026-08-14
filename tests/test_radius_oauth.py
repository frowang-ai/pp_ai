"""Python port of `packages/ai/test/radius-oauth.test.ts`."""

from __future__ import annotations

import time
from urllib.parse import parse_qs

import httpx
import pytest
from pi_ai.auth.oauth import radius
from pi_ai.auth.types import AuthEvent, AuthInteraction, AuthPrompt, Credential
from pi_ai.utils.abort import AbortController

GATEWAY = "https://radius.example"

_REAL_ASYNC_CLIENT = httpx.AsyncClient


class ScriptedInteraction(AuthInteraction):
    """TS `interaction(loginMethod, events)`: prompt() always answers with the
    chosen login method, notify() records events."""

    def __init__(self, login_method: str) -> None:
        self.signal = AbortController().signal
        self._login_method = login_method
        self.events: list[AuthEvent] = []

    async def prompt(self, prompt: AuthPrompt) -> str:
        return self._login_method

    def notify(self, event: AuthEvent) -> None:
        self.events.append(event)


def stub_fetch(monkeypatch: pytest.MonkeyPatch, handler) -> list[str]:
    """TS stubs the global `fetch`; the port constructs its own `httpx.AsyncClient`
    inside `create_radius_oauth`'s `login`/`refresh`, so the equivalent seam is to
    swap the class for one bound to a `MockTransport`. Returns the recorded URLs."""
    urls: list[str] = []

    def recording_handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return handler(request)

    def factory(**kwargs: object) -> httpx.AsyncClient:
        return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(recording_handler))

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return urls


def form(request: httpx.Request) -> dict[str, str]:
    return {key: values[0] for key, values in parse_qs(request.content.decode()).items()}


async def test_uses_gateway_endpoints_directly_for_device_login(monkeypatch: pytest.MonkeyPatch) -> None:
    interaction = ScriptedInteraction("device-code")

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        body = form(request)
        if url == f"{GATEWAY}/v1/oauth/device":
            assert body["client_id"] == "pi-gateway"
            assert body["scope"] == "gateway offline_access"
            return httpx.Response(
                200,
                json={
                    "device_code": "device-code",
                    "user_code": "ABCD-1234",
                    "verification_uri": "https://radius-ui.example/pair",
                    "expires_in": 600,
                    "interval": 5,
                },
            )
        if url == f"{GATEWAY}/v1/oauth/token":
            assert body["grant_type"] == "urn:ietf:params:oauth:grant-type:device_code"
            assert body["client_id"] == "pi-gateway"
            assert body["device_code"] == "device-code"
            return httpx.Response(
                200,
                json={
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                    "expires_in": 3600,
                    "scope": "gateway offline_access",
                },
            )
        raise AssertionError(f"Unexpected request: {url}")

    urls = stub_fetch(monkeypatch, handler)

    oauth = radius.create_radius_oauth("Radius", GATEWAY)
    before_ms = time.time() * 1000
    credential = await oauth.login(interaction)
    after_ms = time.time() * 1000

    assert credential.type == "oauth"
    assert credential.access == "access-token"
    assert credential.refresh == "refresh-token"
    # TS pins `expires` exactly using fake timers; the port has no fake clock, so
    # bracket it against the real clock instead of weakening the formula.
    assert before_ms + 3600 * 1000 - 60_000 <= credential.expires <= after_ms + 3600 * 1000 - 60_000
    # TS's OAuthCredential has an index signature so `scope` sits at the top level;
    # the port carries provider extras in `Credential.data` (see auth/types.py).
    assert credential.data == {"scope": "gateway offline_access"}

    assert interaction.events == [
        AuthEvent(
            type="device_code",
            user_code="ABCD-1234",
            verification_uri="https://radius-ui.example/pair",
            interval_seconds=5,
            expires_in_seconds=600,
        )
    ]
    assert urls == [f"{GATEWAY}/v1/oauth/device", f"{GATEWAY}/v1/oauth/token"]


async def test_refreshes_directly_through_the_gateway_without_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{GATEWAY}/v1/oauth/token"
        body = form(request)
        assert body["grant_type"] == "refresh_token"
        assert body["client_id"] == "pi-gateway"
        assert body["refresh_token"] == "old-refresh"
        return httpx.Response(
            200,
            json={"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600},
        )

    urls = stub_fetch(monkeypatch, handler)

    oauth = radius.create_radius_oauth("Radius", GATEWAY)
    credential = await oauth.refresh(
        Credential(type="oauth", access="old-access", refresh="old-refresh", expires=0),
        AbortController().signal,
    )
    assert credential.access == "new-access"
    assert credential.refresh == "new-refresh"
    assert len(urls) == 1


async def test_discovers_only_the_interactive_browser_authorization_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{GATEWAY}/v1/oauth"
        return httpx.Response(200, json={"issuer": "https://radius-ui.example"})

    urls = stub_fetch(monkeypatch, handler)

    oauth = radius.create_radius_oauth("Radius", GATEWAY)
    with pytest.raises(RuntimeError, match=f"Invalid Radius OAuth config from {GATEWAY}"):
        await oauth.login(ScriptedInteraction("browser"))
    assert len(urls) == 1
