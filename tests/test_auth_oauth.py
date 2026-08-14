"""Tests for the OAuth authentication layer: device-code polling, the local
callback server, `resolve_provider_auth`'s refresh-on-expiry logic, and
`Models` OAuth integration. Per-provider request-construction tests live in
`test_auth_oauth_providers.py`.

The device-code cases inject a `VirtualClock` instead of waiting: the poller
clamps any interval up to `MINIMUM_INTERVAL_MS`, so a "fast" 0.01s interval
still costs a real second per poll, and the deadline would otherwise be judged
against real elapsed time.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest
from pi_ai.auth.helpers import lazy_oauth
from pi_ai.auth.oauth.device_code import (
    DeviceCodeClock,
    DeviceCodeError,
    DeviceCodePollResult,
    poll_oauth_device_code_flow,
)
from pi_ai.auth.oauth.oauth_page import OAuthCallbackServer
from pi_ai.auth.resolve import resolve_provider_auth
from pi_ai.auth.types import (
    ApiKeyAuth,
    AuthEvent,
    AuthInteraction,
    AuthPrompt,
    Credential,
    InMemoryCredentialStore,
    OAuthAuth,
    ProviderAuth,
    ResolvedAuth,
)
from pi_ai.models import ModelsError
from pi_ai.registry import Models, create_provider
from pi_ai.types import Context, Model, ModelCost
from pi_ai.utils.abort import AbortController, AbortSignal


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
# device-code polling
# --------------------------------------------------------------------------


async def test_poll_device_code_pending_then_complete():
    calls = 0

    async def poll() -> DeviceCodePollResult[str]:
        nonlocal calls
        calls += 1
        if calls < 3:
            return DeviceCodePollResult(status="pending")
        return DeviceCodePollResult(status="complete", value="the-token")

    signal = AbortController().signal
    clock = VirtualClock()
    result = await poll_oauth_device_code_flow(poll, signal, interval_seconds=0.01, clock=clock.as_device_code_clock())
    assert result == "the-token"
    assert calls == 3
    # Any interval below `MINIMUM_INTERVAL_MS` is clamped to one second.
    assert clock.sleeps == [1.0, 1.0]


async def test_poll_device_code_slow_down_then_success():
    calls = 0
    statuses: list[str] = []

    async def poll() -> DeviceCodePollResult[str]:
        nonlocal calls
        calls += 1
        statuses.append("call")
        if calls == 1:
            return DeviceCodePollResult(status="pending")
        if calls == 2:
            return DeviceCodePollResult(status="slow_down", interval_seconds=0.01)
        return DeviceCodePollResult(status="complete", value="ok")

    signal = AbortController().signal
    clock = VirtualClock()
    result = await poll_oauth_device_code_flow(poll, signal, interval_seconds=0.01, clock=clock.as_device_code_clock())
    assert result == "ok"
    assert calls == 3
    assert clock.sleeps == [1.0, 1.0]


async def test_poll_device_code_denied_raises_device_code_error():
    async def poll() -> DeviceCodePollResult[str]:
        return DeviceCodePollResult(status="failed", message="access_denied")

    signal = AbortController().signal
    with pytest.raises(DeviceCodeError, match="access_denied"):
        await poll_oauth_device_code_flow(
            poll, signal, interval_seconds=0.01, clock=VirtualClock().as_device_code_clock()
        )


async def test_poll_device_code_expired_times_out():
    async def poll() -> DeviceCodePollResult[str]:
        return DeviceCodePollResult(status="pending")

    signal = AbortController().signal
    with pytest.raises(DeviceCodeError, match="timed out"):
        await poll_oauth_device_code_flow(
            poll, signal, interval_seconds=0.01, expires_in_seconds=0.03, clock=VirtualClock().as_device_code_clock()
        )


async def test_poll_device_code_slow_down_timeout_has_distinct_message():
    async def poll() -> DeviceCodePollResult[str]:
        return DeviceCodePollResult(status="slow_down", interval_seconds=0.02)

    signal = AbortController().signal
    with pytest.raises(DeviceCodeError, match="clock drift"):
        await poll_oauth_device_code_flow(
            poll, signal, interval_seconds=0.01, expires_in_seconds=0.03, clock=VirtualClock().as_device_code_clock()
        )


async def test_poll_device_code_cancelled_via_signal():
    controller = AbortController()

    async def poll() -> DeviceCodePollResult[str]:
        controller.abort()
        return DeviceCodePollResult(status="pending")

    with pytest.raises(DeviceCodeError, match="cancelled"):
        await poll_oauth_device_code_flow(
            poll, controller.signal, interval_seconds=0.01, clock=VirtualClock().as_device_code_clock()
        )


# --------------------------------------------------------------------------
# local OAuth callback server
# --------------------------------------------------------------------------


async def test_oauth_callback_server_receives_code_and_state():
    server = OAuthCallbackServer("/callback", port=0)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://{server.host}:{server.port}/callback",
                params={"code": "abc123", "state": "xyz"},
            )
        assert response.status_code == 200
        assert "successful" in response.text.lower() or "completed" in response.text.lower()

        callback = await asyncio.wait_for(server.wait_for_callback(), timeout=5)
        assert callback is not None
        assert callback.params["code"] == "abc123"
        assert callback.params["state"] == "xyz"
    finally:
        server.close()


async def test_oauth_callback_server_rejects_wrong_path():
    server = OAuthCallbackServer("/callback", port=0)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://{server.host}:{server.port}/not-the-callback")
        assert response.status_code == 404
    finally:
        server.close()


async def test_oauth_callback_server_can_reject_state_mismatch_via_on_callback():
    expected_state = "expected-state"
    rejected: list[str] = []

    def on_callback(result):
        state = result.params.get("state")
        if state != expected_state:
            rejected.append(state or "")
            return (400, "state mismatch")
        return None

    server = OAuthCallbackServer("/callback", port=0, on_callback=on_callback)
    try:
        async with httpx.AsyncClient() as client:
            bad_response = await client.get(
                f"http://{server.host}:{server.port}/callback",
                params={"code": "c1", "state": "wrong-state"},
            )
            assert bad_response.status_code == 400
            assert rejected == ["wrong-state"]

            good_response = await client.get(
                f"http://{server.host}:{server.port}/callback",
                params={"code": "c1", "state": expected_state},
            )
            assert good_response.status_code == 200

        callback = await asyncio.wait_for(server.wait_for_callback(), timeout=5)
        assert callback is not None
        assert callback.params["state"] == expected_state
    finally:
        server.close()


async def test_oauth_callback_server_cancel_settles_future_without_error():
    server = OAuthCallbackServer("/callback", port=0)
    server.cancel()
    result = await asyncio.wait_for(server.wait_for_callback(), timeout=5)
    assert result is None
    server.close()


# --------------------------------------------------------------------------
# resolve_provider_auth: precedence + refresh-on-expiry
# --------------------------------------------------------------------------


def make_oauth_provider_auth(
    *,
    login=None,
    refresh=None,
    to_auth=None,
) -> ProviderAuth:
    async def default_login(interaction: AuthInteraction) -> Credential:
        raise NotImplementedError

    async def default_refresh(credential: Credential, signal: AbortSignal) -> Credential:
        raise NotImplementedError

    async def default_to_auth(credential: Credential) -> ResolvedAuth:
        return ResolvedAuth(api_key=credential.access)

    return ProviderAuth(
        api_key=ApiKeyAuth(name="fake", env_vars=("FAKE_API_KEY",)),
        oauth=OAuthAuth(
            name="fake-oauth",
            login=login or default_login,
            refresh=refresh or default_refresh,
            to_auth=to_auth or default_to_auth,
        ),
    )


async def test_resolve_provider_auth_prefers_stored_over_env(monkeypatch):
    auth = make_oauth_provider_auth()
    store = InMemoryCredentialStore()
    await store.set("p1", Credential(type="api_key", key="stored-key"))

    result = await resolve_provider_auth("p1", auth, store, env={"FAKE_API_KEY": "env-key"}.get)
    assert result is not None
    assert result.auth.api_key == "stored-key"
    assert result.source == "stored credential"


async def test_resolve_provider_auth_falls_back_to_env_when_nothing_stored():
    auth = make_oauth_provider_auth()
    store = InMemoryCredentialStore()

    result = await resolve_provider_auth("p1", auth, store, env={"FAKE_API_KEY": "env-key"}.get)
    assert result is not None
    assert result.auth.api_key == "env-key"
    assert result.source == "FAKE_API_KEY"


async def test_resolve_provider_auth_api_key_override_wins_over_everything():
    auth = make_oauth_provider_auth()
    store = InMemoryCredentialStore()
    await store.set("p1", Credential(type="api_key", key="stored-key"))

    result = await resolve_provider_auth(
        "p1", auth, store, env={"FAKE_API_KEY": "env-key"}.get, api_key_override="override-key"
    )
    assert result is not None
    assert result.auth.api_key == "override-key"


async def test_resolve_provider_auth_no_refresh_when_token_still_valid():
    refresh_calls = 0

    async def refresh(credential: Credential, signal: AbortSignal) -> Credential:
        nonlocal refresh_calls
        refresh_calls += 1
        raise AssertionError("refresh should not be called for a still-valid token")

    auth = make_oauth_provider_auth(refresh=refresh)
    store = InMemoryCredentialStore()
    await store.set(
        "p1",
        Credential(type="oauth", access="valid-access", refresh="r1", expires=now_ms() + 60 * 60 * 1000),
    )

    result = await resolve_provider_auth("p1", auth, store)
    assert result is not None
    assert result.auth.api_key == "valid-access"
    assert refresh_calls == 0


async def test_resolve_provider_auth_refreshes_when_expired():
    async def refresh(credential: Credential, signal: AbortSignal) -> Credential:
        assert credential.refresh == "r1"
        return Credential(type="oauth", access="refreshed-access", refresh="r2", expires=now_ms() + 60 * 60 * 1000)

    auth = make_oauth_provider_auth(refresh=refresh)
    store = InMemoryCredentialStore()
    await store.set(
        "p1",
        Credential(type="oauth", access="stale-access", refresh="r1", expires=now_ms() - 1000),
    )

    result = await resolve_provider_auth("p1", auth, store)
    assert result is not None
    assert result.auth.api_key == "refreshed-access"

    stored = await store.get("p1")
    assert stored.access == "refreshed-access"
    assert stored.refresh == "r2"


async def test_resolve_provider_auth_refreshes_when_near_expiry_within_skew():
    async def refresh(credential: Credential, signal: AbortSignal) -> Credential:
        return Credential(type="oauth", access="refreshed", refresh="r2", expires=now_ms() + 60 * 60 * 1000)

    auth = make_oauth_provider_auth(refresh=refresh)
    store = InMemoryCredentialStore()
    # Expires in 1 minute -- inside the 5 minute minimum-validity margin, so
    # a refresh must be triggered even though the token isn't expired yet.
    await store.set(
        "p1",
        Credential(type="oauth", access="soon-to-expire", refresh="r1", expires=now_ms() + 60 * 1000),
    )

    result = await resolve_provider_auth("p1", auth, store)
    assert result is not None
    assert result.auth.api_key == "refreshed"


async def test_resolve_provider_auth_refresh_failure_raises_models_error_oauth():
    async def refresh(credential: Credential, signal: AbortSignal) -> Credential:
        raise RuntimeError("network exploded")

    auth = make_oauth_provider_auth(refresh=refresh)
    store = InMemoryCredentialStore()
    await store.set(
        "p1",
        Credential(type="oauth", access="stale", refresh="r1", expires=now_ms() - 1000),
    )

    with pytest.raises(ModelsError) as exc_info:
        await resolve_provider_auth("p1", auth, store)
    assert exc_info.value.code == "oauth"


async def test_resolve_provider_auth_to_auth_failure_raises_models_error_oauth():
    async def to_auth(credential: Credential):
        raise RuntimeError("derivation exploded")

    auth = make_oauth_provider_auth(to_auth=to_auth)
    store = InMemoryCredentialStore()
    await store.set(
        "p1",
        Credential(type="oauth", access="valid", refresh="r1", expires=now_ms() + 60 * 60 * 1000),
    )

    with pytest.raises(ModelsError) as exc_info:
        await resolve_provider_auth("p1", auth, store)
    assert exc_info.value.code == "oauth"


async def test_resolve_provider_auth_concurrent_refreshes_are_serialized():
    """Two concurrent resolutions of an expired credential must refresh only once."""
    refresh_calls = 0

    async def refresh(credential: Credential, signal: AbortSignal) -> Credential:
        nonlocal refresh_calls
        refresh_calls += 1
        await asyncio.sleep(0.05)
        return Credential(type="oauth", access="refreshed-once", refresh="r2", expires=now_ms() + 60 * 60 * 1000)

    auth = make_oauth_provider_auth(refresh=refresh)
    store = InMemoryCredentialStore()
    await store.set(
        "p1",
        Credential(type="oauth", access="stale", refresh="r1", expires=now_ms() - 1000),
    )

    results = await asyncio.gather(
        resolve_provider_auth("p1", auth, store),
        resolve_provider_auth("p1", auth, store),
    )
    assert refresh_calls == 1
    assert all(r.auth.api_key == "refreshed-once" for r in results)


# --------------------------------------------------------------------------
# lazy_oauth
# --------------------------------------------------------------------------


async def test_lazy_oauth_loads_once_and_caches():
    load_calls = 0

    async def load() -> OAuthAuth:
        nonlocal load_calls
        load_calls += 1
        await asyncio.sleep(0.01)

        async def login(interaction):
            return Credential(type="oauth", access="a", refresh="r", expires=now_ms() + 1000)

        async def refresh(credential, signal):
            return credential

        async def to_auth(credential):
            return ResolvedAuth(api_key=credential.access)

        return OAuthAuth(name="lazy", login=login, refresh=refresh, to_auth=to_auth)

    lazy = lazy_oauth("lazy", load)

    class FakeInteraction(AuthInteraction):
        signal = AbortController().signal

        async def prompt(self, prompt: AuthPrompt) -> str:
            raise NotImplementedError

        def notify(self, event: AuthEvent) -> None:
            pass

    interaction = FakeInteraction()

    results = await asyncio.gather(
        lazy.login(interaction),
        lazy.login(interaction),
    )
    assert load_calls == 1
    assert all(r.access == "a" for r in results)


# --------------------------------------------------------------------------
# Models integration: login/get_auth for an OAuth-configured provider
# --------------------------------------------------------------------------


class FakeApi:
    def stream(self, model, context, options=None, **kwargs):
        return "stream-result"

    def stream_simple(self, model, context, options=None, **kwargs):
        return "stream-simple-result"


def make_oauth_model() -> Model:
    return Model(
        id="m1",
        name="Model One",
        api="openai-completions",
        provider="",
        base_url="",
        context_window=1000,
        max_tokens=100,
        cost=ModelCost(input=1.0, output=2.0),
    )


class FakeAuthInteraction(AuthInteraction):
    def __init__(self) -> None:
        self.signal = AbortController().signal
        self.events: list[AuthEvent] = []

    async def prompt(self, prompt: AuthPrompt) -> str:
        raise NotImplementedError

    def notify(self, event: AuthEvent) -> None:
        self.events.append(event)


async def test_models_login_oauth_stores_credential_and_get_auth_resolves_it():
    async def login(interaction: AuthInteraction) -> Credential:
        return Credential(type="oauth", access="oauth-access", refresh="r1", expires=now_ms() + 60 * 60 * 1000)

    async def refresh(credential: Credential, signal: AbortSignal) -> Credential:
        raise AssertionError("should not refresh a fresh token")

    async def to_auth(credential: Credential):
        return ResolvedAuth(api_key=credential.access, headers={"X-From": "oauth"})

    provider = create_provider(
        id="oauth-provider",
        name="OAuth Provider",
        auth=ProviderAuth(
            api_key=ApiKeyAuth(name="oauth-provider", env_vars=("OAUTH_PROVIDER_API_KEY",)),
            oauth=OAuthAuth(name="OAuth Provider", login=login, refresh=refresh, to_auth=to_auth),
        ),
        api=FakeApi(),
        models=[make_oauth_model()],
        base_url="https://oauth-provider.invalid/v1",
    )
    models = Models(credential_store=InMemoryCredentialStore(), env={}.get)
    models.add(provider)

    credential = await models.login_oauth("oauth-provider", FakeAuthInteraction())
    assert credential.access == "oauth-access"

    stored = await models.credentials.get("oauth-provider")
    assert stored is not None
    assert stored.type == "oauth"

    auth = await models.get_auth("oauth-provider")
    assert auth is not None
    assert auth.auth.api_key == "oauth-access"
    assert auth.auth.headers["X-From"] == "oauth"
    assert auth.source == "OAuth"

    check = await models.check_auth("oauth-provider")
    assert check is not None
    assert check.configured
    assert check.type == "oauth"


async def test_models_login_oauth_unknown_provider_raises_models_error():
    models = Models(credential_store=InMemoryCredentialStore(), env={}.get)
    with pytest.raises(ModelsError) as exc_info:
        await models.login_oauth("nonexistent", FakeAuthInteraction())
    assert exc_info.value.code == "provider"


async def test_models_login_oauth_provider_without_oauth_support_raises_models_error():
    provider = create_provider(
        id="api-key-only",
        name="Api Key Only",
        auth=ProviderAuth(api_key=ApiKeyAuth(name="api-key-only", env_vars=("SOME_KEY",))),
        api=FakeApi(),
        models=[make_oauth_model()],
        base_url="https://api-key-only.invalid/v1",
    )
    models = Models(credential_store=InMemoryCredentialStore(), env={}.get)
    models.add(provider)

    with pytest.raises(ModelsError) as exc_info:
        await models.login_oauth("api-key-only", FakeAuthInteraction())
    assert exc_info.value.code == "provider"


async def test_models_get_auth_refreshes_expired_oauth_credential():
    async def login(interaction: AuthInteraction) -> Credential:
        raise NotImplementedError

    async def refresh(credential: Credential, signal: AbortSignal) -> Credential:
        return Credential(type="oauth", access="new-access", refresh="new-refresh", expires=now_ms() + 60 * 60 * 1000)

    async def to_auth(credential: Credential):
        return ResolvedAuth(api_key=credential.access)

    provider = create_provider(
        id="oauth-provider-2",
        name="OAuth Provider 2",
        auth=ProviderAuth(
            api_key=ApiKeyAuth(name="oauth-provider-2", env_vars=("UNUSED",)),
            oauth=OAuthAuth(name="OAuth Provider 2", login=login, refresh=refresh, to_auth=to_auth),
        ),
        api=FakeApi(),
        models=[make_oauth_model()],
        base_url="https://oauth-provider-2.invalid/v1",
    )
    models = Models(credential_store=InMemoryCredentialStore(), env={}.get)
    models.add(provider)
    await models.credentials.set(
        "oauth-provider-2",
        Credential(type="oauth", access="stale", refresh="r1", expires=now_ms() - 1000),
    )

    auth = await models.get_auth("oauth-provider-2")
    assert auth is not None
    assert auth.auth.api_key == "new-access"


async def test_models_stream_simple_uses_oauth_base_url_override():
    async def login(interaction: AuthInteraction) -> Credential:
        raise NotImplementedError

    async def refresh(credential: Credential, signal: AbortSignal) -> Credential:
        return credential

    async def to_auth(credential: Credential):
        return ResolvedAuth(api_key=credential.access, base_url="https://dynamic.invalid")

    api = FakeApi()
    seen_models = []

    def stream_simple(model, context, options=None, **kwargs):
        seen_models.append(model)
        return "ok"

    api.stream_simple = stream_simple

    provider = create_provider(
        id="oauth-provider-3",
        name="OAuth Provider 3",
        auth=ProviderAuth(
            api_key=ApiKeyAuth(name="oauth-provider-3", env_vars=("UNUSED",)),
            oauth=OAuthAuth(name="OAuth Provider 3", login=login, refresh=refresh, to_auth=to_auth),
        ),
        api=api,
        models=[make_oauth_model()],
        base_url="https://static.invalid/v1",
    )
    models = Models(credential_store=InMemoryCredentialStore(), env={}.get)
    models.add(provider)
    await models.credentials.set(
        "oauth-provider-3",
        Credential(type="oauth", access="tok", refresh="r1", expires=now_ms() + 60 * 60 * 1000),
    )

    model = provider.get_models()[0]
    await models.stream_simple(model, Context(messages=[]))
    assert len(seen_models) == 1
    assert seen_models[0].base_url == "https://dynamic.invalid"
