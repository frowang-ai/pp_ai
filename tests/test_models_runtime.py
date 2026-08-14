"""Python port of `packages/ai/test/models-runtime.test.ts`.

Cases without a Python subject are called out inline where they would have
gone. In short:

- everything built on `models.refresh()` / `ModelsStore` (dynamic catalog
  publication, restore-before-network, superseded refreshes): `pi_ai.registry`
  has no dynamic-catalog refresh and there is no `models_store` module.
- `CredentialStore.modify()`: the port's store only has `get`/`set`/`delete`,
  and `auth/resolve.py` documents that refresh serialization uses a
  module-level `asyncio.Lock` instead. The observable "no double refresh"
  behavior is ported; the queue-cancellation case is not.
- `ApiKeyAuth.login` / `ApiKeyAuth.check` and the `signal` argument on
  `ApiKeyAuth.resolve`: the port's api-key auth is `resolve(credential, env)`
  only.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from pi_ai.auth.types import (
    ApiKeyAuth,
    AuthInteraction,
    AuthResult,
    Credential,
    CredentialInfo,
    CredentialStore,
    InMemoryCredentialStore,
    OAuthAuth,
    ProviderAuth,
    ResolvedAuth,
)
from pi_ai.models import ModelsError, calculate_cost, has_api
from pi_ai.registry import Models, Provider
from pi_ai.types import (
    Context,
    Cost,
    DoneEvent,
    Model,
    ModelCost,
    ModelCostTier,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
    TextContent,
    Usage,
    UserMessage,
    now_ms,
)
from pi_ai.utils.abort import AbortError, AbortSignal
from pi_ai.utils.event_stream import AssistantMessageEventStream


def make_model(provider: str, model_id: str) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="test-api",
        provider=provider,
        base_url="https://example.test/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=10000,
        max_tokens=1000,
    )


def done_message(model: Model, text: str):
    from pi_ai.types import AssistantMessage

    return AssistantMessage(
        role="assistant",
        content=[TextContent(text=text)],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="stop",
        timestamp=now_ms(),
    )


@dataclass
class ProviderCall:
    model: Model
    options: StreamOptions | None


@dataclass
class RecordingApi:
    calls: list[ProviderCall] = field(default_factory=list)

    def _respond(self, model: Model, options: StreamOptions | None) -> AssistantMessageEventStream:
        self.calls.append(ProviderCall(model=model, options=options))
        stream = AssistantMessageEventStream()
        message = done_message(model, "ok")
        stream.push(StartEvent(partial=message))
        stream.push(DoneEvent(reason="stop", message=message))
        stream.end(message)
        return stream

    def stream(
        self, model: Model, context: Context, options: StreamOptions | None = None, **_kwargs: Any
    ) -> AssistantMessageEventStream:
        return self._respond(model, options)

    def stream_simple(
        self, model: Model, context: Context, options: SimpleStreamOptions | None = None, **_kwargs: Any
    ) -> AssistantMessageEventStream:
        return self._respond(model, options)


async def _ambient_resolve(credential: Credential | None = None, env: Any = None) -> AuthResult:
    """Ambient auth for keyless test providers: configured, with no auth values."""
    return AuthResult(auth=ResolvedAuth(), source="ambient")


AMBIENT_AUTH = ApiKeyAuth(name="Ambient", resolve=_ambient_resolve)


class ListProvider(Provider):
    """Provider whose catalog comes from a caller-supplied callable."""

    def __init__(self, source: Callable[[], list[Model]], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._source = source

    def get_models(self) -> list[Model]:
        return self._source()


def make_provider(
    provider_id: str,
    models: list[Model] | None = None,
    auth: ProviderAuth | None = None,
    api: RecordingApi | None = None,
    get_models: Callable[[], list[Model]] | None = None,
) -> Provider:
    catalog = models if models is not None else [make_model(provider_id, "model-a")]
    kwargs: dict[str, Any] = {
        "id": provider_id,
        "name": provider_id,
        "auth": auth if auth is not None else ProviderAuth(api_key=AMBIENT_AUTH),
        "api": api if api is not None else RecordingApi(),
        "models": catalog,
    }
    if get_models is not None:
        return ListProvider(get_models, **kwargs)
    return Provider(**kwargs)


def make_context() -> Context:
    return Context(messages=[UserMessage(content="hi", timestamp=now_ms())])


def env_key_auth(key: str | None) -> ApiKeyAuth:
    async def resolve(credential: Credential | None = None, env: Any = None) -> AuthResult | None:
        resolved = credential.key if credential is not None and credential.key else key
        if not resolved:
            return None
        return AuthResult(auth=ResolvedAuth(api_key=resolved), source="stored" if credential else "env")

    return ApiKeyAuth(name="Test API key", resolve=resolve)


async def _default_refresh(credential: Credential, signal: AbortSignal | None = None) -> Credential:
    return credential


async def _default_to_auth(credential: Credential) -> ResolvedAuth:
    return ResolvedAuth(api_key=credential.access)


async def _unused_login(interaction: Any) -> Credential:
    raise RuntimeError("not used")


def make_oauth(
    refresh: Callable[..., Any] | None = None,
    to_auth: Callable[..., Any] | None = None,
) -> OAuthAuth:
    return OAuthAuth(
        name="Test OAuth",
        login=_unused_login,
        refresh=refresh if refresh is not None else _default_refresh,
        to_auth=to_auth if to_auth is not None else _default_to_auth,
    )


def oauth_credential(access: str, expires: float, refresh: str = "r") -> Credential:
    return Credential(type="oauth", access=access, refresh=refresh, expires=expires)


# --------------------------------------------------------------------------
# credential store
# --------------------------------------------------------------------------


async def test_enumerates_credential_metadata_without_exposing_secrets() -> None:
    credentials = InMemoryCredentialStore()
    await credentials.set("api-provider", Credential(type="api_key", key="secret"))
    await credentials.set(
        "oauth-provider",
        oauth_credential("access", now_ms() + 60_000, refresh="refresh"),
    )

    assert await credentials.list() == [
        CredentialInfo(provider_id="api-provider", type="api_key"),
        CredentialInfo(provider_id="oauth-provider", type="oauth"),
    ]


def test_applies_request_wide_pricing_tiers_above_the_configured_input_threshold() -> None:
    model = make_model("openai", "gpt-5.6-sol")
    model.cost = ModelCost(
        input=5,
        output=30,
        cache_read=0.5,
        cache_write=6.25,
        tiers=[
            ModelCostTier(
                input_tokens_above=272000,
                input=10,
                output=45,
                cache_read=1,
                cache_write=12.5,
            )
        ],
    )

    def create_usage(cache_write: int) -> Usage:
        return Usage(
            input=200000,
            output=100000,
            cache_read=72000,
            cache_write=cache_write,
            total_tokens=372000 + cache_write,
            cost=Cost(),
        )

    short = calculate_cost(model, create_usage(0))
    assert short.input == 1
    assert short.output == 3
    assert short.cache_read == 0.036
    assert short.cache_write == 0

    long = calculate_cost(model, create_usage(1))
    assert long.input == 2
    assert long.output == 4.5
    assert long.cache_read == 0.072
    assert long.cache_write == 0.0000125


# --------------------------------------------------------------------------
# provider registration and listing
# --------------------------------------------------------------------------


def test_registers_replaces_and_deletes_providers() -> None:
    models = Models()
    models.add(make_provider("p1"))
    models.add(make_provider("p2"))
    assert [provider.id for provider in models.get_providers()] == ["p1", "p2"]

    replacement = make_provider("p1")
    models.add(replacement)
    assert models.get_provider("p1") is replacement
    assert len(models.get_providers()) == 2

    models.delete_provider("p1")
    assert models.get_provider("p1") is None

    models.clear_providers()
    assert models.get_providers() == []


def test_lists_and_finds_models_per_provider() -> None:
    models = Models()
    models.add(make_provider("p1", models=[make_model("p1", "m1"), make_model("p1", "m2")]))
    models.add(make_provider("p2", models=[make_model("p2", "m3")]))

    assert [model.id for model in models.get_models()] == ["m1", "m2", "m3"]
    assert [model.id for model in models.get_models("p1")] == ["m1", "m2"]
    assert models.get_models("nope") == []
    assert models.get_model("p2", "m3") is not None
    assert models.get_model("p2", "m3").id == "m3"
    assert models.get_model("p2", "missing") is None

    found = models.get_model("p2", "m3")
    assert found is not None
    # `hasApi` narrows the static type in TypeScript; in Python it is only the
    # runtime check that remains.
    assert has_api(found, "openai-completions") is False
    assert has_api(found, "test-api") is True


def test_swallows_provider_source_failures_for_listing() -> None:
    def boom() -> list[Model]:
        raise RuntimeError("boom")

    models = Models()
    models.add(make_provider("broken", get_models=boom))
    models.add(make_provider("ok", models=[make_model("ok", "m1")]))

    assert [model.id for model in models.get_models()] == ["m1"]
    assert models.get_models("broken") == []
    # precise failures come from the provider directly
    with pytest.raises(RuntimeError, match="boom"):
        models.get_provider("broken").get_models()


# The `refresh()` / `ModelsStore` group of TypeScript cases has no Python
# subject at all (`Models` has no `refresh()`, no `refresh_models` provider
# hook and there is no models store), so these are not ported:
#   - refresh() updates every configured dynamic provider and reports failures
#   - restricts refresh work to selected providers
#   - restores cached models before waiting for network auth
#   - lets providers choose persistent deletion and ephemeral publication
#   - persists dynamic catalogs and restores them without network access
#   - passes effective API-key credentials and refresh options
#   - refreshes expired OAuth before refreshing models
#   - always gives providers a concrete signal
#   - binds model-store waits to the provider refresh signal
#   - returns aborted state without reporting cancellation as a provider error
#   - stops waiting on abort when a provider ignores its signal
#   - rejects late publication from a superseded non-cooperative provider


# --------------------------------------------------------------------------
# auth resolution
# --------------------------------------------------------------------------


# TypeScript's "passes caller signals to provider auth callbacks" asserts that
# one caller signal reaches all three api-key auth callbacks: `check`,
# `resolve` and `login`. Only part of that claim exists here:
#   - `ApiKeyAuth` has no `check` hook at all (`Models.check_auth` answers from
#     the stored credential plus `resolve`), and it takes no signal.
#   - `ApiKeyAuth.resolve` is `(credential, env)` -- no signal parameter.
#   - `ApiKeyAuth.login(interaction)` does carry a signal, on the interaction,
#     and that half is asserted below.
# The signal a caller passes to `Models.get_auth(..., signal=...)` is still
# honoured for OAuth refresh; see
# `test_passes_cancellation_to_oauth_refresh_and_preserves_the_previous_credential`.


async def test_passes_the_caller_signal_to_the_api_key_login_callback() -> None:
    received: list[AbortSignal | None] = []

    async def login(interaction: AuthInteraction) -> Credential:
        received.append(interaction.signal)
        return Credential(type="api_key", key="saved")

    auth = ApiKeyAuth(name="Signal auth", resolve=_ambient_resolve, login=login)
    signal = AbortSignal()

    class SignalInteraction(AuthInteraction):
        def __init__(self) -> None:
            self.signal = signal

        async def prompt(self, prompt: Any) -> str:
            return "unused"

        def notify(self, event: Any) -> None:
            return None

    assert await auth.login(SignalInteraction()) == Credential(type="api_key", key="saved")
    assert received == [signal]


async def test_stops_waiting_for_non_cooperative_auth_resolution() -> None:
    blocked_resolve: asyncio.Event = asyncio.Event()
    resolve_started: asyncio.Event = asyncio.Event()

    async def resolve(credential: Credential | None = None, env: Any = None) -> AuthResult:
        resolve_started.set()
        await blocked_resolve.wait()
        return AuthResult(auth=ResolvedAuth(api_key="key"), source="blocked")

    models = Models()
    models.add(make_provider("p1", auth=ProviderAuth(api_key=ApiKeyAuth(name="Blocked auth", resolve=resolve))))

    signal = AbortSignal()
    auth = asyncio.ensure_future(models.get_auth("p1", signal=signal))
    await resolve_started.wait()
    signal.abort()
    with pytest.raises(AbortError):
        await auth

    blocked_resolve.set()


# "cancels queued credential mutations without running them later" is not
# ported: `CredentialStore` has no `modify()` queue in this port.


async def test_passes_cancellation_to_oauth_refresh_and_preserves_the_previous_credential() -> None:
    credentials = InMemoryCredentialStore()
    previous = oauth_credential("old", 0, refresh="old-refresh")
    await credentials.set("p1", previous)

    refresh_started: asyncio.Event = asyncio.Event()
    blocked_refresh: asyncio.Future[Credential] = asyncio.get_event_loop().create_future()
    received: dict[str, AbortSignal | None] = {}

    async def refresh(credential: Credential, signal: AbortSignal | None = None) -> Credential:
        received["signal"] = signal
        refresh_started.set()
        return await blocked_refresh

    models = Models(credential_store=credentials)
    models.add(make_provider("p1", auth=ProviderAuth(api_key=AMBIENT_AUTH, oauth=make_oauth(refresh=refresh))))

    signal = AbortSignal()
    auth = asyncio.ensure_future(models.get_auth("p1", signal=signal))
    await refresh_started.wait()
    signal.abort()

    with pytest.raises(AbortError):
        await auth
    assert isinstance(received["signal"], AbortSignal)
    assert received["signal"].aborted is True

    blocked_refresh.set_result(oauth_credential("new", now_ms() + 60_000, refresh="old-refresh"))
    await asyncio.sleep(0)
    assert await credentials.get("p1") == previous


async def test_resolves_auth_stored_credential_owns_the_provider() -> None:
    credentials = InMemoryCredentialStore()
    models = Models(credential_store=credentials)
    models.add(make_provider("p1", auth=ProviderAuth(api_key=env_key_auth("env-key"), oauth=make_oauth())))
    model = make_model("p1", "model-a")

    assert (await models.get_auth(model)).auth.api_key == "env-key"
    assert (await models.get_auth(model.provider)).auth.api_key == "env-key"
    assert (await models.get_auth(model, api_key="explicit-key")).auth.api_key == "explicit-key"

    await credentials.set("p1", oauth_credential("oauth-token", now_ms() + 10 * 60_000))
    resolution = await models.get_auth(model.provider)
    assert resolution.auth.api_key == "oauth-token"
    assert resolution.source == "OAuth"

    await credentials.set("p1", Credential(type="api_key", key="stored-key"))
    api_key_resolution = await models.get_auth(model.provider)
    assert api_key_resolution.auth.api_key == "stored-key"
    assert api_key_resolution.source == "stored"


async def test_checks_provider_auth_without_refreshing_oauth_and_filters_available_models() -> None:
    credentials = InMemoryCredentialStore()
    refreshes = {"count": 0}

    async def refresh(credential: Credential, signal: AbortSignal | None = None) -> Credential:
        refreshes["count"] += 1
        return credential

    models = Models(credential_store=credentials)
    models.add(make_provider("ambient", auth=ProviderAuth(api_key=env_key_auth("env-key"))))
    models.add(make_provider("missing", auth=ProviderAuth(api_key=env_key_auth(None))))
    models.add(make_provider("oauth", auth=ProviderAuth(api_key=env_key_auth(None), oauth=make_oauth(refresh=refresh))))
    await credentials.set("oauth", oauth_credential("expired", 0, refresh="refresh"))

    ambient = await models.check_auth("ambient")
    assert (ambient.configured, ambient.source, ambient.type) == (True, "env", "api_key")
    # TypeScript returns `undefined` for an unconfigured provider; the port
    # distinguishes "unknown provider" (None) from "not configured".
    missing = await models.check_auth("missing")
    assert missing.configured is False
    oauth_check = await models.check_auth("oauth")
    assert (oauth_check.configured, oauth_check.source, oauth_check.type) == (True, "OAuth", "oauth")
    assert refreshes["count"] == 0

    assert [model.provider for model in await models.get_available()] == ["ambient", "oauth"]
    assert [model.provider for model in await models.get_available("ambient")] == ["ambient"]


async def test_runs_provider_login_and_logout_through_the_credential_store() -> None:
    credentials = InMemoryCredentialStore()

    async def login(interaction: AuthInteraction) -> Credential:
        return Credential(type="api_key", key="logged-in")

    api_key = env_key_auth(None)
    api_key.login = login
    models = Models(credential_store=credentials)
    models.add(make_provider("p1", auth=ProviderAuth(api_key=api_key)))

    class NoopInteraction(AuthInteraction):
        def __init__(self) -> None:
            self.signal = AbortSignal()

        async def prompt(self, prompt: Any) -> str:
            return "unused"

        def notify(self, event: Any) -> None:
            return None

    credential = await models.login("p1", interaction=NoopInteraction())
    assert credential == Credential(type="api_key", key="logged-in")
    assert await credentials.get("p1") == credential

    await models.logout("p1")
    assert await credentials.get("p1") is None

    # TypeScript has no key-only overload; the port keeps one because callers
    # that already hold a pasted key should not have to fake an interaction.
    assert await models.login("p1", "pasted") == Credential(type="api_key", key="pasted")
    assert await credentials.get("p1") == Credential(type="api_key", key="pasted")


async def test_a_stored_credential_without_a_matching_handler_blocks_ambient_fallback() -> None:
    credentials = InMemoryCredentialStore()
    models = Models(credential_store=credentials)
    models.add(make_provider("p1", auth=ProviderAuth(api_key=env_key_auth("env-key"))))
    await credentials.set("p1", oauth_credential("a", 0))

    assert await models.get_auth("p1") is None


async def test_refreshes_expired_oauth_credentials_and_persists_the_rotated_credential() -> None:
    credentials = InMemoryCredentialStore()

    async def refresh(credential: Credential, signal: AbortSignal | None = None) -> Credential:
        return oauth_credential("new-token", now_ms() + 60 * 60_000)

    models = Models(credential_store=credentials)
    models.add(make_provider("p1", auth=ProviderAuth(api_key=AMBIENT_AUTH, oauth=make_oauth(refresh=refresh))))
    await credentials.set("p1", oauth_credential("old-token", 0))

    resolution = await models.get_auth("p1")
    assert resolution.auth.api_key == "new-token"
    assert (await credentials.get("p1")).access == "new-token"


async def test_refreshes_oauth_credentials_with_less_than_five_minutes_remaining() -> None:
    credentials = InMemoryCredentialStore()
    calls = {"count": 0}

    async def refresh(credential: Credential, signal: AbortSignal | None = None) -> Credential:
        calls["count"] += 1
        return oauth_credential("new-token", now_ms() + 60 * 60_000)

    models = Models(credential_store=credentials)
    models.add(make_provider("p1", auth=ProviderAuth(api_key=AMBIENT_AUTH, oauth=make_oauth(refresh=refresh))))
    await credentials.set("p1", oauth_credential("old-token", now_ms() + 60_000))

    assert (await models.get_auth("p1")).auth.api_key == "new-token"
    assert calls["count"] == 1


async def test_honors_a_callers_longer_oauth_minimum_validity() -> None:
    credentials = InMemoryCredentialStore()
    calls = {"count": 0}

    async def refresh(credential: Credential, signal: AbortSignal | None = None) -> Credential:
        calls["count"] += 1
        return oauth_credential("new-token", now_ms() + 60 * 60_000)

    models = Models(credential_store=credentials)
    models.add(make_provider("p1", auth=ProviderAuth(api_key=AMBIENT_AUTH, oauth=make_oauth(refresh=refresh))))
    await credentials.set("p1", oauth_credential("old-token", now_ms() + 10 * 60_000))

    resolution = await models.get_auth("p1", min_oauth_validity_ms=30 * 60_000)
    assert resolution.auth.api_key == "new-token"
    assert calls["count"] == 1


async def test_rejects_with_code_oauth_when_refresh_fails_preserving_the_stored_credential() -> None:
    credentials = InMemoryCredentialStore()

    async def refresh(credential: Credential, signal: AbortSignal | None = None) -> Credential:
        raise RuntimeError("invalid_grant")

    models = Models(credential_store=credentials)
    models.add(make_provider("p1", auth=ProviderAuth(api_key=AMBIENT_AUTH, oauth=make_oauth(refresh=refresh))))
    await credentials.set("p1", oauth_credential("old", 0))

    with pytest.raises(ModelsError) as excinfo:
        await models.get_auth("p1")
    assert excinfo.value.code == "oauth"
    # credential preserved for retry / re-login
    assert (await credentials.get("p1")).access == "old"


async def test_serializes_concurrent_oauth_refreshes_without_double_refreshing() -> None:
    credentials = InMemoryCredentialStore()
    await credentials.set("p1", oauth_credential("old", 0, refresh="r1"))
    refreshes = {"count": 0}
    entered = asyncio.Event()
    release = asyncio.Event()

    # The first refresh blocks on an event rather than a real sleep: the second
    # caller has to arrive while it is still in flight for this test to mean
    # anything, and a wall-clock sleep only makes that likely, not certain.
    async def refresh(credential: Credential, signal: AbortSignal | None = None) -> Credential:
        refreshes["count"] += 1
        entered.set()
        await release.wait()
        return oauth_credential(f"new-{refreshes['count']}", now_ms() + 60 * 60_000, refresh="r2")

    models = Models(credential_store=credentials)
    models.add(make_provider("p1", auth=ProviderAuth(api_key=AMBIENT_AUTH, oauth=make_oauth(refresh=refresh))))
    model = make_model("p1", "model-a")

    first = asyncio.ensure_future(models.get_auth(model.provider))
    second = asyncio.ensure_future(models.get_auth(model.provider))
    await asyncio.wait_for(entered.wait(), timeout=5)
    # Drain the ready queue so the second caller is parked on the refresh lock
    # before the first refresh is allowed to finish.
    for _ in range(5):
        await asyncio.sleep(0)
    release.set()

    a, b = await asyncio.gather(first, second)
    assert refreshes["count"] == 1
    assert a.auth.api_key == "new-1"
    assert b.auth.api_key == "new-1"
    assert refreshes["count"] == 1
    assert a.auth.api_key == "new-1"
    assert b.auth.api_key == "new-1"


async def test_valid_oauth_tokens_resolve_without_writing_to_the_store() -> None:
    # TypeScript counts `store.modify` calls; the port has no `modify`, so the
    # equivalent observation is that no write happens at all.
    base = InMemoryCredentialStore()
    writes = {"count": 0}

    class CountingStore(CredentialStore):
        async def get(self, provider_id: str) -> Credential | None:
            return await base.get(provider_id)

        async def set(self, provider_id: str, credential: Credential) -> None:
            writes["count"] += 1
            await base.set(provider_id, credential)

        async def delete(self, provider_id: str) -> None:
            await base.delete(provider_id)

        async def list(self) -> list[CredentialInfo]:
            return await base.list()

    await base.set("p1", oauth_credential("valid", now_ms() + 10 * 60_000))
    models = Models(credential_store=CountingStore())
    models.add(make_provider("p1", auth=ProviderAuth(api_key=AMBIENT_AUTH, oauth=make_oauth())))

    assert (await models.get_auth("p1")).auth.api_key == "valid"
    assert writes["count"] == 0


async def test_wraps_credential_store_failures_in_models_error() -> None:
    class ReadFailingStore(CredentialStore):
        async def get(self, provider_id: str) -> Credential | None:
            raise RuntimeError("disk on fire")

        async def set(self, provider_id: str, credential: Credential) -> None:
            return None

        async def delete(self, provider_id: str) -> None:
            return None

        async def list(self) -> list[CredentialInfo]:
            return []

    models = Models(credential_store=ReadFailingStore())
    models.add(make_provider("p1", auth=ProviderAuth(api_key=env_key_auth("env-key"))))
    with pytest.raises(ModelsError) as excinfo:
        await models.get_auth("p1")
    assert excinfo.value.code == "auth"

    class WriteFailingStore(CredentialStore):
        async def get(self, provider_id: str) -> Credential | None:
            return oauth_credential("old", 0)

        async def set(self, provider_id: str, credential: Credential) -> None:
            raise RuntimeError("disk on fire")

        async def delete(self, provider_id: str) -> None:
            return None

        async def list(self) -> list[CredentialInfo]:
            return [CredentialInfo(provider_id="p1", type="oauth")]

    oauth_models = Models(credential_store=WriteFailingStore())
    oauth_models.add(make_provider("p1", auth=ProviderAuth(api_key=AMBIENT_AUTH, oauth=make_oauth())))
    with pytest.raises(Exception) as write_excinfo:
        await oauth_models.get_auth("p1")
    assert "disk on fire" in str(write_excinfo.value)


async def test_keeps_the_underlying_reason_in_wrapped_oauth_refresh_errors() -> None:
    credentials = InMemoryCredentialStore()
    await credentials.set("p1", oauth_credential("old", 0))

    async def refresh(credential: Credential, signal: AbortSignal | None = None) -> Credential:
        raise RuntimeError("token refresh failed (400): invalid_grant")

    models = Models(credential_store=credentials)
    models.add(make_provider("p1", auth=ProviderAuth(api_key=AMBIENT_AUTH, oauth=make_oauth(refresh=refresh))))

    with pytest.raises(ModelsError) as excinfo:
        await models.get_auth("p1")
    assert str(excinfo.value) == "OAuth refresh failed for p1: token refresh failed (400): invalid_grant"


async def test_wraps_api_key_auth_failures_in_models_error() -> None:
    async def failing(credential: Credential | None = None, env: Any = None) -> AuthResult:
        raise RuntimeError("nope")

    models = Models()
    models.add(make_provider("p1", auth=ProviderAuth(api_key=ApiKeyAuth(name="Failing", resolve=failing))))
    with pytest.raises(ModelsError) as excinfo:
        await models.get_auth("p1")
    assert excinfo.value.code == "auth"


async def test_uses_explicit_request_api_key_and_env_during_provider_auth_resolution() -> None:
    api = RecordingApi()

    async def resolve(credential: Credential | None = None, env: Any = None) -> AuthResult | None:
        account = (credential.env.get("ACCOUNT_ID") if credential is not None else None) or (
            env("ACCOUNT_ID") if env is not None else None
        )
        if credential is None or not credential.key or not account:
            return None
        return AuthResult(
            auth=ResolvedAuth(api_key=credential.key, base_url=f"https://example.test/{account}"),
            source="scoped",
            env={"ACCOUNT_ID": account},
        )

    models = Models()
    models.add(make_provider("p1", auth=ProviderAuth(api_key=ApiKeyAuth(name="Scoped", resolve=resolve)), api=api))
    model = make_model("p1", "model-a")

    stream = await models.stream_simple(
        model, make_context(), SimpleStreamOptions(api_key="explicit-key", env={"ACCOUNT_ID": "acct"})
    )
    await stream.result()

    assert api.calls[0].model.base_url == "https://example.test/acct"
    assert api.calls[0].options.api_key == "explicit-key"
    assert api.calls[0].options.env == {"ACCOUNT_ID": "acct"}


async def test_merges_resolved_auth_into_stream_options_explicit_options_win_per_field() -> None:
    api = RecordingApi()

    async def resolve(credential: Credential | None = None, env: Any = None) -> AuthResult:
        return AuthResult(
            auth=ResolvedAuth(
                api_key="resolved-key",
                headers={"Authorization": "Bearer auth", "x-a": "auth", "x-b": "auth"},
                base_url="https://auth.test/v1",
            ),
            source="test",
        )

    models = Models()
    models.add(make_provider("p1", auth=ProviderAuth(api_key=ApiKeyAuth(name="Test", resolve=resolve)), api=api))
    model = make_model("p1", "model-a")

    stream = await models.stream_simple(
        model,
        make_context(),
        SimpleStreamOptions(api_key="explicit-key", headers={"authorization": "Explicit token", "x-b": "explicit"}),
    )
    result = await stream.result()
    assert result.stop_reason == "stop"
    assert len(api.calls) == 1
    assert api.calls[0].options.api_key == "explicit-key"
    assert api.calls[0].options.headers == {"authorization": "Explicit token", "x-a": "auth", "x-b": "explicit"}
    assert api.calls[0].model.base_url == "https://auth.test/v1"

    result2 = await (await models.stream_simple(model, make_context())).result()
    assert result2.stop_reason == "stop"
    assert api.calls[1].options.api_key == "resolved-key"


async def test_adds_model_headers_only_for_model_auth() -> None:
    api = RecordingApi()
    models = Models()
    models.add(make_provider("p1", auth=ProviderAuth(api_key=env_key_auth("key")), api=api))
    model = make_model("p1", "model-a")
    model.headers = {"x-model": "model", "x-shared": "model"}

    assert (await models.get_auth("p1")).auth.headers == {}
    assert (await models.get_auth(model)).auth.headers == {"x-model": "model", "x-shared": "model"}

    # `transformHeaders` (a Models-only post-merge hook) has no counterpart on
    # `StreamOptions` in this port, so that half of the case is dropped.
    await (
        await models.stream_simple(
            model,
            make_context(),
            SimpleStreamOptions(headers={"x-explicit": "explicit", "X-Shared": "explicit"}),
        )
    ).result()

    assert api.calls[0].options.headers == {
        "x-model": "model",
        "x-explicit": "explicit",
        "X-Shared": "explicit",
    }


async def test_produces_an_error_stream_for_unknown_providers_instead_of_throwing() -> None:
    models = Models()
    result = await (await models.stream_simple(make_model("ghost", "model-a"), make_context())).result()
    assert result.stop_reason == "error"
    assert "Unknown provider: ghost" in (result.error_message or "")


async def test_streams_through_the_provider() -> None:
    models = Models()
    models.add(make_provider("p1"))
    model = make_model("p1", "model-a")

    stream = await models.stream_simple(model, make_context())
    events = [event.type async for event in stream]
    assert events == ["start", "done"]
    message = await stream.result()
    assert message.stop_reason == "stop"
