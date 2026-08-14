"""Python port of `packages/ai/test/providers.test.ts`.

Two TypeScript cases have no full Python subject and are called out inline
where they would have gone:

- `lazyApi` ("lazily exposes only declared deferred capabilities"): the port has
  no `api/lazy.py` because API modules are imported directly, so only the
  capability-declaration half of that case is portable.
- `models.refresh()` / `InMemoryModelsStore` ("lets a newer dynamic refresh
  bypass and supersede older network work"): `pi_ai.registry.Models` has no
  dynamic-catalog refresh, so there is nothing to drive.

The provider-owned interactive `login` flows (Bedrock bearer/profile, Vertex
API key/ADC, `envApiKeyAuth`'s secret prompt) *were* unportable when this file
was first written: `pi_ai.auth.types.ApiKeyAuth` had no `login` hook. It has one
now, and those cases are ported.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from pi_ai.auth.helpers import env_api_key_auth, resolve_api_key_auth
from pi_ai.auth.types import (
    ApiKeyAuth,
    AuthEvent,
    AuthInteraction,
    AuthPrompt,
    AuthResult,
    Credential,
    ProviderAuth,
    ResolvedAuth,
)
from pi_ai.models import complete
from pi_ai.providers.all import builtin_models, builtin_providers, get_builtin_model
from pi_ai.providers.amazon_bedrock import amazon_bedrock_provider
from pi_ai.providers.anthropic import anthropic_provider
from pi_ai.providers.cloudflare_ai_gateway import cloudflare_ai_gateway_provider
from pi_ai.providers.cloudflare_workers_ai import cloudflare_workers_ai_provider
from pi_ai.providers.faux import (
    FauxDeferredOptions,
    RegisterFauxProviderOptions,
    faux_assistant_message,
    faux_provider,
)
from pi_ai.providers.google_vertex import google_vertex_provider
from pi_ai.registry import Models, create_provider
from pi_ai.types import (
    Context,
    DeferredHandle,
    DoneEvent,
    Model,
    ModelCost,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
    TextContent,
    UserMessage,
    now_ms,
)
from pi_ai.utils.abort import AbortController
from pi_ai.utils.event_stream import AssistantMessageEventStream

VERTEX_ADC_PATH = "~/.config/gcloud/application_default_credentials.json"


def fake_env(env: dict[str, str]) -> Callable[[str], str | None]:
    return lambda name: env.get(name)


def make_context() -> Context:
    return Context(messages=[UserMessage(content="hi", timestamp=now_ms())])


def make_model(api: str, model_id: str, provider: str = "mixed") -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api=api,
        provider=provider,
        base_url="https://example.test/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=10000,
        max_tokens=1000,
    )


# --------------------------------------------------------------------------
# builtin providers
# --------------------------------------------------------------------------


def test_builtin_models_registers_every_builtin_provider_with_models() -> None:
    models = builtin_models()
    providers = models.get_providers()
    assert len(providers) == len(builtin_providers())
    assert "anthropic" in [provider.id for provider in providers]

    anthropic = models.get_model("anthropic", "claude-haiku-4-5")
    assert anthropic is not None
    assert anthropic.api == "anthropic-messages"

    assert len(models.get_models()) > 500

    # Static providers list models immediately; Radius is purely dynamic.
    for provider in providers:
        listed = models.get_models(provider.id)
        if provider.id == "radius":
            assert listed == []
        else:
            assert len(listed) > 0
        assert all(model.provider == provider.id for model in listed)


def test_stores_native_constrained_sampling_capabilities_in_model_metadata() -> None:
    gpt4o = get_builtin_model("openai", "gpt-4o")
    assert gpt4o is not None
    assert gpt4o.compat.get("supportsStrictMode") is True
    assert "supportsOpenAIGrammarTools" not in gpt4o.compat

    gpt54 = get_builtin_model("openai", "gpt-5.4")
    assert gpt54 is not None
    assert gpt54.compat.get("supportsStrictMode") is True
    assert gpt54.compat.get("supportsOpenAIGrammarTools") is True

    haiku = get_builtin_model("anthropic", "claude-haiku-4-5")
    assert haiku is not None
    assert haiku.compat.get("supportsStrictTools") is True


def test_uses_official_kimi_k3_pricing_for_moonshot_providers() -> None:
    models = builtin_models()
    for provider in ("moonshotai", "moonshotai-cn"):
        model = models.get_model(provider, "kimi-k3")
        assert model is not None
        assert model.cost == ModelCost(input=3, output=15, cache_read=0.3, cache_write=0)


def test_uses_api_equivalent_implied_pricing_for_kimi_coding_subscription_models() -> None:
    models = builtin_models()
    expected = {
        "k3": ModelCost(input=3, output=15, cache_read=0.3, cache_write=0),
        "kimi-for-coding-highspeed": ModelCost(input=1.9, output=8, cache_read=0.38, cache_write=0),
    }
    for model_id, cost in expected.items():
        model = models.get_model("kimi-coding", model_id)
        assert model is not None
        assert model.cost == cost


async def test_resolves_anthropic_bearer_auth_from_env_with_auth_token_precedence() -> None:
    models = Models(
        env=fake_env(
            {
                "ANTHROPIC_AUTH_TOKEN": "auth-token",
                "ANTHROPIC_OAUTH_TOKEN": "oauth-token",
                "ANTHROPIC_API_KEY": "api-key",
            }
        )
    )
    models.add(anthropic_provider())

    result = await models.get_auth("anthropic")
    assert result is not None
    assert result.auth.api_key is None
    assert result.auth.headers == {"Authorization": "Bearer auth-token"}
    assert result.source == "ANTHROPIC_AUTH_TOKEN"


async def test_preserves_anthropic_oauth_token_precedence_over_the_api_key() -> None:
    models = Models(env=fake_env({"ANTHROPIC_API_KEY": "key", "ANTHROPIC_OAUTH_TOKEN": "oauth-token"}))
    models.add(anthropic_provider())

    result = await models.get_auth("anthropic")
    assert result is not None
    assert result.auth.api_key == "oauth-token"
    assert result.source == "ANTHROPIC_OAUTH_TOKEN"


class ScriptedInteraction(AuthInteraction):
    """Answers prompts from a fixed list and records notifications.

    Port of the inline `{ signal, prompt, notify }` objects the TypeScript test
    passes to `auth.login`.
    """

    def __init__(self, answers: list[str]) -> None:
        self.signal = AbortController().signal
        self._answers = list(answers)
        self.events: list[AuthEvent] = []
        self.prompts: list[AuthPrompt] = []

    async def prompt(self, prompt: AuthPrompt) -> str:
        self.prompts.append(prompt)
        return self._answers.pop(0)

    def notify(self, event: AuthEvent) -> None:
        self.events.append(event)


async def test_runs_provider_owned_bedrock_bearer_token_and_aws_profile_login_flows() -> None:
    auth = amazon_bedrock_provider().auth.api_key
    assert auth.login is not None

    bearer = await auth.login(ScriptedInteraction(["bearer-token", "bedrock-token"]))
    assert bearer == Credential(type="api_key", key="bedrock-token")

    profile_interaction = ScriptedInteraction(["aws-profile", "work"])
    profile = await auth.login(profile_interaction)
    assert profile == Credential(type="api_key", env={"AWS_PROFILE": "work"})
    assert len(profile_interaction.events) == 1
    assert profile_interaction.events[0].type == "info"
    assert [link.label for link in profile_interaction.events[0].links] == ["AWS credential provider chain"]

    result = await resolve_api_key_auth(
        auth,
        Credential(type="api_key", env={"AWS_PROFILE": "work"}),
        fake_env({}),
    )
    assert result is not None
    assert result.auth == ResolvedAuth()
    assert result.env == {"AWS_PROFILE": "work"}


async def test_bedrock_login_stores_an_empty_credential_for_the_ambient_chain() -> None:
    # The third branch of the same TypeScript `login`: "Existing AWS credential
    # chain" only confirms, then stores a credential with neither key nor env so
    # `resolve` keeps detecting ambient credentials.
    auth = amazon_bedrock_provider().auth.api_key
    assert auth.login is not None

    interaction = ScriptedInteraction(["credential-chain", ""])
    assert await auth.login(interaction) == Credential(type="api_key")
    assert [prompt.type for prompt in interaction.prompts] == ["select", "text"]

    with pytest.raises(ValueError, match="Unknown Amazon Bedrock auth method: nonsense"):
        await auth.login(ScriptedInteraction(["nonsense"]))


async def test_reports_bedrock_as_configured_from_ambient_aws_credentials() -> None:
    models = Models(env=fake_env({"AWS_PROFILE": "dev"}))
    models.add(amazon_bedrock_provider())
    model = models.get_models("amazon-bedrock")[0]

    result = await models.get_auth(model.provider)
    assert result is not None
    assert result.auth == ResolvedAuth()
    assert result.source == "AWS_PROFILE"

    unconfigured = Models(env=fake_env({}))
    unconfigured.add(amazon_bedrock_provider())
    assert await unconfigured.get_auth(model.provider) is None


async def test_requires_cloudflare_workers_ai_account_config_and_returns_scoped_env() -> None:
    missing_account = Models(env=fake_env({"CLOUDFLARE_API_KEY": "cf-key"}))
    missing_account.add(cloudflare_workers_ai_provider())
    model = missing_account.get_models("cloudflare-workers-ai")[0]
    assert await missing_account.get_auth(model.provider) is None

    configured = Models(env=fake_env({"CLOUDFLARE_API_KEY": "cf-key", "CLOUDFLARE_ACCOUNT_ID": "account-id"}))
    configured.add(cloudflare_workers_ai_provider())
    result = await configured.get_auth(model.provider)
    assert result is not None
    assert result.auth == ResolvedAuth(api_key="cf-key")
    assert result.env == {"CLOUDFLARE_ACCOUNT_ID": "account-id"}


async def test_requires_cloudflare_ai_gateway_config_and_returns_scoped_env_headers() -> None:
    missing_gateway = Models(env=fake_env({"CLOUDFLARE_API_KEY": "cf-key", "CLOUDFLARE_ACCOUNT_ID": "account-id"}))
    missing_gateway.add(cloudflare_ai_gateway_provider())
    model = missing_gateway.get_models("cloudflare-ai-gateway")[0]
    assert await missing_gateway.get_auth(model.provider) is None

    configured = Models(
        env=fake_env(
            {
                "CLOUDFLARE_API_KEY": "cf-key",
                "CLOUDFLARE_ACCOUNT_ID": "account-id",
                "CLOUDFLARE_GATEWAY_ID": "gateway-id",
            }
        )
    )
    configured.add(cloudflare_ai_gateway_provider())
    result = await configured.get_auth(model.provider)
    assert result is not None
    assert result.auth.api_key is None
    assert result.auth.headers == {
        "cf-aig-authorization": "Bearer cf-key",
        "Authorization": None,
        "x-api-key": None,
    }
    assert result.env == {"CLOUDFLARE_ACCOUNT_ID": "account-id", "CLOUDFLARE_GATEWAY_ID": "gateway-id"}


def patch_adc_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the ADC file look present.

    TypeScript injects `fileExists` through `AuthContext`; the port checks the
    ADC path on disk directly, so the existence check is what gets faked.
    """
    import pathlib

    real_exists = pathlib.Path.exists

    def fake_exists(self: pathlib.Path) -> bool:
        if str(self) == str(pathlib.Path(VERTEX_ADC_PATH).expanduser()):
            return True
        return real_exists(self)

    monkeypatch.setattr(pathlib.Path, "exists", fake_exists)


async def test_runs_provider_owned_vertex_api_key_and_adc_login_flows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_adc_exists(monkeypatch)
    auth = google_vertex_provider().auth.api_key
    assert auth.login is not None

    key_credential = await auth.login(ScriptedInteraction(["api-key", "vertex-key"]))
    assert key_credential == Credential(type="api_key", key="vertex-key")

    adc_interaction = ScriptedInteraction(["adc", "project-id", "us-central1"])
    adc_credential = await auth.login(adc_interaction)
    assert adc_credential == Credential(
        type="api_key",
        env={"GOOGLE_CLOUD_PROJECT": "project-id", "GOOGLE_CLOUD_LOCATION": "us-central1"},
    )
    assert len(adc_interaction.events) == 1
    assert adc_interaction.events[0].type == "info"
    assert [link.label for link in adc_interaction.events[0].links] == ["Application Default Credentials"]

    result = await resolve_api_key_auth(
        auth,
        Credential(
            type="api_key",
            env={"GOOGLE_CLOUD_PROJECT": "project-id", "GOOGLE_CLOUD_LOCATION": "us-central1"},
        ),
        fake_env({}),
    )
    assert result is not None
    assert result.auth == ResolvedAuth()
    assert result.env == {"GOOGLE_CLOUD_PROJECT": "project-id", "GOOGLE_CLOUD_LOCATION": "us-central1"}


async def test_vertex_login_asks_for_the_service_account_file_path() -> None:
    # The third branch of the same TypeScript `login`.
    auth = google_vertex_provider().auth.api_key
    assert auth.login is not None

    credential = await auth.login(ScriptedInteraction(["service-account", "project-id", "us-central1", "/tmp/sa.json"]))
    assert credential == Credential(
        type="api_key",
        env={
            "GOOGLE_CLOUD_PROJECT": "project-id",
            "GOOGLE_CLOUD_LOCATION": "us-central1",
            "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/sa.json",
        },
    )

    with pytest.raises(ValueError, match="Unknown Google Vertex AI auth method: nonsense"):
        await auth.login(ScriptedInteraction(["nonsense"]))


async def test_resolves_vertex_via_adc_file_plus_project_and_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_adc_exists(monkeypatch)

    configured = Models(env=fake_env({"GOOGLE_CLOUD_PROJECT": "proj", "GOOGLE_CLOUD_LOCATION": "us-central1"}))
    configured.add(google_vertex_provider())
    model = configured.get_models("google-vertex")[0]

    result = await configured.get_auth(model.provider)
    assert result is not None
    assert result.auth == ResolvedAuth()
    assert result.source is not None
    assert "application default" in result.source

    # ADC without project/location is not configured.
    partial = Models(env=fake_env({"GOOGLE_CLOUD_PROJECT": "proj"}))
    partial.add(google_vertex_provider())
    assert await partial.get_auth(model.provider) is None

    # An explicit key wins over ADC.
    keyed = Models(env=fake_env({"GOOGLE_CLOUD_API_KEY": "vertex-key"}))
    keyed.add(google_vertex_provider())
    keyed_result = await keyed.get_auth(model.provider)
    assert keyed_result is not None
    assert keyed_result.auth.api_key == "vertex-key"


# --------------------------------------------------------------------------
# env_api_key_auth
# --------------------------------------------------------------------------


async def test_env_api_key_auth_prefers_the_stored_credential_and_falls_back_in_order() -> None:
    auth = env_api_key_auth("Test key", ["FIRST_KEY", "SECOND_KEY"])

    stored = await resolve_api_key_auth(auth, Credential(type="api_key", key="stored"), fake_env({"FIRST_KEY": "env"}))
    assert stored is not None
    assert stored.auth.api_key == "stored"
    assert stored.source == "stored credential"

    second = await resolve_api_key_auth(auth, None, fake_env({"SECOND_KEY": "second"}))
    assert second is not None
    assert second.auth.api_key == "second"
    assert second.source == "SECOND_KEY"

    assert await resolve_api_key_auth(auth, None, fake_env({})) is None


async def test_env_api_key_auth_login_prompts_for_a_secret_and_returns_a_credential() -> None:
    auth = env_api_key_auth("Test key", ["FIRST_KEY"])
    assert auth.login is not None

    interaction = ScriptedInteraction(["secret-key"])
    assert await auth.login(interaction) == Credential(type="api_key", key="secret-key")
    assert interaction.prompts == [AuthPrompt(type="secret", message="Enter Test key")]


# --------------------------------------------------------------------------
# create_provider
# --------------------------------------------------------------------------


@dataclass
class RecordingStreams:
    """Port of the TypeScript `recordingStreams` helper."""

    label: str
    calls: list[str]
    captured: dict[str, Any] = field(default_factory=dict)
    fetch_deferred: Callable[..., AssistantMessageEventStream] | None = None
    cancel_deferred: Callable[..., Awaitable[None]] | None = None

    def _respond(
        self, model: Model, context: Context, options: StreamOptions | None = None, **_kwargs: Any
    ) -> AssistantMessageEventStream:
        self.calls.append(f"{self.label}:{model.id}")
        self.captured["options"] = options
        stream = AssistantMessageEventStream()
        message = faux_assistant_message("ok")
        stream.push(StartEvent(partial=message))
        stream.push(DoneEvent(reason="stop", message=message))
        stream.end(message)
        return stream

    def stream(
        self, model: Model, context: Context, options: StreamOptions | None = None, **kwargs: Any
    ) -> AssistantMessageEventStream:
        return self._respond(model, context, options, **kwargs)

    def stream_simple(
        self, model: Model, context: Context, options: SimpleStreamOptions | None = None, **kwargs: Any
    ) -> AssistantMessageEventStream:
        return self._respond(model, context, options, **kwargs)


def always_resolve() -> ApiKeyAuth:
    async def resolve(credential: Credential | None = None, env: Any = None) -> AuthResult:
        return AuthResult(auth=ResolvedAuth(), source="test")

    return ApiKeyAuth(name="Test", resolve=resolve)


async def test_exposes_only_declared_deferred_capabilities() -> None:
    # TS counterpart: "lazily exposes only declared deferred capabilities". The
    # *lazy* half (a `lazyApi(load, {fetchDeferred: true})` wrapper that defers a
    # dynamic `import()` until the first call, and counts loads) has no Python
    # analogue: `packages/ai/src/api/lazy.ts` exists to keep Node-only modules
    # out of TypeScript's browser/Bun bundles, and this port imports API modules
    # directly. What does carry over is the capability half: a provider must
    # expose `fetch_deferred`/`cancel_deferred` only when its API implements
    # them, and the exposed one must work.
    calls: list[str] = []
    streams = RecordingStreams("deferred", calls)
    streams.fetch_deferred = lambda model, handle, options=None, **kwargs: streams.stream_simple(model, make_context())
    provider = create_provider(
        id="mixed",
        name="mixed",
        auth=ProviderAuth(api_key=always_resolve()),
        models=[make_model("api-a", "model-a")],
        api=streams,
    )
    model = make_model("api-a", "model-a")
    handle = DeferredHandle(provider=model.provider, model_id=model.id, api=model.api, id="response-1")

    assert provider.cancel_deferred is None
    assert provider.fetch_deferred is not None
    result = await provider.fetch_deferred(model, handle).result()
    assert result.stop_reason == "stop"
    assert calls == ["deferred:model-a"]


async def test_dispatches_on_model_api_for_mixed_api_providers() -> None:
    calls: list[str] = []
    provider = create_provider(
        id="mixed",
        name="mixed",
        auth=ProviderAuth(api_key=always_resolve()),
        models=[make_model("api-a", "model-a"), make_model("api-b", "model-b")],
        api={"api-a": RecordingStreams("a", calls), "api-b": RecordingStreams("b", calls)},
    )
    models = Models()
    models.add(provider)

    await complete(await models.stream_simple(make_model("api-a", "model-a"), make_context()))
    await complete(await models.stream_simple(make_model("api-b", "model-b"), make_context()))
    assert calls == ["a:model-a", "b:model-b"]


async def test_merges_provider_resolved_env_into_stream_options() -> None:
    env_model = make_model("api-a", "model-a", provider="env-provider")

    async def resolve(credential: Credential | None = None, env: Any = None) -> AuthResult:
        return AuthResult(
            auth=ResolvedAuth(api_key="provider-key"),
            source="test",
            env={"PROVIDER_ONLY": "provider", "SHARED": "provider"},
        )

    streams = RecordingStreams("a", [])
    provider = create_provider(
        id="env-provider",
        name="env-provider",
        auth=ProviderAuth(api_key=ApiKeyAuth(name="Test", resolve=resolve)),
        models=[env_model],
        api=streams,
    )
    models = Models()
    models.add(provider)

    await complete(
        await models.stream_simple(
            env_model,
            make_context(),
            SimpleStreamOptions(api_key="request-key", env={"REQUEST_ONLY": "request", "SHARED": "request"}),
        )
    )

    options = streams.captured["options"]
    assert options.api_key == "request-key"
    assert options.env == {"PROVIDER_ONLY": "provider", "REQUEST_ONLY": "request", "SHARED": "request"}

    streams.captured.clear()
    await complete(
        await models.stream_simple(
            env_model, make_context(), SimpleStreamOptions(env={"REQUEST_ONLY": "request", "SHARED": "request"})
        )
    )
    options = streams.captured["options"]
    assert options.api_key == "provider-key"
    assert options.env == {"PROVIDER_ONLY": "provider", "REQUEST_ONLY": "request", "SHARED": "request"}


async def test_applies_resolved_request_options_to_deferred_fetch_and_cancellation() -> None:
    deferred_model = make_model("api-a", "model-a", provider="deferred-provider")
    fetched: dict[str, Any] = {}
    cancelled: dict[str, Any] = {}
    streams = RecordingStreams("deferred", [])

    def fetch_deferred(
        model: Model, handle: DeferredHandle, options: StreamOptions | None = None, **_kwargs: Any
    ) -> AssistantMessageEventStream:
        fetched["model"] = model
        fetched["options"] = options
        return streams.stream_simple(model, make_context())

    async def cancel_deferred(
        model: Model, handle: DeferredHandle, options: StreamOptions | None = None, **_kwargs: Any
    ) -> None:
        cancelled["options"] = options

    streams.fetch_deferred = fetch_deferred
    streams.cancel_deferred = cancel_deferred

    async def resolve(credential: Credential | None = None, env: Any = None) -> AuthResult:
        return AuthResult(
            auth=ResolvedAuth(
                api_key="provider-key",
                base_url="https://resolved.test/v1",
                headers={"Authorization": "Bearer provider", "X-Shared": "provider"},
            ),
            source="test",
            env={"PROVIDER_ONLY": "provider", "SHARED": "provider"},
        )

    provider = create_provider(
        id="deferred-provider",
        name="deferred-provider",
        auth=ProviderAuth(api_key=ApiKeyAuth(name="Test", resolve=resolve)),
        models=[deferred_model],
        api=streams,
    )
    models = Models()
    models.add(provider)
    handle = DeferredHandle(
        provider=deferred_model.provider, model_id=deferred_model.id, api=deferred_model.api, id="response-1"
    )

    await models.fetch_deferred(
        deferred_model,
        handle,
        StreamOptions(
            timeout_ms=100,
            api_key="request-key",
            headers={"X-Request": "request", "x-shared": "request"},
            env={"REQUEST_ONLY": "request", "SHARED": "request"},
            extra={"wait": 50},
        ),
    )
    await models.cancel_deferred(deferred_model, handle, StreamOptions(timeout_ms=200))

    # `transformHeaders` (a Models-only hook applied after the merge) has no
    # counterpart on `StreamOptions` in this port, so its two assertions are
    # dropped rather than faked.
    assert fetched["model"].base_url == "https://resolved.test/v1"
    fetch_options = fetched["options"]
    # TypeScript's `DeferredFetchOptions` has a dedicated `wait` field; this
    # port carries provider-specific request options in `extra`.
    assert fetch_options.extra == {"wait": 50}
    assert fetch_options.timeout_ms == 100
    assert fetch_options.api_key == "request-key"
    assert fetch_options.headers == {
        "Authorization": "Bearer provider",
        "X-Request": "request",
        "x-shared": "request",
    }
    assert fetch_options.env == {"PROVIDER_ONLY": "provider", "REQUEST_ONLY": "request", "SHARED": "request"}

    cancel_options = cancelled["options"]
    assert cancel_options.timeout_ms == 200
    assert cancel_options.api_key == "provider-key"
    assert cancel_options.headers == {"Authorization": "Bearer provider", "X-Shared": "provider"}
    assert cancel_options.env == {"PROVIDER_ONLY": "provider", "SHARED": "provider"}


async def test_produces_a_stream_error_for_a_model_whose_api_has_no_implementation() -> None:
    provider = create_provider(
        id="mixed",
        name="mixed",
        auth=ProviderAuth(api_key=always_resolve()),
        models=[make_model("api-a", "model-a")],
        api={"api-a": RecordingStreams("a", [])},
    )
    result = await complete(provider.stream_simple(make_model("api-ghost", "model-x"), make_context()))
    assert result.stop_reason == "error"
    assert result.error_message is not None
    assert "no API implementation" in result.error_message


# The next TypeScript case, "lets a newer dynamic refresh bypass and supersede
# older network work", has no Python counterpart. It drives
# `models.refresh({providers})` against an `InMemoryModelsStore` and asserts
# that a second in-flight refresh wins over a slower first one (both resolve
# with `aborted: false`, `provider.getModels()` and the store both end up with
# the *second* fetch's models, and the late-finishing first fetch does not
# overwrite them afterwards). `pi_ai.registry.Models` has no dynamic-catalog
# refresh and there is no `ModelsStore` in this port -- providers only expose
# their static `models` list -- so there is no subject to assert against.
# Faking one would pin invented behaviour rather than verify the port.


# --------------------------------------------------------------------------
# faux provider
# --------------------------------------------------------------------------


async def test_faux_streams_queued_responses_through_a_models_collection() -> None:
    faux = faux_provider()
    models = Models()
    models.add(faux.provider)
    faux.set_responses([faux_assistant_message("hello from faux")])

    model = models.get_models(faux.provider.id)[0]
    result = await complete(await models.stream_simple(model, make_context()))
    assert result.stop_reason == "stop"
    assert result.content == [TextContent(text="hello from faux")]
    assert faux.state.call_count == 1


async def test_faux_submits_polls_and_redeems_deferred_responses() -> None:
    faux = faux_provider(RegisterFauxProviderOptions(deferred=FauxDeferredOptions(pending_fetches=1, poll_after_ms=25)))
    models = Models()
    models.add(faux.provider)
    faux.set_responses([faux_assistant_message("ready")])
    model = faux.get_model(None)
    assert model is not None

    submission = await models.stream_simple(model, make_context(), SimpleStreamOptions(deferred={"window": "1h"}))
    event_types = [event.type async for event in submission]
    deferred = await submission.result()
    assert event_types == ["start", "done"]
    assert deferred.stop_reason == "deferred"
    assert deferred.content == []
    assert deferred.deferred is not None
    assert deferred.deferred.provider == model.provider
    assert deferred.deferred.model_id == model.id
    assert deferred.deferred.api == model.api
    assert isinstance(deferred.deferred.id, str)
    assert deferred.deferred.poll_after_ms == 25

    pending = await complete(await models.fetch_deferred(model, deferred.deferred))
    assert pending.stop_reason == "deferred"
    assert pending.deferred == deferred.deferred

    ready = await complete(await models.fetch_deferred(model, deferred.deferred, StreamOptions(extra={"wait": 0})))
    assert ready.stop_reason == "stop"
    assert ready.content == [TextContent(text="ready")]
    assert ready.usage.total_tokens > 0
    assert faux.state.call_count == 1
    assert faux.state.deferred_fetch_count == 2


async def test_faux_records_cancellation_and_returns_deferred_fetch_failures_in_band() -> None:
    faux = faux_provider()
    models = Models()
    models.add(faux.provider)

    def reject(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("deferred failed")

    faux.set_responses([reject, faux_assistant_message("cancelled")])
    model = faux.get_model(None)
    assert model is not None

    failed_submission = await complete(
        await models.stream_simple(model, make_context(), SimpleStreamOptions(deferred=True))
    )
    assert failed_submission.deferred is not None
    failed = await complete(await models.fetch_deferred(model, failed_submission.deferred))
    assert failed.stop_reason == "error"
    assert failed.error_message == "deferred failed"

    cancelled_submission = await complete(
        await models.stream_simple(model, make_context(), SimpleStreamOptions(deferred=True))
    )
    assert cancelled_submission.deferred is not None
    await models.cancel_deferred(model, cancelled_submission.deferred)
    assert faux.state.cancelled_deferred == [cancelled_submission.deferred]
    cancelled = await complete(await models.fetch_deferred(model, cancelled_submission.deferred))
    assert cancelled.stop_reason == "error"
    assert cancelled.error_message is not None
    assert "was cancelled" in cancelled.error_message
