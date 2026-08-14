import pytest
from pi_ai.auth.helpers import env_api_key_auth, resolve_api_key_auth
from pi_ai.auth.types import (
    ApiKeyAuth,
    AuthResult,
    Credential,
    InMemoryCredentialStore,
    ProviderAuth,
    ResolvedAuth,
)
from pi_ai.models import ModelsError
from pi_ai.registry import Models, create_provider
from pi_ai.types import Context, Model, ModelCost


class FakeApi:
    """Records the arguments the registry forwards to a provider API module."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def stream(self, model, context, options=None, **kwargs):
        self.calls.append({"kind": "stream", "model": model, "context": context, "options": options})
        return "stream-result"

    def stream_simple(self, model, context, options=None, **kwargs):
        self.calls.append({"kind": "stream_simple", "model": model, "context": context, "options": options})
        return "stream-simple-result"


def make_provider(provider_id: str = "fake", env_var: str = "FAKE_API_KEY", api=None):
    model = Model(
        id="m1",
        name="Model One",
        api="openai-completions",
        provider="",
        base_url="",
        context_window=1000,
        max_tokens=100,
        cost=ModelCost(input=1.0, output=2.0),
    )
    return create_provider(
        id=provider_id,
        name=provider_id.title(),
        auth=ProviderAuth(api_key=env_api_key_auth(f"{provider_id} key", [env_var])),
        api=api or FakeApi(),
        models=[model],
        base_url=f"https://{provider_id}.invalid/v1",
    )


# --------------------------------------------------------------------------
# create_provider
# --------------------------------------------------------------------------


def test_create_provider_stamps_provider_and_base_url_onto_models():
    provider = make_provider()
    model = provider.get_models()[0]
    assert model.provider == "fake"
    assert model.base_url == "https://fake.invalid/v1"


def test_create_provider_keeps_explicit_model_provider_and_base_url():
    model = Model(id="m", provider="explicit", base_url="https://explicit.invalid")
    provider = create_provider(
        id="fake",
        name="Fake",
        auth=ProviderAuth(api_key=env_api_key_auth("k", ["X"])),
        api=FakeApi(),
        models=[model],
        base_url="https://fake.invalid/v1",
    )
    stamped = provider.get_models()[0]
    assert stamped.provider == "explicit"
    assert stamped.base_url == "https://explicit.invalid"


def test_create_provider_does_not_mutate_the_input_models():
    model = Model(id="m")
    create_provider(
        id="fake",
        name="Fake",
        auth=ProviderAuth(api_key=env_api_key_auth("k", ["X"])),
        api=FakeApi(),
        models=[model],
        base_url="https://fake.invalid/v1",
    )
    assert model.provider == ""
    assert model.base_url == ""


def test_provider_get_model_by_id():
    provider = make_provider()
    assert provider.get_model("m1").id == "m1"
    assert provider.get_model("missing") is None


# --------------------------------------------------------------------------
# Models registry lookups
# --------------------------------------------------------------------------


def test_registry_lists_providers_and_models():
    models = Models([make_provider("a", "A_KEY"), make_provider("b", "B_KEY")])
    assert [p.id for p in models.get_providers()] == ["a", "b"]
    assert len(models.get_models()) == 2
    assert [m.provider for m in models.get_models("a")] == ["a"]


def test_registry_get_models_for_unknown_provider_is_empty():
    assert Models([]).get_models("nope") == []


def test_registry_get_model():
    models = Models([make_provider("a", "A_KEY")])
    assert models.get_model("a", "m1").id == "m1"
    assert models.get_model("a", "nope") is None
    assert models.get_model("nope", "m1") is None


def test_find_model_by_qualified_reference():
    models = Models([make_provider("a", "A_KEY"), make_provider("b", "B_KEY")])
    found = models.find_model("b/m1")
    assert found is not None
    assert found.provider == "b"


def test_find_model_by_bare_id_uses_the_first_provider():
    models = Models([make_provider("a", "A_KEY"), make_provider("b", "B_KEY")])
    assert models.find_model("m1").provider == "a"


def test_find_model_returns_none_for_unknown_reference():
    assert Models([make_provider()]).find_model("nope/nope") is None


def test_registry_add_replaces_a_provider_with_the_same_id():
    models = Models([make_provider("a", "A_KEY")])
    models.add(make_provider("a", "OTHER_KEY"))
    assert len(models.get_providers()) == 1
    assert models.get_provider("a").auth.api_key.env_vars == ("OTHER_KEY",)


# --------------------------------------------------------------------------
# auth resolution
# --------------------------------------------------------------------------


async def test_get_auth_resolves_from_the_environment(monkeypatch):
    monkeypatch.setenv("FAKE_API_KEY", "env-key")
    models = Models([make_provider()])
    result = await models.get_auth("fake")
    assert result.auth.api_key == "env-key"
    assert result.source == "FAKE_API_KEY"


async def test_get_auth_prefers_a_stored_credential(monkeypatch):
    monkeypatch.setenv("FAKE_API_KEY", "env-key")
    store = InMemoryCredentialStore({"fake": Credential(key="stored-key")})
    models = Models([make_provider()], credential_store=store)
    result = await models.get_auth("fake")
    assert result.auth.api_key == "stored-key"
    assert result.source == "stored credential"


async def test_get_auth_returns_none_when_unconfigured(monkeypatch):
    monkeypatch.delenv("FAKE_API_KEY", raising=False)
    assert await Models([make_provider()]).get_auth("fake") is None


async def test_get_auth_returns_none_for_unknown_provider():
    assert await Models([]).get_auth("nope") is None


async def test_get_auth_accepts_a_model_and_merges_model_headers(monkeypatch):
    monkeypatch.setenv("FAKE_API_KEY", "env-key")
    provider = make_provider()
    provider.models[0].headers = {"x-model": "1"}
    models = Models([provider])
    result = await models.get_auth(provider.models[0])
    assert result.auth.headers == {"x-model": "1"}


async def test_get_auth_wraps_resolution_failures_in_models_error():
    def boom(**_kwargs):
        raise RuntimeError("store exploded")

    provider = make_provider()
    provider.auth = ProviderAuth(api_key=ApiKeyAuth(name="k", resolve=boom))
    models = Models([provider])

    with pytest.raises(ModelsError) as excinfo:
        await models.get_auth("fake")
    assert excinfo.value.code == "auth"


async def test_check_auth_reports_configuration(monkeypatch):
    models = Models([make_provider()])
    monkeypatch.delenv("FAKE_API_KEY", raising=False)
    assert (await models.check_auth("fake")).configured is False
    monkeypatch.setenv("FAKE_API_KEY", "k")
    check = await models.check_auth("fake")
    assert check.configured is True
    assert check.source == "FAKE_API_KEY"


async def test_check_auth_returns_none_for_unknown_provider():
    assert await Models([]).check_auth("nope") is None


async def test_get_available_only_returns_configured_providers(monkeypatch):
    monkeypatch.setenv("A_KEY", "k")
    monkeypatch.delenv("B_KEY", raising=False)
    models = Models([make_provider("a", "A_KEY"), make_provider("b", "B_KEY")])
    available = await models.get_available()
    assert [m.provider for m in available] == ["a"]


async def test_login_and_logout_round_trip(monkeypatch):
    monkeypatch.delenv("FAKE_API_KEY", raising=False)
    models = Models([make_provider()])

    await models.login("fake", "typed-key")
    assert (await models.get_auth("fake")).auth.api_key == "typed-key"

    await models.logout("fake")
    assert await models.get_auth("fake") is None


async def test_login_rejects_an_unknown_provider():
    with pytest.raises(ModelsError) as excinfo:
        await Models([]).login("nope", "k")
    assert excinfo.value.code == "provider"


async def test_custom_env_lookup_is_used(monkeypatch):
    monkeypatch.delenv("FAKE_API_KEY", raising=False)
    models = Models([make_provider()], env=lambda name: "scoped" if name == "FAKE_API_KEY" else None)
    assert (await models.get_auth("fake")).auth.api_key == "scoped"


async def test_resolve_api_key_auth_supports_an_async_custom_resolver():
    async def resolver(credential=None, env=None):
        return AuthResult(auth=ResolvedAuth(api_key="async-key"), source="custom")

    result = await resolve_api_key_auth(ApiKeyAuth(name="k", resolve=resolver))
    assert result.auth.api_key == "async-key"


# --------------------------------------------------------------------------
# stream delegation
# --------------------------------------------------------------------------


async def test_stream_simple_injects_the_resolved_api_key(monkeypatch):
    monkeypatch.setenv("FAKE_API_KEY", "env-key")
    api = FakeApi()
    models = Models([make_provider(api=api)])
    model = models.get_models()[0]

    result = await models.stream_simple(model, Context(messages=[]))

    assert result == "stream-simple-result"
    assert api.calls[0]["options"].api_key == "env-key"


async def test_stream_simple_keeps_an_explicit_api_key(monkeypatch):
    from pi_ai.types import SimpleStreamOptions

    monkeypatch.setenv("FAKE_API_KEY", "env-key")
    api = FakeApi()
    models = Models([make_provider(api=api)])
    model = models.get_models()[0]

    await models.stream_simple(model, Context(messages=[]), SimpleStreamOptions(api_key="explicit"))

    assert api.calls[0]["options"].api_key == "explicit"


async def test_stream_simple_reports_an_unconfigured_provider_in_band(monkeypatch):
    # Setup failures are stream errors, not raises: `packages/ai/test/models-runtime.test.ts`
    # ("produces an error stream for unknown providers instead of throwing") pins
    # this, and the images registry port already behaved this way.
    monkeypatch.delenv("FAKE_API_KEY", raising=False)
    models = Models([make_provider()])
    model = models.get_models()[0]

    result = await (await models.stream_simple(model, Context(messages=[]))).result()
    assert result.stop_reason == "error"
    assert "not configured" in (result.error_message or "")


async def test_stream_simple_reports_an_unknown_provider_in_band():
    models = Models([])
    result = await (await models.stream_simple(Model(id="m", provider="ghost"), Context(messages=[]))).result()
    assert result.stop_reason == "error"
    assert "Unknown provider: ghost" in (result.error_message or "")


def test_provider_stream_delegates_to_the_api_module():
    api = FakeApi()
    provider = make_provider(api=api)
    provider.stream(provider.models[0], Context(messages=[]))
    assert api.calls[0]["kind"] == "stream"


# --------------------------------------------------------------------------
# built-in providers
# --------------------------------------------------------------------------


def test_built_in_providers_have_distinct_ids_and_models():
    from pi_ai.providers import all_providers

    providers = all_providers()
    ids = [p.id for p in providers]
    assert len(ids) == len(set(ids))
    for provider in providers:
        if provider.id == "radius":
            # Radius has no static catalog; its models come from the gateway.
            continue
        assert provider.get_models(), f"{provider.id} has no models"
        for model in provider.get_models():
            # Azure OpenAI has no fixed endpoint: the resource host is supplied
            # per deployment at request time, so its catalog carries no base URL.
            if provider.id != "azure-openai-responses":
                assert model.base_url.startswith("https://")
            assert model.provider == provider.id


def test_openai_compatible_provider_builds_a_usable_provider():
    from pi_ai.providers import openai_compatible_provider

    provider = openai_compatible_provider(
        "local", "Local", "http://127.0.0.1:8080/v1", ["LOCAL_KEY"], [Model(id="tiny")]
    )
    assert provider.id == "local"
    assert provider.get_models()[0].base_url == "http://127.0.0.1:8080/v1"
    assert provider.auth.api_key.env_vars == ("LOCAL_KEY",)


# --------------------------------------------------------------------------
# regressions found by review against the TypeScript source
# --------------------------------------------------------------------------


async def test_model_headers_override_auth_headers(monkeypatch):
    """A model may deliberately override an auth-supplied header."""

    async def resolver(credential=None, env=None):
        return AuthResult(
            auth=ResolvedAuth(api_key="k", headers={"authorization": "Bearer auth", "x-shared": "auth"}),
            source="custom",
        )

    provider = make_provider()
    provider.auth = ProviderAuth(api_key=ApiKeyAuth(name="k", resolve=resolver))
    provider.models[0].headers = {"authorization": "Bearer model", "x-model-only": "1"}
    models = Models([provider])

    result = await models.get_auth(provider.models[0])

    assert result.auth.headers["authorization"] == "Bearer model"
    assert result.auth.headers["x-shared"] == "auth"
    assert result.auth.headers["x-model-only"] == "1"


async def test_model_headers_replace_auth_headers_case_insensitively(monkeypatch):
    async def resolver(credential=None, env=None):
        return AuthResult(auth=ResolvedAuth(api_key="k", headers={"authorization": "Bearer auth"}), source="c")

    provider = make_provider()
    provider.auth = ProviderAuth(api_key=ApiKeyAuth(name="k", resolve=resolver))
    provider.models[0].headers = {"Authorization": "Bearer model"}
    models = Models([provider])

    result = await models.get_auth(provider.models[0])

    # Exactly one authorization header survives, with the model's value.
    keys = [key for key in result.auth.headers if key.lower() == "authorization"]
    assert keys == ["Authorization"]
    assert result.auth.headers["Authorization"] == "Bearer model"
