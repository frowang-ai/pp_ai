"""Tests for the image-generation registry, catalog and api dispatch.

Covers the Python ports of `packages/ai/src/images-models.ts`,
`packages/ai/src/images-api-registry.ts`, `packages/ai/src/image-models.ts`,
`packages/ai/src/images.ts` and
`packages/ai/src/providers/images/register-builtins.ts`.

No network: every provider is either backed by the committed catalog or by a
local stub function.
"""

import json

import pytest
from pi_ai.auth.helpers import env_api_key_auth
from pi_ai.auth.types import Credential, InMemoryCredentialStore, ProviderAuth
from pi_ai.image_models import (
    IMAGES_DATA_DIR,
    get_image_model,
    get_image_models,
    get_image_providers,
    images_model_from_data,
    load_image_catalog,
)
from pi_ai.images import generate_images, resolve_images_api_provider
from pi_ai.images_api_registry import (
    ImagesApiProvider,
    get_images_api_provider,
    get_images_api_provider_source_id,
    register_images_api_provider,
)
from pi_ai.images_registry import ImagesModels, ImagesProvider, create_images_models, create_images_provider
from pi_ai.models import ModelsError
from pi_ai.providers import (
    OPENROUTER_IMAGES_MODELS,
    builtin_images_models,
    builtin_images_providers,
    openrouter_images_provider,
)
from pi_ai.types import AssistantImages, ImagesContext, ImagesModel, ImagesOptions, TextContent

CONTEXT = ImagesContext(input=[TextContent(text="a cat")])


def make_model(**overrides) -> ImagesModel:
    defaults = dict(
        id="stub/model",
        name="Stub Model",
        api="stub-images",
        provider="stub",
        base_url="https://stub.invalid/v1",
    )
    defaults.update(overrides)
    return ImagesModel(**defaults)


def make_generator(record: list[tuple[ImagesModel, ImagesOptions | None]] | None = None):
    async def generate(model, context, options=None):
        if record is not None:
            record.append((model, options))
        return AssistantImages(api=model.api, provider=model.provider, model=model.id, output=[])

    return generate


def make_auth(env_var: str = "STUB_IMAGES_KEY") -> ProviderAuth:
    return ProviderAuth(api_key=env_api_key_auth("Stub key", [env_var]))


def make_provider(provider_id: str = "stub", env_var: str = "STUB_IMAGES_KEY", **kwargs) -> ImagesProvider:
    return create_images_provider(
        id=provider_id,
        name="Stub",
        auth=make_auth(env_var),
        api=kwargs.pop("api", make_generator()),
        models=kwargs.pop("models", [make_model(provider=provider_id)]),
        **kwargs,
    )


# --------------------------------------------------------------------------
# the generated image catalog
# --------------------------------------------------------------------------


def test_the_committed_image_catalog_has_exactly_one_provider():
    assert get_image_providers() == ["openrouter"]


def test_every_catalog_model_is_a_well_formed_openrouter_image_model():
    models = get_image_models("openrouter")
    assert models
    for model in models:
        assert model.api == "openrouter-images"
        assert model.provider == "openrouter"
        assert model.base_url == "https://openrouter.ai/api/v1"
        assert "image" in model.output
        assert model.input
        assert set(model.input) <= {"text", "image"}
        assert model.name


def test_image_models_are_keyed_by_id():
    catalog = load_image_catalog("openrouter")
    for model_id, model in catalog.items():
        assert model.id == model_id


def test_get_image_model_finds_a_model_by_id():
    model_id = get_image_models("openrouter")[0].id
    assert get_image_model("openrouter", model_id).id == model_id


def test_get_image_model_returns_none_for_unknown_ids():
    assert get_image_model("openrouter", "does/not-exist") is None
    assert get_image_model("nope", "does/not-exist") is None


def test_an_unknown_provider_has_an_empty_catalog():
    assert get_image_models("nope") == []
    assert load_image_catalog("nope") == {}


def test_the_catalog_reads_from_a_custom_data_dir(tmp_path):
    (tmp_path / "demo.json").write_text(
        json.dumps(
            {
                "demo/one": {
                    "id": "demo/one",
                    "name": "Demo One",
                    "api": "demo-images",
                    "provider": "demo",
                    "baseUrl": "https://demo.invalid/v1",
                    "input": ["text"],
                    "output": ["image"],
                    "cost": {"input": 1, "output": 2, "cacheRead": 3, "cacheWrite": 4},
                }
            }
        ),
        encoding="utf-8",
    )
    assert get_image_providers(tmp_path) == ["demo"]
    model = get_image_model("demo", "demo/one", tmp_path)
    assert model.name == "Demo One"
    assert (model.cost.input, model.cost.output, model.cost.cache_read, model.cost.cache_write) == (1, 2, 3, 4)


def test_model_json_keys_keep_their_typescript_spelling():
    raw = json.loads((IMAGES_DATA_DIR / "openrouter.json").read_text(encoding="utf-8"))
    entry = next(iter(raw.values()))
    assert set(entry) == {"id", "name", "api", "provider", "baseUrl", "input", "output", "cost"}
    assert set(entry["cost"]) == {"input", "output", "cacheRead", "cacheWrite"}


def test_images_model_from_data_falls_back_to_sensible_defaults():
    model = images_model_from_data({"id": "bare/model"})
    assert model.name == "bare/model"
    assert model.api == "openrouter-images"
    assert model.input == ["text"]
    assert model.output == ["image"]
    assert model.cost.input == 0


# --------------------------------------------------------------------------
# the api registry
# --------------------------------------------------------------------------


def test_the_builtin_openrouter_images_api_is_registered():
    provider = get_images_api_provider("openrouter-images")
    assert provider is not None
    assert provider.api == "openrouter-images"


def test_an_unregistered_api_has_no_provider():
    assert get_images_api_provider("not-an-api") is None


async def test_a_registered_api_dispatches_to_its_function():
    calls: list[ImagesModel] = []

    async def generate(model, context, options=None):
        calls.append(model)
        return AssistantImages(api=model.api, provider=model.provider, model=model.id)

    register_images_api_provider(ImagesApiProvider(api="test-images", generate_images=generate))
    model = make_model(api="test-images")
    result = await generate_images(model, CONTEXT)
    assert result.model == model.id
    assert calls == [model]


async def test_a_model_from_another_api_is_rejected():
    register_images_api_provider(ImagesApiProvider(api="guard-images", generate_images=make_generator()))
    provider = get_images_api_provider("guard-images")
    with pytest.raises(ValueError, match="Mismatched api: other-images expected guard-images"):
        await provider.generate_images(make_model(api="other-images"), CONTEXT)


def test_registering_an_api_records_its_source_id():
    register_images_api_provider(
        ImagesApiProvider(api="sourced-images", generate_images=make_generator()), "extension-a"
    )
    assert get_images_api_provider_source_id("sourced-images") == "extension-a"
    register_images_api_provider(ImagesApiProvider(api="sourced-images", generate_images=make_generator()))
    assert get_images_api_provider_source_id("sourced-images") is None


def test_resolving_an_unregistered_api_raises():
    with pytest.raises(ValueError, match="No API provider registered for api: nope-images"):
        resolve_images_api_provider("nope-images")


# --------------------------------------------------------------------------
# provider construction
# --------------------------------------------------------------------------


def test_a_provider_lists_and_looks_up_its_models():
    provider = make_provider()
    assert [model.id for model in provider.get_models()] == ["stub/model"]
    assert provider.get_model("stub/model").id == "stub/model"
    assert provider.get_model("nope") is None


def test_a_provider_name_falls_back_to_its_id():
    provider = create_images_provider(id="bare", name="", auth=make_auth(), api=make_generator())
    assert provider.name == "bare"
    assert provider.get_models() == []


def test_get_models_returns_a_copy():
    provider = make_provider()
    provider.get_models().clear()
    assert len(provider.get_models()) == 1


# --------------------------------------------------------------------------
# the provider collection
# --------------------------------------------------------------------------


def test_providers_can_be_added_replaced_and_removed():
    models = ImagesModels()
    models.add(make_provider())
    assert [provider.id for provider in models.get_providers()] == ["stub"]

    models.add(make_provider(env_var="OTHER_KEY"))
    assert len(models.get_providers()) == 1
    assert models.get_provider("stub").auth.api_key.env_vars == ("OTHER_KEY",)

    models.remove("stub")
    assert models.get_providers() == []
    assert models.get_provider("stub") is None


def test_clear_removes_every_provider():
    models = create_images_models([make_provider(), make_provider("second")])
    models.clear()
    assert models.get_providers() == []


def test_get_models_spans_every_provider():
    models = create_images_models(
        [
            make_provider("a", models=[make_model(id="a/one", provider="a")]),
            make_provider("b", models=[make_model(id="b/one", provider="b")]),
        ]
    )
    assert {model.id for model in models.get_models()} == {"a/one", "b/one"}
    assert [model.id for model in models.get_models("a")] == ["a/one"]
    assert models.get_models("nope") == []


def test_get_model_looks_up_within_one_provider():
    models = create_images_models([make_provider()])
    assert models.get_model("stub", "stub/model").id == "stub/model"
    assert models.get_model("stub", "nope") is None
    assert models.get_model("nope", "stub/model") is None


def test_an_ill_behaved_provider_yields_no_models():
    class Broken(ImagesProvider):
        def get_models(self):
            raise RuntimeError("broken")

    broken = Broken(id="broken", name="Broken", auth=make_auth(), api=make_generator())
    models = create_images_models([broken, make_provider()])
    assert [model.id for model in models.get_models()] == ["stub/model"]
    assert models.get_models("broken") == []


# --------------------------------------------------------------------------
# refresh
# --------------------------------------------------------------------------


async def test_refresh_replaces_a_dynamic_providers_models():
    async def fetch():
        return [make_model(id="stub/fresh")]

    models = create_images_models([make_provider(models=[], refresh=fetch)])
    await models.refresh("stub")
    assert [model.id for model in models.get_models("stub")] == ["stub/fresh"]


async def test_refresh_is_a_no_op_for_static_and_unknown_providers():
    models = create_images_models([make_provider()])
    await models.refresh("stub")
    await models.refresh("nope")
    assert [model.id for model in models.get_models("stub")] == ["stub/model"]


async def test_a_targeted_refresh_failure_becomes_a_models_error():
    async def fetch():
        raise RuntimeError("network down")

    models = create_images_models([make_provider(models=[make_model()], refresh=fetch)])
    with pytest.raises(ModelsError) as excinfo:
        await models.refresh("stub")
    assert excinfo.value.code == "model_source"
    # The last-known list survives a failed refresh.
    assert [model.id for model in models.get_models("stub")] == ["stub/model"]


async def test_refreshing_everything_swallows_failures():
    async def boom():
        raise RuntimeError("network down")

    async def ok():
        return [make_model(id="b/fresh", provider="b")]

    models = create_images_models(
        [
            make_provider("a", models=[], refresh=boom),
            make_provider("b", models=[], refresh=ok),
        ]
    )
    await models.refresh()
    assert [model.id for model in models.get_models("b")] == ["b/fresh"]
    assert models.get_models("a") == []


# --------------------------------------------------------------------------
# auth resolution
# --------------------------------------------------------------------------


async def test_get_auth_resolves_from_the_environment(monkeypatch):
    monkeypatch.setenv("STUB_IMAGES_KEY", "env-key")
    result = await create_images_models([make_provider()]).get_auth("stub")
    assert result.auth.api_key == "env-key"
    assert result.source == "STUB_IMAGES_KEY"


async def test_get_auth_prefers_a_stored_credential(monkeypatch):
    monkeypatch.setenv("STUB_IMAGES_KEY", "env-key")
    store = InMemoryCredentialStore({"stub": Credential(key="stored-key")})
    models = create_images_models([make_provider()], credential_store=store)
    result = await models.get_auth("stub")
    assert result.auth.api_key == "stored-key"
    assert result.source == "stored credential"


async def test_get_auth_accepts_a_model(monkeypatch):
    monkeypatch.setenv("STUB_IMAGES_KEY", "env-key")
    provider = make_provider()
    result = await create_images_models([provider]).get_auth(provider.get_models()[0])
    assert result.auth.api_key == "env-key"


async def test_get_auth_returns_none_when_unknown_or_unconfigured(monkeypatch):
    monkeypatch.delenv("STUB_IMAGES_KEY", raising=False)
    models = create_images_models([make_provider()])
    assert await models.get_auth("stub") is None
    assert await models.get_auth("nope") is None


async def test_get_auth_wraps_resolution_failures_in_models_error():
    def boom(**_kwargs):
        raise RuntimeError("store exploded")

    provider = make_provider()
    provider.auth = ProviderAuth(api_key=type(provider.auth.api_key)(name="k", resolve=boom))
    with pytest.raises(ModelsError) as excinfo:
        await create_images_models([provider]).get_auth("stub")
    assert excinfo.value.code == "auth"


async def test_login_stores_a_credential_and_logout_removes_it():
    models = create_images_models([make_provider()])
    await models.login("stub", "typed-key")
    assert (await models.get_auth("stub")).auth.api_key == "typed-key"
    await models.logout("stub")
    assert await models.get_auth("stub") is None


async def test_login_rejects_an_unknown_provider():
    with pytest.raises(ModelsError) as excinfo:
        await create_images_models([]).login("nope", "k")
    assert excinfo.value.code == "provider"


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------


async def test_generation_passes_the_resolved_api_key_through(monkeypatch):
    monkeypatch.setenv("STUB_IMAGES_KEY", "env-key")
    calls: list[tuple[ImagesModel, ImagesOptions | None]] = []
    models = create_images_models([make_provider(api=make_generator(calls))])
    result = await models.generate_images(models.get_models()[0], CONTEXT)
    assert result.stop_reason == "stop"
    assert calls[0][1].api_key == "env-key"


async def test_explicit_options_win_over_resolved_auth(monkeypatch):
    monkeypatch.setenv("STUB_IMAGES_KEY", "env-key")
    calls: list[tuple[ImagesModel, ImagesOptions | None]] = []
    models = create_images_models([make_provider(api=make_generator(calls))])
    await models.generate_images(models.get_models()[0], CONTEXT, ImagesOptions(api_key="explicit"))
    assert calls[0][1].api_key == "explicit"


async def test_generation_runs_unauthenticated_when_the_provider_is_unconfigured(monkeypatch):
    monkeypatch.delenv("STUB_IMAGES_KEY", raising=False)
    calls: list[tuple[ImagesModel, ImagesOptions | None]] = []
    models = create_images_models([make_provider(api=make_generator(calls))])
    result = await models.generate_images(models.get_models()[0], CONTEXT)
    assert result.stop_reason == "stop"
    assert calls[0][1].api_key is None


async def test_an_unknown_provider_comes_back_as_an_error(monkeypatch):
    models = create_images_models([make_provider()])
    result = await models.generate_images(make_model(provider="nope"), CONTEXT)
    assert result.stop_reason == "error"
    assert result.error_message == "Unknown provider: nope"
    assert result.provider == "nope"
    assert result.output == []


async def test_a_failing_provider_comes_back_as_an_error_instead_of_raising(monkeypatch):
    monkeypatch.setenv("STUB_IMAGES_KEY", "env-key")

    async def boom(model, context, options=None):
        raise RuntimeError("provider exploded")

    models = create_images_models([make_provider(api=boom)])
    result = await models.generate_images(models.get_models()[0], CONTEXT)
    assert result.stop_reason == "error"
    assert result.error_message == "provider exploded"
    assert result.model == "stub/model"


async def test_an_auth_failure_comes_back_as_an_error_instead_of_raising():
    def boom(**_kwargs):
        raise RuntimeError("store exploded")

    provider = make_provider()
    provider.auth = ProviderAuth(api_key=type(provider.auth.api_key)(name="k", resolve=boom))
    models = create_images_models([provider])
    result = await models.generate_images(provider.get_models()[0], CONTEXT)
    assert result.stop_reason == "error"
    assert "store exploded" in result.error_message


# --------------------------------------------------------------------------
# the built-in image providers
# --------------------------------------------------------------------------


def test_the_builtin_image_provider_set_is_openrouter():
    providers = builtin_images_providers()
    assert [provider.id for provider in providers] == ["openrouter"]
    assert providers[0].name == "OpenRouter"


def test_the_builtin_image_provider_carries_the_generated_catalog():
    provider = openrouter_images_provider()
    assert len(provider.get_models()) == len(OPENROUTER_IMAGES_MODELS)
    assert provider.get_models() == get_image_models("openrouter")


def test_the_builtin_image_provider_supports_an_api_key_and_oauth():
    auth = openrouter_images_provider().auth
    assert auth.api_key.env_vars == ("OPENROUTER_API_KEY",)
    assert auth.oauth is not None


def test_builtin_images_models_registers_every_builtin_provider():
    models = builtin_images_models()
    assert [provider.id for provider in models.get_providers()] == ["openrouter"]
    assert len(models.get_models()) == len(OPENROUTER_IMAGES_MODELS)


def test_builtin_images_models_accepts_a_credential_store():
    store = InMemoryCredentialStore({"openrouter": Credential(key="stored-key")})
    assert builtin_images_models(credential_store=store).credentials is store


async def test_every_builtin_image_model_resolves_through_its_api():
    models = builtin_images_models()
    for model in models.get_models():
        provider = resolve_images_api_provider(model.api)
        assert provider.api == model.api
