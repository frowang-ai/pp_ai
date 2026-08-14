"""Python port of `packages/ai/test/images-models.test.ts`."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest
from pi_ai.auth.types import ApiKeyAuth, AuthResult, Credential, EnvLookup, ProviderAuth, ResolvedAuth
from pi_ai.images_registry import ImagesProvider, create_images_models, create_images_provider
from pi_ai.models import ModelsError
from pi_ai.providers.all import builtin_images_models
from pi_ai.types import (
    AssistantImages,
    ImageContent,
    ImagesContext,
    ImagesModel,
    ImagesOptions,
    ModelCost,
    TextContent,
    now_ms,
)


def fake_auth_context(env: dict[str, str]) -> EnvLookup:
    async def lookup(name: str) -> str | None:
        return env.get(name)

    return lookup


def make_test_image_model(provider: str, model_id: str) -> ImagesModel:
    return ImagesModel(
        id=model_id,
        name=model_id,
        api="test-images",
        provider=provider,
        base_url="https://example.test/v1",
        input=["text"],
        output=["image"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    )


def ok_result(model: ImagesModel) -> AssistantImages:
    return AssistantImages(
        api=model.api,
        provider=model.provider,
        model=model.id,
        output=[ImageContent(data="aGk=", mime_type="image/png")],
        stop_reason="stop",
        timestamp=now_ms(),
    )


@dataclass
class GenerateCall:
    model: ImagesModel
    options: ImagesOptions | None


@dataclass
class ProviderSpec:
    id: str
    models: list[ImagesModel] | None = None
    env_var: str | None = None
    calls: list[GenerateCall] = field(default_factory=list)


def build_test_provider(spec: ProviderSpec) -> ImagesProvider:
    async def resolve(credential: Credential | None = None, env: EnvLookup | None = None) -> AuthResult | None:
        if not spec.env_var:
            return AuthResult(auth=ResolvedAuth(), source="test")
        key = credential.key if credential is not None and credential.key else None
        if key is None and env is not None:
            key = await env(spec.env_var)
        if not key:
            return None
        return AuthResult(auth=ResolvedAuth(api_key=key), source="stored" if credential else spec.env_var)

    async def generate_images(
        model: ImagesModel, _context: ImagesContext, options: ImagesOptions | None = None
    ) -> AssistantImages:
        spec.calls.append(GenerateCall(model=model, options=options))
        return ok_result(model)

    return create_images_provider(
        id=spec.id,
        name=spec.id,
        auth=ProviderAuth(api_key=ApiKeyAuth(name="Test key", resolve=resolve)),
        models=spec.models if spec.models is not None else [make_test_image_model(spec.id, "model-a")],
        api=generate_images,
    )


CONTEXT = ImagesContext(input=[TextContent(text="a red circle")])


def test_registers_providers_and_reads_models_synchronously():
    models = create_images_models()
    models.add(
        build_test_provider(
            ProviderSpec(id="p1", models=[make_test_image_model("p1", "m1"), make_test_image_model("p1", "m2")])
        )
    )
    models.add(build_test_provider(ProviderSpec(id="p2", models=[make_test_image_model("p2", "m3")])))

    assert [provider.id for provider in models.get_providers()] == ["p1", "p2"]
    assert [model.id for model in models.get_models()] == ["m1", "m2", "m3"]
    assert [model.id for model in models.get_models("p1")] == ["m1", "m2"]
    assert models.get_model("p2", "m3").id == "m3"
    assert models.get_model("p2", "missing") is None

    models.remove("p1")
    assert models.get_provider("p1") is None


async def test_resolves_auth_and_merges_it_into_requests_explicit_options_win():
    spec = ProviderSpec(id="p1", env_var="TEST_KEY")
    models = create_images_models(env=fake_auth_context({"TEST_KEY": "env-key"}))
    models.add(build_test_provider(spec))
    model = models.get_model("p1", "model-a")

    assert (await models.get_auth(model)).auth.api_key == "env-key"
    assert (await models.get_auth(model.provider)).auth.api_key == "env-key"
    assert (await models.get_auth(model, api_key_override="explicit-key")).auth.api_key == "explicit-key"

    result = await models.generate_images(model, CONTEXT)
    assert result.stop_reason == "stop"
    assert spec.calls[0].options.api_key == "env-key"

    await models.generate_images(model, CONTEXT, ImagesOptions(api_key="explicit"))
    assert spec.calls[1].options.api_key == "explicit"


async def test_merges_provider_resolved_env_into_image_options():
    calls: list[GenerateCall] = []

    async def resolve(credential: Credential | None = None, env: EnvLookup | None = None) -> AuthResult | None:
        return AuthResult(
            auth=ResolvedAuth(api_key="provider-key"),
            source="test",
            env={"PROVIDER_ONLY": "provider", "SHARED": "provider"},
        )

    async def generate_images(
        model: ImagesModel, _context: ImagesContext, options: ImagesOptions | None = None
    ) -> AssistantImages:
        calls.append(GenerateCall(model=model, options=options))
        return ok_result(model)

    models = create_images_models()
    models.add(
        create_images_provider(
            id="p1",
            name="p1",
            auth=ProviderAuth(api_key=ApiKeyAuth(name="Test key", resolve=resolve)),
            models=[make_test_image_model("p1", "model-a")],
            api=generate_images,
        )
    )
    model = models.get_model("p1", "model-a")

    await models.generate_images(
        model,
        CONTEXT,
        ImagesOptions(api_key="request-key", env={"REQUEST_ONLY": "request", "SHARED": "request"}),
    )

    assert calls[0].options.api_key == "request-key"
    assert calls[0].options.env == {
        "PROVIDER_ONLY": "provider",
        "REQUEST_ONLY": "request",
        "SHARED": "request",
    }


async def test_returns_error_result_for_unknown_providers_and_unconfigured_auth():
    models = create_images_models(env=fake_auth_context({}))
    ghost = await models.generate_images(make_test_image_model("ghost", "m"), CONTEXT)
    assert ghost.stop_reason == "error"
    assert "Unknown provider: ghost" in (ghost.error_message or "")

    # unconfigured (resolve -> None) still dispatches; provider decides what to do
    spec = ProviderSpec(id="p1", env_var="MISSING")
    models.add(build_test_provider(spec))
    model = models.get_model("p1", "model-a")
    assert await models.get_auth(model) is None
    await models.generate_images(model, CONTEXT)
    assert spec.calls[0].options.api_key is None


async def test_supports_dynamic_providers_via_refresh_with_in_flight_dedupe():
    fetches = 0
    # TypeScript uses `setTimeout(resolve, 5)` to hold the fetch open. A real
    # sleep is not needed and is load-sensitive: an event the test releases
    # itself guarantees both refreshes are in flight before either completes,
    # which is the condition this case is actually about.
    release = asyncio.Event()

    async def resolve_empty(credential: Credential | None = None, env: EnvLookup | None = None) -> AuthResult | None:
        return AuthResult(auth=ResolvedAuth(), source="test")

    async def refresh_models() -> list[ImagesModel]:
        nonlocal fetches
        fetches += 1
        await release.wait()
        return [make_test_image_model("dyn", "listed")]

    async def generate_images(
        model: ImagesModel, _context: ImagesContext, options: ImagesOptions | None = None
    ) -> AssistantImages:
        return ok_result(model)

    provider = create_images_provider(
        id="dyn",
        name="dyn",
        auth=ProviderAuth(api_key=ApiKeyAuth(name="Test", resolve=resolve_empty)),
        models=[],
        refresh=refresh_models,
        api=generate_images,
    )
    models = create_images_models()
    models.add(provider)

    assert models.get_models("dyn") == []
    first = asyncio.ensure_future(models.refresh("dyn"))
    second = asyncio.ensure_future(models.refresh("dyn"))
    # Not a timing sleep: one loop turn runs both scheduled tasks up to their
    # first suspension point, which is where each registers itself against the
    # shared in-flight fetch. Only then is it safe to let the fetch finish.
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(first, second)
    assert fetches == 1
    assert models.get_model("dyn", "listed") is not None

    async def failing_refresh() -> list[ImagesModel]:
        raise RuntimeError("fetch failed")

    models.add(
        create_images_provider(
            id="flaky",
            name="flaky",
            auth=ProviderAuth(api_key=ApiKeyAuth(name="Test", resolve=resolve_empty)),
            models=[],
            refresh=failing_refresh,
            api=generate_images,
        )
    )
    with pytest.raises(ModelsError) as excinfo:
        await models.refresh("flaky")
    assert excinfo.value.code == "model_source"
    assert await models.refresh() is None


async def test_builtin_images_models_registers_the_openrouter_provider():
    models = builtin_images_models(env=fake_auth_context({"OPENROUTER_API_KEY": "or-key"}))
    assert [provider.id for provider in models.get_providers()] == ["openrouter"]

    listed = models.get_models("openrouter")
    assert len(listed) > 0
    assert all(model.api == "openrouter-images" for model in listed)

    assert (await models.get_auth(listed[0])).auth.api_key == "or-key"
