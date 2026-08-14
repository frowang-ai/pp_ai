"""Python port of `packages/ai/test/telemetry-options.test.ts`."""

from __future__ import annotations

from typing import Any

from pi_ai.api.simple_options import build_base_options
from pi_ai.auth.types import ApiKeyAuth, AuthResult, ProviderAuth, ResolvedAuth
from pi_ai.image_models import ImagesModel
from pi_ai.images import generate_images
from pi_ai.images_api_registry import ImagesApiProvider, register_images_api_provider
from pi_ai.images_registry import create_images_models, create_images_provider
from pi_ai.registry import Models, create_provider
from pi_ai.types import (
    AssistantImages,
    AssistantMessage,
    Context,
    Cost,
    DeferredHandle,
    DoneEvent,
    ImagesContext,
    ImagesOptions,
    Model,
    ModelCost,
    SimpleStreamOptions,
    StreamOptions,
    TextContent,
    Usage,
)
from pi_ai.utils.event_stream import AssistantMessageEventStream
from pi_telemetry import NOOP_TELEMETRY_CONTEXT

TELEMETRY_CONTEXT = NOOP_TELEMETRY_CONTEXT
CONTEXT = Context(messages=[])
IMAGES_CONTEXT = ImagesContext(input=[TextContent(text="circle")])

MODEL = Model(
    id="model",
    name="Model",
    api="telemetry-test",
    provider="telemetry-provider",
    base_url="https://example.test",
    reasoning=False,
    input=["text"],
    cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    context_window=1000,
    max_tokens=100,
)

IMAGE_MODEL = ImagesModel(
    id="image-model",
    name="Image Model",
    api="telemetry-test-images",
    provider="telemetry-image-provider",
    base_url="https://example.test",
    input=["text"],
    output=["image"],
    cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
)


def _ambient_resolve(**_kwargs: object) -> AuthResult:
    """TS: `{ name: "Test", resolve: async () => ({ auth: {} }) }` -- reports
    configured with no auth values."""
    return AuthResult(auth=ResolvedAuth(), source="Ambient")


AMBIENT_AUTH = ProviderAuth(api_key=ApiKeyAuth(name="Test", resolve=_ambient_resolve))


def completed_stream(request_model: Model) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    message = AssistantMessage(
        content=[],
        api=request_model.api,
        provider=request_model.provider,
        model=request_model.id,
        usage=Usage(
            input=0,
            output=0,
            cache_read=0,
            cache_write=0,
            total_tokens=0,
            cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
        ),
        stop_reason="stop",
        timestamp=0,
    )
    stream.push(DoneEvent(reason="stop", message=message))
    stream.end(message)
    return stream


def test_is_inherited_by_every_request_option_surface_and_simple_stream_conversion() -> None:
    options = StreamOptions(telemetry_context=TELEMETRY_CONTEXT)
    assert options.telemetry_context is TELEMETRY_CONTEXT
    base = build_base_options(MODEL, CONTEXT, SimpleStreamOptions(telemetry_context=TELEMETRY_CONTEXT))
    assert base.telemetry_context is TELEMETRY_CONTEXT


async def test_survives_provider_and_models_stream_deferred_dispatch() -> None:
    observed: list[Any] = []
    handle = DeferredHandle(provider=MODEL.provider, model_id=MODEL.id, api=MODEL.api, id="response")

    class TelemetryApi:
        def stream(self, request_model, _context, options=None, **_kwargs):
            observed.append(options.telemetry_context if options else None)
            return completed_stream(request_model)

        def stream_simple(self, request_model, _context, options=None, **_kwargs):
            observed.append(options.telemetry_context if options else None)
            return completed_stream(request_model)

        def fetch_deferred(self, request_model, _handle, options=None, **_kwargs):
            observed.append(options.telemetry_context if options else None)
            return completed_stream(request_model)

        async def cancel_deferred(self, _request_model, _handle, options=None, **_kwargs):
            observed.append(options.telemetry_context if options else None)

    provider = create_provider(
        id=MODEL.provider,
        # TS's `createProvider` defaults `name` to `id`; the port requires it.
        name=MODEL.provider,
        auth=AMBIENT_AUTH,
        models=[MODEL],
        api=TelemetryApi(),
    )

    await provider.stream(MODEL, CONTEXT, StreamOptions(telemetry_context=TELEMETRY_CONTEXT)).result()
    await provider.stream_simple(MODEL, CONTEXT, SimpleStreamOptions(telemetry_context=TELEMETRY_CONTEXT)).result()
    fetch = provider.fetch_deferred
    assert fetch is not None
    await fetch(MODEL, handle, StreamOptions(telemetry_context=TELEMETRY_CONTEXT)).result()
    cancel = provider.cancel_deferred
    assert cancel is not None
    await cancel(MODEL, handle, StreamOptions(telemetry_context=TELEMETRY_CONTEXT))

    models = Models()
    models.add(provider)
    await (await models.stream(MODEL, CONTEXT, StreamOptions(telemetry_context=TELEMETRY_CONTEXT))).result()
    await (
        await models.stream_simple(MODEL, CONTEXT, SimpleStreamOptions(telemetry_context=TELEMETRY_CONTEXT))
    ).result()
    await (await models.fetch_deferred(MODEL, handle, StreamOptions(telemetry_context=TELEMETRY_CONTEXT))).result()
    await models.cancel_deferred(MODEL, handle, StreamOptions(telemetry_context=TELEMETRY_CONTEXT))

    assert len(observed) == 8
    assert all(value is TELEMETRY_CONTEXT for value in observed)


async def test_survives_direct_and_images_models_image_dispatch() -> None:
    observed: list[Any] = []

    async def direct_generate_images(request_model, _context, options=None):
        observed.append(options.telemetry_context if options else None)
        return AssistantImages(
            api=request_model.api,
            provider=request_model.provider,
            model=request_model.id,
            output=[],
            stop_reason="stop",
            timestamp=0,
        )

    register_images_api_provider(ImagesApiProvider(api=IMAGE_MODEL.api, generate_images=direct_generate_images))

    await generate_images(IMAGE_MODEL, IMAGES_CONTEXT, ImagesOptions(telemetry_context=TELEMETRY_CONTEXT))

    async def provider_generate_images(request_model, _context, options=None):
        observed.append(options.telemetry_context if options else None)
        return AssistantImages(
            api=request_model.api,
            provider=request_model.provider,
            model=request_model.id,
            output=[],
            stop_reason="stop",
            timestamp=0,
        )

    models = create_images_models()
    models.add(
        create_images_provider(
            id=IMAGE_MODEL.provider,
            name=IMAGE_MODEL.provider,
            auth=AMBIENT_AUTH,
            models=[IMAGE_MODEL],
            api=provider_generate_images,
        )
    )
    result = await models.generate_images(
        IMAGE_MODEL, IMAGES_CONTEXT, ImagesOptions(telemetry_context=TELEMETRY_CONTEXT)
    )
    assert result.stop_reason == "stop"

    assert observed == [TELEMETRY_CONTEXT, TELEMETRY_CONTEXT]
