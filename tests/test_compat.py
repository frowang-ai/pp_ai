"""Test ported from packages/ai/test/compat-env.test.ts, plus coverage for
the global api-provider registry in `pi_ai.compat` that the TypeScript source
also exercises indirectly through `registerFauxProvider`."""

from __future__ import annotations

import dataclasses

import pytest
from pi_ai import AssistantMessage, Context, Cost, Model, StartEvent, Usage, UserMessage
from pi_ai.compat import (
    complete,
    get_api_provider,
    get_api_providers,
    register_api_provider,
    register_faux_provider,
    reset_api_providers,
    stream,
)
from pi_ai.types import DoneEvent, StreamOptions
from pi_ai.utils.event_stream import AssistantMessageEventStream


@pytest.fixture(autouse=True)
def _reset_registry():
    yield
    reset_api_providers()


def make_model(**overrides) -> Model:
    defaults = dict(
        id="test-model",
        name="Test Model",
        api="openai-responses",
        provider="custom-openai",
        base_url="https://example.test/v1",
        reasoning=False,
        input=["text"],
        context_window=128_000,
        max_tokens=4096,
    )
    defaults.update(overrides)
    return Model(**defaults)


def make_message(model: Model) -> AssistantMessage:
    return AssistantMessage(
        api=model.api,
        provider=model.provider,
        model=model.id,
        content=[],
        usage=Usage(cost=Cost()),
        stop_reason="stop",
    )


async def test_dispatches_unknown_providers_through_the_legacy_api_registry() -> None:
    model = make_model()
    context = Context(messages=[UserMessage(content="hi")])
    captured: dict = {}

    def stream_fn(_model, _context, options=None):
        captured["api_key"] = options.api_key if options else None
        out = AssistantMessageEventStream()
        output = make_message(model)
        out.push(StartEvent(partial=output))
        out.push(DoneEvent(reason="stop", message=output))
        out.end(output)
        return out

    register_api_provider(model.api, stream_fn, stream_fn)

    await complete(model, context, StreamOptions(api_key="request-key"))

    assert captured["api_key"] == "request-key"


async def test_raises_for_an_unknown_api() -> None:
    model = make_model()
    model = dataclasses.replace(model, api="totally-unknown-api")
    context = Context(messages=[UserMessage(content="hi")])

    with pytest.raises(RuntimeError, match="No API provider registered for api: totally-unknown-api"):
        stream(model, context)


async def test_register_api_provider_rejects_a_mismatched_model_api() -> None:
    model = make_model(api="custom-mismatch-api")
    context = Context(messages=[UserMessage(content="hi")])

    def stream_fn(_model, _context, options=None):
        out = AssistantMessageEventStream()
        output = make_message(model)
        out.end(output)
        return out

    register_api_provider(model.api, stream_fn, stream_fn)
    wrong_model = dataclasses.replace(model, api="anthropic-messages")

    provider = get_api_provider(model.api)
    with pytest.raises(RuntimeError, match="Mismatched api"):
        provider.stream(wrong_model, context)


def test_register_builtin_api_providers_registers_every_ported_api() -> None:
    reset_api_providers()
    apis = {provider.api for provider in get_api_providers()}
    assert apis == {
        "anthropic-messages",
        "openai-completions",
        "openai-responses",
        "azure-openai-responses",
        "google-generative-ai",
        "google-vertex",
        "mistral-conversations",
        "pi-messages",
    }


def test_unregister_api_providers_only_removes_its_own_source() -> None:
    reset_api_providers()
    register_api_provider("custom-api", lambda *a, **k: None, lambda *a, **k: None, source_id="source-a")
    register_api_provider("custom-api-2", lambda *a, **k: None, lambda *a, **k: None, source_id="source-b")

    from pi_ai.compat import unregister_api_providers

    unregister_api_providers("source-a")

    assert get_api_provider("custom-api") is None
    assert get_api_provider("custom-api-2") is not None


async def test_register_faux_provider_dispatches_through_the_global_registry() -> None:
    from pi_ai.providers.faux import faux_assistant_message

    registration = register_faux_provider()
    registration.set_responses([faux_assistant_message("hi from faux")])

    context = Context(messages=[UserMessage(content="hello")])
    model = next(m for m in registration.models if m.id == registration.get_model().id)
    response = await complete(model, context)

    assert response.content[0].text == "hi from faux"
    assert registration.state.call_count == 1


async def test_register_faux_provider_unregister_removes_it_from_the_registry() -> None:
    from pi_ai.providers.faux import faux_assistant_message

    registration = register_faux_provider()
    registration.set_responses([faux_assistant_message("hello")])
    registration.unregister()

    context = Context(messages=[UserMessage(content="hi")])
    with pytest.raises(RuntimeError, match=f"No API provider registered for api: {registration.api}"):
        await complete(registration.get_model(), context)
