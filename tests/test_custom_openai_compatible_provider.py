"""End-to-end coverage for a hand-configured OpenAI-compatible provider.

`pi-coding-agent` lets a user declare a provider in `models.json` with nothing
but a `baseUrl`, `api: "openai-completions"`, an `apiKey` and some `headers`.
Everything after parsing is `pi-ai`'s job: `create_provider` /
`registry.Models` must carry those four things all the way into the outbound
HTTP request. Individual pieces were covered (`test_registry.py` asserts the
stamped `base_url`, `test_openai_completions.py` asserts header construction),
but nothing asserted the whole chain, so a regression anywhere along it would
only show up in manual testing.

Every request goes through `httpx.MockTransport`; nothing reaches the network.

Exercises `packages/ai/src/models.ts` and `packages/ai/src/api/openai-completions.ts`.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pi_ai.api import openai_completions
from pi_ai.auth.types import ApiKeyAuth, InMemoryCredentialStore, ProviderAuth
from pi_ai.registry import Models, create_provider
from pi_ai.types import Context, Model, ModelCost, SimpleStreamOptions, UserMessage

BASE_URL = "https://llm.internal.example/openai/v1"
API_KEY = "sk-custom-1234"


def sse_body() -> str:
    chunks = [
        {"choices": [{"delta": {"content": "hi"}, "index": 0}]},
        {"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]},
    ]
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"


def make_client(capture: dict[str, httpx.Request]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        capture["request"] = request
        return httpx.Response(200, text=sse_body(), headers={"content-type": "text/event-stream"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def custom_model(**overrides: object) -> Model:
    defaults: dict[str, object] = {
        "id": "internal-llama",
        "name": "Internal Llama",
        "api": "openai-completions",
        "base_url": BASE_URL,
        "reasoning": False,
        "input": ["text"],
        "cost": ModelCost(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0),
        "context_window": 32_000,
        "max_tokens": 4096,
        "headers": {"X-Tenant": "acme", "X-Trace": "on"},
    }
    defaults.update(overrides)
    return Model(**defaults)  # type: ignore[arg-type]


def custom_models(env: dict[str, str], model: Model | None = None) -> tuple[Models, Model]:
    """Register one API-key provider that reads its key from ``env``."""
    resolved_model = model if model is not None else custom_model()
    provider = create_provider(
        id="internal",
        name="Internal Gateway",
        auth=ProviderAuth(api_key=ApiKeyAuth(name="Internal Gateway", env_vars=("INTERNAL_API_KEY",))),
        api=openai_completions,
        models=[resolved_model],
        base_url=BASE_URL,
    )
    models = Models(credential_store=InMemoryCredentialStore(), env=env.get)
    models.add(provider)
    return models, provider.get_models()[0]


async def collect(event_stream) -> None:
    async for _event in event_stream:
        pass
    await event_stream.result()


async def test_base_url_api_key_and_headers_all_reach_the_outbound_request() -> None:
    capture: dict[str, httpx.Request] = {}
    models, model = custom_models({"INTERNAL_API_KEY": API_KEY})

    async with make_client(capture) as client:
        await collect(
            await models.stream_simple(model, Context(messages=[UserMessage(content="hello")]), client=client)
        )

    request = capture["request"]
    assert str(request.url) == f"{BASE_URL}/chat/completions"
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    assert request.headers["x-tenant"] == "acme"
    assert request.headers["x-trace"] == "on"
    assert json.loads(request.content)["model"] == "internal-llama"


async def test_provider_base_url_is_stamped_onto_a_model_that_omits_one() -> None:
    capture: dict[str, httpx.Request] = {}
    models, model = custom_models({"INTERNAL_API_KEY": API_KEY}, custom_model(base_url=""))
    assert model.base_url == BASE_URL

    async with make_client(capture) as client:
        await collect(
            await models.stream_simple(model, Context(messages=[UserMessage(content="hello")]), client=client)
        )

    assert str(capture["request"].url) == f"{BASE_URL}/chat/completions"


async def test_an_explicit_api_key_option_overrides_the_environment() -> None:
    capture: dict[str, httpx.Request] = {}
    models, model = custom_models({"INTERNAL_API_KEY": API_KEY})

    async with make_client(capture) as client:
        await collect(
            await models.stream_simple(
                model,
                Context(messages=[UserMessage(content="hello")]),
                SimpleStreamOptions(api_key="sk-explicit"),
                client=client,
            )
        )

    assert capture["request"].headers["authorization"] == "Bearer sk-explicit"


async def test_request_headers_win_over_model_headers_case_insensitively() -> None:
    capture: dict[str, httpx.Request] = {}
    models, model = custom_models({"INTERNAL_API_KEY": API_KEY})

    async with make_client(capture) as client:
        await collect(
            await models.stream_simple(
                model,
                Context(messages=[UserMessage(content="hello")]),
                SimpleStreamOptions(headers={"x-tenant": "override"}),
                client=client,
            )
        )

    request = capture["request"]
    assert request.headers["x-tenant"] == "override"
    # The model's spelling must not survive alongside the override.
    assert request.headers.get_list("x-tenant") == ["override"]
    assert request.headers["x-trace"] == "on"


async def test_a_header_set_to_none_is_removed_from_the_request() -> None:
    capture: dict[str, httpx.Request] = {}
    models, model = custom_models({"INTERNAL_API_KEY": API_KEY})

    async with make_client(capture) as client:
        await collect(
            await models.stream_simple(
                model,
                Context(messages=[UserMessage(content="hello")]),
                SimpleStreamOptions(headers={"X-Trace": None}),
                client=client,
            )
        )

    assert "x-trace" not in capture["request"].headers
    assert capture["request"].headers["x-tenant"] == "acme"


async def test_an_unconfigured_custom_provider_reports_an_auth_error_in_band() -> None:
    models, model = custom_models({})

    stream = await models.stream_simple(model, Context(messages=[UserMessage(content="hello")]))
    async for _event in stream:
        pass
    result = await stream.result()

    assert result.stop_reason == "error"
    assert result.error_message is not None
    assert "not configured" in result.error_message


async def test_get_auth_reports_the_environment_variable_as_the_source() -> None:
    models, model = custom_models({"INTERNAL_API_KEY": API_KEY})

    auth = await models.get_auth(model)
    assert auth is not None
    assert auth.auth.api_key == API_KEY
    assert "INTERNAL_API_KEY" in auth.source


@pytest.mark.parametrize("base_url", ["https://llm.internal.example/openai/v1", "http://127.0.0.1:11434/v1"])
async def test_any_openai_compatible_base_url_is_used_verbatim(base_url: str) -> None:
    capture: dict[str, httpx.Request] = {}
    models, model = custom_models({"INTERNAL_API_KEY": API_KEY}, custom_model(base_url=base_url))

    async with make_client(capture) as client:
        await collect(
            await models.stream_simple(model, Context(messages=[UserMessage(content="hello")]), client=client)
        )

    assert str(capture["request"].url) == f"{base_url}/chat/completions"
