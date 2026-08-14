import json

import httpx
import pytest
from pi_ai import ImageContent, TextContent
from pi_ai.api.openrouter_images import (
    ImagesOptions,
    build_params,
    generate_images,
    parse_usage,
)
from pi_ai.providers.openrouter import OPENROUTER_MODELS, openrouter_provider
from pi_ai.types import ImagesContext, ImagesModel, ModelCost


def make_images_model(**overrides) -> ImagesModel:
    defaults = dict(
        id="black-forest-labs/flux.2-pro",
        name="FLUX.2 Pro",
        api="openrouter-images",
        provider="openrouter",
        base_url="https://openrouter.example.com/api/v1",
        input=["text", "image"],
        output=["image"],
        cost=ModelCost(input=2.0, output=4.0, cache_read=1.0, cache_write=0.5),
    )
    defaults.update(overrides)
    return ImagesModel(**defaults)


def make_client(status: int, body: dict | str, capture: dict | None = None) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture["request"] = request
            capture["json"] = json.loads(request.content)
        if isinstance(body, str):
            return httpx.Response(status, text=body)
        return httpx.Response(status, json=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def data_url(mime_type: str, data: str) -> str:
    return f"data:{mime_type};base64,{data}"


# --------------------------------------------------------------------------
# build_params
# --------------------------------------------------------------------------


def test_build_params_text_only_content():
    model = make_images_model()
    context = ImagesContext(input=[TextContent(text="a red cat")])
    params = build_params(model, context)
    assert params == {
        "model": "black-forest-labs/flux.2-pro",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "a red cat"}]}],
        "stream": False,
        "modalities": ["image"],
    }


def test_build_params_includes_text_modality_when_model_outputs_text():
    model = make_images_model(output=["image", "text"])
    context = ImagesContext(input=[TextContent(text="describe this")])
    params = build_params(model, context)
    assert params["modalities"] == ["image", "text"]


def test_build_params_converts_image_content_to_data_url():
    model = make_images_model()
    context = ImagesContext(input=[ImageContent(data="AAAA", mime_type="image/png")])
    params = build_params(model, context)
    assert params["messages"][0]["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    ]


def test_build_params_mixed_text_and_image_content():
    model = make_images_model()
    context = ImagesContext(input=[TextContent(text="edit this"), ImageContent(data="BBBB", mime_type="image/jpeg")])
    params = build_params(model, context)
    content = params["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "edit this"}
    assert content[1] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BBBB"}}


# --------------------------------------------------------------------------
# parse_usage
# --------------------------------------------------------------------------


def test_parse_usage_computes_cost_from_per_million_rates():
    model = make_images_model()
    raw_usage = {"prompt_tokens": 1000, "completion_tokens": 500}
    usage = parse_usage(raw_usage, model)
    assert usage.input == 1000
    assert usage.output == 500
    assert usage.cache_read == 0
    assert usage.cache_write == 0
    assert usage.total_tokens == 1500
    assert usage.cost.input == pytest.approx((2.0 / 1_000_000) * 1000)
    assert usage.cost.output == pytest.approx((4.0 / 1_000_000) * 500)
    assert usage.cost.total == pytest.approx(usage.cost.input + usage.cost.output)


def test_parse_usage_splits_cache_read_and_write_tokens():
    model = make_images_model()
    raw_usage = {
        "prompt_tokens": 1200,
        "completion_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 700, "cache_write_tokens": 200},
    }
    usage = parse_usage(raw_usage, model)
    # cached_tokens (700) includes cache_write_tokens (200), so cache_read is
    # the remainder: 700 - 200 = 500.
    assert usage.cache_write == 200
    assert usage.cache_read == 500
    assert usage.input == 1200 - 500 - 200
    assert usage.cost.cache_read == pytest.approx((1.0 / 1_000_000) * 500)
    assert usage.cost.cache_write == pytest.approx((0.5 / 1_000_000) * 200)


def test_parse_usage_treats_cached_tokens_as_cache_read_when_no_cache_write():
    model = make_images_model()
    raw_usage = {"prompt_tokens": 900, "completion_tokens": 50, "prompt_tokens_details": {"cached_tokens": 300}}
    usage = parse_usage(raw_usage, model)
    assert usage.cache_read == 300
    assert usage.cache_write == 0
    assert usage.input == 900 - 300


def test_parse_usage_handles_missing_fields():
    model = make_images_model()
    usage = parse_usage({}, model)
    assert usage.input == 0
    assert usage.output == 0
    assert usage.cache_read == 0
    assert usage.cache_write == 0
    assert usage.total_tokens == 0
    assert usage.cost.total == 0.0


# --------------------------------------------------------------------------
# generate_images
# --------------------------------------------------------------------------


async def test_generate_images_success_with_image_output():
    body = {
        "id": "resp_1",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "choices": [
            {
                "message": {
                    "content": "Here is your image",
                    "images": [{"image_url": {"url": data_url("image/png", "AAAA")}}],
                }
            }
        ],
    }
    async with make_client(200, body) as client:
        result = await generate_images(
            make_images_model(), ImagesContext(input=[TextContent(text="a cat")]), ImagesOptions(api_key="k"), client
        )

    assert result.stop_reason == "stop"
    assert result.response_id == "resp_1"
    assert result.usage is not None
    assert result.usage.input == 10
    assert isinstance(result.output[0], TextContent)
    assert result.output[0].text == "Here is your image"
    assert isinstance(result.output[1], ImageContent)
    assert result.output[1].mime_type == "image/png"
    assert result.output[1].data == "AAAA"


async def test_generate_images_ignores_non_data_url_images():
    body = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "images": [{"image_url": {"url": "https://example.com/image.png"}}],
                }
            }
        ]
    }
    async with make_client(200, body) as client:
        result = await generate_images(
            make_images_model(), ImagesContext(input=[TextContent(text="a cat")]), ImagesOptions(api_key="k"), client
        )
    assert result.stop_reason == "stop"
    assert result.output == []


async def test_generate_images_handles_string_image_url_field():
    body = {"choices": [{"message": {"content": "", "images": [{"image_url": data_url("image/jpeg", "ZZZZ")}]}}]}
    async with make_client(200, body) as client:
        result = await generate_images(
            make_images_model(), ImagesContext(input=[TextContent(text="a cat")]), ImagesOptions(api_key="k"), client
        )
    assert result.output[0].mime_type == "image/jpeg"
    assert result.output[0].data == "ZZZZ"


async def test_generate_images_reports_http_error():
    async with make_client(400, {"error": {"message": "bad request"}}) as client:
        result = await generate_images(
            make_images_model(), ImagesContext(input=[TextContent(text="a cat")]), ImagesOptions(api_key="k"), client
        )
    assert result.stop_reason == "error"
    assert result.error_message is not None
    assert result.output == []


async def test_generate_images_reports_error_without_api_key():
    result = await generate_images(
        make_images_model(), ImagesContext(input=[TextContent(text="a cat")]), ImagesOptions(api_key=None)
    )
    assert result.stop_reason == "error"
    assert "No API key" in result.error_message


async def test_generate_images_reports_aborted_when_signal_aborted_and_request_fails():
    from pi_ai.utils.abort import AbortSignal

    signal = AbortSignal()
    signal.abort()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("connection reset", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await generate_images(
            make_images_model(),
            ImagesContext(input=[TextContent(text="a cat")]),
            ImagesOptions(api_key="k", signal=signal),
            client,
        )
    assert result.stop_reason == "aborted"


async def test_generate_images_sends_bearer_auth_and_request_payload():
    body = {"choices": [{"message": {"content": "ok"}}]}
    capture: dict = {}
    async with make_client(200, body, capture=capture) as client:
        await generate_images(
            make_images_model(), ImagesContext(input=[TextContent(text="a cat")]), ImagesOptions(api_key="k"), client
        )
    assert capture["request"].headers["authorization"] == "Bearer k"
    assert capture["json"]["model"] == "black-forest-labs/flux.2-pro"
    assert capture["json"]["stream"] is False


async def test_generate_images_on_payload_hook_can_replace_payload():
    body = {"choices": [{"message": {"content": "ok"}}]}
    capture: dict = {}

    def on_payload(payload, model):
        payload["extra_marker"] = "injected"
        return payload

    async with make_client(200, body, capture=capture) as client:
        await generate_images(
            make_images_model(),
            ImagesContext(input=[TextContent(text="a cat")]),
            ImagesOptions(api_key="k", on_payload=on_payload),
            client,
        )
    assert capture["json"]["extra_marker"] == "injected"


async def test_generate_images_on_response_hook_invoked_on_success():
    body = {"choices": [{"message": {"content": "ok"}}]}
    seen: dict = {}

    def on_response(response, model):
        seen["status"] = response.status

    async with make_client(200, body) as client:
        await generate_images(
            make_images_model(),
            ImagesContext(input=[TextContent(text="a cat")]),
            ImagesOptions(api_key="k", on_response=on_response),
            client,
        )
    assert seen["status"] == 200


async def test_generate_images_on_response_hook_not_invoked_on_error():
    seen: dict = {}

    def on_response(response, model):
        seen["called"] = True

    async with make_client(400, {"error": {"message": "bad"}}) as client:
        await generate_images(
            make_images_model(),
            ImagesContext(input=[TextContent(text="a cat")]),
            ImagesOptions(api_key="k", on_response=on_response),
            client,
        )
    assert "called" not in seen


# --------------------------------------------------------------------------
# openrouter_provider() factory
# --------------------------------------------------------------------------


def test_openrouter_provider_metadata():
    provider = openrouter_provider()
    assert provider.id == "openrouter"
    assert provider.base_url == "https://openrouter.ai/api/v1"
    model_ids = {model.id for model in provider.models}
    assert "openai/gpt-4o" in model_ids
    assert "anthropic/claude-sonnet-4.5" in model_ids
    for model in provider.models:
        assert model.provider == "openrouter"
        assert model.base_url == "https://openrouter.ai/api/v1"
        assert model.api == "openai-completions"


def test_openrouter_models_catalog_is_nonempty_and_matches_provider():
    assert len(OPENROUTER_MODELS) > 0
    provider = openrouter_provider()
    assert len(provider.models) == len(OPENROUTER_MODELS)


async def test_openrouter_provider_resolves_api_key_from_env(monkeypatch):
    from pi_ai.auth.helpers import resolve_api_key_auth

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    provider = openrouter_provider()
    result = await resolve_api_key_auth(provider.auth.api_key)
    assert result.auth.api_key == "sk-or-test"
    assert result.source == "OPENROUTER_API_KEY"
