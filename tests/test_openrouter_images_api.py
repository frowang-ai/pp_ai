"""Python port of `packages/ai/test/openrouter-images.test.ts`.

The TypeScript file mocks the `openai` SDK module. This port has no vendor SDK:
`pi_ai.api.openrouter_images` posts with `httpx` and `pi_ai.images.generate_images`
does not accept a client, so the equivalent seam is swapping `httpx.AsyncClient`
for one bound to a `MockTransport`.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pi_ai.images import generate_images
from pi_ai.types import ImagesContext, ImagesModel, ImagesOptions, ModelCost, TextContent
from pi_ai.utils.abort import AbortController

_REAL_ASYNC_CLIENT = httpx.AsyncClient

_RESPONSE_BODY: dict[str, Any] = {
    "id": "img-1",
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 34,
        "prompt_tokens_details": {"cached_tokens": 0},
    },
    "choices": [
        {
            "message": {
                "content": "Here is your image.",
                "images": [{"image_url": "data:image/png;base64,ZmFrZS1wbmc="}],
            }
        }
    ],
}


def stub_client(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Record the request payload and always answer with `_RESPONSE_BODY`."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = json.loads(request.content)
        return httpx.Response(200, json=_RESPONSE_BODY)

    def factory(**kwargs: object) -> httpx.AsyncClient:
        kwargs.pop("transport", None)
        kwargs.pop("proxy", None)
        return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return captured


def text_image_model() -> ImagesModel:
    return ImagesModel(
        id="google/gemini-3.1-flash-image-preview",
        name="Gemini 3.1 Flash Image Preview",
        api="openrouter-images",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        input=["text", "image"],
        output=["text", "image"],
        cost=ModelCost(input=0.015, output=0.03, cache_read=0, cache_write=0),
        headers={"HTTP-Referer": "https://example.com"},
    )


def image_only_model() -> ImagesModel:
    return ImagesModel(
        id="black-forest-labs/flux.2-pro",
        name="FLUX.2 Pro",
        api="openrouter-images",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        input=["text", "image"],
        output=["image"],
        cost=ModelCost(input=0.015, output=0.03, cache_read=0, cache_write=0),
    )


def dog_context() -> ImagesContext:
    return ImagesContext(input=[TextContent(text="Generate a dog")])


async def test_returns_text_plus_images_in_final_output(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = stub_client(monkeypatch)

    output = await generate_images(text_image_model(), dog_context(), ImagesOptions(api_key="test"))

    assert output.stop_reason == "stop"
    assert output.response_id == "img-1"
    assert output.output[0].type == "text"
    assert output.output[0].text == "Here is your image."
    assert output.output[1].type == "image"
    assert output.output[1].mime_type == "image/png"
    assert output.output[1].data == "ZmFrZS1wbmc="

    params = captured["params"]
    assert params["stream"] is False
    assert params["modalities"] == ["image", "text"]
    assert params["messages"][0]["content"][0] == {"type": "text", "text": "Generate a dog"}


async def test_passes_through_abort_signal_and_returns_aborted_result(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = stub_client(monkeypatch)
    controller = AbortController()
    controller.abort()

    output = await generate_images(
        image_only_model(), dog_context(), ImagesOptions(api_key="test", signal=controller.signal)
    )

    assert output.stop_reason == "aborted"
    # TypeScript's exact message ("Request aborted") comes from the mocked openai
    # SDK; this port raises its own abort error, using the same wording as the
    # streaming adapters (`openai_completions`).
    assert output.error_message == "Request was aborted"
    # TypeScript asserts the signal reached the SDK via `requestOptions.signal`.
    # There is no vendor SDK here, so the observable equivalent is that the
    # request was never sent at all.
    assert "params" not in captured


async def test_generate_images_resolves_the_final_assistant_images_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_client(monkeypatch)

    output = await generate_images(image_only_model(), dog_context(), ImagesOptions(api_key="test"))

    assert any(item.type == "image" for item in output.output)
