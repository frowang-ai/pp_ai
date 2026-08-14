"""Python port of `packages/ai/test/provider-error-body-passthrough.test.ts`.

Regression test for `issues/provider-error-body-passthrough`: when an endpoint
behind a proxy/gateway returns a non-2xx response with a body the SDK cannot
fold into its message, a body-blind `except` block that reads only the message
surfaces the opaque `"403 status code (no body)"` and hides the real reason.

The TypeScript test mocks the `openai` SDK module so the request throws an
`APIError` whose parsed body lives on `.error`. The port has no SDK layer, so
the 403-with-body is served by an `httpx.MockTransport`; `ProviderHttpError`
carries the same status/body/parsed-body attributes that `normalize_provider_
error` probes.
"""

from __future__ import annotations

import httpx
import pytest
from pi_ai import images as images_module
from pi_ai.api import openrouter_images
from pi_ai.images import generate_images
from pi_ai.types import ImagesContext, ImagesModel, ImagesOptions, ModelCost, TextContent


@pytest.fixture
def gateway_403(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        # 403 from a gateway/proxy carrying the real reason in the body.
        return httpx.Response(403, json={"error": "blocked by gateway WAF"})

    real_async_client = httpx.AsyncClient

    def fake_client(*_args: object, **kwargs: object) -> httpx.AsyncClient:
        return real_async_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(openrouter_images.httpx, "AsyncClient", fake_client)


async def test_surfaces_the_http_body_reason_instead_of_the_opaque_message(gateway_403: None) -> None:
    assert images_module.generate_images is generate_images
    model = ImagesModel(
        id="black-forest-labs/flux.2-pro",
        name="FLUX.2 Pro",
        api="openrouter-images",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        input=["text", "image"],
        output=["image"],
        cost=ModelCost(input=0.015, output=0.03, cache_read=0, cache_write=0),
    )
    context = ImagesContext(input=[TextContent(text="Generate a dog")])

    output = await generate_images(model, context, ImagesOptions(api_key="test"))

    assert output.stop_reason == "error"
    assert "403" in (output.error_message or "")
    assert "blocked by gateway WAF" in (output.error_message or "")
    assert output.error_message != "403 status code (no body)"
