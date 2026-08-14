"""Port of `packages/ai/test/azure-openai-base-url.test.ts`.

TypeScript mocks the `openai` SDK and reads `baseURL` off the captured
`AzureOpenAI` constructor options during a real `stream()` call. This port has
no SDK client, so it drives the real `stream()` through an
`httpx.MockTransport` and recovers the base URL from the actual request URL
sent on the wire (built from the same `_resolve_azure_config` call `stream()`
uses internally). The payload cases (`prompt_cache_key`, `store`, `strict`)
capture the real request body the same way.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pi_ai.api.azure_openai_responses import AzureOpenAIResponsesOptions
from pi_ai.api.azure_openai_responses import stream as stream_azure_openai_responses
from pi_ai.providers.all import get_builtin_model
from pi_ai.types import Context, JsonSchemaConstrainedSampling, Model, Tool, UserMessage

CONTEXT = Context(messages=[UserMessage(content="hello")])

AZURE_ENV_VARS = (
    "AZURE_OPENAI_BASE_URL",
    "AZURE_OPENAI_RESOURCE_NAME",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_azure_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in AZURE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def get_test_model() -> Model:
    return get_builtin_model("azure-openai-responses", "gpt-4o-mini")


async def _run_and_capture_base_url() -> str:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text="", headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await stream_azure_openai_responses(
        get_test_model(), CONTEXT, AzureOpenAIResponsesOptions(api_key="test-api-key"), client=client
    ).result()

    assert len(requests) == 1
    request_url = str(requests[0].url)
    base, _, _rest = request_url.partition("/deployments/")
    return base


async def capture_client_base_url(base_url: str, monkeypatch: pytest.MonkeyPatch) -> str:
    # TS captures `azureMock.constructorCalls[0].baseURL`, i.e. the `baseURL` the
    # real `AzureOpenAI` SDK client is constructed with for an actual request.
    # This port has no SDK client, so instead it runs the real `stream()` call
    # through a MockTransport and recovers the same base URL from the actual
    # request URL sent on the wire (`{base_url}/deployments/...`), which is
    # built from the identical `_resolve_azure_config` call inside `stream()`.
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", base_url)
    return await _run_and_capture_base_url()


async def capture_client_base_url_from_env(monkeypatch: pytest.MonkeyPatch) -> str:
    # Same seam as `capture_client_base_url`, used by the
    # AZURE_OPENAI_RESOURCE_NAME default-URL case (no explicit base URL env var).
    return await _run_and_capture_base_url()


async def capture_payload(model: Model, context: Context, options: AzureOpenAIResponsesOptions) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, text="", headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await stream_azure_openai_responses(model, context, options, client=client).result()
    assert captured, "Azure request payload was not captured"
    return captured


async def test_normalizes_cognitive_services_root_endpoints(monkeypatch: pytest.MonkeyPatch):
    assert (
        await capture_client_base_url("https://marc-quicktests-resource.cognitiveservices.azure.com", monkeypatch)
        == "https://marc-quicktests-resource.cognitiveservices.azure.com/openai/v1"
    )


async def test_normalizes_microsoft_foundry_root_endpoints(monkeypatch: pytest.MonkeyPatch):
    assert (
        await capture_client_base_url("https://marc-quicktests-resource.ai.azure.com", monkeypatch)
        == "https://marc-quicktests-resource.ai.azure.com/openai/v1"
    )


async def test_normalizes_azure_openai_root_endpoints(monkeypatch: pytest.MonkeyPatch):
    assert (
        await capture_client_base_url("https://my-resource.openai.azure.com", monkeypatch)
        == "https://my-resource.openai.azure.com/openai/v1"
    )


async def test_normalizes_openai_to_openai_v1(monkeypatch: pytest.MonkeyPatch):
    assert (
        await capture_client_base_url("https://my-resource.cognitiveservices.azure.com/openai", monkeypatch)
        == "https://my-resource.cognitiveservices.azure.com/openai/v1"
    )


async def test_preserves_openai_v1_endpoints(monkeypatch: pytest.MonkeyPatch):
    assert (
        await capture_client_base_url("https://my-resource.cognitiveservices.azure.com/openai/v1", monkeypatch)
        == "https://my-resource.cognitiveservices.azure.com/openai/v1"
    )


async def test_normalizes_openai_v1_responses_to_openai_v1(monkeypatch: pytest.MonkeyPatch):
    assert (
        await capture_client_base_url("https://my-resource.services.ai.azure.com/openai/v1/responses", monkeypatch)
        == "https://my-resource.services.ai.azure.com/openai/v1"
    )


async def test_preserves_explicit_non_azure_proxy_paths(monkeypatch: pytest.MonkeyPatch):
    assert (
        await capture_client_base_url("https://my-proxy.example.com/v1", monkeypatch)
        == "https://my-proxy.example.com/v1"
    )


async def test_strips_query_params_when_normalizing_azure_host_urls(monkeypatch: pytest.MonkeyPatch):
    assert (
        await capture_client_base_url("https://my-resource.openai.azure.com/openai?api-version=2024-12-01", monkeypatch)
        == "https://my-resource.openai.azure.com/openai/v1"
    )


async def test_preserves_query_params_on_non_azure_proxy_urls(monkeypatch: pytest.MonkeyPatch):
    assert (
        await capture_client_base_url("https://my-proxy.example.com/v1?custom=true", monkeypatch)
        == "https://my-proxy.example.com/v1?custom=true"
    )


async def test_throws_on_invalid_urls(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "not-a-url")
    result = await stream_azure_openai_responses(
        get_test_model(), CONTEXT, AzureOpenAIResponsesOptions(api_key="test-api-key")
    ).result()
    assert result.stop_reason == "error"
    assert "Invalid Azure OpenAI base URL" in (result.error_message or "")


async def test_clamps_prompt_cache_key_to_64_characters():
    payload = await capture_payload(
        get_test_model(),
        CONTEXT,
        AzureOpenAIResponsesOptions(
            api_key="test-api-key",
            azure_base_url="https://my-resource.openai.azure.com",
            session_id="x" * 67,
        ),
    )
    assert payload["prompt_cache_key"] == "x" * 64


async def test_disables_server_side_response_storage():
    payload = await capture_payload(
        get_test_model(),
        CONTEXT,
        AzureOpenAIResponsesOptions(api_key="test-api-key", azure_base_url="https://my-resource.openai.azure.com"),
    )
    assert payload["store"] is False


async def test_honors_supports_strict_mode_false():
    base_model = get_test_model()
    model = Model(
        **{
            **base_model.__dict__,
            "compat": {**base_model.compat, "supportsStrictMode": False},
        }
    )
    context = Context(
        messages=CONTEXT.messages,
        tools=[
            Tool(
                name="preferred",
                description="Preferred constrained tool",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
                constrained_sampling=JsonSchemaConstrainedSampling(strict="prefer"),
            )
        ],
    )

    payload = await capture_payload(
        model,
        context,
        AzureOpenAIResponsesOptions(api_key="test-api-key", azure_base_url="https://my-resource.openai.azure.com"),
    )
    assert "strict" not in payload["tools"][0]


async def test_builds_default_url_from_azure_openai_resource_name(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AZURE_OPENAI_RESOURCE_NAME", "my-resource")
    assert await capture_client_base_url_from_env(monkeypatch) == "https://my-resource.openai.azure.com/openai/v1"
