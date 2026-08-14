"""Python port of `packages/ai/test/fetch-option.test.ts`.

TypeScript's `StreamOptions.fetch` lets a caller swap the HTTP function each
provider SDK uses. This port has no `fetch` option: every adapter instead takes
an optional `client: httpx.AsyncClient` argument, which is the Python analogue
(the same "route all provider HTTP through the caller's transport" seam). Each
case therefore asserts that the injected client is the one used and that the
ambient/default `httpx` transport is never touched -- the port of TypeScript's
`expect(fallback).not.toHaveBeenCalled()`.

Two TypeScript cases are not ported as written:

* "rejects custom fetch for Google adapters instead of silently bypassing it".
  TypeScript rejects it because the vendored `@google/genai` SDK ignores a
  custom `fetch`, so accepting one would silently bypass it. This port talks to
  Google over `httpx` directly and threads `client` all the way into
  `iterate_google_chunks`, so there is nothing to bypass and no reason to
  reject. The inverted assertion (the injected client *is* used) is kept below.
* "uses fetch for Codex SSE" (the Codex third of the Mistral/Codex/pi-messages
  case): this port omits the `openai-codex-responses` provider entirely
  (`stream` raises `NotImplementedError`). It is declared below as an explicit
  skip so the gap stays visible in `pytest -rs` output.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pytest
from pi_ai.api.anthropic_messages import stream_simple as stream_anthropic
from pi_ai.api.azure_openai_responses import stream_simple as stream_azure_openai_responses
from pi_ai.api.google_generative_ai import stream_simple as stream_google_generative_ai
from pi_ai.api.google_vertex import stream_simple as stream_google_vertex
from pi_ai.api.mistral_conversations import stream_simple as stream_mistral
from pi_ai.api.openai_completions import stream_simple as stream_openai_completions
from pi_ai.api.openai_responses import stream_simple as stream_openai_responses
from pi_ai.api.openrouter_images import generate_images
from pi_ai.api.pi_messages import stream_simple as stream_pi_messages
from pi_ai.types import (
    Context,
    ImagesContext,
    ImagesModel,
    ImagesOptions,
    Model,
    ModelCost,
    SimpleStreamOptions,
    TextContent,
    UserMessage,
)

ERROR_BODY = json.dumps({"error": {"message": "upstream rejected request"}})


def make_context() -> Context:
    return Context(messages=[UserMessage(content="hello", timestamp=1)])


def create_model(api: str) -> Model:
    return Model(
        id="test-model",
        name="Test Model",
        api=api,
        provider="test-provider",
        base_url="https://upstream.test/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=10_000,
        max_tokens=1_000,
    )


class CountingTransport(httpx.MockTransport):
    """MockTransport that records how many requests it served."""

    def __init__(self) -> None:
        self.calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            self.calls += 1
            return httpx.Response(401, headers={"content-type": "application/json"}, text=ERROR_BODY)

        super().__init__(handler)


@pytest.fixture(autouse=True)
def forbid_ambient_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Port of `vi.stubGlobal("fetch", fallback)` with a throwing fallback."""

    async def forbidden(self: object, request: httpx.Request) -> httpx.Response:
        raise AssertionError("ambient httpx transport must not be called")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    yield


def make_client() -> tuple[httpx.AsyncClient, CountingTransport]:
    transport = CountingTransport()
    return httpx.AsyncClient(transport=transport), transport


async def test_passes_the_client_through_stream_simple_to_the_anthropic_adapter():
    client, transport = make_client()
    async with client:
        await stream_anthropic(
            create_model("anthropic-messages"),
            make_context(),
            SimpleStreamOptions(api_key="test-key", max_retries=0),
            client=client,
        ).result()
    assert transport.calls == 1


async def test_passes_the_client_through_stream_simple_to_openai_adapters():
    adapters = [
        (stream_openai_completions, "openai-completions"),
        (stream_openai_responses, "openai-responses"),
        (stream_azure_openai_responses, "azure-openai-responses"),
    ]
    client, transport = make_client()
    async with client:
        for stream_fn, api in adapters:
            await stream_fn(
                create_model(api),
                make_context(),
                SimpleStreamOptions(api_key="test-key", max_retries=0),
                client=client,
            ).result()
    assert transport.calls == len(adapters)


async def test_uses_the_client_for_mistral_and_pi_messages_http_requests():
    client, transport = make_client()
    async with client:
        await stream_mistral(
            create_model("mistral-conversations"),
            make_context(),
            SimpleStreamOptions(api_key="test-key", max_retries=0),
            client=client,
        ).result()
        await stream_pi_messages(
            create_model("pi-messages"),
            make_context(),
            SimpleStreamOptions(api_key="test-key", max_retries=0),
            client=client,
        ).result()
    assert transport.calls == 2


async def test_google_adapters_use_the_injected_client():
    # Inverted from TS's `it("rejects custom fetch for Google adapters instead
    # of silently bypassing it")`: TS rejects a custom fetch there because the
    # vendored `@google/genai` SDK ignores it, so honoring one would silently
    # bypass it (see `google-generative-ai.ts`/`google-vertex.ts`, which never
    # forward `fetch` to the SDK client). This port talks to Google over
    # `httpx` directly (`iterate_google_chunks` threads `client` into
    # `stream_sse`, verified in `google_shared.py`), so there is nothing to
    # silently bypass; the injected client is used for real, which this test
    # asserts via `transport.calls`. This also subsumes TS's
    # `it("allows Google adapters to receive globalThis.fetch explicitly")`,
    # whose only content was `errorMessage` not containing the rejection text.
    client, transport = make_client()
    async with client:
        google = await stream_google_generative_ai(
            create_model("google-generative-ai"),
            make_context(),
            SimpleStreamOptions(api_key="test-key", max_retries=0),
            client=client,
        ).result()
        vertex = await stream_google_vertex(
            create_model("google-vertex"),
            make_context(),
            SimpleStreamOptions(api_key="test-key", max_retries=0),
            client=client,
        ).result()

    assert transport.calls == 2
    assert google.error_message is not None
    assert "Custom fetch is not supported" not in google.error_message
    assert vertex.error_message is not None
    assert "Custom fetch is not supported" not in vertex.error_message


async def test_uses_the_client_for_image_generation():
    model = ImagesModel(
        id="test-model",
        name="Test Model",
        api="openrouter-images",
        provider="openrouter",
        base_url="https://upstream.test/v1",
        input=["text"],
        output=["image"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    )
    client, transport = make_client()
    async with client:
        await generate_images(
            model,
            ImagesContext(input=[TextContent(text="draw")]),
            ImagesOptions(api_key="test-key", max_retries=0),
            client=client,
        )
    assert transport.calls == 1


@pytest.mark.skip(
    reason="This port deliberately omits the `openai-codex-responses` provider "
    "(`pi_ai.api.openai_codex_responses.stream` raises NotImplementedError; see the README's "
    "list of omissions), so there is no Python code path to exercise."
)
async def test_uses_the_client_for_codex_sse_requests():
    """The Codex third of `it("uses fetch for Mistral, Codex SSE, and pi-messages HTTP requests")`.

    Asserts the Codex Responses adapter issues its SSE request through the
    caller-supplied HTTP function rather than the ambient one.
    """
