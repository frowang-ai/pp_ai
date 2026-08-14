"""Python port of `packages/ai/test/openai-completions-raw-stop-reason.test.ts`.

The TypeScript test mocks the `openai` SDK module to yield canned chunks. The
port has no SDK layer: the same chunks are served as an SSE body through an
`httpx.MockTransport`, so nothing leaves the process.
"""

from __future__ import annotations

import json

import httpx
from pi_ai.api.openai_completions import OpenAICompletionsOptions, stream
from pi_ai.types import Context, Model, ModelCost, UserMessage, now_ms

MODEL = Model(
    id="test-model",
    name="Test Model",
    api="openai-completions",
    provider="openai",
    base_url="https://api.openai.com/v1",
    reasoning=False,
    input=["text"],
    cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    context_window=128_000,
    max_tokens=4096,
)


def make_context() -> Context:
    return Context(messages=[UserMessage(content="hello", timestamp=now_ms())])


def sse_client(chunks: list[dict]) -> httpx.AsyncClient:
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_preserves_raw_finish_reasons_for_successful_stops() -> None:
    chunks = [{"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]
    message = await stream(
        MODEL,
        make_context(),
        OpenAICompletionsOptions(api_key="test"),
        client=sse_client(chunks),
    ).result()

    assert message.stop_reason == "stop"
    assert message.raw_stop_reason == "stop"
    assert message.error_message is None


async def test_preserves_raw_finish_reasons_for_provider_error_stops() -> None:
    chunks = [{"id": "chatcmpl-2", "choices": [{"index": 0, "delta": {}, "finish_reason": "content_filter"}]}]
    message = await stream(
        MODEL,
        make_context(),
        OpenAICompletionsOptions(api_key="test"),
        client=sse_client(chunks),
    ).result()

    assert message.stop_reason == "error"
    assert message.raw_stop_reason == "content_filter"
    assert message.error_message == "Provider finish_reason: content_filter"
