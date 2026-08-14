"""Python port of `packages/ai/test/openai-completions-response-model.test.ts`.

Router/virtual ids (for example OpenRouter `auto`) keep `model` pinned to the
requested id and surface the routed concrete id on `response_model`.

The TypeScript test mocks the `openai` SDK; the port serves the same chunks as
an SSE body through an `httpx.MockTransport`.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from pi_ai.api.openai_completions import OpenAICompletionsOptions, stream
from pi_ai.types import Context, Model, ModelCost, UserMessage, now_ms


def open_router_auto() -> Model:
    return Model(
        id="openrouter/auto",
        name="OpenRouter Auto",
        api="openai-completions",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=200_000,
        max_tokens=8192,
    )


def sse_client(chunks: list[dict[str, Any]]) -> httpx.AsyncClient:
    body = "".join(f"data: {json.dumps(item)}\n\n" for item in chunks) + "data: [DONE]\n\n"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def complete_with(chunks: list[dict[str, Any]]):
    return await stream(
        open_router_auto(),
        Context(messages=[UserMessage(content="hi", timestamp=now_ms())]),
        OpenAICompletionsOptions(api_key="test"),
        client=sse_client(chunks),
    ).result()


USAGE = {
    "prompt_tokens": 1,
    "completion_tokens": 1,
    "prompt_tokens_details": {"cached_tokens": 0},
    "completion_tokens_details": {"reasoning_tokens": 0},
}


async def test_surfaces_routed_chunk_model_on_response_model_without_changing_model() -> None:
    message = await complete_with(
        [
            {
                "id": "chatcmpl-1",
                "model": "anthropic/claude-opus-4.8",
                "choices": [{"index": 0, "delta": {"content": "hi"}}],
            },
            {
                "id": "chatcmpl-1",
                "model": "anthropic/claude-opus-4.8",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {**USAGE, "prompt_tokens": 10, "completion_tokens": 5},
            },
        ]
    )

    assert message.model == "openrouter/auto"
    assert message.response_model == "anthropic/claude-opus-4.8"
    assert message.provider == "openrouter"
    assert message.stop_reason == "stop"


async def test_leaves_response_model_unset_when_chunks_echo_the_requested_id() -> None:
    message = await complete_with(
        [
            {"id": "chatcmpl-2", "model": "openrouter/auto", "choices": [{"index": 0, "delta": {"content": "hi"}}]},
            {
                "id": "chatcmpl-2",
                "model": "openrouter/auto",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": USAGE,
            },
        ]
    )

    assert message.model == "openrouter/auto"
    assert message.response_model is None


async def test_ignores_empty_or_missing_chunk_model() -> None:
    message = await complete_with(
        [
            {"id": "chatcmpl-3", "choices": [{"index": 0, "delta": {"content": "hi"}}]},
            {"id": "chatcmpl-3", "model": "", "choices": [{"index": 0, "delta": {"content": "!"}}]},
            {
                "id": "chatcmpl-3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {**USAGE, "completion_tokens": 2},
            },
        ]
    )

    assert message.model == "openrouter/auto"
    assert message.response_model is None
