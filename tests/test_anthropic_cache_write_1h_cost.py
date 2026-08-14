"""Python port of `packages/ai/test/anthropic-cache-write-1h-cost.test.ts`."""

from __future__ import annotations

import json

import httpx
from pi_ai.api.anthropic_messages import AnthropicOptions, stream
from pi_ai.providers.all import get_builtin_model
from pi_ai.types import Context, UserMessage


def sse_body(events: list[tuple[str, dict]]) -> str:
    return "\n".join(f"event: {name}\ndata: {json.dumps(data)}\n" for name, data in events)


def make_client(body: str) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def events_with_cache_creation(cache_creation: dict[str, int] | None) -> list[tuple[str, dict]]:
    start_usage: dict[str, object] = {
        "input_tokens": 100,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 1_000_000,
    }
    if cache_creation:
        start_usage["cache_creation"] = cache_creation
    return [
        ("message_start", {"type": "message_start", "message": {"id": "msg_test", "usage": start_usage}}),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 1_000_000,
                },
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]


# claude-opus-4-8: input 5, cacheWrite (5m) 6.25 per Mtok. 1h write = 2x input = 10.
CONTEXT = Context(messages=[UserMessage(content="hi")])


async def test_prices_the_1h_portion_at_2x_input_and_the_rest_at_the_5m_rate():
    model = get_builtin_model("anthropic", "claude-opus-4-8")
    client = make_client(
        sse_body(
            events_with_cache_creation({"ephemeral_5m_input_tokens": 600_000, "ephemeral_1h_input_tokens": 400_000})
        )
    )
    result = await stream(model, CONTEXT, AnthropicOptions(api_key="test"), client=client).result()

    assert result.usage.cache_write == 1_000_000
    assert result.usage.cache_write_1h == 400_000
    # 600k * 6.25/Mtok + 400k * 10/Mtok = 3.75 + 4.0 = 7.75
    assert round(result.usage.cost.cache_write, 10) == 7.75


async def test_falls_back_to_the_5m_rate_when_no_breakdown_is_reported():
    model = get_builtin_model("anthropic", "claude-opus-4-8")
    client = make_client(sse_body(events_with_cache_creation(None)))
    result = await stream(model, CONTEXT, AnthropicOptions(api_key="test"), client=client).result()

    assert result.usage.cache_write == 1_000_000
    assert (result.usage.cache_write_1h or 0) == 0
    # 1M * 6.25/Mtok = 6.25
    assert round(result.usage.cost.cache_write, 10) == 6.25
