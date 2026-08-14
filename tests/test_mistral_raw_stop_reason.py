"""Python port of `packages/ai/test/mistral-raw-stop-reason.test.ts`."""

from __future__ import annotations

import json

import httpx
from pi_ai.api.mistral_conversations import MistralOptions, stream
from pi_ai.providers.all import get_builtin_model
from pi_ai.types import Context, UserMessage

MODEL = get_builtin_model("mistral", "devstral-medium-latest")
CONTEXT = Context(messages=[UserMessage(content="hello")])


def make_client(finish_reason: str) -> httpx.AsyncClient:
    body = (
        "data: "
        + json.dumps(
            {
                "id": "mistral-response-id",
                "model": MODEL.id,
                "choices": [{"index": 0, "finish_reason": finish_reason, "delta": {}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
            }
        )
        + "\n\ndata: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_preserves_raw_mistral_finish_reasons_for_successful_stops():
    message = await stream(MODEL, CONTEXT, MistralOptions(api_key="test"), client=make_client("stop")).result()

    assert message.stop_reason == "stop"
    assert message.raw_stop_reason == "stop"
    assert message.error_message is None


async def test_preserves_raw_mistral_finish_reasons_for_provider_error_stops():
    message = await stream(MODEL, CONTEXT, MistralOptions(api_key="test"), client=make_client("error")).result()

    assert message.stop_reason == "error"
    assert message.raw_stop_reason == "error"
    assert message.error_message == "Provider stopped with: error"


async def test_treats_unknown_mistral_finish_reasons_as_provider_error_stops():
    message = await stream(
        MODEL, CONTEXT, MistralOptions(api_key="test"), client=make_client("unmapped_error")
    ).result()

    assert message.stop_reason == "error"
    assert message.raw_stop_reason == "unmapped_error"
    assert message.error_message == "Provider stopped with: unmapped_error"
