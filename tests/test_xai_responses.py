"""Python port of `packages/ai/test/xai-responses.test.ts`.

The TypeScript test stubs `globalThis.fetch`. The port passes an
`httpx.AsyncClient` backed by a `MockTransport` through the provider's
`stream(..., client=...)` keyword, which the registry forwards to the api
module -- no socket is opened.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from pi_ai.api.openai_responses import OpenAIResponsesOptions
from pi_ai.models import get_supported_thinking_levels
from pi_ai.providers.all import get_builtin_model, get_builtin_models
from pi_ai.providers.xai import xai_provider
from pi_ai.types import Context, Model, UserMessage

COMPLETED_EVENT = {
    "type": "response.completed",
    "sequence_number": 0,
    "response": {
        "id": "resp_xai_test",
        "status": "completed",
        "output": [],
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
            "input_tokens_details": {"cached_tokens": 0},
        },
    },
}


def completed_response_body() -> str:
    return f"data: {json.dumps(COMPLETED_EVENT)}\n\ndata: [DONE]\n\n"


async def capture_request(model: Model, context: Context, options: OpenAIResponsesOptions) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            text=completed_response_body(),
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = await xai_provider().stream(model, context, options, client=client).result()
    assert result.stop_reason == "stop", result.error_message
    assert captured
    return captured


def test_excludes_retired_and_redundant_models_from_the_builtin_catalog() -> None:
    model_ids = [model.id for model in get_builtin_models("xai")]
    for model_id in [
        "grok-3",
        "grok-3-fast",
        "grok-4.20-0309-non-reasoning",
        "grok-4.20-0309-reasoning",
        "grok-code-fast-1",
    ]:
        assert model_id not in model_ids


def test_uses_responses_with_low_medium_high_efforts_only_for_grok_45() -> None:
    grok_45 = get_builtin_model("xai", "grok-4.5")
    assert grok_45 is not None
    assert grok_45.api == "openai-responses"
    assert get_supported_thinking_levels(grok_45) == ["low", "medium", "high"]

    grok_43 = get_builtin_model("xai", "grok-4.3")
    assert grok_43 is not None
    assert grok_43.api == "openai-completions"


async def test_uses_responses_with_bearer_auth_and_xai_compatible_request_fields() -> None:
    model = get_builtin_model("xai", "grok-4.5")
    assert model is not None
    captured = await capture_request(
        model,
        Context(
            system_prompt="You are a careful coding assistant.",
            messages=[UserMessage(content="hello", timestamp=1)],
        ),
        OpenAIResponsesOptions(
            api_key="xai-test-token",
            session_id="pi-session-123",
            cache_retention="long",
            reasoning_effort="medium",
        ),
    )

    assert captured["url"] == "https://api.x.ai/v1/responses"
    assert captured["headers"]["authorization"] == "Bearer xai-test-token"
    assert captured["headers"]["session_id"] == "pi-session-123"

    body = captured["body"]
    assert body["model"] == "grok-4.5"
    assert body["store"] is False
    assert body["stream"] is True
    assert body["prompt_cache_key"] == "pi-session-123"
    # `toMatchObject` is a recursive partial match, so the TypeScript
    # assertion on `reasoning` only pins `effort`.
    assert body["reasoning"]["effort"] == "medium"
    assert body["include"] == ["reasoning.encrypted_content"]
    assert "prompt_cache_retention" not in body
    assert any(
        item.get("role") == "developer" and item.get("content") == "You are a careful coding assistant."
        for item in body["input"]
    )
