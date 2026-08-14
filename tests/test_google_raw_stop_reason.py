"""Python port of `packages/ai/test/google-raw-stop-reason.test.ts`.

TypeScript mocks `@google/genai` so `generateContentStream` yields one chunk
carrying only a `finishReason`. This port speaks the REST API directly, so the
same chunk is served as an SSE body over an `httpx.MockTransport`.
"""

from __future__ import annotations

import json

import httpx
from pi_ai.api.google_generative_ai import GoogleOptions
from pi_ai.api.google_generative_ai import stream as stream_google_generative_ai
from pi_ai.api.google_vertex import GoogleVertexOptions
from pi_ai.api.google_vertex import stream as stream_google_vertex
from pi_ai.providers.all import get_builtin_model
from pi_ai.types import Context, UserMessage

CONTEXT = Context(messages=[UserMessage(content="hello")])


def make_client(finish_reason: str) -> httpx.AsyncClient:
    chunk = {
        "responseId": "google-response-id",
        "candidates": [{"finishReason": finish_reason}],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 0, "totalTokenCount": 1},
    }
    body = f"data: {json.dumps(chunk)}\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_preserves_raw_gemini_finish_reasons_for_google_generative_ai_errors():
    model = get_builtin_model("google", "gemini-2.5-flash")
    event_stream = stream_google_generative_ai(
        model,
        CONTEXT,
        GoogleOptions(api_key="test-api-key"),
        client=make_client("MALFORMED_FUNCTION_CALL"),
    )

    message = await event_stream.result()

    assert message.stop_reason == "error"
    assert message.raw_stop_reason == "MALFORMED_FUNCTION_CALL"
    assert message.error_message == "Provider stopped with: MALFORMED_FUNCTION_CALL"


async def test_preserves_raw_gemini_finish_reasons_for_google_vertex_errors():
    model = get_builtin_model("google-vertex", "gemini-3-flash-preview")
    # TypeScript passes only `{project, location}`: its mocked `GoogleGenAI` class
    # never authenticates. The Python adapter really builds the request, so it needs
    # an access token to reach the transport. Every assertion below is unchanged.
    event_stream = stream_google_vertex(
        model,
        CONTEXT,
        GoogleVertexOptions(project="test-project", location="us-central1", access_token="test-token"),
        client=make_client("SAFETY"),
    )

    message = await event_stream.result()

    assert message.stop_reason == "error"
    assert message.raw_stop_reason == "SAFETY"
    assert message.error_message == "Provider stopped with: SAFETY"
