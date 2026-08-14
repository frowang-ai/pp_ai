"""Python port of `packages/ai/test/openai-completions-reasoning-details.test.ts`.

The TypeScript test mocks the `openai` SDK to serve two scripted chunk sets and
record the payloads. The port serves the same chunks as SSE bodies through an
`httpx.MockTransport` and records the payloads with `on_payload`.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from pi_ai.api.openai_completions import OpenAICompletionsOptions, stream
from pi_ai.types import AssistantMessage, Context, Model, ModelCost, Tool, ToolCall
from pi_ai.utils.json_stringify import json_stringify

REASONING_DETAIL = {"type": "reasoning.encrypted", "id": "call_1", "data": "encrypted-signature"}

READ_TOOL = Tool(
    name="read",
    description="Read a file",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
)


def make_model() -> Model:
    return Model(
        id="google/gemini-test",
        name="Gemini Test",
        api="openai-completions",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=100_000,
        max_tokens=4096,
    )


def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "model": "google/gemini-test",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def tool_call_chunk() -> dict[str, Any]:
    return chunk(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path":"README.md"}'},
                }
            ]
        }
    )


def sse_client(chunks: list[dict[str, Any]], payloads: list[dict[str, Any]]) -> httpx.AsyncClient:
    body = "".join(f"data: {json.dumps(item)}\n\n" for item in chunks) + "data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def run_stream(
    chunks: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    messages: list[AssistantMessage] | None = None,
) -> AssistantMessage:
    return await stream(
        make_model(),
        Context(messages=list(messages or []), tools=[READ_TOOL]),
        OpenAICompletionsOptions(api_key="test"),
        client=sse_client(chunks, payloads),
    ).result()


def get_assistant_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    return next((message for message in payload.get("messages", []) if message.get("role") == "assistant"), None)


async def test_preserves_reasoning_details_that_arrive_before_their_matching_tool_call() -> None:
    payloads: list[dict[str, Any]] = []

    assistant_message = await run_stream(
        [chunk({"reasoning_details": [REASONING_DETAIL]}), tool_call_chunk(), chunk({}, "tool_calls")],
        payloads,
    )
    tool_call = next((block for block in assistant_message.content if block.type == "toolCall"), None)
    assert isinstance(tool_call, ToolCall)
    assert tool_call.id == "call_1"
    assert tool_call.name == "read"
    assert tool_call.arguments == {"path": "README.md"}
    assert tool_call.thought_signature == json_stringify(REASONING_DETAIL)

    await run_stream([chunk({"content": "ok"}), chunk({}, "stop")], payloads, [assistant_message])

    assert get_assistant_payload(payloads[1])["reasoning_details"] == [REASONING_DETAIL]
