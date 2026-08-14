"""Python port of `packages/ai/test/mistral-http-transport.test.ts`.

TypeScript injects a `fetch` function and hands back `Response` objects. The
Python adapter talks to `httpx` directly, so the equivalent seam is an injected
`httpx.AsyncClient` with a mock transport; the payload/header/wire-format
assertions are otherwise identical.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

import httpx
from pi_ai.api.mistral_conversations import MistralOptions, stream
from pi_ai.providers.all import get_builtin_model
from pi_ai.types import (
    AssistantMessage,
    Context,
    ImageContent,
    Model,
    ProviderResponse,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pi_ai.utils.abort import AbortSignal

MODEL = get_builtin_model("mistral", "mistral-large-latest")


def sse_body(events: list[object]) -> str:
    joined = "\r\n\r\n".join(f"data: {json.dumps(event)}" for event in events)
    return f"{joined}\r\n\r\ndata: [DONE]\r\n\r\n"


def terminal_event(finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "id": "mistral-response-id",
        "model": "mistral-large-latest",
        "choices": [{"index": 0, "finish_reason": finish_reason, "delta": {}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def sse_client(
    events: list[object],
    headers: dict[str, str] | None = None,
    seen: list[httpx.Request] | None = None,
) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream", **(headers or {})},
            text=sse_body(events),
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class _ByteAtATimeStream(httpx.AsyncByteStream):
    """Delivers one byte per chunk, so SSE frames and UTF-8 runes split across reads."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for byte in self.payload:
            yield bytes([byte])


class _HangingStream(httpx.AsyncByteStream):
    """A response body that never produces a chunk.

    `entered` fires once the adapter is actually awaiting the first chunk, so a
    test can abort at exactly that point instead of guessing with a sleep or a
    bare `sleep(0)` yield, which is a scheduling assumption that can break when
    the suite runs under parallel load.
    """

    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.entered.set()
        await asyncio.Event().wait()
        yield b""


class _TimingOutTransport(httpx.AsyncBaseTransport):
    """Honors the read timeout httpx was configured with, like a real transport.

    The configured timeout is asserted rather than slept through: nothing here
    observes elapsed time, so a real `asyncio.sleep(read)` would only make the
    test slower and load-sensitive. Reading the value out of
    `request.extensions` is what actually proves `timeout_ms` reached httpx.
    """

    def __init__(self, expected_read_timeout: float) -> None:
        self.expected_read_timeout = expected_read_timeout
        self.seen_read_timeout: float | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        timeout = request.extensions.get("timeout", {})
        self.seen_read_timeout = timeout.get("read")
        assert self.seen_read_timeout == self.expected_read_timeout
        raise httpx.ReadTimeout("", request=request)


async def test_serializes_sdk_style_payloads_to_the_mistral_wire_format():
    context = Context(
        system_prompt="Be precise",
        messages=[
            UserMessage(
                content=[
                    TextContent(text="describe"),
                    ImageContent(data="aGVsbG8=", mime_type="image/png"),
                ],
                timestamp=1,
            )
        ],
        tools=[
            Tool(
                name="lookup",
                description="Look something up",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            )
        ],
    )
    requests: list[httpx.Request] = []
    callback_payload: dict[str, Any] | None = None
    callback_response: ProviderResponse | None = None

    def on_payload(payload: dict[str, Any], model: Model) -> dict[str, Any]:
        nonlocal callback_payload
        callback_payload = payload
        return {
            **payload,
            "topP": 0.9,
            "randomSeed": 42,
            "responseFormat": {
                "type": "json_schema",
                "jsonSchema": {
                    "name": "result",
                    "schemaDefinition": {"type": "object", "properties": {"maxTokens": {"type": "number"}}},
                },
            },
            "presencePenalty": 0.1,
            "frequencyPenalty": 0.2,
            "parallelToolCalls": True,
            "safePrompt": True,
        }

    def on_response(response: ProviderResponse, model: Model) -> None:
        nonlocal callback_response
        callback_response = response

    async with sse_client([terminal_event()], {"x-request-id": "request-1"}, requests) as client:
        message = await stream(
            MODEL,
            context,
            MistralOptions(
                api_key="secret",
                headers={"x-custom": "value"},
                max_tokens=123,
                prompt_mode="reasoning",
                reasoning_effort="high",
                tool_choice={"type": "function", "function": {"name": "lookup"}},
                session_id="session-1",
                on_payload=on_payload,
                on_response=on_response,
            ),
            client=client,
        ).result()

    assert message.stop_reason == "stop"
    request = requests[0]
    assert str(request.url) == "https://api.mistral.ai/v1/chat/completions"
    scheme, _, token = request.headers["authorization"].partition(" ")
    assert (scheme, token) == ("Bearer", "secret")
    assert request.headers["accept"] == "text/event-stream"
    assert request.headers["x-affinity"] == "session-1"
    assert request.headers["x-custom"] == "value"

    assert callback_payload is not None
    assert callback_payload["maxTokens"] == 123
    assert callback_payload["promptMode"] == "reasoning"
    assert callback_payload["promptCacheKey"] == "session-1"
    assert callback_response is not None
    assert callback_response.status == 200
    # TS: expect(callbackResponse).toEqual({ status: 200, headers: {"content-type": ...,
    # "x-request-id": ...} }) is full deep equality on the headers dict. That can't be
    # reproduced verbatim here: httpx.Response auto-computes and injects a
    # "content-length" header for a text body (verified: dict(httpx.Response(...).headers)
    # includes "content-length" even though only content-type/x-request-id were passed to
    # the constructor), whereas Node's Response used by the TS test does not add one. So
    # pin the full key set modulo that one known, environment-specific extra key, plus the
    # exact values TS checks.
    assert set(callback_response.headers) - {"content-length"} == {"content-type", "x-request-id"}
    assert callback_response.headers["content-type"] == "text/event-stream"
    assert callback_response.headers["x-request-id"] == "request-1"

    wire_payload = json.loads(request.content)
    assert wire_payload["max_tokens"] == 123
    assert wire_payload["prompt_mode"] == "reasoning"
    assert wire_payload["reasoning_effort"] == "high"
    assert wire_payload["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}
    assert wire_payload["prompt_cache_key"] == "session-1"
    assert wire_payload["top_p"] == 0.9
    assert wire_payload["random_seed"] == 42
    assert wire_payload["presence_penalty"] == 0.1
    assert wire_payload["frequency_penalty"] == 0.2
    assert wire_payload["parallel_tool_calls"] is True
    assert wire_payload["safe_prompt"] is True
    assert wire_payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "result",
            "schema": {"type": "object", "properties": {"maxTokens": {"type": "number"}}},
        },
    }
    assert "maxTokens" not in wire_payload
    assert "promptMode" not in wire_payload
    assert "promptCacheKey" not in wire_payload
    assert wire_payload["messages"] == [
        {"role": "system", "content": "Be precise"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": "data:image/png;base64,aGVsbG8="},
            ],
        },
    ]


async def test_serializes_assistant_thinking_tool_calls_and_tool_results_for_replay():
    context = Context(
        messages=[
            AssistantMessage(
                api="mistral-conversations",
                provider="mistral",
                model=MODEL.id,
                content=[
                    ThinkingContent(thinking="reason"),
                    TextContent(text="answer"),
                    ToolCall(id="abc123456", name="lookup", arguments={"query": "pi"}),
                ],
                usage=Usage(),
                stop_reason="toolUse",
                timestamp=1,
            ),
            ToolResultMessage(
                tool_call_id="abc123456",
                tool_name="lookup",
                content=[
                    TextContent(text="found"),
                    ImageContent(data="aGVsbG8=", mime_type="image/png"),
                ],
                is_error=False,
                timestamp=2,
            ),
        ]
    )
    requests: list[httpx.Request] = []

    async with sse_client([terminal_event()], None, requests) as client:
        message = await stream(MODEL, context, MistralOptions(api_key="test"), client=client).result()

    assert message.stop_reason == "stop"
    wire_payload = json.loads(requests[0].content)
    assert wire_payload["messages"] == [
        {
            "role": "assistant",
            "prefix": False,
            "content": [
                {"type": "thinking", "thinking": [{"type": "text", "text": "reason"}]},
                {"type": "text", "text": "answer"},
            ],
            "tool_calls": [
                {
                    "id": "abc123456",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"query":"pi"}'},
                    "index": 0,
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "abc123456",
            "name": "lookup",
            "content": [
                {"type": "text", "text": "found"},
                {"type": "image_url", "image_url": "data:image/png;base64,aGVsbG8="},
            ],
        },
    ]


async def test_parses_native_thinking_text_tool_calls_and_cached_token_usage():
    context = Context(messages=[UserMessage(content="hello", timestamp=1)])
    events: list[object] = [
        {
            "id": "response-1",
            "model": MODEL.id,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "delta": {"content": [{"type": "thinking", "thinking": [{"type": "text", "text": "reason"}]}]},
                }
            ],
        },
        {
            "id": "response-1",
            "model": MODEL.id,
            "choices": [
                {"index": 0, "finish_reason": None, "delta": {"content": [{"type": "text", "text": "answer"}]}}
            ],
        },
        {
            "id": "response-1",
            "model": MODEL.id,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "delta": {
                        "tool_calls": [
                            {"id": "abc123456", "index": 0, "function": {"name": "lookup", "arguments": '{"query":'}}
                        ]
                    },
                }
            ],
        },
        {
            "id": "response-1",
            "model": MODEL.id,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "delta": {
                        "tool_calls": [
                            {"id": "abc123456", "index": 0, "function": {"name": "lookup", "arguments": '"pi"}'}}
                        ]
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
                "prompt_tokens_details": {"cached_tokens": 3},
            },
        },
    ]

    async with sse_client(events) as client:
        message = await stream(MODEL, context, MistralOptions(api_key="test"), client=client).result()

    assert message.stop_reason == "toolUse"
    assert message.raw_stop_reason == "tool_calls"
    assert message.response_id == "response-1"
    assert message.content == [
        ThinkingContent(thinking="reason"),
        TextContent(text="answer"),
        ToolCall(id="abc123456", name="lookup", arguments={"query": "pi"}),
    ]
    assert message.usage.input == 7
    assert message.usage.output == 4
    assert message.usage.cache_read == 3
    assert message.usage.cache_write == 0
    assert message.usage.total_tokens == 14


async def test_parses_sse_and_utf8_sequences_split_across_transport_chunks():
    context = Context(messages=[UserMessage(content="hello", timestamp=1)])
    event = {
        "id": "response-bytewise",
        "model": MODEL.id,
        "choices": [{"index": 0, "finish_reason": "stop", "delta": {"content": "héllo 🌍"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    payload = f"data: {json.dumps(event)}\r\n\r\ndata: [DONE]\r\n\r\n".encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_ByteAtATimeStream(payload))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        message = await stream(MODEL, context, MistralOptions(api_key="test"), client=client).result()

    assert message.stop_reason == "stop"
    assert message.content == [TextContent(text="héllo 🌍")]


async def test_honors_case_insensitive_header_overrides_and_explicit_affinity_suppression():
    # TS uses `Authorization: "Bearer model-key"` as the model's default header value
    # (verified against the raw source bytes, since the CLI's redaction filter displays
    # any auth-looking string as asterisks). Match that input exactly, not just its shape.
    model = replace(MODEL, headers={"Authorization": " ".join(["Bearer", "model-key"]), "X-Affinity": "model-affinity"})
    context = Context(messages=[UserMessage(content="hello", timestamp=1)])
    requests: list[httpx.Request] = []

    async with sse_client([terminal_event()], None, requests) as client:
        await stream(
            model,
            context,
            MistralOptions(
                api_key="request-key",
                session_id="automatic-affinity",
                headers={"authorization": None, "x-affinity": None},
            ),
            client=client,
        ).result()

    assert "authorization" not in requests[0].headers
    assert "x-affinity" not in requests[0].headers


async def test_aborts_while_waiting_for_an_sse_chunk():
    context = Context(messages=[UserMessage(content="hello", timestamp=1)])
    signal = AbortSignal()
    body = _HangingStream()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = stream(MODEL, context, MistralOptions(api_key="test", signal=signal), client=client).result()
        await asyncio.wait_for(body.entered.wait(), timeout=5)
        signal.abort()
        message = await result

    assert message.stop_reason == "aborted"


async def test_applies_the_request_timeout_while_waiting_for_an_sse_chunk():
    context = Context(messages=[UserMessage(content="hello", timestamp=1)])

    transport = _TimingOutTransport(expected_read_timeout=0.005)
    async with httpx.AsyncClient(transport=transport) as client:
        message = await stream(MODEL, context, MistralOptions(api_key="test", timeout_ms=5), client=client).result()

    assert transport.seen_read_timeout == 0.005
    assert message.stop_reason == "error"
    assert message.error_message is not None
    assert "timeout" in message.error_message.lower()


async def test_preserves_http_status_and_response_bodies_in_errors():
    context = Context(messages=[UserMessage(content="hello", timestamp=1)])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text='{"message":"blocked by gateway"}')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        message = await stream(MODEL, context, MistralOptions(api_key="test"), client=client).result()

    assert message.stop_reason == "error"
    assert message.error_message == 'Mistral API error (403): {"message":"blocked by gateway"}'
