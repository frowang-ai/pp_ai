import json

import httpx
from pi_ai import (
    AssistantMessage,
    Context,
    ImageContent,
    Model,
    ModelCost,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pi_ai.api.pi_messages import (
    PiMessagesOptions,
    PiMessagesResponseError,
    _context_to_wire,
    _message_to_wire,
    _tool_to_wire,
    _usage_from_wire,
    _usage_to_wire,
    stream,
    stream_simple,
)


def make_model(**overrides) -> Model:
    defaults = dict(
        id="pi-backend-model",
        name="Pi Backend Model",
        api="pi-messages",
        provider="radius",
        base_url="https://radius.example.com",
        reasoning=False,
        input=["text", "image"],
        cost=ModelCost(input=1.0, output=2.0),
        context_window=100_000,
        max_tokens=8192,
    )
    defaults.update(overrides)
    return Model(**defaults)


def sse_body(events: list[dict]) -> str:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events)


def make_client(body: str, status: int = 200, capture: dict | None = None) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture["request"] = request
            capture["json"] = json.loads(request.content)
        return httpx.Response(status, text=body, headers={"content-type": "text/event-stream"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def collect(event_stream):
    events = [event async for event in event_stream]
    return events, await event_stream.result()


def start_event() -> dict:
    return {"type": "start"}


def text_start(index: int = 0) -> dict:
    return {"type": "text_start", "contentIndex": index}


def text_delta(index: int, delta: str) -> dict:
    return {"type": "text_delta", "contentIndex": index, "delta": delta}


def text_end(index: int, content: str) -> dict:
    return {"type": "text_end", "contentIndex": index, "content": content}


def done_event(reason: str = "stop", usage: dict | None = None) -> dict:
    return {
        "type": "done",
        "reason": reason,
        "usage": usage
        or {"input": 10, "output": 5, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 15, "cost": {"total": 0.01}},
    }


def error_event(reason: str = "error", error_message: str = "backend failed") -> dict:
    return {
        "type": "error",
        "reason": reason,
        "errorMessage": error_message,
        "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0},
    }


# --------------------------------------------------------------------------
# Wire serialization
# --------------------------------------------------------------------------


def test_context_to_wire_user_string_message():
    context = Context(messages=[UserMessage(content="hi", timestamp=123)])
    wire = _context_to_wire(context)
    assert wire["messages"] == [{"role": "user", "content": "hi", "timestamp": 123}]
    assert "systemPrompt" not in wire


def test_context_to_wire_includes_system_prompt_and_tools():
    context = Context(
        messages=[],
        system_prompt="be helpful",
        tools=[Tool(name="t", description="d", parameters={"type": "object", "properties": {}})],
    )
    wire = _context_to_wire(context)
    assert wire["systemPrompt"] == "be helpful"
    assert wire["tools"] == [{"name": "t", "description": "d", "parameters": {"type": "object", "properties": {}}}]


def test_message_to_wire_user_with_image_content():
    message = UserMessage(content=[TextContent(text="look"), ImageContent(data="AAAA", mime_type="image/png")])
    wire = _message_to_wire(message)
    assert wire["content"][0] == {"type": "text", "text": "look"}
    assert wire["content"][1] == {"type": "image", "data": "AAAA", "mimeType": "image/png"}


def test_message_to_wire_assistant_omits_none_fields():
    message = AssistantMessage(
        api="pi-messages", provider="radius", model="m", content=[TextContent(text="hi")], stop_reason="stop"
    )
    wire = _message_to_wire(message)
    assert wire["stopReason"] == "stop"
    assert "responseId" not in wire
    assert "errorMessage" not in wire


def test_message_to_wire_assistant_with_tool_call():
    message = AssistantMessage(
        api="pi-messages",
        provider="radius",
        model="m",
        content=[ToolCall(id="call_1", name="lookup", arguments={"q": "x"})],
        stop_reason="toolUse",
    )
    wire = _message_to_wire(message)
    assert wire["content"] == [{"type": "toolCall", "id": "call_1", "name": "lookup", "arguments": {"q": "x"}}]


def test_message_to_wire_tool_result():
    message = ToolResultMessage(tool_call_id="call_1", tool_name="lookup", content=[TextContent(text="ok")])
    wire = _message_to_wire(message)
    assert wire == {
        "role": "toolResult",
        "toolCallId": "call_1",
        "toolName": "lookup",
        "content": [{"type": "text", "text": "ok"}],
        "isError": False,
        "timestamp": message.timestamp,
    }


def test_tool_to_wire_basic():
    tool = Tool(name="t", description="d", parameters={"type": "object", "properties": {}})
    assert _tool_to_wire(tool) == {
        "name": "t",
        "description": "d",
        "parameters": {"type": "object", "properties": {}},
    }


def test_usage_round_trip():
    usage = Usage(input=10, output=5, cache_read=1, cache_write=2, total_tokens=18)
    usage.cost.total = 0.5
    wire = _usage_to_wire(usage)
    restored = _usage_from_wire(wire)
    assert restored.input == 10
    assert restored.output == 5
    assert restored.cache_read == 1
    assert restored.cache_write == 2
    assert restored.total_tokens == 18
    assert restored.cost.total == 0.5


def test_usage_from_wire_handles_missing_data():
    assert _usage_from_wire(None) == Usage()
    assert _usage_from_wire({}) == Usage()


# --------------------------------------------------------------------------
# Full streaming state machine
# --------------------------------------------------------------------------


async def test_stream_text_events():
    body = sse_body(
        [
            start_event(),
            text_start(0),
            text_delta(0, "Hel"),
            text_delta(0, "lo"),
            text_end(0, "Hello"),
            done_event(),
        ]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), PiMessagesOptions(api_key="k"), client=client)
        )

    assert events[0].type == "start"
    assert events[-1].type == "done"
    assert message.stop_reason == "stop"
    assert message.content[0].text == "Hello"
    assert message.usage.input == 10
    assert message.usage.output == 5


async def test_stream_thinking_events():
    body = sse_body(
        [
            start_event(),
            {"type": "thinking_start", "contentIndex": 0},
            {"type": "thinking_delta", "contentIndex": 0, "delta": "hmm"},
            {"type": "thinking_end", "contentIndex": 0, "content": "hmm", "redacted": False},
            done_event(),
        ]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), PiMessagesOptions(api_key="k"), client=client)
        )
    assert any(e.type == "thinking_start" for e in events)
    assert message.content[0].thinking == "hmm"
    assert message.content[0].redacted is False


async def test_stream_tool_call_events():
    body = sse_body(
        [
            start_event(),
            {"type": "toolcall_start", "contentIndex": 0, "id": "call_1", "toolName": "lookup"},
            {"type": "toolcall_delta", "contentIndex": 0, "delta": '{"q":'},
            {"type": "toolcall_delta", "contentIndex": 0, "delta": '"x"}'},
            {
                "type": "toolcall_end",
                "contentIndex": 0,
                "toolCall": {"id": "call_1", "name": "lookup", "arguments": {"q": "x"}},
            },
            done_event(reason="toolUse"),
        ]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), PiMessagesOptions(api_key="k"), client=client)
        )
    assert any(e.type == "toolcall_start" for e in events)
    tool_call_delta_events = [e for e in events if e.type == "toolcall_delta"]
    assert tool_call_delta_events[-1].partial.content[0].arguments == {"q": "x"}
    assert any(e.type == "toolcall_end" for e in events)
    assert message.stop_reason == "toolUse"
    assert message.content[0].type == "toolCall"
    assert message.content[0].arguments == {"q": "x"}


async def test_stream_backend_error_event():
    body = sse_body([start_event(), error_event(reason="error", error_message="backend exploded")])
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), PiMessagesOptions(api_key="k"), client=client)
        )
    assert events[-1].type == "error"
    assert message.stop_reason == "error"
    assert message.error_message == "backend exploded"


async def test_stream_reports_http_error():
    body = json.dumps({"error": {"message": "invalid request", "code": "bad_request"}})
    async with make_client(body, status=400) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), PiMessagesOptions(api_key="k"), client=client)
        )
    assert events[-1].type == "error"
    assert message.stop_reason == "error"
    assert "invalid request" in message.error_message
    assert "bad_request" in message.error_message
    # On an HTTP failure, the error assistant message starts fresh (no partial
    # content carried over), matching the TypeScript `createErrorEvent`.
    assert message.content == []


async def test_stream_reports_http_error_without_json_body():
    async with make_client("plain text failure", status=500) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), PiMessagesOptions(api_key="k"), client=client)
        )
    assert events[-1].type == "error"
    assert "plain text failure" in message.error_message


async def test_stream_no_api_key_reports_error():
    events, message = await collect(stream(make_model(), Context(messages=[]), PiMessagesOptions(api_key=None)))
    assert events[-1].type == "error"
    assert message.stop_reason == "error"
    assert "No API key" in message.error_message


async def test_stream_reports_aborted_when_signal_is_aborted_and_request_fails():
    # pi-messages.ts has no explicit pre-flight/post-loop abort check (unlike
    # Mistral/Anthropic): it relies solely on the fetch `AbortController`
    # rejecting the in-flight request, and the catch-all error handler then
    # reports `reason: "aborted"` by checking `signal.aborted`. This exercises
    # that same catch-all path: a transport failure while the signal happens
    # to already be aborted is reported as "aborted", not "error".
    from pi_ai.utils.abort import AbortSignal

    signal = AbortSignal()
    signal.abort()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("connection reset", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), PiMessagesOptions(api_key="k", signal=signal), client=client)
        )
    assert events[-1].type == "error"
    assert message.stop_reason == "aborted"


async def test_stream_reports_plain_error_when_request_fails_without_abort():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("connection reset", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), PiMessagesOptions(api_key="k"), client=client)
        )
    assert events[-1].type == "error"
    assert message.stop_reason == "error"


async def test_stream_ended_without_terminal_event_reports_error():
    body = sse_body([start_event(), text_start(0), text_delta(0, "partial")])
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), PiMessagesOptions(api_key="k"), client=client)
        )
    assert events[-1].type == "error"
    assert message.stop_reason == "error"
    assert "terminal event" in message.error_message


async def test_stream_sends_request_payload_and_debug_query_param():
    body = sse_body([start_event(), done_event()])
    capture: dict = {}
    async with make_client(body, capture=capture) as client:
        await collect(
            stream(
                make_model(),
                Context(messages=[UserMessage(content="hi")]),
                PiMessagesOptions(api_key="k", debug=True, reasoning="high", tool_choice="auto"),
                client=client,
            )
        )
    assert capture["request"].url.path.endswith("/messages")
    assert capture["request"].url.params.get("debug") == "1"
    assert capture["request"].headers["authorization"] == "Bearer k"
    assert capture["json"]["model"] == "pi-backend-model"
    assert capture["json"]["context"]["messages"][0]["role"] == "user"
    assert capture["json"]["context"]["messages"][0]["content"] == "hi"
    assert capture["json"]["options"]["reasoning"] == "high"
    assert capture["json"]["options"]["toolChoice"] == "auto"


async def test_stream_on_payload_hook_can_replace_payload():
    body = sse_body([start_event(), done_event()])
    capture: dict = {}

    def on_payload(payload, model):
        payload["extra_marker"] = "injected"
        return payload

    async with make_client(body, capture=capture) as client:
        await collect(
            stream(
                make_model(),
                Context(messages=[]),
                PiMessagesOptions(api_key="k", on_payload=on_payload),
                client=client,
            )
        )
    assert capture["json"]["extra_marker"] == "injected"


async def test_stream_on_response_hook_invoked_on_success():
    body = sse_body([start_event(), done_event()])
    seen: dict = {}

    def on_response(response, model):
        seen["status"] = response.status

    async with make_client(body) as client:
        await collect(
            stream(
                make_model(),
                Context(messages=[]),
                PiMessagesOptions(api_key="k", on_response=on_response),
                client=client,
            )
        )
    assert seen["status"] == 200


# --------------------------------------------------------------------------
# stream_simple
# --------------------------------------------------------------------------


async def test_stream_simple_forwards_reasoning_and_tool_choice():
    from pi_ai import SimpleStreamOptions

    body = sse_body([start_event(), done_event()])
    capture: dict = {}
    async with make_client(body, capture=capture) as client:
        await collect(
            stream_simple(
                make_model(), Context(messages=[]), SimpleStreamOptions(api_key="k", reasoning="high"), client=client
            )
        )
    assert capture["json"]["options"]["reasoning"] == "high"


async def test_stream_simple_does_not_require_api_key_up_front():
    # Unlike Mistral's `stream_simple`, pi-messages defers the missing-API-key
    # check into the async body (matching TypeScript, which has no early
    # synchronous check here).
    from pi_ai import SimpleStreamOptions

    events, message = await collect(
        stream_simple(make_model(), Context(messages=[]), SimpleStreamOptions(api_key=None))
    )
    assert events[-1].type == "error"
    assert "No API key" in message.error_message


# --------------------------------------------------------------------------
# PiMessagesResponseError
# --------------------------------------------------------------------------


def test_pi_messages_response_error_carries_code_and_details():
    error = PiMessagesResponseError("400 Bad Request: invalid request (bad_request)", "bad_request", {"status": 400})
    assert error.code == "bad_request"
    assert error.diagnostic_details == {"status": 400}
    assert str(error) == "400 Bad Request: invalid request (bad_request)"


# ==========================================================================
# Python port of `packages/ai/test/pi-messages.test.ts`.
#
# TypeScript starts a real `node:http` server; this port uses
# `httpx.MockTransport`, which observes the same request and response.
# ==========================================================================

TS_USAGE = {
    "input": 10,
    "output": 5,
    "cacheRead": 0,
    "cacheWrite": 0,
    "totalTokens": 15,
    "cost": {"input": 0.1, "output": 0.2, "cacheRead": 0, "cacheWrite": 0, "total": 0.3},
}


def ts_model() -> Model:
    return make_model(
        id="auto",
        name="Radius Auto",
        base_url="http://127.0.0.1:1/v1",
        input=["text"],
        cost=ModelCost(input=1, output=2, cache_read=0.1, cache_write=0.2),
        context_window=128000,
        max_tokens=16384,
    )


def ts_context() -> Context:
    return Context(messages=[UserMessage(content="Hello", timestamp=1)])


async def test_ts_streams_text_and_tool_calls_and_resolves_the_terminal_message():
    events_body = sse_body(
        [
            {"type": "start"},
            {"type": "text_start", "contentIndex": 0},
            {"type": "text_delta", "contentIndex": 0, "delta": "Hel"},
            {"type": "text_delta", "contentIndex": 0, "delta": "lo"},
            {"type": "text_end", "contentIndex": 0, "content": "Hello"},
            {"type": "toolcall_start", "contentIndex": 1, "id": "call_1", "toolName": "read"},
            {"type": "toolcall_delta", "contentIndex": 1, "delta": '{"path":'},
            {"type": "toolcall_delta", "contentIndex": 1, "delta": '"a.txt"}'},
            {
                "type": "toolcall_end",
                "contentIndex": 1,
                "toolCall": {"type": "toolCall", "id": "call_1", "name": "read", "arguments": {"path": "a.txt"}},
            },
            {"type": "done", "reason": "toolUse", "usage": TS_USAGE, "responseId": "resp_1"},
        ]
    )
    capture: dict = {}
    client = make_client(events_body, capture=capture)
    model = ts_model()
    context = ts_context()

    event_stream = stream(
        model,
        context,
        PiMessagesOptions(
            api_key="test-key",
            session_id="session-1",
            tool_choice="auto",
            max_tokens=100,
            headers={"x-custom": "1"},
        ),
        client=client,
    )
    events = []
    partial_stop_reasons = []
    async for event in event_stream:
        partial = getattr(event, "partial", None)
        if partial is not None:
            partial_stop_reasons.append(partial.stop_reason)
        events.append(event)
    message = await event_stream.result()

    # TypeScript asserts `partialStopReasons[0] === "pending"`. Both languages
    # attach the *same mutable* assistant message as every event's `partial`
    # (see `Object.assign(partial, ...)` in `pi-messages.ts`), so that only
    # holds because vitest's real socket lets the consumer drain each event
    # before the next arrives. `httpx.MockTransport` answers from a buffer, so
    # the producer task runs to completion first and every recorded `partial`
    # is the finished message. Assert the timing-independent part instead.
    assert events[0].type == "start"
    assert all(reason == message.stop_reason for reason in partial_stop_reasons)
    assert message.stop_reason == "toolUse"
    assert message.usage == _usage_from_wire(TS_USAGE)
    assert message.response_id == "resp_1"
    assert message.model == "auto"
    assert message.provider == "radius"
    assert message.content == [
        TextContent(text="Hello"),
        ToolCall(id="call_1", name="read", arguments={"path": "a.txt"}),
    ]
    assert any(event.type == "text_delta" for event in events)
    assert len([event for event in events if event.type == "toolcall_end"]) == 1

    request = capture["request"]
    assert request.url.path == "/v1/messages"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["x-custom"] == "1"
    assert capture["json"] == {
        "model": "auto",
        "context": _context_to_wire(context),
        "options": {"maxTokens": 100, "sessionId": "session-1", "toolChoice": "auto"},
    }


async def test_ts_appends_debug_1_and_reports_response_headers_via_on_response():
    from pi_ai import SimpleStreamOptions

    capture: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        capture["request"] = request
        return httpx.Response(
            200,
            text=sse_body([{"type": "done", "reason": "stop", "usage": TS_USAGE}]),
            headers={
                "content-type": "text/event-stream",
                "x-pi-gateway-upstream-provider": "anthropic",
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    observed: dict = {}

    def on_response(response, _model=None):
        observed["headers"] = response.headers

    options = SimpleStreamOptions(api_key="test-key", on_response=on_response)
    options.debug = True
    message = await stream_simple(ts_model(), ts_context(), options, client=client).result()

    assert message.stop_reason == "stop"
    assert capture["request"].url.path == "/v1/messages"
    assert capture["request"].url.query.decode() == "debug=1"
    assert observed["headers"]["x-pi-gateway-upstream-provider"] == "anthropic"


async def test_ts_surfaces_backend_error_responses_with_diagnostics():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            text=json.dumps({"error": {"message": "Token expired", "code": "unauthorized"}}),
            headers={"content-type": "application/json"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    message = await stream(ts_model(), ts_context(), PiMessagesOptions(api_key="stale"), client=client).result()

    assert message.stop_reason == "error"
    assert "401" in message.error_message
    assert "Token expired" in message.error_message
    assert "unauthorized" in message.error_message
    # `pi_ai.utils.diagnostics` documents that the port flattens TypeScript's
    # `{type, error, details}` diagnostic into `{kind, message, detail}`.
    assert message.diagnostics[0].kind == "pi_messages_response_failure"
    assert message.diagnostics[0].detail["status"] == 401


async def test_ts_propagates_server_sent_error_events():
    body = sse_body(
        [
            {"type": "start"},
            {"type": "error", "reason": "error", "usage": TS_USAGE, "errorMessage": "Upstream failed"},
        ]
    )
    message = await stream(
        ts_model(), ts_context(), PiMessagesOptions(api_key="test-key"), client=make_client(body)
    ).result()

    assert message.stop_reason == "error"
    assert message.error_message == "Upstream failed"
    assert message.usage == _usage_from_wire(TS_USAGE)


async def test_ts_errors_when_no_api_key_is_provided():
    message = await stream(ts_model(), ts_context(), client=make_client(sse_body([]))).result()

    assert message.stop_reason == "error"
    assert "No API key provided" in message.error_message


async def test_ts_errors_when_the_stream_ends_without_a_terminal_event():
    body = sse_body(
        [
            {"type": "start"},
            {"type": "text_start", "contentIndex": 0},
            {"type": "text_delta", "contentIndex": 0, "delta": "partial"},
        ]
    )
    message = await stream(
        ts_model(), ts_context(), PiMessagesOptions(api_key="test-key"), client=make_client(body)
    ).result()

    assert message.stop_reason == "error"
    assert "stream ended without a terminal event" in message.error_message


def test_ts_pi_messages_is_registered_as_a_builtin_api_provider():
    from pi_ai.compat import get_api_provider, register_builtin_api_providers

    register_builtin_api_providers()
    assert get_api_provider("pi-messages") is not None


# The TypeScript file's last case (`is a known api usable on models`) only
# asserts `const api: Api = "pi-messages"; expect(api).toBe("pi-messages")`,
# i.e. it is a compile-time check that the string literal is a member of the
# `Api` union. `Api` is a `Literal[...]` alias in Python and is erased at
# runtime, so the runtime half of that assertion is a tautology. The
# equivalent runtime claim -- that a model may declare `api="pi-messages"` --
# is what `make_model()` exercises in every case above.
