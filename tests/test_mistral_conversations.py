import json

import httpx
import pytest
from pi_ai import (
    AssistantMessage,
    Context,
    ImageContent,
    Model,
    ModelCost,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from pi_ai.api.mistral_conversations import (
    MistralOptions,
    build_chat_payload,
    build_mistral_headers,
    build_tool_result_text,
    create_mistral_tool_call_id_normalizer,
    derive_mistral_tool_call_id,
    format_mistral_error,
    map_chat_stop_reason,
    map_tool_choice,
    stream,
    stream_simple,
    to_chat_messages,
    to_function_tools,
    to_mistral_wire_content_chunk,
    to_mistral_wire_message,
    to_mistral_wire_payload,
)
from pi_ai.providers.mistral import MISTRAL_API_KEY_ENV, mistral_provider
from pi_ai.types import JsonSchemaConstrainedSampling
from pi_ai.utils.http import ProviderHttpError


def make_model(**overrides) -> Model:
    defaults = dict(
        id="mistral-large-latest",
        name="Mistral Large",
        api="mistral-conversations",
        provider="mistral",
        base_url="https://api.mistral.ai",
        reasoning=False,
        input=["text", "image"],
        cost=ModelCost(input=0.50, output=1.50),
        context_window=256_000,
        max_tokens=8192,
    )
    defaults.update(overrides)
    return Model(**defaults)


def sse_body(events: list[dict]) -> str:
    lines = []
    for data in events:
        payload = "[DONE]" if data == "[DONE]" else json.dumps(data)
        lines.append(f"data: {payload}\n\n")
    return "".join(lines)


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


def chunk(
    delta: dict | None = None, finish_reason: str | None = None, usage: dict | None = None, chunk_id: str = "cmpl-1"
) -> dict:
    choice: dict = {"index": 0, "delta": delta or {}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    result: dict = {"id": chunk_id, "choices": [choice]}
    if usage is not None:
        result["usage"] = usage
    return result


# --------------------------------------------------------------------------
# Message conversion
# --------------------------------------------------------------------------


def test_to_chat_messages_user_string_content():
    messages = [UserMessage(content="hello")]
    result = to_chat_messages(messages, supports_images=True)
    assert result == [{"role": "user", "content": "hello"}]


def test_to_chat_messages_user_with_images_supported():
    messages = [UserMessage(content=[TextContent(text="look"), ImageContent(data="AAAA", mime_type="image/png")])]
    result = to_chat_messages(messages, supports_images=True)
    assert result[0]["content"][0] == {"type": "text", "text": "look"}
    assert result[0]["content"][1] == {"type": "image_url", "imageUrl": "data:image/png;base64,AAAA"}


def test_to_chat_messages_user_images_omitted_when_unsupported():
    messages = [UserMessage(content=[ImageContent(data="AAAA", mime_type="image/png")])]
    result = to_chat_messages(messages, supports_images=False)
    assert result == [{"role": "user", "content": "(image omitted: model does not support images)"}]


def test_to_chat_messages_assistant_text_and_tool_call():
    assistant = AssistantMessage(
        content=[TextContent(text="hi"), ToolCall(id="call_1", name="lookup", arguments={"q": "x"})],
        api="mistral-conversations",
        provider="mistral",
        model="mistral-large-latest",
        stop_reason="toolUse",
    )
    result = to_chat_messages([assistant], supports_images=True)
    assert result[0]["role"] == "assistant"
    assert result[0]["content"] == [{"type": "text", "text": "hi"}]
    assert result[0]["toolCalls"][0]["id"] == "call_1"
    assert json.loads(result[0]["toolCalls"][0]["function"]["arguments"]) == {"q": "x"}


def test_to_chat_messages_assistant_thinking_block():
    assistant = AssistantMessage(
        content=[ThinkingContent(thinking="pondering")],
        api="mistral-conversations",
        provider="mistral",
        model="mistral-large-latest",
        stop_reason="stop",
    )
    result = to_chat_messages([assistant], supports_images=True)
    assert result[0]["content"] == [{"type": "thinking", "thinking": [{"type": "text", "text": "pondering"}]}]


def test_to_chat_messages_tool_result_text():
    tool_result = ToolResultMessage(tool_call_id="call_1", tool_name="lookup", content=[TextContent(text="result")])
    result = to_chat_messages([tool_result], supports_images=True)
    assert result[0] == {
        "role": "tool",
        "toolCallId": "call_1",
        "name": "lookup",
        "content": [{"type": "text", "text": "result"}],
    }


def test_to_chat_messages_tool_result_error_prefix():
    tool_result = ToolResultMessage(
        tool_call_id="call_1", tool_name="lookup", content=[TextContent(text="boom")], is_error=True
    )
    result = to_chat_messages([tool_result], supports_images=True)
    assert result[0]["content"][0]["text"] == "[tool error] boom"


def test_to_chat_messages_tool_result_image_included_when_supported():
    tool_result = ToolResultMessage(
        tool_call_id="call_1",
        tool_name="lookup",
        content=[TextContent(text="see"), ImageContent(data="BBBB", mime_type="image/jpeg")],
    )
    result = to_chat_messages([tool_result], supports_images=True)
    assert result[0]["content"][1] == {"type": "image_url", "imageUrl": "data:image/jpeg;base64,BBBB"}


def test_build_tool_result_text_no_output():
    assert build_tool_result_text("", has_images=False, supports_images=True, is_error=False) == "(no tool output)"
    assert (
        build_tool_result_text("", has_images=False, supports_images=True, is_error=True)
        == "[tool error] (no tool output)"
    )


def test_build_tool_result_text_image_only():
    assert build_tool_result_text("", has_images=True, supports_images=True, is_error=False) == "(see attached image)"
    assert (
        build_tool_result_text("", has_images=True, supports_images=False, is_error=False)
        == "(image omitted: model does not support images)"
    )


# --------------------------------------------------------------------------
# Tool schema conversion
# --------------------------------------------------------------------------


def test_to_function_tools_basic():
    tools = [Tool(name="lookup", description="Look something up", parameters={"type": "object", "properties": {}})]
    result = to_function_tools(tools)
    assert result == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look something up",
                "parameters": {"type": "object", "properties": {}},
                "strict": False,
            },
        }
    ]


def test_to_function_tools_strict_json_schema():
    tools = [
        Tool(
            name="lookup",
            description="d",
            parameters={"type": "object", "properties": {}},
            constrained_sampling=JsonSchemaConstrainedSampling(strict="require"),
        )
    ]
    result = to_function_tools(tools)
    assert result[0]["function"]["strict"] is True


# --------------------------------------------------------------------------
# Tool call id normalization
# --------------------------------------------------------------------------


def test_derive_mistral_tool_call_id_alnum_length_match():
    derived = derive_mistral_tool_call_id("abcdefghi", 0)
    assert derived == "abcdefghi"
    assert len(derived) == 9


def test_derive_mistral_tool_call_id_hashes_when_length_mismatch():
    derived = derive_mistral_tool_call_id("short", 0)
    assert len(derived) == 9


def test_tool_call_id_normalizer_is_stable_and_bijective():
    normalize = create_mistral_tool_call_id_normalizer()
    first = normalize("toolu_01ABCDEFG")
    again = normalize("toolu_01ABCDEFG")
    assert first == again


def test_tool_call_id_normalizer_resolves_collisions():
    normalize = create_mistral_tool_call_id_normalizer()
    # Two different original ids that could hash-collide must resolve to
    # distinct normalized ids.
    first = normalize("call-aaaaaaaaa")
    second = normalize("call-aaaaaaaab")
    assert first != second


# --------------------------------------------------------------------------
# Header / URL construction
# --------------------------------------------------------------------------


def test_build_mistral_headers_basic():
    headers = build_mistral_headers(make_model(), "sk-test")
    assert headers["authorization"] == "Bearer sk-test"
    assert headers["accept"] == "text/event-stream"
    assert "x-affinity" not in headers


def test_build_mistral_headers_model_header_override():
    model = make_model(headers={"x-custom": "abc"})
    headers = build_mistral_headers(model, "sk-test")
    assert headers["x-custom"] == "abc"


def test_build_mistral_headers_options_header_removal():
    model = make_model(headers={"x-custom": "abc"})
    options = MistralOptions(api_key="sk-test", headers={"x-custom": None})
    headers = build_mistral_headers(model, "sk-test", options)
    assert "x-custom" not in headers


def test_build_mistral_headers_affinity_set_with_session_and_caching():
    options = MistralOptions(api_key="sk-test", session_id="sess-1", cache_retention="short")
    headers = build_mistral_headers(make_model(), "sk-test", options)
    assert headers["x-affinity"] == "sess-1"


def test_build_mistral_headers_affinity_skipped_when_cache_retention_none():
    options = MistralOptions(api_key="sk-test", session_id="sess-1", cache_retention="none")
    headers = build_mistral_headers(make_model(), "sk-test", options)
    assert "x-affinity" not in headers


def test_build_mistral_headers_explicit_affinity_override_wins():
    options = MistralOptions(api_key="sk-test", session_id="sess-1", headers={"x-affinity": "manual"})
    headers = build_mistral_headers(make_model(), "sk-test", options)
    assert headers["x-affinity"] == "manual"


# --------------------------------------------------------------------------
# Wire payload remapping
# --------------------------------------------------------------------------


def test_to_mistral_wire_payload_remaps_top_level_keys():
    payload = {
        "model": "mistral-large-latest",
        "stream": True,
        "maxTokens": 100,
        "toolChoice": "auto",
        "messages": [{"role": "user", "content": "hi"}],
    }
    wire = to_mistral_wire_payload(payload)
    assert wire["max_tokens"] == 100
    assert wire["tool_choice"] == "auto"
    assert "maxTokens" not in wire
    assert "toolChoice" not in wire


def test_to_mistral_wire_message_remaps_tool_calls():
    message = {"role": "assistant", "toolCalls": [{"id": "1"}]}
    wire = to_mistral_wire_message(message)
    assert wire["tool_calls"] == [{"id": "1"}]
    assert "toolCalls" not in wire


def test_to_mistral_wire_message_remaps_tool_call_id():
    message = {"role": "tool", "toolCallId": "call_1", "content": []}
    wire = to_mistral_wire_message(message)
    assert wire["tool_call_id"] == "call_1"


def test_to_mistral_wire_content_chunk_remaps_image_url():
    chunk_dict = {"type": "image_url", "imageUrl": "data:image/png;base64,AAAA"}
    wire = to_mistral_wire_content_chunk(chunk_dict)
    assert wire["image_url"] == "data:image/png;base64,AAAA"
    assert "imageUrl" not in wire


def test_to_mistral_wire_payload_remaps_content_chunks_in_messages():
    payload = {
        "model": "m",
        "stream": True,
        "messages": [{"role": "user", "content": [{"type": "image_url", "imageUrl": "data:x;base64,Y"}]}],
    }
    wire = to_mistral_wire_payload(payload)
    assert wire["messages"][0]["content"][0]["image_url"] == "data:x;base64,Y"


# --------------------------------------------------------------------------
# Payload building
# --------------------------------------------------------------------------


def test_build_chat_payload_includes_system_prompt_and_tools():
    model = make_model()
    context = Context(
        messages=[UserMessage(content="hi")],
        system_prompt="be nice",
        tools=[Tool(name="t", description="d", parameters={"type": "object", "properties": {}})],
    )
    payload = build_chat_payload(model, context, context.messages, MistralOptions(api_key="k"))
    assert payload["messages"][0] == {"role": "system", "content": "be nice"}
    assert payload["tools"][0]["function"]["name"] == "t"


def test_build_chat_payload_prompt_cache_key_set():
    model = make_model()
    context = Context(messages=[UserMessage(content="hi")])
    options = MistralOptions(api_key="k", session_id="sess-1", cache_retention="short")
    payload = build_chat_payload(model, context, context.messages, options)
    assert payload["promptCacheKey"] == "sess-1"


# --------------------------------------------------------------------------
# Stop reason / tool choice mapping
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected_reason", "has_error"),
    [
        ("stop", "stop", False),
        ("length", "length", False),
        ("model_length", "length", False),
        ("tool_calls", "toolUse", False),
        ("error", "error", True),
        ("something_else", "error", True),
        (None, "stop", False),
    ],
)
def test_map_chat_stop_reason(raw, expected_reason, has_error):
    reason, error_message = map_chat_stop_reason(raw)
    assert reason == expected_reason
    assert (error_message is not None) == has_error


def test_map_tool_choice_passthrough_literals():
    assert map_tool_choice("auto") == "auto"
    assert map_tool_choice("none") == "none"
    assert map_tool_choice(None) is None


def test_map_tool_choice_function_object():
    choice = {"type": "function", "function": {"name": "lookup"}}
    assert map_tool_choice(choice) == {"type": "function", "function": {"name": "lookup"}}


# --------------------------------------------------------------------------
# Error formatting
# --------------------------------------------------------------------------


def test_format_mistral_error_with_body():
    error = ProviderHttpError(400, "bad request")
    assert format_mistral_error(error) == "Mistral API error (400): bad request"


def test_format_mistral_error_without_body():
    error = ProviderHttpError(500, "")
    assert "Mistral API error (500)" in format_mistral_error(error)


def test_format_mistral_error_generic_exception():
    assert format_mistral_error(ValueError("oops")) == "oops"


# --------------------------------------------------------------------------
# Full streaming state machine
# --------------------------------------------------------------------------


async def test_stream_text_only():
    body = sse_body(
        [
            chunk({"content": "Hel"}),
            chunk({"content": "lo"}),
            chunk({}, finish_reason="stop", usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}),
            "[DONE]",
        ]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(
                make_model(), Context(messages=[UserMessage(content="hi")]), MistralOptions(api_key="k"), client=client
            )
        )

    assert events[0].type == "start"
    assert any(e.type == "text_start" for e in events)
    assert any(e.type == "text_delta" for e in events)
    assert events[-1].type == "done"
    assert message.stop_reason == "stop"
    assert message.content[0].text == "Hello"
    assert message.usage.input == 10
    assert message.usage.output == 2


async def test_stream_thinking_content():
    body = sse_body(
        [
            chunk({"content": [{"type": "thinking", "thinking": [{"text": "hmm"}]}]}),
            chunk({}, finish_reason="stop", usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}),
            "[DONE]",
        ]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), MistralOptions(api_key="k"), client=client)
        )

    assert any(e.type == "thinking_start" for e in events)
    assert message.content[0].thinking == "hmm"


async def test_stream_tool_call_accumulation():
    body = sse_body(
        [
            chunk({"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "lookup", "arguments": '{"q":'}}]}),
            chunk({"tool_calls": [{"index": 0, "id": "call_1", "function": {"arguments": '"x"}'}}]}),
            chunk(
                {}, finish_reason="tool_calls", usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
            ),
            "[DONE]",
        ]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), MistralOptions(api_key="k"), client=client)
        )

    assert any(e.type == "toolcall_start" for e in events)
    assert any(e.type == "toolcall_end" for e in events)
    assert message.stop_reason == "toolUse"
    tool_call = message.content[0]
    assert tool_call.type == "toolCall"
    assert tool_call.name == "lookup"
    assert tool_call.arguments == {"q": "x"}


async def test_stream_reports_http_error():
    async with make_client("not found", status=404) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), MistralOptions(api_key="k"), client=client)
        )
    assert events[-1].type == "error"
    assert message.stop_reason == "error"
    assert "Mistral API error (404)" in message.error_message


async def test_stream_no_api_key_reports_error():
    events, message = await collect(stream(make_model(), Context(messages=[]), MistralOptions(api_key=None)))
    assert events[-1].type == "error"
    assert message.stop_reason == "error"
    assert "No API key" in message.error_message


async def test_stream_reports_error_when_signal_already_aborted():
    from pi_ai.utils.abort import AbortSignal

    signal = AbortSignal()
    signal.abort()
    body = sse_body([chunk({}, finish_reason="stop", usage={"prompt_tokens": 1, "completion_tokens": 1}), "[DONE]"])
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), MistralOptions(api_key="k", signal=signal), client=client)
        )
    assert events[-1].type == "error"
    assert message.stop_reason == "aborted"


async def test_stream_invalid_event_shape_reports_error():
    body = 'data: {"not": "a choices array"}\n\n'
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), MistralOptions(api_key="k"), client=client)
        )
    assert events[-1].type == "error"
    assert message.stop_reason == "error"


async def test_stream_sends_request_payload():
    body = sse_body([chunk({}, finish_reason="stop", usage={"prompt_tokens": 1, "completion_tokens": 1}), "[DONE]"])
    capture: dict = {}
    async with make_client(body, capture=capture) as client:
        await collect(
            stream(
                make_model(),
                Context(messages=[UserMessage(content="hi")]),
                MistralOptions(api_key="k"),
                client=client,
            )
        )
    assert capture["json"]["model"] == "mistral-large-latest"
    assert capture["json"]["stream"] is True
    assert capture["request"].url.path.endswith("/v1/chat/completions")
    assert capture["request"].headers["authorization"] == "Bearer k"


# --------------------------------------------------------------------------
# stream_simple reasoning routing
# --------------------------------------------------------------------------


async def test_stream_simple_uses_reasoning_effort_for_small_model():
    from pi_ai import SimpleStreamOptions

    model = make_model(id="mistral-small-2603", reasoning=True, thinking_level_map={"high": "high", "low": "none"})
    body = sse_body([chunk({}, finish_reason="stop", usage={"prompt_tokens": 1, "completion_tokens": 1}), "[DONE]"])
    capture: dict = {}
    async with make_client(body, capture=capture) as client:
        await collect(
            stream_simple(
                model, Context(messages=[]), SimpleStreamOptions(api_key="k", reasoning="high"), client=client
            )
        )
    assert capture["json"]["reasoning_effort"] == "high"
    assert "prompt_mode" not in capture["json"]


async def test_stream_simple_uses_prompt_mode_for_magistral():
    from pi_ai import SimpleStreamOptions

    model = make_model(id="magistral-medium-latest", reasoning=True)
    body = sse_body([chunk({}, finish_reason="stop", usage={"prompt_tokens": 1, "completion_tokens": 1}), "[DONE]"])
    capture: dict = {}
    async with make_client(body, capture=capture) as client:
        await collect(
            stream_simple(
                model, Context(messages=[]), SimpleStreamOptions(api_key="k", reasoning="high"), client=client
            )
        )
    assert capture["json"]["prompt_mode"] == "reasoning"
    assert "reasoning_effort" not in capture["json"]


def test_stream_simple_no_api_key_raises():
    from pi_ai import SimpleStreamOptions

    with pytest.raises(ValueError, match="No API key"):
        stream_simple(make_model(), Context(messages=[]), SimpleStreamOptions(api_key=None))


# --------------------------------------------------------------------------
# Provider factory
# --------------------------------------------------------------------------


def test_mistral_provider_metadata():
    provider = mistral_provider()
    assert provider.id == "mistral"
    assert provider.base_url == "https://api.mistral.ai"
    assert all(model.provider == "mistral" for model in provider.models)
    assert all(model.base_url == "https://api.mistral.ai" for model in provider.models)


async def test_mistral_provider_resolves_api_key_from_env(monkeypatch):
    from pi_ai.auth.helpers import resolve_api_key_auth

    monkeypatch.setenv(MISTRAL_API_KEY_ENV, "env-mistral-key")
    provider = mistral_provider()
    result = await resolve_api_key_auth(provider.auth.api_key)
    assert result is not None
    assert result.auth.api_key == "env-mistral-key"
    assert result.source == MISTRAL_API_KEY_ENV
