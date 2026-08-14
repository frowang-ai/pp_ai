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
from pi_ai.api.google_generative_ai import (
    GoogleOptions,
    GoogleThinkingOptions,
    build_headers,
    build_params,
    build_url,
    stream,
    stream_simple,
)
from pi_ai.api.google_shared import (
    convert_messages,
    convert_tools,
    get_gemini_major_version,
    is_gemini3_flash_model,
    is_gemini3_pro_model,
    is_thinking_part,
    map_stop_reason,
    map_stop_reason_string,
    map_tool_choice,
    requires_tool_call_id,
    resolve_google_function_calling_mode,
    resolve_thought_signature,
    retain_thought_signature,
    supports_google_strict_tool_sampling,
    supports_multimodal_function_response,
)
from pi_ai.providers.google import google_provider
from pi_ai.types import JsonSchemaConstrainedSampling


def make_model(**overrides) -> Model:
    defaults = dict(
        id="gemini-2.5-flash",
        name="Gemini 2.5 Flash",
        api="google-generative-ai",
        provider="google",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        reasoning=False,
        input=["text", "image"],
        cost=ModelCost(input=0.3, output=2.5),
        context_window=1_048_576,
        max_tokens=65_536,
    )
    defaults.update(overrides)
    return Model(**defaults)


def sse_body(chunks: list[dict]) -> str:
    """Gemini's SSE stream carries unlabelled `data:` lines (no `event:` field)."""
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)


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


def text_chunk(text: str, thought: bool = False, thought_signature: str | None = None, **extra) -> dict:
    part: dict = {"text": text}
    if thought:
        part["thought"] = True
    if thought_signature:
        part["thoughtSignature"] = thought_signature
    chunk = {"candidates": [{"content": {"role": "model", "parts": [part]}}]}
    chunk["candidates"][0].update(extra)
    return chunk


def finish_chunk(reason: str, usage: dict | None = None) -> dict:
    chunk: dict = {"candidates": [{"finishReason": reason}]}
    if usage:
        chunk["usageMetadata"] = usage
    return chunk


# --------------------------------------------------------------------------
# thought signature helpers
# --------------------------------------------------------------------------


def test_is_thinking_part():
    assert is_thinking_part({"thought": True}) is True
    assert is_thinking_part({"thought": False}) is False
    assert is_thinking_part({}) is False


def test_retain_thought_signature_keeps_last_non_empty():
    assert retain_thought_signature(None, "sig1") == "sig1"
    assert retain_thought_signature("sig1", None) == "sig1"
    assert retain_thought_signature("sig1", "") == "sig1"
    assert retain_thought_signature("sig1", "sig2") == "sig2"


def test_resolve_thought_signature_requires_same_model_and_valid_base64():
    assert resolve_thought_signature(True, "QUJD") == "QUJD"
    assert resolve_thought_signature(False, "QUJD") is None
    assert resolve_thought_signature(True, "not base64!!") is None
    assert resolve_thought_signature(True, None) is None


# --------------------------------------------------------------------------
# model id helpers
# --------------------------------------------------------------------------


def test_requires_tool_call_id():
    assert requires_tool_call_id("claude-opus-5") is True
    assert requires_tool_call_id("gpt-oss-120b") is True
    assert requires_tool_call_id("gemini-3-pro") is True
    assert requires_tool_call_id("gemini-2.5-flash") is False


def test_get_gemini_major_version():
    assert get_gemini_major_version("gemini-3-pro") == 3
    assert get_gemini_major_version("gemini-2.5-flash") == 2
    assert get_gemini_major_version("gemini-live-3-flash") == 3
    assert get_gemini_major_version("claude-opus-5") is None


def test_supports_multimodal_function_response():
    assert supports_multimodal_function_response("gemini-3-pro") is True
    assert supports_multimodal_function_response("gemini-2.5-flash") is False
    assert supports_multimodal_function_response("claude-opus-5") is True


def test_is_gemini3_pro_and_flash_model():
    assert is_gemini3_pro_model("gemini-3-pro") is True
    assert is_gemini3_pro_model("gemini-3.5-pro") is True
    assert is_gemini3_pro_model("gemini-2.5-pro") is False
    assert is_gemini3_flash_model("gemini-3-flash") is True
    assert is_gemini3_flash_model("gemini-flash-latest") is True
    assert is_gemini3_flash_model("gemini-2.5-flash") is False


# --------------------------------------------------------------------------
# stop reason mapping
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("STOP", "stop"),
        ("MAX_TOKENS", "length"),
        ("SAFETY", "error"),
        ("RECITATION", "error"),
        ("PROHIBITED_CONTENT", "error"),
        ("SOME_FUTURE_REASON", "error"),
    ],
)
def test_map_stop_reason(reason, expected):
    assert map_stop_reason(reason) == expected


def test_map_stop_reason_string_matches_map_stop_reason():
    assert map_stop_reason_string("STOP") == "stop"
    assert map_stop_reason_string("MAX_TOKENS") == "length"
    assert map_stop_reason_string("ANYTHING_ELSE") == "error"


# --------------------------------------------------------------------------
# message conversion
# --------------------------------------------------------------------------


def test_convert_messages_user_string_content():
    contents = convert_messages(make_model(), Context(messages=[UserMessage(content="hi")]))
    assert contents == [{"role": "user", "parts": [{"text": "hi"}]}]


def test_convert_messages_user_content_with_image():
    context = Context(
        messages=[UserMessage(content=[TextContent(text="look"), ImageContent(data="AAA", mime_type="image/png")])]
    )
    contents = convert_messages(make_model(), context)
    parts = contents[0]["parts"]
    assert parts[0] == {"text": "look"}
    assert parts[1] == {"inlineData": {"mimeType": "image/png", "data": "AAA"}}


def test_convert_messages_assistant_text_and_tool_call():
    assistant = AssistantMessage(
        api="google-generative-ai",
        provider="google",
        model="gemini-2.5-flash",
        content=[TextContent(text="calling"), ToolCall(id="t1", name="read", arguments={"path": "a.txt"})],
        stop_reason="toolUse",
    )
    contents = convert_messages(make_model(), Context(messages=[assistant]))
    parts = contents[0]["parts"]
    assert parts[0] == {"text": "calling"}
    assert parts[1]["functionCall"]["name"] == "read"
    assert parts[1]["functionCall"]["args"] == {"path": "a.txt"}
    # gemini-2.5 doesn't require explicit tool call ids.
    assert "id" not in parts[1]["functionCall"]


def test_convert_messages_tool_call_id_included_for_gemini3():
    assistant = AssistantMessage(
        api="google-generative-ai",
        provider="google",
        model="gemini-3-pro",
        content=[ToolCall(id="t1", name="read", arguments={})],
        stop_reason="toolUse",
    )
    contents = convert_messages(make_model(id="gemini-3-pro"), Context(messages=[assistant]))
    assert contents[0]["parts"][0]["functionCall"]["id"] == "t1"


def test_convert_messages_tool_call_thought_signature_kept_for_same_model_valid_base64():
    assistant = AssistantMessage(
        api="google-generative-ai",
        provider="google",
        model="gemini-2.5-flash",
        content=[ToolCall(id="t1", name="read", arguments={}, thought_signature="QUJD")],
        stop_reason="toolUse",
    )
    contents = convert_messages(make_model(), Context(messages=[assistant]))
    assert contents[0]["parts"][0]["thoughtSignature"] == "QUJD"


def test_convert_messages_thinking_kept_for_same_model():
    assistant = AssistantMessage(
        api="google-generative-ai",
        provider="google",
        model="gemini-2.5-flash",
        content=[ThinkingContent(thinking="pondering", thinking_signature="QUJD")],
        stop_reason="stop",
    )
    contents = convert_messages(make_model(), Context(messages=[assistant]))
    part = contents[0]["parts"][0]
    assert part["thought"] is True
    assert part["text"] == "pondering"
    assert part["thoughtSignature"] == "QUJD"


def test_convert_messages_thinking_converted_to_text_cross_model():
    assistant = AssistantMessage(
        api="anthropic-messages",
        provider="anthropic",
        model="claude-test",
        content=[ThinkingContent(thinking="pondering", thinking_signature="sig")],
        stop_reason="stop",
    )
    contents = convert_messages(make_model(), Context(messages=[assistant]))
    part = contents[0]["parts"][0]
    assert "thought" not in part
    assert part["text"] == "pondering"
    assert "thoughtSignature" not in part


def test_convert_messages_tool_result_text_output():
    tool_result = ToolResultMessage(tool_call_id="t1", tool_name="read", content=[TextContent(text="file contents")])
    contents = convert_messages(make_model(), Context(messages=[tool_result]))
    function_response = contents[0]["parts"][0]["functionResponse"]
    assert function_response["name"] == "read"
    assert function_response["response"] == {"output": "file contents"}


def test_convert_messages_tool_result_error_output():
    tool_result = ToolResultMessage(
        tool_call_id="t1", tool_name="read", content=[TextContent(text="boom")], is_error=True
    )
    contents = convert_messages(make_model(), Context(messages=[tool_result]))
    function_response = contents[0]["parts"][0]["functionResponse"]
    assert function_response["response"] == {"error": "boom"}


def test_convert_messages_tool_result_images_merged_for_gemini3():
    tool_result = ToolResultMessage(
        tool_call_id="t1", tool_name="see", content=[ImageContent(data="AAA", mime_type="image/png")]
    )
    contents = convert_messages(make_model(id="gemini-3-pro"), Context(messages=[tool_result]))
    function_response = contents[0]["parts"][0]["functionResponse"]
    assert function_response["response"] == {"output": "(see attached image)"}
    assert function_response["parts"] == [{"inlineData": {"mimeType": "image/png", "data": "AAA"}}]
    # No separate image turn for Gemini 3+.
    assert len(contents) == 1


def test_convert_messages_tool_result_images_separate_turn_for_gemini_below_3():
    tool_result = ToolResultMessage(
        tool_call_id="t1", tool_name="see", content=[ImageContent(data="AAA", mime_type="image/png")]
    )
    contents = convert_messages(make_model(), Context(messages=[tool_result]))
    assert "parts" not in contents[0]["parts"][0]["functionResponse"]
    assert len(contents) == 2
    assert contents[1]["parts"][0] == {"text": "Tool result image:"}


def test_convert_messages_consecutive_tool_results_merged_into_one_user_turn():
    messages = [
        ToolResultMessage(tool_call_id="t1", tool_name="read", content=[TextContent(text="a")]),
        ToolResultMessage(tool_call_id="t2", tool_name="read", content=[TextContent(text="b")]),
    ]
    contents = convert_messages(make_model(), Context(messages=messages))
    assert len(contents) == 1
    assert len(contents[0]["parts"]) == 2


# --------------------------------------------------------------------------
# tool conversion
# --------------------------------------------------------------------------


def test_convert_tools_uses_parameters_json_schema_by_default():
    tools = [Tool(name="read", description="reads a file", parameters={"type": "object", "properties": {}})]
    result = convert_tools(tools)
    declaration = result[0]["functionDeclarations"][0]
    assert declaration["name"] == "read"
    assert declaration["parametersJsonSchema"] == {"type": "object", "properties": {}}
    assert "parameters" not in declaration


def test_convert_tools_use_parameters_strips_meta_declarations():
    tools = [
        Tool(
            name="read",
            description="reads a file",
            parameters={"type": "object", "$schema": "http://json-schema.org/draft-07/schema#", "properties": {}},
        )
    ]
    result = convert_tools(tools, use_parameters=True)
    declaration = result[0]["functionDeclarations"][0]
    assert "parametersJsonSchema" not in declaration
    assert "$schema" not in declaration["parameters"]
    assert declaration["parameters"] == {"type": "object", "properties": {}}


def test_convert_tools_empty_returns_none():
    assert convert_tools([]) is None


def test_map_tool_choice():
    assert map_tool_choice("auto") == "AUTO"
    assert map_tool_choice("none") == "NONE"
    assert map_tool_choice("any") == "ANY"


def test_resolve_google_function_calling_mode_none_and_any():
    tools = [Tool(name="read", description="d")]
    assert resolve_google_function_calling_mode(tools, "none", False) == "NONE"
    assert resolve_google_function_calling_mode(tools, "any", False) == "ANY"


def test_resolve_google_function_calling_mode_validated_for_strict_tools():
    tools = [
        Tool(
            name="read",
            description="d",
            constrained_sampling=JsonSchemaConstrainedSampling(strict="require"),
        )
    ]
    assert resolve_google_function_calling_mode(tools, None, True) == "VALIDATED"


def test_resolve_google_function_calling_mode_default_none():
    tools = [Tool(name="read", description="d")]
    assert resolve_google_function_calling_mode(tools, None, False) is None


def test_supports_google_strict_tool_sampling():
    assert supports_google_strict_tool_sampling("gemini-3-pro") is True
    assert supports_google_strict_tool_sampling("gemini-2.5-flash") is False


# --------------------------------------------------------------------------
# build_params
# --------------------------------------------------------------------------


def test_build_params_basic_shape():
    params = build_params(make_model(), Context(messages=[UserMessage(content="hi")]))
    assert params["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert "generationConfig" not in params


def test_build_params_temperature_and_max_tokens():
    params = build_params(
        make_model(), Context(messages=[]), GoogleOptions(api_key="k", temperature=0.4, max_tokens=512)
    )
    assert params["generationConfig"]["temperature"] == 0.4
    assert params["generationConfig"]["maxOutputTokens"] == 512


def test_build_params_system_prompt():
    params = build_params(make_model(), Context(messages=[], system_prompt="be nice"))
    assert params["systemInstruction"] == {"parts": [{"text": "be nice"}]}


def test_build_params_tools_and_tool_config():
    tools = [Tool(name="read", description="d")]
    params = build_params(make_model(), Context(messages=[], tools=tools), GoogleOptions(tool_choice="any"))
    assert params["tools"][0]["functionDeclarations"][0]["name"] == "read"
    assert params["toolConfig"] == {"functionCallingConfig": {"mode": "ANY"}}


def test_build_params_thinking_budget_based():
    model = make_model(reasoning=True)
    params = build_params(
        model,
        Context(messages=[]),
        GoogleOptions(thinking=GoogleThinkingOptions(enabled=True, budget_tokens=2048)),
    )
    assert params["generationConfig"]["thinkingConfig"] == {"includeThoughts": True, "thinkingBudget": 2048}


def test_build_params_thinking_level_based():
    model = make_model(id="gemini-3-pro", reasoning=True)
    params = build_params(
        model,
        Context(messages=[]),
        GoogleOptions(thinking=GoogleThinkingOptions(enabled=True, level="HIGH")),
    )
    assert params["generationConfig"]["thinkingConfig"] == {"includeThoughts": True, "thinkingLevel": "HIGH"}


def test_build_params_thinking_disabled_uses_budget_zero_for_25_models():
    model = make_model(reasoning=True)
    params = build_params(model, Context(messages=[]), GoogleOptions(thinking=GoogleThinkingOptions(enabled=False)))
    assert params["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}


def test_build_params_thinking_disabled_uses_low_level_for_gemini3_pro():
    model = make_model(id="gemini-3-pro", reasoning=True)
    params = build_params(model, Context(messages=[]), GoogleOptions(thinking=GoogleThinkingOptions(enabled=False)))
    assert params["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "LOW"}


def test_build_params_signal_already_aborted_raises():
    from pi_ai.utils.abort import AbortController

    controller = AbortController()
    controller.abort()
    with pytest.raises(RuntimeError):
        build_params(make_model(), Context(messages=[]), GoogleOptions(signal=controller.signal))


# --------------------------------------------------------------------------
# headers / url
# --------------------------------------------------------------------------


def test_build_headers_sets_api_key():
    headers = build_headers(make_model(), "gm-key")
    assert headers["x-goog-api-key"] == "gm-key"


def test_build_url_uses_model_base_url():
    url = build_url(make_model())
    assert (
        url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    )


def test_build_url_defaults_when_base_url_empty():
    url = build_url(make_model(base_url=""))
    assert (
        url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    )


# --------------------------------------------------------------------------
# streaming
# --------------------------------------------------------------------------


async def test_stream_emits_text_events_and_final_message():
    body = sse_body(
        [
            text_chunk("Hel"),
            text_chunk("lo"),
            finish_chunk("STOP", usage={"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}),
        ]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(
                make_model(), Context(messages=[UserMessage(content="hi")]), GoogleOptions(api_key="k"), client=client
            )
        )
    assert [e.type for e in events] == ["start", "text_start", "text_delta", "text_delta", "text_end", "done"]
    assert message.stop_reason == "stop"
    assert message.content[0].text == "Hello"
    assert message.usage.input == 10
    assert message.usage.output == 5
    assert message.usage.total_tokens == 15
    assert message.usage.cost.total > 0


async def test_stream_thought_parts_become_thinking_events():
    body = sse_body(
        [
            text_chunk("pondering...", thought=True, thought_signature="QUJD"),
            text_chunk("answer"),
            finish_chunk("STOP"),
        ]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(reasoning=True), Context(messages=[]), GoogleOptions(api_key="k"), client=client)
        )
    assert [e.type for e in events] == [
        "start",
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        "text_start",
        "text_delta",
        "text_end",
        "done",
    ]
    thinking_block, text_block = message.content
    assert thinking_block.thinking == "pondering..."
    assert thinking_block.thinking_signature == "QUJD"
    assert text_block.text == "answer"


async def test_stream_function_call_with_thought_signature():
    chunk = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {"name": "read", "args": {"path": "a.txt"}},
                            "thoughtSignature": "QUJD",
                        }
                    ],
                }
            }
        ]
    }
    body = sse_body([chunk, finish_chunk("STOP")])
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), GoogleOptions(api_key="k"), client=client)
        )
    assert [e.type for e in events] == ["start", "toolcall_start", "toolcall_delta", "toolcall_end", "done"]
    assert message.stop_reason == "toolUse"
    tool_call = message.content[0]
    assert tool_call.name == "read"
    assert tool_call.arguments == {"path": "a.txt"}
    assert tool_call.thought_signature == "QUJD"


async def test_stream_safety_blocked_reports_error_through_stream():
    body = sse_body([finish_chunk("SAFETY")])
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), GoogleOptions(api_key="k"), client=client)
        )
    assert events[-1].type == "error"
    assert message.stop_reason == "error"
    assert message.raw_stop_reason == "SAFETY"
    assert "SAFETY" in message.error_message


async def test_stream_reports_http_error_through_stream():
    async with make_client('{"error": {"message": "invalid api key"}}', status=401) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), GoogleOptions(api_key="k"), client=client)
        )
    assert [e.type for e in events] == ["error"]
    assert message.stop_reason == "error"
    assert "invalid api key" in message.error_message


async def test_stream_missing_api_key_reports_error_without_raising():
    _events, message = await collect(stream(make_model(), Context(messages=[]), GoogleOptions()))
    assert message.stop_reason == "error"
    assert "No API key" in message.error_message


async def test_stream_reports_pending_error_when_no_finish_reason():
    body = sse_body([text_chunk("partial")])
    async with make_client(body) as client:
        _events, message = await collect(
            stream(make_model(), Context(messages=[]), GoogleOptions(api_key="k"), client=client)
        )
    assert message.stop_reason == "error"
    assert "finish reason" in message.error_message


async def test_stream_sends_expected_request_and_body():
    body = sse_body([finish_chunk("STOP")])
    capture: dict = {}
    async with make_client(body, capture=capture) as client:
        await collect(
            stream(
                make_model(),
                Context(messages=[UserMessage(content="hi")]),
                GoogleOptions(api_key="secret-key"),
                client=client,
            )
        )
    assert capture["request"].headers["x-goog-api-key"] == "secret-key"
    assert str(capture["request"].url).endswith(":streamGenerateContent?alt=sse")
    assert capture["json"]["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]


async def test_stream_async_on_payload_replacement():
    body = sse_body([finish_chunk("STOP")])
    capture: dict = {}

    async def on_payload(params, model):
        params["generationConfig"] = {"temperature": 0.9}
        return params

    async with make_client(body, capture=capture) as client:
        await collect(
            stream(
                make_model(),
                Context(messages=[]),
                GoogleOptions(api_key="k", on_payload=on_payload),
                client=client,
            )
        )
    assert capture["json"]["generationConfig"] == {"temperature": 0.9}


# --------------------------------------------------------------------------
# stream_simple
# --------------------------------------------------------------------------


async def test_stream_simple_disables_thinking_without_reasoning():
    body = sse_body([finish_chunk("STOP")])
    capture: dict = {}
    async with make_client(body, capture=capture) as client:
        await collect(
            stream_simple(make_model(reasoning=True), Context(messages=[]), api_key_options("k"), client=client)
        )
    assert capture["json"]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}


async def test_stream_simple_budget_based_thinking_for_flash():
    from pi_ai.types import SimpleStreamOptions

    body = sse_body([finish_chunk("STOP")])
    capture: dict = {}
    async with make_client(body, capture=capture) as client:
        await collect(
            stream_simple(
                make_model(reasoning=True),
                Context(messages=[]),
                SimpleStreamOptions(api_key="k", reasoning="high"),
                client=client,
            )
        )
    assert capture["json"]["generationConfig"]["thinkingConfig"] == {"includeThoughts": True, "thinkingBudget": 24576}


async def test_stream_simple_level_based_thinking_for_gemini3_pro():
    from pi_ai.types import SimpleStreamOptions

    body = sse_body([finish_chunk("STOP")])
    capture: dict = {}
    async with make_client(body, capture=capture) as client:
        await collect(
            stream_simple(
                make_model(id="gemini-3-pro", reasoning=True),
                Context(messages=[]),
                SimpleStreamOptions(api_key="k", reasoning="high"),
                client=client,
            )
        )
    assert capture["json"]["generationConfig"]["thinkingConfig"] == {"includeThoughts": True, "thinkingLevel": "HIGH"}


def api_key_options(key: str):
    from pi_ai.types import SimpleStreamOptions

    return SimpleStreamOptions(api_key=key)


# --------------------------------------------------------------------------
# provider factory
# --------------------------------------------------------------------------


async def test_google_provider_resolves_auth_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gm-key-from-env")
    provider = google_provider()
    assert provider.id == "google"
    from pi_ai.auth.helpers import resolve_api_key_auth

    result = await resolve_api_key_auth(provider.auth.api_key)
    assert result is not None
    assert result.auth.api_key == "gm-key-from-env"
    assert result.source == "GEMINI_API_KEY"


def test_google_provider_models_have_real_ids_and_costs():
    provider = google_provider()
    ids = {m.id for m in provider.models}
    assert "gemini-2.5-pro" in ids
    assert "gemini-2.5-flash" in ids
    # Gemma models are served free of charge, so only the Gemini tiers are priced.
    assert all(m.cost.input > 0 for m in provider.models if m.id.startswith("gemini-"))
    assert all(m.base_url == "https://generativelanguage.googleapis.com/v1beta" for m in provider.models)
