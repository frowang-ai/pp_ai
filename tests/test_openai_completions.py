import json

import httpx
import pytest
from pi_ai import (
    AssistantMessage,
    Context,
    ImageContent,
    Model,
    ModelCost,
    SimpleStreamOptions,
    TextContent,
    ThinkingBudgets,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from pi_ai.api.openai_completions import (
    OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH,
    OpenAICompletionsOptions,
    _build_chat_template_values,
    _mapped_effort,
    _mapped_effort_or_raw,
    build_headers,
    build_params,
    clamp_openai_prompt_cache_key,
    convert_messages,
    convert_tools,
    detect_compat,
    get_client_api_key,
    get_compat,
    get_compat_cache_control,
    get_deferred_tool_names,
    has_tool_history,
    map_stop_reason,
    normalize_tool_call_id,
    parse_chunk_usage,
    stream,
    stream_simple,
)


def make_model(**overrides) -> Model:
    defaults = dict(
        id="gpt-test",
        name="GPT Test",
        api="openai-completions",
        provider="openai",
        base_url="https://api.openai.com/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=1.0, output=2.0, cache_read=0.5, cache_write=1.5),
        context_window=100_000,
        max_tokens=4096,
    )
    defaults.update(overrides)
    return Model(**defaults)


def sse_body(chunks: list[dict], done: bool = True) -> str:
    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    if done:
        lines.append("data: [DONE]\n\n")
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


# --------------------------------------------------------------------------
# compat detection
# --------------------------------------------------------------------------


def test_detect_compat_defaults_for_openai():
    compat = detect_compat(make_model())
    assert compat.supports_store is True
    assert compat.max_tokens_field == "max_completion_tokens"
    assert compat.thinking_format == "openai"
    assert compat.session_affinity_format == "openai"


def test_detect_compat_deepseek_uses_deepseek_thinking_and_max_tokens():
    compat = detect_compat(make_model(provider="deepseek", base_url="https://api.deepseek.com/v1"))
    assert compat.thinking_format == "deepseek"
    assert compat.max_tokens_field == "max_tokens"
    assert compat.requires_reasoning_content_on_assistant_messages is True
    assert compat.supports_store is False


def test_detect_compat_openrouter():
    compat = detect_compat(make_model(provider="openrouter", base_url="https://openrouter.ai/api/v1"))
    assert compat.thinking_format == "openrouter"
    assert compat.session_affinity_format == "openrouter"
    assert compat.supports_developer_role is False


def test_detect_compat_openrouter_anthropic_model_enables_developer_role_and_cache_control():
    compat = detect_compat(
        make_model(provider="openrouter", base_url="https://openrouter.ai/api/v1", id="anthropic/claude")
    )
    assert compat.supports_developer_role is True
    assert compat.cache_control_format == "anthropic"


def test_detect_compat_zai_and_together():
    zai = detect_compat(make_model(provider="zai", base_url="https://api.z.ai/v1"))
    assert zai.thinking_format == "zai"
    assert zai.supports_reasoning_effort is False

    together = detect_compat(make_model(provider="together", base_url="https://api.together.ai/v1"))
    assert together.thinking_format == "together"
    assert together.supports_long_cache_retention is False


def test_get_compat_applies_model_overrides_in_both_spellings():
    camel = get_compat(make_model(compat={"maxTokensField": "max_tokens", "supportsStore": False}))
    assert camel.max_tokens_field == "max_tokens"
    assert camel.supports_store is False

    snake = get_compat(make_model(compat={"max_tokens_field": "max_tokens"}))
    assert snake.max_tokens_field == "max_tokens"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def test_get_client_api_key_prefers_explicit_key():
    assert get_client_api_key("openai", "sk-1", None) == "sk-1"


def test_get_client_api_key_accepts_authorization_header():
    assert get_client_api_key("openai", None, {"Authorization": "Bearer x"}) == "unused"
    assert get_client_api_key("openai", None, {"cf-aig-authorization": "Bearer x"}) == "unused"


def test_get_client_api_key_rejects_blank_header():
    with pytest.raises(ValueError, match="No API key for provider: openai"):
        get_client_api_key("openai", None, {"authorization": "   "})


def test_has_tool_history():
    assert has_tool_history([UserMessage(content="hi")]) is False
    assert has_tool_history([ToolResultMessage(tool_call_id="1", tool_name="t")]) is True
    assistant = AssistantMessage(content=[ToolCall(id="1", name="t")])
    assert has_tool_history([assistant]) is True


def test_normalize_tool_call_id_combines_pipe_separated_ids():
    assert normalize_tool_call_id(make_model(), "call_1|item_2") == "call_1_item_2"


def test_normalize_tool_call_id_hashes_long_pipe_ids():
    long_id = "call_abc|" + "x" * 200
    result = normalize_tool_call_id(make_model(), long_id)
    assert len(result) <= 40
    assert result.startswith("call_abc")


def test_normalize_tool_call_id_truncates_long_openai_ids():
    assert len(normalize_tool_call_id(make_model(provider="openai"), "c" * 60)) == 40


def test_normalize_tool_call_id_passes_through_other_providers():
    assert normalize_tool_call_id(make_model(provider="groq"), "c" * 60) == "c" * 60


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (None, ("stop", None)),
        ("stop", ("stop", None)),
        ("end", ("stop", None)),
        ("length", ("length", None)),
        ("tool_calls", ("toolUse", None)),
        ("function_call", ("toolUse", None)),
    ],
)
def test_map_stop_reason(reason, expected):
    assert map_stop_reason(reason) == expected


def test_map_stop_reason_error_variants():
    assert map_stop_reason("content_filter") == ("error", "Provider finish_reason: content_filter")
    assert map_stop_reason("weird") == ("error", "Provider finish_reason: weird")


def test_parse_chunk_usage_splits_cache_tokens_and_computes_cost():
    model = make_model(cost=ModelCost(input=1.0, output=2.0, cache_read=0.5, cache_write=1.5))
    usage = parse_chunk_usage(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "prompt_tokens_details": {"cached_tokens": 300, "cache_write_tokens": 100},
            "completion_tokens_details": {"reasoning_tokens": 50},
        },
        model,
    )
    assert usage.input == 600
    assert usage.cache_read == 300
    assert usage.cache_write == 100
    assert usage.output == 200
    assert usage.reasoning == 50
    assert usage.total_tokens == 1200
    assert usage.cost.input == pytest.approx(600 / 1_000_000)
    assert usage.cost.total == pytest.approx(
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
    )


def test_parse_chunk_usage_falls_back_to_prompt_cache_hit_tokens():
    usage = parse_chunk_usage({"prompt_tokens": 100, "prompt_cache_hit_tokens": 40}, make_model())
    assert usage.cache_read == 40
    assert usage.input == 60


# --------------------------------------------------------------------------
# message conversion
# --------------------------------------------------------------------------


def test_convert_messages_uses_system_role_for_non_reasoning_models():
    model = make_model()
    params = convert_messages(model, Context(system_prompt="be nice", messages=[]), get_compat(model))
    assert params[0] == {"role": "system", "content": "be nice"}


def test_convert_messages_uses_developer_role_for_reasoning_models():
    model = make_model(reasoning=True)
    params = convert_messages(model, Context(system_prompt="be nice", messages=[]), get_compat(model))
    assert params[0]["role"] == "developer"


def test_convert_messages_string_and_block_user_content():
    model = make_model(input=["text", "image"])
    context = Context(
        messages=[
            UserMessage(content="plain"),
            UserMessage(content=[TextContent(text="hello"), ImageContent(data="AAA", mime_type="image/png")]),
        ]
    )
    params = convert_messages(model, context, get_compat(model))
    assert params[0] == {"role": "user", "content": "plain"}
    assert params[1]["content"][0] == {"type": "text", "text": "hello"}
    assert params[1]["content"][1]["image_url"]["url"] == "data:image/png;base64,AAA"


def test_convert_messages_downgrades_images_for_text_only_models():
    model = make_model(input=["text"])
    context = Context(messages=[UserMessage(content=[ImageContent(data="AAA", mime_type="image/png")])])
    params = convert_messages(model, context, get_compat(model))
    assert params[0]["content"] == [{"type": "text", "text": "(image omitted: model does not support images)"}]


def test_convert_messages_serializes_assistant_tool_calls():
    model = make_model()
    assistant = AssistantMessage(
        api="openai-completions",
        provider="openai",
        model="gpt-test",
        content=[TextContent(text="calling"), ToolCall(id="c1", name="read", arguments={"path": "a.txt"})],
        stop_reason="toolUse",
    )
    context = Context(
        messages=[assistant, ToolResultMessage(tool_call_id="c1", tool_name="read", content=[TextContent(text="ok")])]
    )
    params = convert_messages(model, context, get_compat(model))
    assert params[0]["content"] == "calling"
    assert params[0]["tool_calls"][0]["function"] == {"name": "read", "arguments": '{"path":"a.txt"}'}
    assert params[1] == {"role": "tool", "content": "ok", "tool_call_id": "c1"}


def test_convert_messages_skips_empty_assistant_messages():
    model = make_model()
    assistant = AssistantMessage(
        api="openai-completions", provider="openai", model="gpt-test", content=[], stop_reason="stop"
    )
    context = Context(messages=[assistant, UserMessage(content="next")])
    params = convert_messages(model, context, get_compat(model))
    assert [p["role"] for p in params] == ["user"]


def test_convert_messages_placeholder_for_empty_tool_result():
    model = make_model()
    assistant = AssistantMessage(
        api="openai-completions",
        provider="openai",
        model="gpt-test",
        content=[ToolCall(id="c1", name="read")],
        stop_reason="toolUse",
    )
    context = Context(messages=[assistant, ToolResultMessage(tool_call_id="c1", tool_name="read", content=[])])
    params = convert_messages(model, context, get_compat(model))
    assert params[-1]["content"] == "(no tool output)"


def test_convert_messages_replays_thinking_signature_field_for_same_model():
    model = make_model(id="m", provider="p", api="openai-completions")
    assistant = AssistantMessage(
        api="openai-completions",
        provider="p",
        model="m",
        content=[
            ThinkingContent(thinking="deep", thinking_signature="reasoning_content"),
            TextContent(text="answer"),
        ],
        stop_reason="stop",
    )
    params = convert_messages(model, Context(messages=[assistant]), get_compat(model))
    assert params[0]["reasoning_content"] == "deep"
    assert params[0]["content"] == "answer"


def test_convert_messages_converts_thinking_to_text_across_models():
    model = make_model(id="other", provider="p")
    assistant = AssistantMessage(
        api="openai-completions",
        provider="p",
        model="m",
        content=[ThinkingContent(thinking="deep"), TextContent(text="answer")],
        stop_reason="stop",
    )
    params = convert_messages(model, Context(messages=[assistant]), get_compat(model))
    assert params[0]["content"] == "deepanswer"


def test_convert_messages_synthesizes_missing_tool_results():
    model = make_model()
    assistant = AssistantMessage(
        api="openai-completions",
        provider="openai",
        model="gpt-test",
        content=[ToolCall(id="c1", name="read")],
        stop_reason="toolUse",
    )
    params = convert_messages(model, Context(messages=[assistant]), get_compat(model))
    assert params[-1] == {"role": "tool", "content": "No result provided", "tool_call_id": "c1"}


def test_convert_messages_bridges_user_after_tool_result_when_required():
    model = make_model(compat={"requiresAssistantAfterToolResult": True})
    assistant = AssistantMessage(
        api="openai-completions",
        provider="openai",
        model="gpt-test",
        content=[ToolCall(id="c1", name="read")],
        stop_reason="toolUse",
    )
    context = Context(
        messages=[
            assistant,
            ToolResultMessage(tool_call_id="c1", tool_name="read", content=[TextContent(text="ok")]),
            UserMessage(content="next"),
        ]
    )
    params = convert_messages(model, context, get_compat(model))
    roles = [p["role"] for p in params]
    assert roles == ["assistant", "tool", "assistant", "user"]
    assert params[2]["content"] == "I have processed the tool results."


def test_convert_messages_adds_tool_result_name_when_required():
    model = make_model(compat={"requiresToolResultName": True})
    assistant = AssistantMessage(
        api="openai-completions",
        provider="openai",
        model="gpt-test",
        content=[ToolCall(id="c1", name="read")],
        stop_reason="toolUse",
    )
    context = Context(
        messages=[assistant, ToolResultMessage(tool_call_id="c1", tool_name="read", content=[TextContent(text="ok")])]
    )
    params = convert_messages(model, context, get_compat(model))
    assert params[-1]["name"] == "read"


def test_convert_messages_attaches_tool_result_images_as_user_message():
    model = make_model(input=["text", "image"])
    assistant = AssistantMessage(
        api="openai-completions",
        provider="openai",
        model="gpt-test",
        content=[ToolCall(id="c1", name="shot")],
        stop_reason="toolUse",
    )
    context = Context(
        messages=[
            assistant,
            ToolResultMessage(
                tool_call_id="c1", tool_name="shot", content=[ImageContent(data="AAA", mime_type="image/png")]
            ),
        ]
    )
    params = convert_messages(model, context, get_compat(model))
    assert params[1]["content"] == "(see attached image)"
    assert params[2]["role"] == "user"
    assert params[2]["content"][1]["image_url"]["url"] == "data:image/png;base64,AAA"


# --------------------------------------------------------------------------
# tools and params
# --------------------------------------------------------------------------


def test_convert_tools_emits_function_definitions():
    tool = Tool(name="read", description="Read a file", parameters={"type": "object", "properties": {}})
    converted = convert_tools([tool], get_compat(make_model()))
    assert converted == [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {}},
                "strict": False,
            },
        }
    ]


def test_convert_tools_omits_strict_when_unsupported():
    compat = get_compat(make_model(compat={"supportsStrictMode": False}))
    converted = convert_tools([Tool(name="read", description="d")], compat)
    assert "strict" not in converted[0]["function"]


def test_build_params_basic_shape():
    model = make_model()
    params = build_params(model, Context(messages=[UserMessage(content="hi")]), OpenAICompletionsOptions())
    assert params["model"] == "gpt-test"
    assert params["stream"] is True
    assert params["stream_options"] == {"include_usage": True}
    assert params["store"] is False


def test_build_params_uses_configured_max_tokens_field():
    model = make_model(compat={"maxTokensField": "max_tokens"})
    params = build_params(model, Context(messages=[]), OpenAICompletionsOptions(max_tokens=100))
    assert params["max_tokens"] == 100
    assert "max_completion_tokens" not in params


def test_build_params_sends_empty_tools_when_history_has_tool_calls():
    model = make_model()
    assistant = AssistantMessage(
        api="openai-completions",
        provider="openai",
        model="gpt-test",
        content=[ToolCall(id="c1", name="read")],
        stop_reason="toolUse",
    )
    context = Context(
        messages=[assistant, ToolResultMessage(tool_call_id="c1", tool_name="read", content=[TextContent(text="x")])]
    )
    params = build_params(model, context, OpenAICompletionsOptions())
    assert params["tools"] == []


def test_build_params_sampling_params_override_named_fields():
    model = make_model()
    options = OpenAICompletionsOptions(temperature=0.1, sampling_params={"temperature": 0.9, "top_p": 0.5})
    params = build_params(model, Context(messages=[]), options)
    assert params["temperature"] == 0.9
    assert params["top_p"] == 0.5


def test_build_params_reasoning_effort_for_openai_models():
    model = make_model(reasoning=True)
    params = build_params(model, Context(messages=[]), OpenAICompletionsOptions(reasoning_effort="high"))
    assert params["reasoning_effort"] == "high"


def test_build_params_maps_reasoning_effort_through_thinking_level_map():
    model = make_model(reasoning=True, thinking_level_map={"high": "ultra"})
    params = build_params(model, Context(messages=[]), OpenAICompletionsOptions(reasoning_effort="high"))
    assert params["reasoning_effort"] == "ultra"


def test_build_params_openrouter_reasoning_object():
    model = make_model(provider="openrouter", base_url="https://openrouter.ai/api/v1", reasoning=True)
    params = build_params(model, Context(messages=[]), OpenAICompletionsOptions(reasoning_effort="low"))
    assert params["reasoning"] == {"effort": "low"}


def test_build_params_openrouter_reasoning_off_default():
    model = make_model(provider="openrouter", base_url="https://openrouter.ai/api/v1", reasoning=True)
    params = build_params(model, Context(messages=[]), OpenAICompletionsOptions())
    assert params["reasoning"] == {"effort": "none"}


def test_build_params_deepseek_thinking_toggle():
    model = make_model(provider="deepseek", base_url="https://api.deepseek.com/v1", reasoning=True)
    enabled = build_params(model, Context(messages=[]), OpenAICompletionsOptions(reasoning_effort="medium"))
    assert enabled["thinking"] == {"type": "enabled"}
    disabled = build_params(model, Context(messages=[]), OpenAICompletionsOptions())
    assert disabled["thinking"] == {"type": "disabled"}


def test_build_params_zai_thinking_toggle():
    model = make_model(provider="zai", base_url="https://api.z.ai/v1", reasoning=True)
    enabled = build_params(model, Context(messages=[]), OpenAICompletionsOptions(reasoning_effort="high"))
    assert enabled["thinking"] == {"type": "enabled", "clear_thinking": False}


def test_build_params_prompt_cache_key_for_openai():
    model = make_model()
    params = build_params(model, Context(messages=[]), OpenAICompletionsOptions(session_id="s-1"))
    assert params["prompt_cache_key"] == "s-1"


def test_build_params_long_cache_retention():
    model = make_model()
    params = build_params(model, Context(messages=[]), OpenAICompletionsOptions(session_id="s", cache_retention="long"))
    assert params["prompt_cache_retention"] == "24h"


def test_build_params_omits_cache_key_when_retention_none():
    model = make_model()
    params = build_params(model, Context(messages=[]), OpenAICompletionsOptions(session_id="s", cache_retention="none"))
    assert "prompt_cache_key" not in params


# --------------------------------------------------------------------------
# streaming
# --------------------------------------------------------------------------


async def test_stream_emits_text_events_and_final_message():
    body = sse_body(
        [
            {"id": "resp-1", "model": "gpt-test", "choices": [{"delta": {"content": "Hel"}}]},
            {"id": "resp-1", "choices": [{"delta": {"content": "lo"}}]},
            {
                "id": "resp-1",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        ]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(
                make_model(),
                Context(messages=[UserMessage(content="hi")]),
                OpenAICompletionsOptions(api_key="k"),
                client=client,
            )
        )

    assert [event.type for event in events] == [
        "start",
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
        "done",
    ]
    assert message.stop_reason == "stop"
    assert message.content[0].text == "Hello"
    assert message.response_id == "resp-1"
    assert message.usage.input == 10
    assert message.usage.output == 5


async def test_stream_records_response_model_when_it_differs():
    body = sse_body(
        [
            {"id": "r", "model": "anthropic/claude", "choices": [{"delta": {"content": "x"}}]},
            {"id": "r", "choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
    )
    async with make_client(body) as client:
        _events, message = await collect(
            stream(make_model(), Context(messages=[]), OpenAICompletionsOptions(api_key="k"), client=client)
        )
    assert message.response_model == "anthropic/claude"


async def test_stream_accumulates_tool_call_arguments():
    body = sse_body(
        [
            {
                "id": "r",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read", "arguments": '{"pa'}}]
                        }
                    }
                ],
            },
            {
                "id": "r",
                "choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'th": "a.txt"}'}}]}}],
            },
            {"id": "r", "choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), OpenAICompletionsOptions(api_key="k"), client=client)
        )

    assert [event.type for event in events] == [
        "start",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]
    assert message.stop_reason == "toolUse"
    tool_call = message.content[0]
    assert tool_call.name == "read"
    assert tool_call.arguments == {"path": "a.txt"}


async def test_stream_emits_thinking_from_reasoning_fields():
    body = sse_body(
        [
            {"id": "r", "choices": [{"delta": {"reasoning_content": "think "}}]},
            {"id": "r", "choices": [{"delta": {"reasoning_content": "more"}}]},
            {"id": "r", "choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]},
        ]
    )
    async with make_client(body) as client:
        _events, message = await collect(
            stream(
                make_model(reasoning=True), Context(messages=[]), OpenAICompletionsOptions(api_key="k"), client=client
            )
        )
    thinking = message.content[0]
    assert thinking.type == "thinking"
    assert thinking.thinking == "think more"
    assert thinking.thinking_signature == "reasoning_content"


async def test_stream_uses_only_the_first_non_empty_reasoning_field():
    body = sse_body(
        [
            {"id": "r", "choices": [{"delta": {"reasoning_content": "a", "reasoning": "a"}}]},
            {"id": "r", "choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
    )
    async with make_client(body) as client:
        _events, message = await collect(
            stream(
                make_model(reasoning=True), Context(messages=[]), OpenAICompletionsOptions(api_key="k"), client=client
            )
        )
    assert message.content[0].thinking == "a"


async def test_stream_reports_http_error_through_stream():
    async with make_client('{"error": {"message": "bad key"}}', status=401) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), OpenAICompletionsOptions(api_key="k"), client=client)
        )
    assert events[-1].type == "error"
    assert message.stop_reason == "error"
    assert "bad key" in message.error_message


async def test_stream_errors_when_finish_reason_missing():
    body = sse_body([{"id": "r", "choices": [{"delta": {"content": "x"}}]}])
    async with make_client(body) as client:
        _events, message = await collect(
            stream(make_model(), Context(messages=[]), OpenAICompletionsOptions(api_key="k"), client=client)
        )
    assert message.stop_reason == "error"
    assert "finish_reason" in message.error_message


async def test_stream_infers_stop_reason_when_provider_omits_finish_reason():
    model = make_model(compat={"supportsFinishReason": False})
    body = sse_body([{"id": "r", "choices": [{"delta": {"content": "x"}}]}])
    async with make_client(body) as client:
        _events, message = await collect(
            stream(model, Context(messages=[]), OpenAICompletionsOptions(api_key="k"), client=client)
        )
    assert message.stop_reason == "stop"


async def test_stream_maps_content_filter_to_error():
    body = sse_body([{"id": "r", "choices": [{"delta": {}, "finish_reason": "content_filter"}]}])
    async with make_client(body) as client:
        _events, message = await collect(
            stream(make_model(), Context(messages=[]), OpenAICompletionsOptions(api_key="k"), client=client)
        )
    assert message.stop_reason == "error"
    assert message.error_message == "Provider finish_reason: content_filter"


async def test_stream_sends_expected_request():
    capture: dict = {}
    body = sse_body([{"id": "r", "choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}])
    async with make_client(body, capture=capture) as client:
        await collect(
            stream(
                make_model(),
                Context(system_prompt="sys", messages=[UserMessage(content="hi")]),
                OpenAICompletionsOptions(api_key="sk-test", headers={"x-custom": "1"}),
                client=client,
            )
        )
    request = capture["request"]
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-test"
    assert request.headers["x-custom"] == "1"
    assert capture["json"]["messages"][0] == {"role": "system", "content": "sys"}


async def test_stream_github_copilot_sends_dynamic_headers():
    capture: dict = {}
    body = sse_body([{"id": "r", "choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}])
    model = make_model(provider="github-copilot")
    async with make_client(body, capture=capture) as client:
        await collect(
            stream(
                model,
                Context(messages=[UserMessage(content=[ImageContent(data="AAA", mime_type="image/png")])]),
                OpenAICompletionsOptions(api_key="tok"),
                client=client,
            )
        )
    request = capture["request"]
    assert request.headers["x-initiator"] == "user"
    assert request.headers["openai-intent"] == "conversation-edits"
    assert request.headers["copilot-vision-request"] == "true"


async def test_stream_on_payload_can_replace_request_body():
    capture: dict = {}
    body = sse_body([{"id": "r", "choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}])

    def on_payload(payload, model):
        payload["metadata_marker"] = model.id
        return payload

    async with make_client(body, capture=capture) as client:
        await collect(
            stream(
                make_model(),
                Context(messages=[]),
                OpenAICompletionsOptions(api_key="k", on_payload=on_payload),
                client=client,
            )
        )
    assert capture["json"]["metadata_marker"] == "gpt-test"


async def test_stream_invokes_on_response_hook():
    seen: list = []
    body = sse_body([{"id": "r", "choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}])

    def on_response(response, model):
        seen.append((response.status, model.id))

    async with make_client(body) as client:
        await collect(
            stream(
                make_model(),
                Context(messages=[]),
                OpenAICompletionsOptions(api_key="k", on_response=on_response),
                client=client,
            )
        )
    assert seen == [(200, "gpt-test")]


async def test_stream_reads_usage_from_choice_fallback():
    body = sse_body(
        [
            {
                "id": "r",
                "choices": [
                    {
                        "delta": {"content": "x"},
                        "finish_reason": "stop",
                        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                    }
                ],
            }
        ]
    )
    async with make_client(body) as client:
        _events, message = await collect(
            stream(make_model(), Context(messages=[]), OpenAICompletionsOptions(api_key="k"), client=client)
        )
    assert message.usage.input == 7
    assert message.usage.output == 3


# --------------------------------------------------------------------------
# regressions found by review against the TypeScript source
# --------------------------------------------------------------------------


def test_cache_retention_defaults_to_short(monkeypatch):
    from pi_ai.api.openai_completions import resolve_cache_retention

    monkeypatch.delenv("PI_CACHE_RETENTION", raising=False)
    assert resolve_cache_retention(None, None) == "short"


def test_cache_retention_env_opts_into_long(monkeypatch):
    from pi_ai.api.openai_completions import resolve_cache_retention

    monkeypatch.setenv("PI_CACHE_RETENTION", "long")
    assert resolve_cache_retention(None, None) == "long"
    # A scoped env override takes precedence over the process environment.
    assert resolve_cache_retention(None, {"PI_CACHE_RETENTION": "short"}) == "short"


def test_cache_retention_explicit_value_wins(monkeypatch):
    from pi_ai.api.openai_completions import resolve_cache_retention

    monkeypatch.setenv("PI_CACHE_RETENTION", "long")
    assert resolve_cache_retention("none", None) == "none"


def test_long_retention_from_env_sends_the_24h_field(monkeypatch):
    monkeypatch.setenv("PI_CACHE_RETENTION", "long")
    params = build_params(make_model(), Context(messages=[]), OpenAICompletionsOptions(session_id="s"))
    assert params["prompt_cache_retention"] == "24h"


async def test_aborted_signal_terminates_the_stream_as_aborted():
    from pi_ai.utils.abort import AbortSignal

    signal = AbortSignal()
    signal.abort()
    body = sse_body([{"id": "r", "choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}])
    async with make_client(body) as client:
        events, message = await collect(
            stream(
                make_model(),
                Context(messages=[]),
                OpenAICompletionsOptions(api_key="k", signal=signal),
                client=client,
            )
        )
    assert events[-1].type == "error"
    assert events[-1].reason == "aborted"
    assert message.stop_reason == "aborted"


async def test_unaborted_signal_does_not_change_the_outcome():
    from pi_ai.utils.abort import AbortSignal

    body = sse_body([{"id": "r", "choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}])
    async with make_client(body) as client:
        _events, message = await collect(
            stream(
                make_model(),
                Context(messages=[]),
                OpenAICompletionsOptions(api_key="k", signal=AbortSignal()),
                client=client,
            )
        )
    assert message.stop_reason == "stop"


def test_empty_assistant_turn_still_triggers_the_tool_result_bridge():
    """An aborted turn between tool results and a user message must not
    suppress the synthetic assistant message some providers require."""
    model = make_model(compat={"requiresAssistantAfterToolResult": True})
    tool_turn = AssistantMessage(
        api="openai-completions",
        provider="openai",
        model="gpt-test",
        content=[ToolCall(id="c1", name="read")],
        stop_reason="toolUse",
    )
    empty_turn = AssistantMessage(
        api="openai-completions", provider="openai", model="gpt-test", content=[], stop_reason="stop"
    )
    context = Context(
        messages=[
            tool_turn,
            ToolResultMessage(tool_call_id="c1", tool_name="read", content=[TextContent(text="ok")]),
            empty_turn,
            UserMessage(content="next"),
        ]
    )
    params = convert_messages(model, context, get_compat(model))
    roles = [p["role"] for p in params]
    assert roles == ["assistant", "tool", "assistant", "user"]
    assert params[2]["content"] == "I have processed the tool results."


def test_anthropic_cache_control_is_applied_for_openrouter_anthropic_models():
    model = make_model(provider="openrouter", base_url="https://openrouter.ai/api/v1", id="anthropic/claude")
    tool = Tool(name="read", description="Read")
    context = Context(system_prompt="sys", messages=[UserMessage(content="hi")], tools=[tool])
    params = build_params(model, context, OpenAICompletionsOptions())

    system_message = params["messages"][0]
    assert system_message["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert params["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert params["messages"][-1]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_cache_control_uses_a_1h_ttl_for_long_retention():
    model = make_model(provider="openrouter", base_url="https://openrouter.ai/api/v1", id="anthropic/claude")
    params = build_params(
        model,
        Context(system_prompt="sys", messages=[UserMessage(content="hi")]),
        OpenAICompletionsOptions(cache_retention="long"),
    )
    assert params["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_anthropic_cache_control_is_skipped_when_retention_is_none():
    model = make_model(provider="openrouter", base_url="https://openrouter.ai/api/v1", id="anthropic/claude")
    params = build_params(
        model,
        Context(system_prompt="sys", messages=[UserMessage(content="hi")]),
        OpenAICompletionsOptions(cache_retention="none"),
    )
    assert params["messages"][0]["content"] == "sys"


def test_no_cache_control_for_non_anthropic_openrouter_models():
    model = make_model(provider="openrouter", base_url="https://openrouter.ai/api/v1", id="openai/gpt-4o")
    params = build_params(
        model, Context(system_prompt="sys", messages=[UserMessage(content="hi")]), OpenAICompletionsOptions()
    )
    assert params["messages"][0]["content"] == "sys"


# --------------------------------------------------------------------------
# thinking / reasoning parameter branches
# --------------------------------------------------------------------------


def test_thinking_zai_enabled_and_disabled():
    model = make_model(reasoning=True, compat={"thinkingFormat": "zai"})
    params_on = build_params(model, Context(messages=[]), OpenAICompletionsOptions(reasoning_effort="high"))
    assert params_on["thinking"] == {"type": "enabled", "clear_thinking": False}
    assert params_on["reasoning_effort"] == "high"

    params_off = build_params(model, Context(messages=[]), OpenAICompletionsOptions())
    assert params_off["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in params_off


def test_thinking_zai_null_mapping_omits_reasoning_effort():
    # zai uses strict lookup: an explicit `None` mapping stays unset rather than
    # falling back to the raw effort string.
    model = make_model(reasoning=True, compat={"thinkingFormat": "zai"}, thinking_level_map={"high": None})
    params = build_params(model, Context(messages=[]), OpenAICompletionsOptions(reasoning_effort="high"))
    assert "reasoning_effort" not in params


def test_thinking_qwen_toggle_and_mapping():
    model = make_model(reasoning=True, compat={"thinkingFormat": "qwen"}, thinking_level_map={"high": "max_thinking"})
    on = build_params(model, Context(messages=[]), OpenAICompletionsOptions(reasoning_effort="high"))
    assert on["enable_thinking"] is True
    assert on["reasoning_effort"] == "max_thinking"

    off = build_params(model, Context(messages=[]), OpenAICompletionsOptions())
    assert off["enable_thinking"] is False
    assert "reasoning_effort" not in off


def test_thinking_qwen_null_mapping_falls_back_to_raw_effort():
    # qwen uses nullish-coalescing lookup in TypeScript (`?? effort`): an
    # explicit `None` mapping still falls back to the raw effort string, unlike
    # zai's strict lookup.
    model = make_model(reasoning=True, compat={"thinkingFormat": "qwen"}, thinking_level_map={"high": None})
    params = build_params(model, Context(messages=[]), OpenAICompletionsOptions(reasoning_effort="high"))
    assert params["reasoning_effort"] == "high"


def test_thinking_qwen_chat_template():
    model = make_model(reasoning=True, compat={"thinkingFormat": "qwen-chat-template"})
    on = build_params(model, Context(messages=[]), OpenAICompletionsOptions(reasoning_effort="low"))
    assert on["chat_template_kwargs"] == {"enable_thinking": True, "preserve_thinking": True}
    off = build_params(model, Context(messages=[]), OpenAICompletionsOptions())
    assert off["chat_template_kwargs"] == {"enable_thinking": False, "preserve_thinking": True}


def test_thinking_chat_template_uses_var_values():
    model = make_model(
        reasoning=True,
        compat={
            "thinkingFormat": "chat-template",
            "chatTemplateKwargs": {
                "thinking": {"$var": "thinking.enabled"},
                "effort": {"$var": "thinking.effort"},
                "static": "always",
            },
        },
        thinking_level_map={"high": "deep"},
    )
    params = build_params(model, Context(messages=[]), OpenAICompletionsOptions(reasoning_effort="high"))
    assert params["chat_template_kwargs"] == {"thinking": True, "effort": "deep", "static": "always"}


def test_thinking_chat_template_omit_when_off():
    model = make_model(
        reasoning=True,
        compat={
            "thinkingFormat": "chat-template",
            "chatTemplateKwargs": {"effort": {"$var": "thinking.effort", "omitWhenOff": True}},
        },
    )
    params = build_params(model, Context(messages=[]), OpenAICompletionsOptions())
    assert "chat_template_kwargs" not in params


def test_thinking_chat_template_off_uses_thinking_level_map_off():
    model = make_model(
        reasoning=True,
        compat={
            "thinkingFormat": "chat-template",
            "chatTemplateKwargs": {"effort": {"$var": "thinking.effort"}},
        },
        thinking_level_map={"off": "disabled"},
    )
    params = build_params(model, Context(messages=[]), OpenAICompletionsOptions())
    assert params["chat_template_kwargs"] == {"effort": "disabled"}


def test_thinking_baseten_chat_template_args_and_reasoning_effort():
    model = make_model(
        reasoning=True,
        compat={
            "thinkingFormat": "baseten",
            "chatTemplateArgs": {"thinking": {"$var": "thinking.enabled"}},
        },
        thinking_level_map={"high": "hard"},
    )
    on = build_params(model, Context(messages=[]), OpenAICompletionsOptions(reasoning_effort="high"))
    assert on["chat_template_args"] == {"thinking": True}
    assert on["reasoning_effort"] == "hard"

    off = build_params(model, Context(messages=[]), OpenAICompletionsOptions())
    assert off["chat_template_args"] == {"thinking": False}
    assert "reasoning_effort" not in off


def test_thinking_deepseek_null_mapping_falls_back_to_raw_effort():
    model = make_model(
        reasoning=True, provider="deepseek", base_url="https://api.deepseek.com/v1", thinking_level_map={"high": None}
    )
    params = build_params(model, Context(messages=[]), OpenAICompletionsOptions(reasoning_effort="high"))
    assert params["reasoning_effort"] == "high"


def test_thinking_deepseek_off_explicitly_disabled_skips_thinking_field():
    model = make_model(
        reasoning=True, provider="deepseek", base_url="https://api.deepseek.com/v1", thinking_level_map={"off": None}
    )
    params = build_params(model, Context(messages=[]), OpenAICompletionsOptions())
    assert "thinking" not in params


def test_thinking_openrouter_null_mapping_falls_back_to_raw_effort():
    model = make_model(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        reasoning=True,
        thinking_level_map={"high": None},
    )
    params = build_params(model, Context(messages=[]), OpenAICompletionsOptions(reasoning_effort="high"))
    assert params["reasoning"] == {"effort": "high"}


def test_thinking_openrouter_off_explicitly_disabled_skips_reasoning_field():
    model = make_model(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        reasoning=True,
        thinking_level_map={"off": None},
    )
    params = build_params(model, Context(messages=[]), OpenAICompletionsOptions())
    assert "reasoning" not in params


def test_thinking_ant_ling_mapped_and_unmapped():
    model = make_model(reasoning=True, compat={"thinkingFormat": "ant-ling"}, thinking_level_map={"high": "deep"})
    mapped = build_params(model, Context(messages=[]), OpenAICompletionsOptions(reasoning_effort="high"))
    assert mapped["reasoning"] == {"effort": "deep"}

    # ant-ling never falls back to the raw effort string when unmapped.
    unmapped = build_params(model, Context(messages=[]), OpenAICompletionsOptions(reasoning_effort="low"))
    assert "reasoning" not in unmapped


def test_thinking_together_toggle_and_effort():
    model = make_model(reasoning=True, compat={"thinkingFormat": "together"}, thinking_level_map={"high": None})
    params = build_params(model, Context(messages=[]), OpenAICompletionsOptions(reasoning_effort="high"))
    assert params["reasoning"] == {"enabled": True}
    # nullish fallback: explicit null mapping still falls back to raw effort.
    assert params["reasoning_effort"] == "high"

    off = build_params(model, Context(messages=[]), OpenAICompletionsOptions())
    assert off["reasoning"] == {"enabled": False}
    assert "reasoning_effort" not in off


def test_thinking_string_thinking_effort_and_off_variants():
    model = make_model(reasoning=True, compat={"thinkingFormat": "string-thinking"}, thinking_level_map={"high": None})
    with_effort = build_params(model, Context(messages=[]), OpenAICompletionsOptions(reasoning_effort="high"))
    assert with_effort["thinking"] == "high"

    off_missing = build_params(
        make_model(reasoning=True, compat={"thinkingFormat": "string-thinking"}),
        Context(messages=[]),
        OpenAICompletionsOptions(),
    )
    assert off_missing["thinking"] == "none"

    off_mapped = build_params(
        make_model(reasoning=True, compat={"thinkingFormat": "string-thinking"}, thinking_level_map={"off": "quiet"}),
        Context(messages=[]),
        OpenAICompletionsOptions(),
    )
    assert off_mapped["thinking"] == "quiet"

    off_disabled = build_params(
        make_model(reasoning=True, compat={"thinkingFormat": "string-thinking"}, thinking_level_map={"off": None}),
        Context(messages=[]),
        OpenAICompletionsOptions(),
    )
    assert "thinking" not in off_disabled


def test_thinking_plain_openai_null_mapping_falls_back_to_raw_effort():
    model = make_model(reasoning=True, thinking_level_map={"high": None})
    params = build_params(model, Context(messages=[]), OpenAICompletionsOptions(reasoning_effort="high"))
    assert params["reasoning_effort"] == "high"


def test_thinking_plain_openai_off_uses_thinking_level_map_off_string():
    model = make_model(reasoning=True, thinking_level_map={"off": "disabled"})
    params = build_params(model, Context(messages=[]), OpenAICompletionsOptions())
    assert params["reasoning_effort"] == "disabled"


def test_thinking_plain_openai_off_with_no_mapping_omits_field():
    model = make_model(reasoning=True)
    params = build_params(model, Context(messages=[]), OpenAICompletionsOptions())
    assert "reasoning_effort" not in params


def test_mapped_effort_helpers_strict_vs_nullish():
    model = make_model(thinking_level_map={"high": None, "low": "chill"})
    assert _mapped_effort(model, "high") is None
    assert _mapped_effort(model, "low") == "chill"
    assert _mapped_effort(model, "medium") == "medium"

    assert _mapped_effort_or_raw(model, "high") == "high"
    assert _mapped_effort_or_raw(model, "low") == "chill"
    assert _mapped_effort_or_raw(model, "medium") == "medium"


def test_build_chat_template_values_thinking_effort_null_mapping_omits_key():
    model = make_model(thinking_level_map={"high": None})
    options = OpenAICompletionsOptions(reasoning_effort="high")
    result = _build_chat_template_values(model, options, {"effort": {"$var": "thinking.effort"}})
    assert result is None


def test_build_chat_template_values_returns_none_when_all_omitted():
    model = make_model()
    options = OpenAICompletionsOptions()
    result = _build_chat_template_values(model, options, {"effort": {"$var": "thinking.effort", "omitWhenOff": True}})
    assert result is None


def test_build_chat_template_values_passes_through_plain_values():
    model = make_model()
    options = OpenAICompletionsOptions()
    result = _build_chat_template_values(model, options, {"top_k": 20, "static": "always"})
    assert result == {"top_k": 20, "static": "always"}


def test_supports_thinking_token_budget_clamps_and_caps():
    model = make_model(reasoning=True, compat={"supportsThinkingTokenBudget": True}, max_tokens=20_000)
    params = build_params(
        model,
        Context(messages=[]),
        OpenAICompletionsOptions(reasoning_effort="high", max_tokens=20_000),
    )
    assert params["thinking_token_budget"] == 16384


def test_supports_thinking_token_budget_honors_custom_budgets():
    model = make_model(reasoning=True, compat={"supportsThinkingTokenBudget": True}, max_tokens=100_000)
    params = build_params(
        model,
        Context(messages=[]),
        OpenAICompletionsOptions(
            reasoning_effort="medium", max_tokens=100_000, thinking_budgets=ThinkingBudgets(medium=4000)
        ),
    )
    assert params["thinking_token_budget"] == 4000


def test_supports_thinking_token_budget_clamps_to_leave_room_for_answer():
    model = make_model(reasoning=True, compat={"supportsThinkingTokenBudget": True}, max_tokens=1200)
    params = build_params(
        model, Context(messages=[]), OpenAICompletionsOptions(reasoning_effort="high", max_tokens=1200)
    )
    # ceiling(1200) - MIN_ANSWER_TOKENS(1024) = 176, well below the 16384 budget.
    assert params["thinking_token_budget"] == 176


def test_supports_thinking_token_budget_zero_budget_omits_field():
    model = make_model(reasoning=True, compat={"supportsThinkingTokenBudget": True}, max_tokens=500)
    params = build_params(
        model, Context(messages=[]), OpenAICompletionsOptions(reasoning_effort="high", max_tokens=500)
    )
    assert "thinking_token_budget" not in params


def test_supports_thinking_token_budget_xhigh_clamps_to_high():
    model = make_model(reasoning=True, compat={"supportsThinkingTokenBudget": True}, max_tokens=100_000)
    params = build_params(
        model, Context(messages=[]), OpenAICompletionsOptions(reasoning_effort="xhigh", max_tokens=100_000)
    )
    assert params["thinking_token_budget"] == 16384


# --------------------------------------------------------------------------
# deferred tools ("kimi") path
# --------------------------------------------------------------------------


def test_get_deferred_tool_names_collects_from_tool_results():
    names = get_deferred_tool_names(
        [
            ToolResultMessage(tool_call_id="1", tool_name="a", added_tool_names=["search", "grep"]),
            ToolResultMessage(tool_call_id="2", tool_name="b", added_tool_names=["grep", "ls"]),
        ]
    )
    # Checked before the order assertion on purpose. The invariant is insertion
    # order, not "these three happened to come out right". TypeScript
    # accumulates into a `Set`, which iterates in insertion order; a plain
    # Python `set` does not, and its iteration order depends on the
    # interpreter's hash seed. Since every xdist worker is a separate process
    # with its own seed, a `set` regression fails in some workers and passes in
    # others -- indistinguishable from a load-dependent flake unless the
    # container type is pinned first, which is deterministic under every seed.
    assert not isinstance(names, set | frozenset)
    assert list(names) == ["search", "grep", "ls"]
    assert list(names) == list(dict.fromkeys(["search", "grep", "grep", "ls"]))


def test_get_deferred_tool_names_preserves_first_seen_order_for_many_names():
    # A longer list makes an accidental reordering visible: with 8 names a
    # hash-ordered container is overwhelmingly unlikely to reproduce first-seen
    # order, whereas 3 names can coincide.
    first = ["zeta", "alpha", "mu", "beta"]
    second = ["mu", "omega", "alpha", "kappa", "delta"]
    names = get_deferred_tool_names(
        [
            ToolResultMessage(tool_call_id="1", tool_name="a", added_tool_names=first),
            ToolResultMessage(tool_call_id="2", tool_name="b", added_tool_names=second),
        ]
    )
    assert list(names) == ["zeta", "alpha", "mu", "beta", "omega", "kappa", "delta"]


def test_build_params_excludes_deferred_tools_from_active_tools():
    model = make_model(compat={"deferredToolsMode": "kimi"})
    search_tool = Tool(name="search", description="Search")
    read_tool = Tool(name="read", description="Read")
    context = Context(
        messages=[ToolResultMessage(tool_call_id="1", tool_name="x", added_tool_names=["search"])],
        tools=[search_tool, read_tool],
    )
    params = build_params(model, context, OpenAICompletionsOptions())
    tool_names = {t["function"]["name"] for t in params["tools"]}
    assert tool_names == {"read"}


def test_convert_messages_kimi_inserts_system_message_with_deferred_tools():
    model = make_model(compat={"deferredToolsMode": "kimi"})
    search_tool = Tool(name="search", description="Search the web")
    assistant = AssistantMessage(
        api="openai-completions",
        provider="openai",
        model="gpt-test",
        content=[ToolCall(id="c1", name="read")],
        stop_reason="toolUse",
    )
    context = Context(
        messages=[
            assistant,
            ToolResultMessage(
                tool_call_id="c1", tool_name="read", content=[TextContent(text="ok")], added_tool_names=["search"]
            ),
        ],
        tools=[search_tool],
    )
    params = convert_messages(model, context, get_compat(model))
    system_messages = [p for p in params if p.get("role") == "system" and "tools" in p]
    assert len(system_messages) == 1
    assert system_messages[0]["tools"][0]["function"]["name"] == "search"
    assert "content" not in system_messages[0]


def test_convert_messages_kimi_skips_system_message_when_tool_unknown():
    model = make_model(compat={"deferredToolsMode": "kimi"})
    context = Context(
        messages=[ToolResultMessage(tool_call_id="1", tool_name="x", added_tool_names=["ghost"])],
        tools=[],
    )
    params = convert_messages(model, context, get_compat(model))
    assert not any(p.get("role") == "system" and "tools" in p for p in params)


# --------------------------------------------------------------------------
# build_headers session affinity + removal
# --------------------------------------------------------------------------


def test_build_headers_openai_session_affinity_format():
    model = make_model(compat={"sendSessionAffinityHeaders": True, "sessionAffinityFormat": "openai"})
    headers = build_headers(model, "key", OpenAICompletionsOptions(session_id="sess-1"), get_compat(model))
    assert headers["session_id"] == "sess-1"
    assert headers["x-client-request-id"] == "sess-1"
    assert headers["x-session-affinity"] == "sess-1"
    assert "x-session-id" not in headers


def test_build_headers_openai_nosession_format_omits_session_id():
    model = make_model(compat={"sendSessionAffinityHeaders": True, "sessionAffinityFormat": "openai-nosession"})
    headers = build_headers(model, "key", OpenAICompletionsOptions(session_id="sess-1"), get_compat(model))
    assert "session_id" not in headers
    assert headers["x-client-request-id"] == "sess-1"
    assert headers["x-session-affinity"] == "sess-1"


def test_build_headers_openrouter_session_affinity_format():
    model = make_model(compat={"sendSessionAffinityHeaders": True, "sessionAffinityFormat": "openrouter"})
    headers = build_headers(model, "key", OpenAICompletionsOptions(session_id="sess-1"), get_compat(model))
    assert headers["x-session-id"] == "sess-1"
    assert "session_id" not in headers
    assert "x-client-request-id" not in headers


def test_build_headers_removes_header_via_none_value():
    model = make_model(headers={"x-custom": "value"})
    headers = build_headers(model, "key", OpenAICompletionsOptions(headers={"x-custom": None}), get_compat(model))
    assert "x-custom" not in headers


def test_build_headers_masks_authorization():
    headers = build_headers(make_model(), "sekrit", OpenAICompletionsOptions(), get_compat(make_model()))
    assert headers["authorization"] == "Bearer sekrit"


def test_build_headers_github_copilot_dynamic_headers():
    model = make_model(provider="github-copilot")
    context = Context(messages=[UserMessage(content=[ImageContent(data="AAA", mime_type="image/png")])])
    headers = build_headers(model, "tok", OpenAICompletionsOptions(), get_compat(model), context)
    assert headers["X-Initiator"] == "user"
    assert headers["Openai-Intent"] == "conversation-edits"
    assert headers["Copilot-Vision-Request"] == "true"


def test_build_headers_github_copilot_no_images_omits_vision_header():
    model = make_model(provider="github-copilot")
    context = Context(messages=[UserMessage(content="hi")])
    headers = build_headers(model, "tok", OpenAICompletionsOptions(), get_compat(model), context)
    assert "Copilot-Vision-Request" not in headers


def test_build_headers_non_copilot_provider_omits_copilot_headers():
    model = make_model(provider="openai")
    context = Context(messages=[UserMessage(content="hi")])
    headers = build_headers(model, "tok", OpenAICompletionsOptions(), get_compat(model), context)
    assert "X-Initiator" not in headers
    assert "Openai-Intent" not in headers


# --------------------------------------------------------------------------
# OpenRouter / Vercel gateway routing
# --------------------------------------------------------------------------


def test_build_params_open_router_routing_applied():
    model = make_model(compat={"openRouterRouting": {"order": ["anthropic", "openai"]}})
    params = build_params(model, Context(messages=[]), OpenAICompletionsOptions())
    assert params["provider"] == {"order": ["anthropic", "openai"]}


def test_build_params_vercel_gateway_routing_only():
    model = make_model(compat={"vercelGatewayRouting": {"only": ["bedrock"]}})
    params = build_params(model, Context(messages=[]), OpenAICompletionsOptions())
    assert params["providerOptions"] == {"gateway": {"only": ["bedrock"]}}


def test_build_params_vercel_gateway_routing_order():
    model = make_model(compat={"vercelGatewayRouting": {"order": ["bedrock", "vertex"]}})
    params = build_params(model, Context(messages=[]), OpenAICompletionsOptions())
    assert params["providerOptions"] == {"gateway": {"order": ["bedrock", "vertex"]}}


def test_build_params_vercel_gateway_routing_omitted_without_only_or_order():
    model = make_model(compat={"vercelGatewayRouting": {"irrelevant": True}})
    params = build_params(model, Context(messages=[]), OpenAICompletionsOptions())
    assert "providerOptions" not in params


def test_get_compat_cache_control_no_marker_for_non_anthropic_format():
    assert get_compat_cache_control(get_compat(make_model()), "short") is None


def test_get_compat_cache_control_no_marker_when_retention_none():
    model = make_model(provider="openrouter", base_url="https://openrouter.ai/api/v1", id="anthropic/claude")
    assert get_compat_cache_control(get_compat(model), "none") is None


# --------------------------------------------------------------------------
# stream_simple
# --------------------------------------------------------------------------


async def test_stream_simple_maps_reasoning_level_to_reasoning_effort():
    capture: dict = {}
    body = sse_body([{"id": "r", "choices": [{"delta": {}, "finish_reason": "stop"}]}])
    model = make_model(reasoning=True)
    async with make_client(body, capture=capture) as client:
        _events, message = await collect(
            stream_simple(
                model,
                Context(messages=[UserMessage(content="hi")]),
                SimpleStreamOptions(api_key="k", reasoning="high"),
                client=client,
            )
        )
    assert message.stop_reason == "stop"
    assert capture["json"]["reasoning_effort"] == "high"


async def test_stream_simple_off_reasoning_omits_reasoning_effort():
    capture: dict = {}
    body = sse_body([{"id": "r", "choices": [{"delta": {}, "finish_reason": "stop"}]}])
    model = make_model(reasoning=True)
    async with make_client(body, capture=capture) as client:
        await collect(
            stream_simple(
                model,
                Context(messages=[UserMessage(content="hi")]),
                SimpleStreamOptions(api_key="k", reasoning="off"),
                client=client,
            )
        )
    assert "reasoning_effort" not in capture["json"]


async def test_stream_simple_reaches_provider_with_clamped_reasoning():
    capture: dict = {}
    body = sse_body([{"id": "r", "choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}])
    # thinking_level_map restricts support to only "low"; clamp_thinking_level should
    # snap the requested "high" down to the nearest supported level before it
    # reaches the wire.
    model = make_model(reasoning=True, thinking_level_map={"off": None, "minimal": None, "medium": None, "high": None})
    async with make_client(body, capture=capture) as client:
        await collect(
            stream_simple(
                model,
                Context(messages=[UserMessage(content="hi")]),
                SimpleStreamOptions(api_key="k", reasoning="high"),
                client=client,
            )
        )
    assert capture["json"]["reasoning_effort"] == "low"


# --------------------------------------------------------------------------
# tool-call streaming edge cases
# --------------------------------------------------------------------------


async def test_stream_tool_call_delivered_without_index():
    body = sse_body(
        [
            {
                "id": "r",
                "choices": [{"delta": {"tool_calls": [{"id": "c1", "function": {"name": "read", "arguments": "{}"}}]}}],
            },
            {"id": "r", "choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), OpenAICompletionsOptions(api_key="k"), client=client)
        )
    assert message.content[0].type == "toolCall"
    assert message.content[0].id == "c1"
    assert message.content[0].name == "read"
    assert message.content[0].arguments == {}
    assert [e.type for e in events].count("toolcall_start") == 1


async def test_stream_two_tool_calls_accumulate_independently():
    body = sse_body(
        [
            {
                "id": "r",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "c1", "function": {"name": "read", "arguments": '{"a"'}},
                                {"index": 1, "id": "c2", "function": {"name": "write", "arguments": '{"b"'}},
                            ]
                        }
                    }
                ],
            },
            {
                "id": "r",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": ": 1}"}},
                                {"index": 1, "function": {"arguments": ": 2}"}},
                            ]
                        }
                    }
                ],
            },
            {"id": "r", "choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
    )
    async with make_client(body) as client:
        _events, message = await collect(
            stream(make_model(), Context(messages=[]), OpenAICompletionsOptions(api_key="k"), client=client)
        )
    tool_calls = [b for b in message.content if b.type == "toolCall"]
    assert len(tool_calls) == 2
    assert tool_calls[0].name == "read"
    assert tool_calls[0].arguments == {"a": 1}
    assert tool_calls[1].name == "write"
    assert tool_calls[1].arguments == {"b": 2}


async def test_stream_reasoning_details_attached_after_tool_call_arrives():
    detail = {"type": "reasoning.encrypted", "id": "c1", "data": "secret"}
    body = sse_body(
        [
            {
                "id": "r",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read", "arguments": "{}"}}]
                        }
                    }
                ],
            },
            {"id": "r", "choices": [{"delta": {"reasoning_details": [detail]}}]},
            {"id": "r", "choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
    )
    async with make_client(body) as client:
        _events, message = await collect(
            stream(make_model(), Context(messages=[]), OpenAICompletionsOptions(api_key="k"), client=client)
        )
    tool_call = message.content[0]
    assert json.loads(tool_call.thought_signature) == detail


async def test_stream_reasoning_details_attached_before_tool_call_arrives():
    detail = {"type": "reasoning.encrypted", "id": "c1", "data": "secret"}
    body = sse_body(
        [
            {"id": "r", "choices": [{"delta": {"reasoning_details": [detail]}}]},
            {
                "id": "r",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read", "arguments": "{}"}}]
                        }
                    }
                ],
            },
            {"id": "r", "choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
    )
    async with make_client(body) as client:
        _events, message = await collect(
            stream(make_model(), Context(messages=[]), OpenAICompletionsOptions(api_key="k"), client=client)
        )
    tool_call = message.content[0]
    assert json.loads(tool_call.thought_signature) == detail


async def test_stream_reasoning_details_ignores_non_encrypted_entries():
    body = sse_body(
        [
            {
                "id": "r",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read", "arguments": "{}"}}],
                            "reasoning_details": [{"type": "reasoning.plain", "id": "c1", "data": "x"}],
                        }
                    }
                ],
            },
            {"id": "r", "choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
    )
    async with make_client(body) as client:
        _events, message = await collect(
            stream(make_model(), Context(messages=[]), OpenAICompletionsOptions(api_key="k"), client=client)
        )
    assert message.content[0].thought_signature is None


# --------------------------------------------------------------------------
# normalize_tool_call_id edge cases
# --------------------------------------------------------------------------


def test_normalize_tool_call_id_empty_item_id_after_pipe():
    assert normalize_tool_call_id(make_model(), "call_1|") == "call_1"


def test_normalize_tool_call_id_sanitizes_non_ascii_and_special_chars():
    result = normalize_tool_call_id(make_model(), "call+abc/def=|item.one two")
    assert result.startswith("call_abc_def_")
    assert "/" not in result and "+" not in result and "=" not in result


def test_normalize_tool_call_id_short_ids_untouched_for_non_openai_provider():
    assert normalize_tool_call_id(make_model(provider="anthropic"), "short-id") == "short-id"


# --------------------------------------------------------------------------
# _StreamState.content_index fallback
# --------------------------------------------------------------------------


def test_stream_state_content_index_returns_minus_one_for_unknown_block():
    from pi_ai.api.openai_completions import _StreamState
    from pi_ai.utils.event_stream import AssistantMessageEventStream

    model = make_model()
    output = AssistantMessage(api="openai-completions", provider="openai", model="gpt-test", stop_reason="pending")
    state = _StreamState(AssistantMessageEventStream(), output, model)
    orphan = TextContent(text="not attached")
    assert state.content_index(orphan) == -1


# --------------------------------------------------------------------------
# additional small-branch coverage
# --------------------------------------------------------------------------


def test_clamp_openai_prompt_cache_key_truncates_long_keys():
    long_key = "s" * 100
    result = clamp_openai_prompt_cache_key(long_key)
    assert result == "s" * OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH


def test_clamp_openai_prompt_cache_key_passes_short_keys_and_none():
    assert clamp_openai_prompt_cache_key(None) is None
    assert clamp_openai_prompt_cache_key("short") == "short"


def test_detect_compat_ant_ling_via_provider_heuristic():
    compat = detect_compat(make_model(provider="ant-ling", base_url="https://api.ant-ling.com/v1"))
    assert compat.thinking_format == "ant-ling"
    assert compat.supports_reasoning_effort is False
    assert compat.supports_long_cache_retention is False


def test_get_compat_ignores_none_valued_overrides():
    model = make_model(compat={"supportsStore": None, "maxTokensField": "max_tokens"})
    compat = get_compat(model)
    # supportsStore=None is ignored; the auto-detected default is kept.
    assert compat.supports_store is True
    assert compat.max_tokens_field == "max_tokens"


def test_get_client_api_key_raises_with_no_headers_at_all():
    with pytest.raises(ValueError, match="No API key for provider: openai"):
        get_client_api_key("openai", None, None)


def test_convert_messages_skips_user_message_with_no_content_blocks():
    model = make_model()
    context = Context(messages=[UserMessage(content=[]), UserMessage(content="after")])
    params = convert_messages(model, context, get_compat(model))
    assert len(params) == 1
    assert params[0]["content"] == "after"


def test_convert_messages_requires_thinking_as_text():
    model = make_model(compat={"requiresThinkingAsText": True})
    assistant = AssistantMessage(
        api="openai-completions",
        provider="openai",
        model="gpt-test",
        content=[ThinkingContent(thinking="deep thought"), TextContent(text="answer")],
        stop_reason="stop",
    )
    params = convert_messages(model, Context(messages=[assistant]), get_compat(model))
    assert params[0]["content"] == [
        {"type": "text", "text": "deep thought"},
        {"type": "text", "text": "answer"},
    ]


def test_convert_messages_opencode_go_remaps_reasoning_signature():
    model = make_model(provider="opencode-go")
    assistant = AssistantMessage(
        api="openai-completions",
        provider="opencode-go",
        model="gpt-test",
        content=[ThinkingContent(thinking="deep", thinking_signature="reasoning"), TextContent(text="answer")],
        stop_reason="stop",
    )
    params = convert_messages(model, Context(messages=[assistant]), get_compat(model))
    assert params[0]["reasoning_content"] == "deep"
    assert params[0]["content"] == "answer"


def test_convert_messages_drops_unparsable_reasoning_detail_signature():
    model = make_model()
    assistant = AssistantMessage(
        api="openai-completions",
        provider="openai",
        model="gpt-test",
        content=[ToolCall(id="c1", name="read", thought_signature="not-json{")],
        stop_reason="toolUse",
    )
    params = convert_messages(model, Context(messages=[assistant]), get_compat(model))
    assert "reasoning_details" not in params[0]


def test_convert_messages_requires_reasoning_content_on_assistant_messages():
    model = make_model(reasoning=True, compat={"requiresReasoningContentOnAssistantMessages": True})
    assistant = AssistantMessage(
        api="openai-completions",
        provider="openai",
        model="gpt-test",
        content=[TextContent(text="hi")],
        stop_reason="stop",
    )
    params = convert_messages(model, Context(messages=[assistant]), get_compat(model))
    assert params[0]["reasoning_content"] == ""


def test_convert_tools_strict_prefers_and_requires():
    from pi_ai.types import JsonSchemaConstrainedSampling

    require_tool = Tool(name="a", description="d", constrained_sampling=JsonSchemaConstrainedSampling(strict="require"))
    prefer_tool = Tool(name="b", description="d", constrained_sampling=JsonSchemaConstrainedSampling(strict="prefer"))
    compat = get_compat(make_model())
    converted = convert_tools([require_tool, prefer_tool], compat)
    assert converted[0]["function"]["strict"] is True
    assert converted[1]["function"]["strict"] is True


def test_convert_tools_strict_false_for_grammar_constrained_sampling():
    from pi_ai.types import GrammarConstrainedSampling

    tool = Tool(name="a", description="d", constrained_sampling=GrammarConstrainedSampling())
    converted = convert_tools([tool], get_compat(make_model()))
    assert converted[0]["function"]["strict"] is False


def test_convert_messages_tool_result_name_omitted_when_tool_name_blank():
    model = make_model(compat={"requiresToolResultName": True})
    assistant = AssistantMessage(
        api="openai-completions",
        provider="openai",
        model="gpt-test",
        content=[ToolCall(id="c1", name="read")],
        stop_reason="toolUse",
    )
    context = Context(
        messages=[assistant, ToolResultMessage(tool_call_id="c1", tool_name="", content=[TextContent(text="ok")])]
    )
    params = convert_messages(model, context, get_compat(model))
    assert "name" not in params[-1]
