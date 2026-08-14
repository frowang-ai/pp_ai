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
from pi_ai.api.openai_responses import (
    OPENAI_RESPONSES_MIN_OUTPUT_TOKENS,
    OpenAIResponsesOptions,
    apply_service_tier_pricing,
    build_params,
    detect_compat,
    get_client_api_key,
    get_compat,
    get_prompt_cache_retention,
    get_service_tier_cost_multiplier,
    stream,
    stream_simple,
)
from pi_ai.api.openai_responses_shared import (
    ConvertResponsesMessagesOptions,
    ConvertResponsesToolsOptions,
    _build_foreign_responses_item_id,
    _convert_tool_result_output,
    _encode_text_signature_v1,
    _model_supports_developer_role,
    _normalize_id_part,
    _parse_text_signature,
    convert_responses_messages,
    convert_responses_tools,
    map_stop_reason,
)
from pi_ai.providers import openai_responses_provider
from pi_ai.types import GrammarConstrainedSampling


def make_model(**overrides) -> Model:
    defaults = dict(
        id="gpt-responses-test",
        name="GPT Responses Test",
        api="openai-responses",
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


def sse_body(chunks: list[dict], event_name: bool = False) -> str:
    lines = []
    for chunk in chunks:
        if event_name:
            lines.append(f"event: {chunk['type']}\n")
        lines.append(f"data: {json.dumps(chunk)}\n\n")
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


def test_detect_compat_defaults():
    compat = detect_compat(make_model())
    assert compat.supports_developer_role is True
    assert compat.session_affinity_format == "openai"
    assert compat.supports_long_cache_retention is True
    assert compat.supports_strict_mode is False
    assert compat.supports_openai_grammar_tools is False
    assert compat.supports_additional_tools is False
    assert compat.supports_tool_search is False
    assert compat.supports_explicit_prompt_cache_mode is False


def test_detect_compat_openrouter_session_affinity():
    compat = detect_compat(make_model(provider="openrouter", base_url="https://openrouter.ai/api/v1"))
    assert compat.session_affinity_format == "openrouter"


def test_get_compat_applies_model_overrides_in_both_spellings():
    camel = get_compat(make_model(compat={"supportsStrictMode": True, "supportsOpenAIGrammarTools": True}))
    assert camel.supports_strict_mode is True
    assert camel.supports_openai_grammar_tools is True

    snake = get_compat(make_model(compat={"supports_strict_mode": True}))
    assert snake.supports_strict_mode is True


def test_get_prompt_cache_retention():
    compat = detect_compat(make_model())
    assert get_prompt_cache_retention(compat, "long") == "24h"
    assert get_prompt_cache_retention(compat, "short") is None
    assert get_prompt_cache_retention(compat, "none") is None


# --------------------------------------------------------------------------
# get_client_api_key
# --------------------------------------------------------------------------


def test_get_client_api_key_uses_explicit_key():
    assert get_client_api_key("openai", "sk-1", None) == "sk-1"


def test_get_client_api_key_accepts_authorization_header():
    assert get_client_api_key("openai", None, {"Authorization": "Bearer x"}) == "unused"


def test_get_client_api_key_raises_without_key():
    with pytest.raises(ValueError, match="No API key"):
        get_client_api_key("openai", None, None)


# --------------------------------------------------------------------------
# message conversion
# --------------------------------------------------------------------------


def test_convert_messages_user_text():
    context = Context(messages=[UserMessage(content="hello")])
    items = convert_responses_messages(make_model(), context, {"openai"})
    assert items == [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]


def test_convert_messages_user_image():
    context = Context(
        messages=[UserMessage(content=[TextContent(text="look"), ImageContent(data="Zm9v", mime_type="image/png")])]
    )
    items = convert_responses_messages(make_model(input=["text", "image"]), context, {"openai"})
    assert items[0]["content"][0] == {"type": "input_text", "text": "look"}
    assert items[0]["content"][1] == {
        "type": "input_image",
        "detail": "auto",
        "image_url": "data:image/png;base64,Zm9v",
    }


def test_convert_messages_tool_result_text_and_image():
    context = Context(
        messages=[
            ToolResultMessage(
                tool_call_id="call1|fc_1",
                tool_name="read",
                content=[TextContent(text="result")],
            )
        ]
    )
    items = convert_responses_messages(make_model(), context, {"openai"})
    assert items[0]["type"] == "function_call_output"
    assert items[0]["call_id"] == "call1"
    assert items[0]["output"] == "result"


def test_convert_messages_tool_result_with_image_and_vision_support():
    context = Context(
        messages=[
            ToolResultMessage(
                tool_call_id="call1|fc_1",
                tool_name="read",
                content=[TextContent(text="result"), ImageContent(data="Zm9v", mime_type="image/png")],
            )
        ]
    )
    items = convert_responses_messages(make_model(input=["text", "image"]), context, {"openai"})
    assert items[0]["output"] == [
        {"type": "input_text", "text": "result"},
        {"type": "input_image", "detail": "auto", "image_url": "data:image/png;base64,Zm9v"},
    ]


def test_convert_messages_assistant_reasoning_replay():
    model = make_model(reasoning=True)
    reasoning_item = {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": "enc"}
    context = Context(
        messages=[
            AssistantMessage(
                provider=model.provider,
                api=model.api,
                model=model.id,
                content=[ThinkingContent(thinking="thoughts", thinking_signature=json.dumps(reasoning_item))],
            )
        ]
    )
    items = convert_responses_messages(model, context, {"openai"})
    assert items[0] == reasoning_item


def test_convert_messages_assistant_toolcall_same_model_keeps_id():
    model = make_model()
    context = Context(
        messages=[
            AssistantMessage(
                provider=model.provider,
                api=model.api,
                model=model.id,
                content=[ToolCall(id="call1|fc_1", name="read", arguments={"path": "a"})],
            )
        ]
    )
    items = convert_responses_messages(model, context, {"openai"})
    assert items[0] == {
        "type": "function_call",
        "call_id": "call1",
        "name": "read",
        "arguments": '{"path":"a"}',  # JSON.stringify: no spaces
        "id": "fc_1",
    }


def test_convert_messages_assistant_toolcall_different_model_drops_id():
    model = make_model()
    context = Context(
        messages=[
            AssistantMessage(
                provider=model.provider,
                api=model.api,
                model="a-different-model",
                content=[ToolCall(id="call1|fc_1", name="read", arguments={"path": "a"})],
            )
        ]
    )
    items = convert_responses_messages(model, context, {"openai"})
    assert "id" not in items[0]


def test_convert_messages_system_prompt_uses_developer_role_for_reasoning_models():
    model = make_model(reasoning=True)
    context = Context(system_prompt="sys", messages=[])
    items = convert_responses_messages(model, context, {"openai"})
    assert items[0] == {"role": "developer", "content": "sys"}


def test_convert_messages_system_prompt_uses_system_role_for_non_reasoning_models():
    model = make_model(reasoning=False)
    context = Context(system_prompt="sys", messages=[])
    items = convert_responses_messages(model, context, {"openai"})
    assert items[0] == {"role": "system", "content": "sys"}


def test_convert_messages_system_prompt_uses_system_role_when_developer_role_unsupported():
    model = make_model(reasoning=True, compat={"supportsDeveloperRole": False})
    context = Context(system_prompt="sys", messages=[])
    items = convert_responses_messages(model, context, {"openai"})
    assert items[0] == {"role": "system", "content": "sys"}


def test_convert_messages_long_text_signature_id_is_hashed():
    long_id = "x" * 100
    parsed_signature = _encode_text_signature_v1(long_id)
    model = make_model()
    context = Context(
        messages=[
            AssistantMessage(
                provider=model.provider,
                api=model.api,
                model=model.id,
                content=[TextContent(text="hi", text_signature=parsed_signature)],
            )
        ]
    )
    items = convert_responses_messages(model, context, {"openai"})
    assert len(items[0]["id"]) <= 64
    assert items[0]["id"].startswith("msg_")


def test_convert_messages_deferred_tools_additional_tools_mode():
    read_tool = Tool(name="read", description="reads a file")
    context = Context(
        messages=[
            ToolResultMessage(
                tool_call_id="call1|fc_1",
                tool_name="search",
                content=[TextContent(text="found it")],
                added_tool_names=["read"],
            )
        ]
    )
    items = convert_responses_messages(
        make_model(),
        context,
        {"openai"},
        ConvertResponsesMessagesOptions(deferred_tools={"read": read_tool}, deferred_tools_mode="additional-tools"),
    )
    assert items[0]["type"] == "function_call_output"
    assert items[1]["type"] == "additional_tools"
    assert items[1]["tools"][0]["name"] == "read"


def test_convert_messages_deferred_tools_tool_search_mode():
    read_tool = Tool(name="read", description="reads a file")
    context = Context(
        messages=[
            ToolResultMessage(
                tool_call_id="call1|fc_1",
                tool_name="search",
                content=[TextContent(text="found it")],
                added_tool_names=["read"],
            )
        ]
    )
    items = convert_responses_messages(
        make_model(),
        context,
        {"openai"},
        ConvertResponsesMessagesOptions(deferred_tools={"read": read_tool}, deferred_tools_mode="tool-search"),
    )
    types = [item["type"] for item in items]
    assert types == ["function_call_output", "tool_search_call", "tool_search_output"]
    assert items[2]["tools"][0]["name"] == "read"


def test_convert_messages_custom_tool_call_grammar_replay():
    tool = Tool(
        name="calc",
        description="calculator",
        parameters={
            "type": "object",
            "required": ["expr"],
            "properties": {"expr": {"type": "string"}},
        },
        constrained_sampling=GrammarConstrainedSampling(variants={"openai_lark": "start: NUMBER"}),
    )
    model = make_model()
    grammar_props = {"calc": "expr"}
    context = Context(
        messages=[
            AssistantMessage(
                provider=model.provider,
                api=model.api,
                model=model.id,
                content=[ToolCall(id="call1|ctc_1", name="calc", arguments={"expr": "1+1"})],
            )
        ],
        tools=[tool],
    )
    items = convert_responses_messages(
        model, context, {"openai"}, ConvertResponsesMessagesOptions(grammar_tool_input_properties=grammar_props)
    )
    assert items[0]["type"] == "custom_tool_call"
    assert items[0]["input"] == "1+1"


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def test_encode_and_parse_text_signature_round_trip_with_phase():
    encoded = _encode_text_signature_v1("msg_1", "final_answer")
    parsed = _parse_text_signature(encoded)
    assert parsed == ("msg_1", "final_answer")


def test_encode_and_parse_text_signature_round_trip_without_phase():
    encoded = _encode_text_signature_v1("msg_1")
    parsed = _parse_text_signature(encoded)
    assert parsed == ("msg_1", None)


def test_parse_text_signature_handles_legacy_plain_string():
    assert _parse_text_signature("legacy-signature") == ("legacy-signature", None)


def test_parse_text_signature_handles_none():
    assert _parse_text_signature(None) is None


def test_parse_text_signature_ignores_malformed_json():
    assert _parse_text_signature("{not json") == ("{not json", None)


def test_model_supports_developer_role_default_true():
    assert _model_supports_developer_role(make_model()) is True


def test_model_supports_developer_role_false_camel_case():
    assert _model_supports_developer_role(make_model(compat={"supportsDeveloperRole": False})) is False


def test_model_supports_developer_role_false_snake_case():
    assert _model_supports_developer_role(make_model(compat={"supports_developer_role": False})) is False


def test_convert_tool_result_output_image_only():
    model = make_model(input=["text"])
    result = _convert_tool_result_output(model, [ImageContent(data="Zm9v", mime_type="image/png")])
    assert result == "(see attached image)"


def test_convert_tool_result_output_empty():
    model = make_model()
    assert _convert_tool_result_output(model, []) == "(no tool output)"


def test_normalize_id_part_truncates_and_strips_trailing_underscore():
    assert _normalize_id_part("a" * 70) == "a" * 64
    assert _normalize_id_part("weird!!chars??") == "weird__chars"


def test_build_foreign_responses_item_id_starts_with_fc():
    assert _build_foreign_responses_item_id("some-foreign-id").startswith("fc_")


# --------------------------------------------------------------------------
# tool conversion
# --------------------------------------------------------------------------


def test_convert_tools_strict_mode_on():
    tool = Tool(name="read", description="reads a file", parameters={"type": "object", "properties": {}})
    converted = convert_responses_tools([tool], ConvertResponsesToolsOptions(supports_strict_mode=True))
    assert converted == [
        {
            "type": "function",
            "name": "read",
            "description": "reads a file",
            "parameters": {"type": "object", "properties": {}},
            "strict": False,
        }
    ]


def test_convert_tools_strict_mode_off_omits_strict_key():
    tool = Tool(name="read", description="reads a file")
    converted = convert_responses_tools([tool], ConvertResponsesToolsOptions(supports_strict_mode=False))
    assert "strict" not in converted[0]


# --------------------------------------------------------------------------
# build_params
# --------------------------------------------------------------------------


def test_build_params_max_output_tokens_clamped_to_minimum():
    params = build_params(make_model(), Context(messages=[]), OpenAIResponsesOptions(api_key="k", max_tokens=1))
    assert params["max_output_tokens"] == OPENAI_RESPONSES_MIN_OUTPUT_TOKENS


def test_build_params_temperature():
    params = build_params(make_model(), Context(messages=[]), OpenAIResponsesOptions(api_key="k", temperature=0.5))
    assert params["temperature"] == 0.5


def test_build_params_reasoning_effort_mapped_through_thinking_level_map():
    model = make_model(reasoning=True, thinking_level_map={"medium": "med-effort"})
    params = build_params(model, Context(messages=[]), OpenAIResponsesOptions(api_key="k", reasoning_effort="medium"))
    assert params["reasoning"] == {"effort": "med-effort", "summary": "auto"}
    assert params["include"] == ["reasoning.encrypted_content"]


def test_build_params_reasoning_off_uses_thinking_level_map_off_value():
    model = make_model(reasoning=True, thinking_level_map={"off": "disabled"})
    params = build_params(model, Context(messages=[]), OpenAIResponsesOptions(api_key="k"))
    assert params["reasoning"] == {"effort": "disabled"}


def test_build_params_reasoning_off_skipped_when_map_marks_none():
    model = make_model(reasoning=True, thinking_level_map={"off": None})
    params = build_params(model, Context(messages=[]), OpenAIResponsesOptions(api_key="k"))
    assert "reasoning" not in params


def test_build_params_tool_choice():
    params = build_params(make_model(), Context(messages=[]), OpenAIResponsesOptions(api_key="k", tool_choice="auto"))
    assert params["tool_choice"] == "auto"


def test_build_params_prompt_cache_key_and_retention():
    model = make_model()
    params = build_params(
        model,
        Context(messages=[]),
        OpenAIResponsesOptions(api_key="k", session_id="sess-1", cache_retention="long"),
    )
    assert params["prompt_cache_key"] == "sess-1"
    assert params["prompt_cache_retention"] == "24h"


def test_build_params_no_cache_key_when_retention_none():
    params = build_params(
        make_model(),
        Context(messages=[]),
        OpenAIResponsesOptions(api_key="k", session_id="sess-1", cache_retention="none"),
    )
    assert "prompt_cache_key" not in params


def test_build_params_store_always_false():
    params = build_params(make_model(), Context(messages=[]), OpenAIResponsesOptions(api_key="k"))
    assert params["store"] is False


def test_build_params_sampling_params_override_named_fields():
    params = build_params(
        make_model(),
        Context(messages=[]),
        OpenAIResponsesOptions(api_key="k", temperature=0.1, sampling_params={"temperature": 0.9}),
    )
    assert params["temperature"] == 0.9


# --------------------------------------------------------------------------
# service tier pricing
# --------------------------------------------------------------------------


def test_service_tier_cost_multiplier_flex():
    assert get_service_tier_cost_multiplier("gpt-5.1", "flex") == 0.5


def test_service_tier_cost_multiplier_priority():
    assert get_service_tier_cost_multiplier("gpt-5.1", "priority") == 2.0
    assert get_service_tier_cost_multiplier("gpt-5.5", "priority") == 2.5


def test_service_tier_cost_multiplier_default():
    assert get_service_tier_cost_multiplier("gpt-5.1", None) == 1.0


def test_apply_service_tier_pricing_scales_costs():
    from pi_ai.types import Cost, Usage

    usage = Usage(cost=Cost(input=1.0, output=2.0, cache_read=0.5, cache_write=0.25))
    apply_service_tier_pricing(usage, "flex", "gpt-5.1")
    assert usage.cost.input == 0.5
    assert usage.cost.output == 1.0
    assert usage.cost.total == 0.5 + 1.0 + 0.25 + 0.125


# --------------------------------------------------------------------------
# stop reason mapping
# --------------------------------------------------------------------------


def test_map_stop_reason_completed():
    assert map_stop_reason("completed") == ("stop", None)


def test_map_stop_reason_incomplete_max_output_tokens():
    assert map_stop_reason("incomplete", "max_output_tokens") == ("length", None)


def test_map_stop_reason_incomplete_other_reason():
    reason, message = map_stop_reason("incomplete", "content_filter")
    assert reason == "error"
    assert "content_filter" in message


def test_map_stop_reason_failed_and_cancelled():
    assert map_stop_reason("failed") == ("error", None)
    assert map_stop_reason("cancelled") == ("error", None)


def test_map_stop_reason_in_progress_and_queued():
    assert map_stop_reason("in_progress") == ("stop", None)
    assert map_stop_reason("queued") == ("stop", None)


def test_map_stop_reason_none_status():
    assert map_stop_reason(None) == ("stop", None)


def test_map_stop_reason_unknown_raises():
    with pytest.raises(ValueError):
        map_stop_reason("some-unknown-status")


# --------------------------------------------------------------------------
# streaming
# --------------------------------------------------------------------------


async def test_stream_emits_text_events_and_final_message():
    body = sse_body(
        [
            {"type": "response.created", "response": {"id": "resp_1"}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "message", "id": "msg_1"},
            },
            {"type": "response.output_text.delta", "output_index": 0, "delta": "Hel"},
            {"type": "response.output_text.delta", "output_index": 0, "delta": "lo"},
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "msg_1",
                    "content": [{"type": "output_text", "text": "Hello"}],
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "output": [],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                },
            },
        ]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(
                make_model(),
                Context(messages=[UserMessage(content="hi")]),
                OpenAIResponsesOptions(api_key="k"),
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
    assert message.response_id == "resp_1"
    assert message.usage.input == 10
    assert message.usage.output == 5


async def test_stream_reasoning_summary_and_encrypted_content():
    body = sse_body(
        [
            {"type": "response.created", "response": {"id": "resp_1"}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "reasoning", "id": "rs_1"},
            },
            {"type": "response.reasoning_summary_text.delta", "output_index": 0, "delta": "thinking "},
            {"type": "response.reasoning_summary_text.delta", "output_index": 0, "delta": "hard"},
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"text": "thinking hard"}],
                    "encrypted_content": "enc-abc",
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "output": [],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            },
        ]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(reasoning=True), Context(messages=[]), OpenAIResponsesOptions(api_key="k"), client=client)
        )

    assert [event.type for event in events] == [
        "start",
        "thinking_start",
        "thinking_delta",
        "thinking_delta",
        "thinking_end",
        "done",
    ]
    thinking = message.content[0]
    assert thinking.thinking == "thinking hard"
    signature = json.loads(thinking.thinking_signature)
    assert signature["encrypted_content"] == "enc-abc"


async def test_stream_reasoning_signature_backfilled_from_completed_response():
    body = sse_body(
        [
            {"type": "response.created", "response": {"id": "resp_1"}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "reasoning", "id": "rs_1"},
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "reasoning", "id": "rs_1", "summary": [{"text": "thoughts"}]},
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "output": [{"type": "reasoning", "id": "rs_1", "encrypted_content": "backfilled"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            },
        ]
    )
    async with make_client(body) as client:
        _events, message = await collect(
            stream(make_model(reasoning=True), Context(messages=[]), OpenAIResponsesOptions(api_key="k"), client=client)
        )
    signature = json.loads(message.content[0].thinking_signature)
    assert signature["encrypted_content"] == "backfilled"


async def test_stream_function_call_with_argument_deltas():
    body = sse_body(
        [
            {"type": "response.created", "response": {"id": "resp_1"}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "function_call", "call_id": "call1", "id": "fc_1", "name": "read", "arguments": ""},
            },
            {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"pa'},
            {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": 'th": "a.txt"}'},
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "call_id": "call1",
                    "id": "fc_1",
                    "name": "read",
                    "arguments": '{"path": "a.txt"}',
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "output": [],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            },
        ]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), OpenAIResponsesOptions(api_key="k"), client=client)
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
    assert tool_call.id == "call1|fc_1"


async def test_stream_custom_tool_call_grammar_input_deltas():
    body = sse_body(
        [
            {"type": "response.created", "response": {"id": "resp_1"}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "custom_tool_call", "call_id": "call1", "id": "ctc_1", "name": "calc", "input": ""},
            },
            {"type": "response.custom_tool_call_input.delta", "output_index": 0, "delta": "1+1"},
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "custom_tool_call",
                    "call_id": "call1",
                    "id": "ctc_1",
                    "name": "calc",
                    "input": "1+1",
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "output": [],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            },
        ]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), OpenAIResponsesOptions(api_key="k"), client=client)
        )
    assert [event.type for event in events] == [
        "start",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]
    tool_call = message.content[0]
    assert tool_call.name == "calc"
    assert tool_call.arguments == {"input": "1+1"}


async def test_stream_refusal_delta_appends_to_text():
    body = sse_body(
        [
            {"type": "response.created", "response": {"id": "resp_1"}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "message", "id": "msg_1"},
            },
            {"type": "response.refusal.delta", "output_index": 0, "delta": "I can't help with that"},
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "msg_1",
                    "content": [{"type": "refusal", "refusal": "I can't help with that"}],
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "output": [],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            },
        ]
    )
    async with make_client(body) as client:
        _events, message = await collect(
            stream(make_model(), Context(messages=[]), OpenAIResponsesOptions(api_key="k"), client=client)
        )
    assert message.content[0].text == "I can't help with that"


async def test_stream_final_answer_phase_sets_stop_reason_stop():
    body = sse_body(
        [
            {"type": "response.created", "response": {"id": "resp_1"}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "message", "id": "msg_1", "phase": "final_answer"},
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "msg_1",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "done"}],
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "output": [],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            },
        ]
    )
    async with make_client(body) as client:
        _events, message = await collect(
            stream(make_model(), Context(messages=[]), OpenAIResponsesOptions(api_key="k"), client=client)
        )
    # apply_message_phase_stop_reason marks the running message as "stop" as soon
    # as a final_answer-phase item is seen; finalize_response's incomplete-status
    # mapping runs afterward and determines the final persisted stop_reason, but the
    # phase is still encoded into the replayable text signature.
    assert message.content[0].text_signature is not None
    assert json.loads(message.content[0].text_signature)["phase"] == "final_answer"
    assert message.stop_reason == "length"


async def test_stream_response_failed_event_reports_error():
    body = sse_body(
        [
            {
                "type": "response.failed",
                "response": {
                    "status": "failed",
                    "error": {"code": "server_error", "message": "boom"},
                },
            }
        ]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), OpenAIResponsesOptions(api_key="k"), client=client)
        )
    assert events[-1].type == "error"
    assert message.stop_reason == "error"
    assert "boom" in message.error_message


async def test_stream_reports_http_error_through_stream():
    async with make_client('{"error": {"message": "bad key"}}', status=401) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), OpenAIResponsesOptions(api_key="k"), client=client)
        )
    assert events[-1].type == "error"
    assert message.stop_reason == "error"
    assert "bad key" in message.error_message


async def test_stream_reports_error_named_event():
    body = sse_body([{"type": "error", "code": "rate_limit", "message": "slow down"}])
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), OpenAIResponsesOptions(api_key="k"), client=client)
        )
    assert events[-1].type == "error"
    assert "rate_limit" in message.error_message
    assert "slow down" in message.error_message


async def test_stream_ends_without_terminal_event_is_an_error():
    body = sse_body(
        [
            {"type": "response.created", "response": {"id": "resp_1"}},
        ]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), OpenAIResponsesOptions(api_key="k"), client=client)
        )
    assert events[-1].type == "error"
    assert message.stop_reason == "error"
    assert "terminal response event" in message.error_message


async def test_stream_sends_expected_request():
    capture: dict = {}
    body = sse_body(
        [
            {
                "type": "response.completed",
                "response": {"id": "resp_1", "status": "completed", "output": [], "usage": {}},
            }
        ]
    )
    async with make_client(body, capture=capture) as client:
        await collect(
            stream(
                make_model(),
                Context(system_prompt="sys", messages=[UserMessage(content="hi")]),
                OpenAIResponsesOptions(api_key="sk-test", headers={"x-custom": "1"}),
                client=client,
            )
        )
    request = capture["request"]
    assert str(request.url) == "https://api.openai.com/v1/responses"
    assert request.headers["authorization"] == "Bearer sk-test"
    assert request.headers["x-custom"] == "1"
    assert capture["json"]["input"][0] == {"role": "system", "content": "sys"}
    assert capture["json"]["stream"] is True


async def test_stream_on_payload_can_replace_request_body():
    capture: dict = {}
    body = sse_body(
        [
            {
                "type": "response.completed",
                "response": {"id": "resp_1", "status": "completed", "output": [], "usage": {}},
            }
        ]
    )

    def on_payload(payload, model):
        payload["metadata_marker"] = model.id
        return payload

    async with make_client(body, capture=capture) as client:
        await collect(
            stream(
                make_model(),
                Context(messages=[]),
                OpenAIResponsesOptions(api_key="k", on_payload=on_payload),
                client=client,
            )
        )
    assert capture["json"]["metadata_marker"] == "gpt-responses-test"


async def test_stream_invokes_on_response_hook():
    seen: list = []
    body = sse_body(
        [
            {
                "type": "response.completed",
                "response": {"id": "resp_1", "status": "completed", "output": [], "usage": {}},
            }
        ]
    )

    def on_response(response, model):
        seen.append((response.status, model.id))

    async with make_client(body) as client:
        await collect(
            stream(
                make_model(),
                Context(messages=[]),
                OpenAIResponsesOptions(api_key="k", on_response=on_response),
                client=client,
            )
        )
    assert seen == [(200, "gpt-responses-test")]


async def test_aborted_signal_terminates_the_stream_as_aborted():
    from pi_ai.utils.abort import AbortSignal

    signal = AbortSignal()
    signal.abort()
    body = sse_body(
        [
            {
                "type": "response.completed",
                "response": {"id": "resp_1", "status": "completed", "output": [], "usage": {}},
            }
        ]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(
                make_model(), Context(messages=[]), OpenAIResponsesOptions(api_key="k", signal=signal), client=client
            )
        )
    assert events[-1].type == "error"
    assert events[-1].reason == "aborted"
    assert message.stop_reason == "aborted"


async def test_unaborted_signal_does_not_change_the_outcome():
    from pi_ai.utils.abort import AbortSignal

    body = sse_body(
        [
            {
                "type": "response.completed",
                "response": {"id": "resp_1", "status": "completed", "output": [], "usage": {}},
            }
        ]
    )
    async with make_client(body) as client:
        _events, message = await collect(
            stream(
                make_model(),
                Context(messages=[]),
                OpenAIResponsesOptions(api_key="k", signal=AbortSignal()),
                client=client,
            )
        )
    assert message.stop_reason == "stop"


async def test_stream_simple_maps_reasoning_level_to_reasoning_effort():
    from pi_ai import SimpleStreamOptions

    capture: dict = {}
    body = sse_body(
        [
            {
                "type": "response.completed",
                "response": {"id": "resp_1", "status": "completed", "output": [], "usage": {}},
            }
        ]
    )
    model = make_model(reasoning=True, thinking_level_map={"high": "high"})
    async with make_client(body, capture=capture) as client:
        await collect(
            stream_simple(
                model,
                Context(messages=[]),
                SimpleStreamOptions(api_key="k", reasoning="high"),
                client=client,
            )
        )
    assert capture["json"]["reasoning"]["effort"] == "high"


async def test_stream_simple_off_reasoning_sends_none_effort_when_off_supported():
    from pi_ai import SimpleStreamOptions

    capture: dict = {}
    body = sse_body(
        [
            {
                "type": "response.completed",
                "response": {"id": "resp_1", "status": "completed", "output": [], "usage": {}},
            }
        ]
    )
    # Model does not disable the "off" level, so requesting "off" reasoning
    # sends an explicit `{"effort": "none"}` marker rather than omitting the field.
    model = make_model(reasoning=True)
    async with make_client(body, capture=capture) as client:
        await collect(
            stream_simple(
                model,
                Context(messages=[]),
                SimpleStreamOptions(api_key="k", reasoning="off"),
                client=client,
            )
        )
    assert capture["json"]["reasoning"] == {"effort": "none"}


async def test_stream_simple_no_reasoning_requested_omits_field_when_off_unsupported():
    from pi_ai import SimpleStreamOptions

    capture: dict = {}
    body = sse_body(
        [
            {
                "type": "response.completed",
                "response": {"id": "resp_1", "status": "completed", "output": [], "usage": {}},
            }
        ]
    )
    # thinking_level_map marks "off" as unsupported (always-reasoning model), so no
    # reasoning is requested and the reasoning field is omitted entirely.
    model = make_model(reasoning=True, thinking_level_map={"off": None})
    async with make_client(body, capture=capture) as client:
        await collect(
            stream_simple(
                model,
                Context(messages=[]),
                SimpleStreamOptions(api_key="k"),
                client=client,
            )
        )
    assert "reasoning" not in capture["json"]


# --------------------------------------------------------------------------
# provider factory
# --------------------------------------------------------------------------


async def test_openai_responses_provider_resolves_auth_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-from-env")
    provider = openai_responses_provider()
    assert provider.id == "openai-responses"
    from pi_ai.auth.helpers import resolve_api_key_auth

    result = await resolve_api_key_auth(provider.auth.api_key)
    assert result is not None
    assert result.auth.api_key == "sk-openai-from-env"
    assert result.source == "OPENAI_API_KEY"


def test_openai_responses_provider_models_have_real_ids_and_costs():
    provider = openai_responses_provider()
    ids = {m.id for m in provider.models}
    assert "gpt-5.1" in ids
    assert all(m.api == "openai-responses" for m in provider.models)
    assert all(m.provider == "openai-responses" for m in provider.models)
    assert all(m.base_url == "https://api.openai.com/v1" for m in provider.models)
    assert all(m.cost.input > 0 for m in provider.models)
