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
from pi_ai.api.anthropic_messages import (
    AnthropicOptions,
    build_headers,
    build_params,
    convert_content_blocks,
    convert_messages,
    convert_tools,
    default_supports_tool_references,
    detect_anthropic_compat,
    from_claude_code_name,
    get_anthropic_compat,
    get_cache_control,
    has_header,
    is_oauth_token,
    map_stop_reason,
    map_thinking_level_to_effort,
    merge_headers,
    normalize_tool_call_id,
    resolve_cache_retention,
    stream,
    stream_simple,
    to_claude_code_name,
)
from pi_ai.providers import anthropic_provider


def make_model(**overrides) -> Model:
    defaults = dict(
        id="claude-test",
        name="Claude Test",
        api="anthropic-messages",
        provider="anthropic",
        base_url="https://api.anthropic.com/v1",
        reasoning=False,
        input=["text", "image"],
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
        context_window=200_000,
        max_tokens=8192,
    )
    defaults.update(overrides)
    return Model(**defaults)


def sse_body(events: list[tuple[str, dict]]) -> str:
    lines = []
    for event_name, data in events:
        lines.append(f"event: {event_name}\ndata: {json.dumps(data)}\n\n")
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


def message_start_event(**usage_overrides) -> tuple[str, dict]:
    usage = {"input_tokens": 10, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    usage.update(usage_overrides)
    return (
        "message_start",
        {
            "type": "message_start",
            "message": {"id": "msg_1", "usage": usage},
        },
    )


def message_delta_event(stop_reason: str, **usage_overrides) -> tuple[str, dict]:
    payload: dict = {"type": "message_delta", "delta": {"stop_reason": stop_reason}}
    if usage_overrides:
        payload["usage"] = usage_overrides
    return ("message_delta", payload)


def message_stop_event() -> tuple[str, dict]:
    return ("message_stop", {"type": "message_stop"})


# --------------------------------------------------------------------------
# compat detection
# --------------------------------------------------------------------------


def test_detect_compat_defaults():
    compat = detect_anthropic_compat(make_model())
    assert compat.supports_eager_tool_input_streaming is True
    assert compat.supports_long_cache_retention is True
    assert compat.send_session_affinity_headers is False
    assert compat.supports_cache_control_on_tools is True
    assert compat.supports_temperature is True
    assert compat.allow_empty_signature is False
    assert compat.supports_strict_tools is False


def test_default_supports_tool_references_false_for_haiku_and_non_anthropic():
    assert default_supports_tool_references(make_model(id="claude-haiku-4-5", provider="anthropic")) is False
    assert default_supports_tool_references(make_model(id="claude-opus-5", provider="openrouter")) is False


def test_default_supports_tool_references_true_for_opus_5_and_sonnet_4_5():
    assert default_supports_tool_references(make_model(id="claude-opus-5", provider="anthropic")) is True
    assert default_supports_tool_references(make_model(id="claude-sonnet-4-5", provider="anthropic")) is True


def test_default_supports_tool_references_false_for_old_models():
    assert default_supports_tool_references(make_model(id="claude-opus-4-1", provider="anthropic")) is False
    assert default_supports_tool_references(make_model(id="claude-3-opus", provider="anthropic")) is False


def test_get_compat_applies_model_overrides_in_both_spellings():
    camel = get_anthropic_compat(make_model(compat={"supportsTemperature": False, "allowEmptySignature": True}))
    assert camel.supports_temperature is False
    assert camel.allow_empty_signature is True

    snake = get_anthropic_compat(make_model(compat={"supports_temperature": False}))
    assert snake.supports_temperature is False


def test_get_compat_ignores_force_adaptive_thinking_key():
    compat = get_anthropic_compat(make_model(compat={"forceAdaptiveThinking": True}))
    assert not hasattr(compat, "force_adaptive_thinking")


# --------------------------------------------------------------------------
# cache retention
# --------------------------------------------------------------------------


def test_resolve_cache_retention_defaults_to_short():
    assert resolve_cache_retention(None, {}) == "short"


def test_resolve_cache_retention_reads_env_for_long():
    assert resolve_cache_retention(None, {"PI_CACHE_RETENTION": "long"}) == "long"


def test_resolve_cache_retention_prefers_explicit_value():
    assert resolve_cache_retention("none", {"PI_CACHE_RETENTION": "long"}) == "none"


def test_get_cache_control_none_omits_cache_control():
    result = get_cache_control(make_model(), "none")
    assert result.cache_control is None


def test_get_cache_control_short_has_no_ttl():
    result = get_cache_control(make_model(), "short")
    assert result.cache_control == {"type": "ephemeral"}


def test_get_cache_control_long_sets_1h_ttl_when_supported():
    result = get_cache_control(make_model(), "long")
    assert result.cache_control == {"type": "ephemeral", "ttl": "1h"}


def test_get_cache_control_long_omits_ttl_when_unsupported():
    model = make_model(compat={"supportsLongCacheRetention": False})
    result = get_cache_control(model, "long")
    assert result.cache_control == {"type": "ephemeral"}


# --------------------------------------------------------------------------
# Claude Code tool naming
# --------------------------------------------------------------------------


def test_to_claude_code_name_matches_case_insensitively():
    assert to_claude_code_name("read") == "Read"
    assert to_claude_code_name("BASH") == "Bash"
    assert to_claude_code_name("unknown_tool") == "unknown_tool"


def test_from_claude_code_name_matches_available_tools():
    # from_claude_code_name matches a tool literally named "read" (any casing),
    # not a semantic alias for Claude Code's canonical "Read" tool.
    tools = [Tool(name="read", description="d"), Tool(name="Bash", description="d")]
    assert from_claude_code_name("Read", tools) == "read"
    assert from_claude_code_name("bash", tools) == "Bash"
    assert from_claude_code_name("Grep", tools) == "Grep"
    assert from_claude_code_name("Grep", None) == "Grep"


# --------------------------------------------------------------------------
# header / auth helpers
# --------------------------------------------------------------------------


def test_merge_headers_combines_sources():
    assert merge_headers({"a": "1"}, None, {"b": "2"}) == {"a": "1", "b": "2"}


def test_has_header_case_insensitive():
    assert has_header({"Authorization": "x"}, "authorization") is True
    assert has_header({"authorization": "  "}, "authorization") is False
    assert has_header(None, "authorization") is False


def test_is_oauth_token():
    assert is_oauth_token("sk-ant-oat01-abc") is True
    assert is_oauth_token("sk-ant-api03-abc") is False


def test_build_headers_api_key_path():
    headers, is_oauth = build_headers(make_model(), "sk-ant-api-key", True, False)
    assert headers["x-api-key"] == "sk-ant-api-key"
    assert headers["anthropic-beta"] == "interleaved-thinking-2025-05-14"
    assert is_oauth is False


def test_build_headers_oauth_path_sends_claude_code_identity():
    headers, is_oauth = build_headers(make_model(), "sk-ant-oat-xyz", True, False)
    assert is_oauth is True
    assert headers["authorization"] == "Bearer sk-ant-oat-xyz"
    assert "claude-code-20250219" in headers["anthropic-beta"]
    assert "oauth-2025-04-20" in headers["anthropic-beta"]
    assert headers["x-app"] == "cli"
    assert "x-api-key" not in headers


def test_build_headers_fine_grained_tool_streaming_beta():
    headers, _ = build_headers(make_model(), "sk-ant-key", False, True)
    assert "fine-grained-tool-streaming-2025-05-14" in headers["anthropic-beta"]


def test_build_headers_session_affinity_when_enabled():
    model = make_model(compat={"sendSessionAffinityHeaders": True})
    headers, _ = build_headers(model, "sk-ant-key", False, False, session_id="sess-1")
    assert headers["x-session-affinity"] == "sess-1"


def test_build_headers_options_headers_override_and_delete():
    headers, _ = build_headers(make_model(), "sk-ant-key", False, False, {"x-custom": "1", "accept": None})
    assert headers["x-custom"] == "1"
    assert "accept" not in headers


def test_build_headers_github_copilot_sets_bearer_and_no_api_key():
    model = make_model(provider="github-copilot")
    headers, is_oauth = build_headers(model, "copilot-tok", True, False)
    assert headers["authorization"] == "Bearer copilot-tok"
    assert "x-api-key" not in headers
    assert is_oauth is False


def test_build_headers_github_copilot_merges_dynamic_headers():
    model = make_model(provider="github-copilot")
    headers, _ = build_headers(
        model,
        "copilot-tok",
        True,
        False,
        dynamic_headers={"X-Initiator": "agent", "Openai-Intent": "conversation-edits"},
    )
    assert headers["X-Initiator"] == "agent"
    assert headers["Openai-Intent"] == "conversation-edits"
    assert headers["anthropic-beta"] == "interleaved-thinking-2025-05-14"


def test_build_headers_github_copilot_options_headers_override_dynamic():
    model = make_model(provider="github-copilot")
    headers, _ = build_headers(
        model,
        "copilot-tok",
        True,
        False,
        {"X-Initiator": "override"},
        dynamic_headers={"X-Initiator": "agent"},
    )
    assert headers["X-Initiator"] == "override"


# --------------------------------------------------------------------------
# tool call id normalization
# --------------------------------------------------------------------------


def test_normalize_tool_call_id_sanitizes_and_truncates():
    assert normalize_tool_call_id("call!!!1") == "call___1"
    assert len(normalize_tool_call_id("x" * 100)) == 64


# --------------------------------------------------------------------------
# content block conversion
# --------------------------------------------------------------------------


def test_convert_content_blocks_text_only_returns_string():
    result = convert_content_blocks([TextContent(text="hello"), TextContent(text="world")])
    assert result == "hello\nworld"


def test_convert_content_blocks_with_images_returns_block_list():
    result = convert_content_blocks([TextContent(text="hi"), ImageContent(data="AAA", mime_type="image/png")])
    assert result[0] == {"type": "text", "text": "hi"}
    assert result[1]["source"] == {"type": "base64", "media_type": "image/png", "data": "AAA"}


def test_convert_content_blocks_images_only_adds_placeholder_text():
    result = convert_content_blocks([ImageContent(data="AAA", mime_type="image/png")])
    assert result[0] == {"type": "text", "text": "(see attached image)"}


# --------------------------------------------------------------------------
# message conversion
# --------------------------------------------------------------------------


def test_convert_messages_string_user_content():
    params = convert_messages([UserMessage(content="hi")], is_oauth_token=False)
    assert params == [{"role": "user", "content": "hi"}]


def test_convert_messages_user_content_with_images():
    context_messages = [
        UserMessage(content=[TextContent(text="look"), ImageContent(data="AAA", mime_type="image/png")])
    ]
    params = convert_messages(context_messages, is_oauth_token=False)
    assert params[0]["content"][0] == {"type": "text", "text": "look"}
    assert params[0]["content"][1]["type"] == "image"


def test_convert_messages_skips_blank_user_message():
    params = convert_messages([UserMessage(content="   ")], is_oauth_token=False)
    assert params == []


def test_convert_messages_assistant_tool_call_and_text():
    assistant = AssistantMessage(
        api="anthropic-messages",
        provider="anthropic",
        model="claude-test",
        content=[TextContent(text="calling"), ToolCall(id="t1", name="read", arguments={"path": "a.txt"})],
        stop_reason="toolUse",
    )
    params = convert_messages([assistant], is_oauth_token=False)
    assert params[0]["content"][0] == {"type": "text", "text": "calling"}
    assert params[0]["content"][1] == {"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "a.txt"}}


def test_convert_messages_renames_tool_call_for_oauth():
    assistant = AssistantMessage(
        api="anthropic-messages",
        provider="anthropic",
        model="claude-test",
        content=[ToolCall(id="t1", name="read", arguments={})],
        stop_reason="toolUse",
    )
    params = convert_messages([assistant], is_oauth_token=True)
    assert params[0]["content"][0]["name"] == "Read"


def test_convert_messages_thinking_with_signature_replayed():
    assistant = AssistantMessage(
        api="anthropic-messages",
        provider="anthropic",
        model="claude-test",
        content=[ThinkingContent(thinking="deep thought", thinking_signature="sig-abc")],
        stop_reason="stop",
    )
    params = convert_messages([assistant], is_oauth_token=False)
    assert params[0]["content"][0] == {"type": "thinking", "thinking": "deep thought", "signature": "sig-abc"}


def test_convert_messages_thinking_without_signature_converts_to_text():
    assistant = AssistantMessage(
        api="anthropic-messages",
        provider="anthropic",
        model="claude-test",
        content=[ThinkingContent(thinking="deep thought", thinking_signature="")],
        stop_reason="stop",
    )
    params = convert_messages([assistant], is_oauth_token=False)
    assert params[0]["content"][0] == {"type": "text", "text": "deep thought"}


def test_convert_messages_thinking_without_signature_allow_empty_signature():
    assistant = AssistantMessage(
        api="anthropic-messages",
        provider="anthropic",
        model="claude-test",
        content=[ThinkingContent(thinking="deep thought", thinking_signature=None)],
        stop_reason="stop",
    )
    params = convert_messages([assistant], is_oauth_token=False, allow_empty_signature=True)
    assert params[0]["content"][0] == {"type": "thinking", "thinking": "deep thought", "signature": ""}


def test_convert_messages_redacted_thinking_replayed_as_redacted_block():
    assistant = AssistantMessage(
        api="anthropic-messages",
        provider="anthropic",
        model="claude-test",
        content=[ThinkingContent(thinking="[Reasoning redacted]", thinking_signature="opaque-data", redacted=True)],
        stop_reason="stop",
    )
    params = convert_messages([assistant], is_oauth_token=False)
    assert params[0]["content"][0] == {"type": "redacted_thinking", "data": "opaque-data"}


def test_convert_messages_tool_result_becomes_user_message():
    tool_result = ToolResultMessage(tool_call_id="t1", tool_name="read", content=[TextContent(text="ok")])
    params = convert_messages([tool_result], is_oauth_token=False)
    assert params[0]["role"] == "user"
    assert params[0]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "t1",
        "content": "ok",
        "is_error": False,
    }


def test_convert_messages_consecutive_tool_results_grouped():
    results = [
        ToolResultMessage(tool_call_id="t1", tool_name="a", content=[TextContent(text="1")]),
        ToolResultMessage(tool_call_id="t2", tool_name="b", content=[TextContent(text="2")]),
    ]
    params = convert_messages(results, is_oauth_token=False)
    assert len(params) == 1
    assert [c["tool_use_id"] for c in params[0]["content"]] == ["t1", "t2"]


def test_convert_messages_cache_control_on_last_user_message():
    params = convert_messages([UserMessage(content="hi")], is_oauth_token=False, cache_control={"type": "ephemeral"})
    assert params[-1]["content"] == [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]


def test_convert_messages_cache_control_on_last_block_of_array_content():
    context_messages = [UserMessage(content=[TextContent(text="a"), TextContent(text="b")])]
    params = convert_messages(context_messages, is_oauth_token=False, cache_control={"type": "ephemeral"})
    assert params[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in params[-1]["content"][0]


def test_convert_messages_tool_reference_for_deferred_tool_name():
    tool_result = ToolResultMessage(
        tool_call_id="t1", tool_name="loader", content=[TextContent(text="loaded")], added_tool_names=["search"]
    )
    params = convert_messages([tool_result], is_oauth_token=False, deferred_tool_names={"search"})
    content = params[0]["content"]
    assert content[0]["content"] == [{"type": "tool_reference", "tool_name": "search"}]
    # Sibling content carries the actual tool result text alongside the reference.
    assert {"type": "text", "text": "loaded"} in content


# --------------------------------------------------------------------------
# tool conversion
# --------------------------------------------------------------------------


def test_convert_tools_basic_shape():
    tool = Tool(name="read", description="Read a file", parameters={"type": "object", "properties": {"path": {}}})
    converted = convert_tools(
        [tool], is_oauth_token=False, supports_eager_tool_input_streaming=True, supports_strict_tools=False
    )
    assert converted == [
        {
            "name": "read",
            "description": "Read a file",
            "eager_input_streaming": True,
            "input_schema": {"type": "object", "properties": {"path": {}}, "required": []},
        }
    ]


def test_convert_tools_renames_for_oauth():
    tool = Tool(name="read", description="d")
    converted = convert_tools(
        [tool], is_oauth_token=True, supports_eager_tool_input_streaming=False, supports_strict_tools=False
    )
    assert converted[0]["name"] == "Read"
    assert "eager_input_streaming" not in converted[0]


def test_convert_tools_cache_control_on_last_tool_only():
    tools = [Tool(name="a", description="d"), Tool(name="b", description="d")]
    converted = convert_tools(
        tools,
        is_oauth_token=False,
        supports_eager_tool_input_streaming=False,
        supports_strict_tools=False,
        cache_control={"type": "ephemeral"},
    )
    assert "cache_control" not in converted[0]
    assert converted[1]["cache_control"] == {"type": "ephemeral"}


def test_convert_tools_defer_loading_flag():
    tools = [Tool(name="a", description="d")]
    converted = convert_tools(
        tools,
        is_oauth_token=False,
        supports_eager_tool_input_streaming=False,
        supports_strict_tools=False,
        defer_loading=True,
    )
    assert converted[0]["defer_loading"] is True


# --------------------------------------------------------------------------
# stop reason mapping
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("end_turn", ("stop", None)),
        ("max_tokens", ("length", None)),
        ("tool_use", ("toolUse", None)),
        ("pause_turn", ("stop", None)),
        ("stop_sequence", ("stop", None)),
    ],
)
def test_map_stop_reason(reason, expected):
    assert map_stop_reason(reason) == expected


def test_map_stop_reason_refusal_uses_explanation():
    result = map_stop_reason("refusal", {"explanation": "policy violation"})
    assert result == ("error", "policy violation")


def test_map_stop_reason_refusal_default_message():
    result = map_stop_reason("refusal", None)
    assert result == ("error", "The model refused to complete the request")


def test_map_stop_reason_sensitive():
    assert map_stop_reason("sensitive") == ("error", "Provider stopped with: sensitive")


def test_map_stop_reason_unknown_raises():
    with pytest.raises(ValueError, match="Unhandled stop reason"):
        map_stop_reason("something_new")


def test_map_thinking_level_to_effort_defaults():
    model = make_model(reasoning=True)
    assert map_thinking_level_to_effort(model, "low") == "low"
    assert map_thinking_level_to_effort(model, "minimal") == "low"
    assert map_thinking_level_to_effort(model, "medium") == "medium"
    assert map_thinking_level_to_effort(model, "high") == "high"
    assert map_thinking_level_to_effort(model, None) == "high"


def test_map_thinking_level_to_effort_uses_thinking_level_map_override():
    model = make_model(reasoning=True, thinking_level_map={"high": "xhigh"})
    assert map_thinking_level_to_effort(model, "high") == "xhigh"


# --------------------------------------------------------------------------
# build_params
# --------------------------------------------------------------------------


def test_build_params_basic_shape():
    model = make_model()
    params = build_params(model, Context(messages=[UserMessage(content="hi")]), False, AnthropicOptions())
    assert params["model"] == "claude-test"
    assert params["stream"] is True
    assert params["max_tokens"] == model.max_tokens


def test_build_params_uses_options_max_tokens():
    model = make_model()
    params = build_params(model, Context(messages=[]), False, AnthropicOptions(max_tokens=555))
    assert params["max_tokens"] == 555


def test_build_params_system_prompt_top_level():
    # Default cache retention is "short", which still attaches an ephemeral
    # (no-ttl) cache_control to the system prompt.
    model = make_model()
    params = build_params(model, Context(system_prompt="be nice", messages=[]), False, AnthropicOptions())
    assert params["system"] == [{"type": "text", "text": "be nice", "cache_control": {"type": "ephemeral"}}]


def test_build_params_system_prompt_no_cache_control_when_retention_none():
    model = make_model()
    params = build_params(
        model, Context(system_prompt="be nice", messages=[]), False, AnthropicOptions(cache_retention="none")
    )
    assert params["system"] == [{"type": "text", "text": "be nice"}]


def test_build_params_oauth_injects_claude_code_identity():
    model = make_model()
    params = build_params(model, Context(system_prompt="be nice", messages=[]), True, AnthropicOptions())
    assert params["system"][0]["text"] == "You are Claude Code, Anthropic's official CLI for Claude."
    assert params["system"][1]["text"] == "be nice"


def test_build_params_temperature_omitted_when_thinking_enabled():
    model = make_model()
    params = build_params(model, Context(messages=[]), False, AnthropicOptions(temperature=0.5, thinking_enabled=True))
    assert "temperature" not in params


def test_build_params_temperature_included_when_supported():
    model = make_model()
    params = build_params(model, Context(messages=[]), False, AnthropicOptions(temperature=0.5))
    assert params["temperature"] == 0.5


def test_build_params_temperature_omitted_when_unsupported():
    model = make_model(compat={"supportsTemperature": False})
    params = build_params(model, Context(messages=[]), False, AnthropicOptions(temperature=0.5))
    assert "temperature" not in params


def test_build_params_thinking_budget_based():
    model = make_model(reasoning=True)
    params = build_params(
        model, Context(messages=[]), False, AnthropicOptions(thinking_enabled=True, thinking_budget_tokens=2048)
    )
    assert params["thinking"] == {"type": "enabled", "budget_tokens": 2048, "display": "summarized"}


def test_build_params_thinking_adaptive_with_effort():
    model = make_model(reasoning=True, compat={"forceAdaptiveThinking": True})
    params = build_params(model, Context(messages=[]), False, AnthropicOptions(thinking_enabled=True, effort="high"))
    assert params["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert params["output_config"] == {"effort": "high"}


def test_build_params_thinking_disabled_sends_disabled_type():
    model = make_model(reasoning=True)
    params = build_params(model, Context(messages=[]), False, AnthropicOptions(thinking_enabled=False))
    assert params["thinking"] == {"type": "disabled"}


def test_build_params_thinking_disabled_skipped_when_off_is_null():
    model = make_model(reasoning=True, thinking_level_map={"off": None})
    params = build_params(model, Context(messages=[]), False, AnthropicOptions(thinking_enabled=False))
    assert "thinking" not in params


def test_build_params_tool_choice_string():
    model = make_model()
    params = build_params(model, Context(messages=[]), False, AnthropicOptions(tool_choice="any"))
    assert params["tool_choice"] == {"type": "any"}


def test_build_params_tool_choice_object():
    model = make_model()
    params = build_params(
        model, Context(messages=[]), False, AnthropicOptions(tool_choice={"type": "tool", "name": "read"})
    )
    assert params["tool_choice"] == {"type": "tool", "name": "read"}


def test_build_params_includes_tools():
    model = make_model()
    tool = Tool(name="read", description="Read a file")
    params = build_params(model, Context(messages=[], tools=[tool]), False, AnthropicOptions())
    assert params["tools"][0]["name"] == "read"


def test_build_params_metadata_user_id():
    model = make_model()
    params = build_params(model, Context(messages=[]), False, AnthropicOptions(metadata={"user_id": "u1"}))
    assert params["metadata"] == {"user_id": "u1"}


# --------------------------------------------------------------------------
# provider factory
# --------------------------------------------------------------------------


async def test_anthropic_provider_resolves_auth_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-from-env")
    provider = anthropic_provider()
    assert provider.id == "anthropic"
    from pi_ai.auth.helpers import resolve_api_key_auth

    result = await resolve_api_key_auth(provider.auth.api_key)
    assert result is not None
    assert result.auth.api_key == "sk-ant-api-from-env"
    assert result.source == "ANTHROPIC_API_KEY"


def test_anthropic_provider_models_have_real_ids_and_costs():
    provider = anthropic_provider()
    ids = {m.id for m in provider.models}
    assert "claude-opus-5" in ids
    assert "claude-sonnet-5" in ids
    assert all(m.base_url == "https://api.anthropic.com" for m in provider.models)
    assert all(m.cost.input > 0 for m in provider.models)


# --------------------------------------------------------------------------
# streaming
# --------------------------------------------------------------------------


async def test_stream_emits_text_events_and_final_message():
    body = sse_body(
        [
            message_start_event(),
            (
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            message_delta_event("end_turn", output_tokens=5),
            message_stop_event(),
        ]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(
                make_model(),
                Context(messages=[UserMessage(content="hi")]),
                AnthropicOptions(api_key="k"),
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
    assert message.response_id == "msg_1"
    assert message.usage.input == 10
    assert message.usage.output == 5


async def test_stream_thinking_with_signature_delta():
    body = sse_body(
        [
            message_start_event(),
            (
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}},
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig-1"}},
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            message_delta_event("end_turn"),
            message_stop_event(),
        ]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(
                make_model(reasoning=True),
                Context(messages=[]),
                AnthropicOptions(api_key="k", thinking_enabled=True),
                client=client,
            )
        )
    assert [e.type for e in events] == ["start", "thinking_start", "thinking_delta", "thinking_end", "done"]
    thinking_block = message.content[0]
    assert thinking_block.thinking == "hmm"
    assert thinking_block.thinking_signature == "sig-1"


async def test_stream_tool_use_with_input_json_delta_accumulation():
    body = sse_body(
        [
            message_start_event(),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "tool_use", "id": "t1", "name": "read", "input": {}},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '{"pa'},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": 'th": "a.txt"}'},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            message_delta_event("tool_use"),
            message_stop_event(),
        ]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), AnthropicOptions(api_key="k"), client=client)
        )
    assert [e.type for e in events] == [
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


async def test_stream_renames_tool_from_claude_code_name_for_oauth():
    tools = [Tool(name="read", description="d")]
    body = sse_body(
        [
            message_start_event(),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            message_delta_event("tool_use"),
            message_stop_event(),
        ]
    )
    async with make_client(body) as client:
        _events, message = await collect(
            stream(
                make_model(),
                Context(messages=[], tools=tools),
                AnthropicOptions(api_key="sk-ant-oat-token"),
                client=client,
            )
        )
    assert message.content[0].name == "read"


async def test_stream_cache_and_usage_accumulation():
    body = sse_body(
        [
            message_start_event(cache_read_input_tokens=100, cache_creation_input_tokens=50),
            message_delta_event(
                "end_turn", cache_read_input_tokens=120, cache_creation_input_tokens=60, output_tokens=8
            ),
            message_stop_event(),
        ]
    )
    async with make_client(body) as client:
        _events, message = await collect(
            stream(make_model(), Context(messages=[]), AnthropicOptions(api_key="k"), client=client)
        )
    assert message.usage.cache_read == 120
    assert message.usage.cache_write == 60
    assert message.usage.output == 8
    assert message.usage.total_tokens == message.usage.input + 8 + 120 + 60
    assert message.usage.cost.total > 0


async def test_stream_cache_write_1h_from_message_start():
    body = sse_body(
        [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "usage": {
                            "input_tokens": 5,
                            "output_tokens": 0,
                            "cache_creation": {"ephemeral_1h_input_tokens": 42},
                        },
                    },
                },
            ),
            message_delta_event("end_turn"),
            message_stop_event(),
        ]
    )
    async with make_client(body) as client:
        _events, message = await collect(
            stream(make_model(), Context(messages=[]), AnthropicOptions(api_key="k"), client=client)
        )
    assert message.usage.cache_write_1h == 42


async def test_stream_error_event_reports_through_stream():
    body = "event: message_start\ndata: " + json.dumps(message_start_event()[1]) + "\n\n"
    body += 'event: error\ndata: {"type": "error", "error": {"type": "overloaded_error", "message": "overloaded"}}\n\n'
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), AnthropicOptions(api_key="k"), client=client)
        )
    assert events[-1].type == "error"
    assert message.stop_reason == "error"
    assert "overloaded" in message.error_message


async def test_stream_reports_http_error_through_stream():
    async with make_client('{"error": {"message": "invalid api key"}}', status=401) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), AnthropicOptions(api_key="k"), client=client)
        )
    assert events[-1].type == "error"
    assert message.stop_reason == "error"
    assert "invalid api key" in message.error_message


async def test_stream_errors_without_a_stop_reason():
    body = sse_body([message_start_event()])
    # No message_stop event: iterate_anthropic_events raises before completion.
    async with make_client(body) as client:
        _events, message = await collect(
            stream(make_model(), Context(messages=[]), AnthropicOptions(api_key="k"), client=client)
        )
    assert message.stop_reason == "error"


async def test_stream_async_on_payload_replacement():
    body = sse_body([message_start_event(), message_delta_event("end_turn"), message_stop_event()])
    capture: dict = {}

    async def on_payload(params, model):
        params["metadata"] = {"user_id": "async-injected"}
        return params

    async with make_client(body, capture=capture) as client:
        await collect(
            stream(
                make_model(),
                Context(messages=[]),
                AnthropicOptions(api_key="k", on_payload=on_payload),
                client=client,
            )
        )
    assert capture["json"]["metadata"] == {"user_id": "async-injected"}


async def test_stream_reports_error_when_signal_already_aborted():
    from pi_ai.utils.abort import AbortSignal

    signal = AbortSignal()
    signal.abort()
    body = sse_body([message_start_event(), message_delta_event("end_turn"), message_stop_event()])
    async with make_client(body) as client:
        events, message = await collect(
            stream(
                make_model(),
                Context(messages=[]),
                AnthropicOptions(api_key="k", signal=signal),
                client=client,
            )
        )
    assert events[-1].type == "error"
    assert message.stop_reason == "aborted"


async def test_stream_refusal_stop_reason_reports_error_through_stream():
    body = sse_body(
        [
            message_start_event(),
            ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "refusal"}}),
            message_stop_event(),
        ]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), AnthropicOptions(api_key="k"), client=client)
        )
    assert events[-1].type == "error"
    assert message.stop_reason == "error"
    assert "refused" in message.error_message


async def test_stream_message_stop_without_message_delta_reports_pending_error():
    body = sse_body([message_start_event(), message_stop_event()])
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), AnthropicOptions(api_key="k"), client=client)
        )
    assert events[-1].type == "error"
    assert "without a stop reason" in message.error_message


async def test_stream_empty_response_reports_error_without_start_event():
    async with make_client("") as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), AnthropicOptions(api_key="k"), client=client)
        )
    assert [e.type for e in events] == ["start", "error"]
    assert message.stop_reason == "error"


async def test_stream_unknown_stop_reason_reports_through_stream():
    body = sse_body([message_start_event(), message_delta_event("some_new_reason"), message_stop_event()])
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), AnthropicOptions(api_key="k"), client=client)
        )
    assert events[-1].type == "error"
    assert message.stop_reason == "error"
    assert "some_new_reason" in message.error_message


async def test_stream_redacted_thinking_content_block_start():
    body = sse_body(
        [
            message_start_event(),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "redacted_thinking", "data": "opaque"},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            message_delta_event("end_turn"),
            message_stop_event(),
        ]
    )
    async with make_client(body) as client:
        events, message = await collect(
            stream(make_model(), Context(messages=[]), AnthropicOptions(api_key="k"), client=client)
        )
    assert [e.type for e in events] == ["start", "thinking_start", "thinking_end", "done"]
    block = message.content[0]
    assert block.redacted is True
    assert block.thinking_signature == "opaque"


async def test_stream_invokes_on_payload_and_on_response_callbacks():
    body = sse_body([message_start_event(), message_delta_event("end_turn"), message_stop_event()])
    payload_calls: list[dict] = []
    response_calls: list = []

    def on_payload(params, model):
        payload_calls.append(params)
        params["metadata"] = {"user_id": "injected"}
        return params

    async def on_response(response, model):
        response_calls.append((response.status, model.id))

    async with make_client(body) as client:
        await collect(
            stream(
                make_model(),
                Context(messages=[]),
                AnthropicOptions(api_key="k", on_payload=on_payload, on_response=on_response),
                client=client,
            )
        )
    assert payload_calls and payload_calls[0]["metadata"] == {"user_id": "injected"}
    assert response_calls == [(200, "claude-test")]


async def test_stream_sends_expected_request_and_headers():
    capture: dict = {}
    body = sse_body([message_start_event(), message_delta_event("end_turn"), message_stop_event()])
    async with make_client(body, capture=capture) as client:
        await collect(
            stream(
                make_model(),
                Context(system_prompt="sys", messages=[UserMessage(content="hi")]),
                AnthropicOptions(api_key="sk-ant-test-key", headers={"x-custom": "1"}),
                client=client,
            )
        )
    request = capture["request"]
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == "sk-ant-test-key"
    assert request.headers["x-custom"] == "1"
    assert capture["json"]["system"][0]["text"] == "sys"
    # Cache control is also appended to the last user message's last block.
    assert capture["json"]["messages"][0]["content"] == [
        {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
    ]


async def test_stream_github_copilot_sends_dynamic_headers_and_bearer_auth():
    capture: dict = {}
    body = sse_body([message_start_event(), message_delta_event("end_turn"), message_stop_event()])
    model = make_model(provider="github-copilot")
    async with make_client(body, capture=capture) as client:
        await collect(
            stream(
                model,
                Context(messages=[UserMessage(content=[ImageContent(data="AAA", mime_type="image/png")])]),
                AnthropicOptions(api_key="copilot-tok"),
                client=client,
            )
        )
    request = capture["request"]
    assert request.headers["authorization"] == "Bearer copilot-tok"
    assert "x-api-key" not in request.headers
    assert request.headers["x-initiator"] == "user"
    assert request.headers["openai-intent"] == "conversation-edits"
    assert request.headers["copilot-vision-request"] == "true"


async def test_stream_missing_api_key_reports_error_without_raising():
    events, message = await collect(stream(make_model(), Context(messages=[]), AnthropicOptions()))
    assert events[-1].type == "error"
    assert message.stop_reason == "error"
    assert "No API key" in message.error_message


async def test_stream_simple_disables_thinking_without_reasoning():
    from pi_ai import SimpleStreamOptions

    capture: dict = {}
    body = sse_body([message_start_event(), message_delta_event("end_turn"), message_stop_event()])
    async with make_client(body, capture=capture) as client:
        await collect(
            stream_simple(
                make_model(reasoning=True),
                Context(messages=[UserMessage(content="hi")]),
                SimpleStreamOptions(api_key="k"),
                client=client,
            )
        )
    assert capture["json"]["thinking"] == {"type": "disabled"}


async def test_stream_simple_budget_based_thinking():
    from pi_ai import SimpleStreamOptions

    capture: dict = {}
    body = sse_body([message_start_event(), message_delta_event("end_turn"), message_stop_event()])
    model = make_model(reasoning=True, max_tokens=8192)
    async with make_client(body, capture=capture) as client:
        await collect(
            stream_simple(
                model,
                Context(messages=[UserMessage(content="hi")]),
                SimpleStreamOptions(api_key="k", reasoning="medium"),
                client=client,
            )
        )
    assert capture["json"]["thinking"]["type"] == "enabled"
    # medium budget defaults to 8192, but max_tokens is capped at the model's
    # 8192 ceiling, so the budget is squeezed to leave room for the answer.
    assert capture["json"]["thinking"]["budget_tokens"] == 7168


async def test_stream_simple_adaptive_thinking_effort():
    from pi_ai import SimpleStreamOptions

    capture: dict = {}
    body = sse_body([message_start_event(), message_delta_event("end_turn"), message_stop_event()])
    model = make_model(reasoning=True, compat={"forceAdaptiveThinking": True})
    async with make_client(body, capture=capture) as client:
        await collect(
            stream_simple(
                model,
                Context(messages=[UserMessage(content="hi")]),
                SimpleStreamOptions(api_key="k", reasoning="high"),
                client=client,
            )
        )
    assert capture["json"]["thinking"] == {"type": "adaptive", "display": "summarized"}
