from pi_ai.api.transform_messages import transform_messages
from pi_ai.types import (
    AssistantMessage,
    ImageContent,
    Model,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


def make_model(**overrides) -> Model:
    defaults = dict(
        id="model-a",
        name="Model A",
        api="anthropic-messages",
        provider="anthropic",
        base_url="https://example.test",
        input=["text", "image"],
    )
    defaults.update(overrides)
    return Model(**defaults)


def same_model_assistant(**overrides) -> AssistantMessage:
    defaults = dict(api="anthropic-messages", provider="anthropic", model="model-a")
    defaults.update(overrides)
    return AssistantMessage(**defaults)


def other_model_assistant(**overrides) -> AssistantMessage:
    defaults = dict(api="openai-completions", provider="openai", model="model-b")
    defaults.update(overrides)
    return AssistantMessage(**defaults)


# --------------------------------------------------------------------------
# image downgrade
# --------------------------------------------------------------------------


def test_downgrades_user_message_images_for_text_only_model():
    model = make_model(input=["text"])
    messages = [UserMessage(content=[TextContent(text="look"), ImageContent(data="d", mime_type="image/png")])]
    result = transform_messages(messages, model)
    assert result[0].content == [
        TextContent(text="look"),
        TextContent(text="(image omitted: model does not support images)"),
    ]


def test_downgrade_collapses_consecutive_images_into_single_placeholder():
    model = make_model(input=["text"])
    messages = [
        UserMessage(
            content=[
                ImageContent(data="a", mime_type="image/png"),
                ImageContent(data="b", mime_type="image/png"),
                ImageContent(data="c", mime_type="image/png"),
            ]
        )
    ]
    result = transform_messages(messages, model)
    assert result[0].content == [TextContent(text="(image omitted: model does not support images)")]


def test_downgrade_tool_result_images_use_tool_placeholder():
    model = make_model(input=["text"])
    messages = [
        ToolResultMessage(
            tool_call_id="tc1",
            tool_name="tool",
            content=[ImageContent(data="a", mime_type="image/png")],
        )
    ]
    result = transform_messages(messages, model)
    assert result[0].content == [TextContent(text="(tool image omitted: model does not support images)")]


def test_no_downgrade_when_model_supports_images():
    model = make_model(input=["text", "image"])
    image = ImageContent(data="a", mime_type="image/png")
    messages = [UserMessage(content=[image])]
    result = transform_messages(messages, model)
    assert result[0].content == [image]


def test_downgrade_leaves_assistant_messages_untouched():
    model = make_model(input=["text"])
    assistant = same_model_assistant(content=[TextContent(text="hi")])
    messages = [assistant]
    result = transform_messages(messages, model)
    assert result[0].content == [TextContent(text="hi")]


def test_none_content_is_normalized_to_empty_list():
    model = make_model()
    tool_result = ToolResultMessage(tool_call_id="tc1", tool_name="tool")
    tool_result.content = None
    result = transform_messages([tool_result], model)
    assert result[0].content == []


# --------------------------------------------------------------------------
# thinking blocks
# --------------------------------------------------------------------------


def test_redacted_thinking_kept_for_same_model():
    model = make_model()
    block = ThinkingContent(thinking="", thinking_signature=None, redacted=True)
    messages = [same_model_assistant(content=[block])]
    result = transform_messages(messages, model)
    assert result[0].content == [block]


def test_redacted_thinking_dropped_cross_model():
    model = make_model()
    block = ThinkingContent(thinking="secret", thinking_signature=None, redacted=True)
    messages = [other_model_assistant(content=[block])]
    result = transform_messages(messages, model)
    assert result[0].content == []


def test_signed_thinking_kept_for_same_model_even_when_empty():
    model = make_model()
    block = ThinkingContent(thinking="", thinking_signature="sig-1", redacted=False)
    messages = [same_model_assistant(content=[block])]
    result = transform_messages(messages, model)
    assert result[0].content == [block]


def test_empty_thinking_without_signature_is_dropped():
    model = make_model()
    block = ThinkingContent(thinking="   ", thinking_signature=None, redacted=False)
    messages_same = [same_model_assistant(content=[block])]
    assert transform_messages(messages_same, model)[0].content == []

    messages_other = [other_model_assistant(content=[block])]
    assert transform_messages(messages_other, model)[0].content == []


def test_thinking_converted_to_text_cross_model():
    model = make_model()
    block = ThinkingContent(thinking="reasoning text", thinking_signature=None, redacted=False)
    messages = [other_model_assistant(content=[block])]
    result = transform_messages(messages, model)
    assert result[0].content == [TextContent(text="reasoning text")]


def test_thinking_kept_as_thinking_for_same_model():
    model = make_model()
    block = ThinkingContent(thinking="reasoning text", thinking_signature=None, redacted=False)
    messages = [same_model_assistant(content=[block])]
    result = transform_messages(messages, model)
    assert result[0].content == [block]


# --------------------------------------------------------------------------
# tool call thought_signature / id normalization
# --------------------------------------------------------------------------


def test_thought_signature_stripped_cross_model():
    model = make_model()
    tool_call = ToolCall(id="tc1", name="tool", arguments={}, thought_signature="sig")
    messages = [other_model_assistant(content=[tool_call])]
    result = transform_messages(messages, model)
    assert result[0].content[0].thought_signature is None


def test_thought_signature_kept_for_same_model():
    model = make_model()
    tool_call = ToolCall(id="tc1", name="tool", arguments={}, thought_signature="sig")
    messages = [same_model_assistant(content=[tool_call])]
    result = transform_messages(messages, model)
    assert result[0].content[0].thought_signature == "sig"


def test_tool_call_id_normalization_propagates_to_matching_tool_result():
    model = make_model()
    tool_call = ToolCall(id="original-id", name="tool", arguments={})
    messages = [
        other_model_assistant(content=[tool_call]),
        ToolResultMessage(tool_call_id="original-id", tool_name="tool", content=[TextContent(text="ok")]),
    ]

    def normalize(call_id, model, source):
        return f"normalized-{call_id}"

    result = transform_messages(messages, model, normalize_tool_call_id=normalize)
    assert result[0].content[0].id == "normalized-original-id"
    assert result[1].tool_call_id == "normalized-original-id"


def test_tool_call_id_normalization_not_applied_for_same_model():
    model = make_model()
    tool_call = ToolCall(id="original-id", name="tool", arguments={})
    messages = [same_model_assistant(content=[tool_call])]

    def normalize(call_id, model, source):
        return f"normalized-{call_id}"

    result = transform_messages(messages, model, normalize_tool_call_id=normalize)
    assert result[0].content[0].id == "original-id"


def test_tool_call_id_normalization_noop_when_normalized_id_is_unchanged():
    model = make_model()
    tool_call = ToolCall(id="tc1", name="tool", arguments={})
    messages = [other_model_assistant(content=[tool_call])]

    def normalize(call_id, model, source):
        return call_id

    result = transform_messages(messages, model, normalize_tool_call_id=normalize)
    assert result[0].content[0].id == "tc1"


def test_unrecognized_content_block_type_passes_through_unchanged():
    model = make_model()
    image_block = ImageContent(data="a", mime_type="image/png")
    messages = [other_model_assistant(content=[image_block])]
    result = transform_messages(messages, model)
    assert result[0].content == [image_block]


# --------------------------------------------------------------------------
# errored/aborted assistant messages
# --------------------------------------------------------------------------


def test_errored_assistant_message_is_skipped_entirely():
    model = make_model()
    messages = [
        same_model_assistant(content=[TextContent(text="partial")], stop_reason="error"),
        UserMessage(content="follow up"),
    ]
    result = transform_messages(messages, model)
    assert len(result) == 1
    assert result[0].role == "user"


def test_aborted_assistant_message_is_skipped_entirely():
    model = make_model()
    messages = [same_model_assistant(content=[TextContent(text="partial")], stop_reason="aborted")]
    result = transform_messages(messages, model)
    assert result == []


# --------------------------------------------------------------------------
# synthetic tool results for orphaned tool calls
# --------------------------------------------------------------------------


def test_synthetic_tool_result_inserted_when_conversation_ends_with_unresolved_call():
    model = make_model()
    tool_call = ToolCall(id="tc1", name="tool", arguments={})
    messages = [same_model_assistant(content=[tool_call])]
    result = transform_messages(messages, model)
    assert len(result) == 2
    synthetic = result[1]
    assert synthetic.role == "toolResult"
    assert synthetic.tool_call_id == "tc1"
    assert synthetic.tool_name == "tool"
    assert synthetic.is_error is True
    assert synthetic.content == [TextContent(text="No result provided")]


def test_synthetic_tool_result_inserted_when_user_message_interrupts():
    model = make_model()
    tool_call = ToolCall(id="tc1", name="tool", arguments={})
    messages = [
        same_model_assistant(content=[tool_call]),
        UserMessage(content="interrupting"),
    ]
    result = transform_messages(messages, model)
    assert [m.role for m in result] == ["assistant", "toolResult", "user"]
    assert result[1].tool_call_id == "tc1"


def test_existing_tool_result_suppresses_synthetic_one():
    model = make_model()
    tool_call = ToolCall(id="tc1", name="tool", arguments={})
    messages = [
        same_model_assistant(content=[tool_call]),
        ToolResultMessage(tool_call_id="tc1", tool_name="tool", content=[TextContent(text="real result")]),
    ]
    result = transform_messages(messages, model)
    assert len(result) == 2
    assert result[1].content == [TextContent(text="real result")]


def test_multiple_orphaned_tool_calls_from_same_assistant_message_all_get_synthetic_results():
    model = make_model()
    tool_calls = [
        ToolCall(id="tc1", name="tool1", arguments={}),
        ToolCall(id="tc2", name="tool2", arguments={}),
    ]
    messages = [same_model_assistant(content=tool_calls)]
    result = transform_messages(messages, model)
    assert len(result) == 3
    assert {r.tool_call_id for r in result[1:]} == {"tc1", "tc2"}


def test_next_assistant_message_flushes_previous_orphaned_tool_calls():
    model = make_model()
    tool_call_1 = ToolCall(id="tc1", name="tool", arguments={})
    tool_call_2 = ToolCall(id="tc2", name="tool", arguments={})
    messages = [
        same_model_assistant(content=[tool_call_1]),
        same_model_assistant(content=[tool_call_2]),
    ]
    result = transform_messages(messages, model)
    assert [m.role for m in result] == ["assistant", "toolResult", "assistant", "toolResult"]
    assert result[1].tool_call_id == "tc1"
    assert result[3].tool_call_id == "tc2"
