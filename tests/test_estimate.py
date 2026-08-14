from pi_ai.types import (
    AssistantMessage,
    Context,
    ImageContent,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pi_ai.utils.estimate import (
    CHARS_PER_TOKEN,
    ESTIMATED_IMAGE_CHARS,
    _safe_json_stringify,
    calculate_context_tokens,
    estimate_context_tokens,
    estimate_message_tokens,
    estimate_text_and_image_content_tokens,
    estimate_text_tokens,
)


def test_calculate_context_tokens_prefers_total_tokens():
    usage = Usage(input=10, output=5, cache_read=1, cache_write=1, total_tokens=100)
    assert calculate_context_tokens(usage) == 100


def test_calculate_context_tokens_falls_back_to_sum():
    usage = Usage(input=10, output=5, cache_read=1, cache_write=1, total_tokens=0)
    assert calculate_context_tokens(usage) == 17


def test_estimate_text_tokens_rounds_up():
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("a") == 1
    assert estimate_text_tokens("abcd") == 1
    assert estimate_text_tokens("abcde") == 2


def test_estimate_text_and_image_content_tokens_for_plain_string():
    text = "a" * 40
    assert estimate_text_and_image_content_tokens(text) == 10


def test_estimate_text_and_image_content_tokens_for_blocks():
    content = [TextContent(text="abcd"), ImageContent(data="x", mime_type="image/png")]
    expected_chars = 4 + ESTIMATED_IMAGE_CHARS
    import math

    assert estimate_text_and_image_content_tokens(content) == math.ceil(expected_chars / CHARS_PER_TOKEN)


def test_estimate_message_tokens_user_message():
    message = UserMessage(content="abcdefgh")
    assert estimate_message_tokens(message) == 2


def test_estimate_message_tokens_tool_result_message():
    message = ToolResultMessage(content=[TextContent(text="abcdefgh")])
    assert estimate_message_tokens(message) == 2


def test_estimate_message_tokens_assistant_text_and_thinking():
    message = AssistantMessage(
        content=[TextContent(text="abcd"), TextContent(text="efgh")],
    )
    assert estimate_message_tokens(message) == 2


def test_estimate_message_tokens_assistant_tool_call_uses_name_and_json_arguments():
    message = AssistantMessage(content=[ToolCall(id="1", name="tool", arguments={"a": 1})])
    # "tool" (4 chars) + json.dumps({"a":1}, separators=(",",":")) == '{"a":1}' (7 chars) => 11 chars
    assert estimate_message_tokens(message) == 3


def test_estimate_context_tokens_accepts_message_list():
    messages = [UserMessage(content="abcdefgh")]
    estimate = estimate_context_tokens(messages)
    assert estimate.tokens == 2
    assert estimate.last_usage_index is None


def test_estimate_context_tokens_uses_last_applicable_assistant_usage():
    usage = Usage(input=100, output=50, total_tokens=150)
    assistant = AssistantMessage(content=[TextContent(text="hi")], usage=usage, stop_reason="stop", timestamp=1)
    trailing = UserMessage(content="abcd", timestamp=2)
    context = Context(messages=[UserMessage(content="q", timestamp=0), assistant, trailing])

    estimate = estimate_context_tokens(context)

    assert estimate.usage_tokens == 150
    assert estimate.trailing_tokens == estimate_message_tokens(trailing)
    assert estimate.tokens == 150 + estimate.trailing_tokens
    assert estimate.last_usage_index == 1


def test_estimate_context_tokens_ignores_aborted_and_error_assistant_usage():
    usage = Usage(input=100, output=50, total_tokens=150)
    aborted = AssistantMessage(usage=usage, stop_reason="aborted", timestamp=0)
    errored = AssistantMessage(usage=usage, stop_reason="error", timestamp=1)
    context = Context(messages=[aborted, errored])

    estimate = estimate_context_tokens(context)

    assert estimate.last_usage_index is None
    assert estimate.usage_tokens == 0


def test_estimate_context_tokens_ignores_zero_token_usage():
    zero_usage = Usage()
    assistant = AssistantMessage(usage=zero_usage, stop_reason="stop", timestamp=0)
    context = Context(messages=[assistant])

    estimate = estimate_context_tokens(context)

    assert estimate.last_usage_index is None


def test_estimate_context_tokens_stale_assistant_after_newer_prefix_message_is_ignored():
    # A prefix message inserted with a later timestamp than a prior assistant
    # response (e.g. a compaction summary) shadows any assistant usage whose
    # own timestamp falls before it, even if that assistant message comes
    # later in the list and reports nonzero usage.
    usage = Usage(input=100, total_tokens=100)
    assistant = AssistantMessage(usage=usage, stop_reason="stop", timestamp=10)
    later_prefix = UserMessage(content="summary", timestamp=20)
    stale_assistant = AssistantMessage(usage=Usage(total_tokens=999), stop_reason="stop", timestamp=15)
    context = Context(messages=[assistant, later_prefix, stale_assistant])

    estimate = estimate_context_tokens(context)

    assert estimate.last_usage_index == 0
    assert estimate.usage_tokens == 100


def test_estimate_context_tokens_adds_prefix_tokens_when_no_usage_present():
    tool = Tool(name="t", description="d", parameters={"type": "object", "properties": {}})
    context = Context(messages=[UserMessage(content="abcd")], system_prompt="abcdefgh", tools=[tool])

    estimate = estimate_context_tokens(context)

    message_tokens = estimate_message_tokens(context.messages[0])
    system_tokens = estimate_text_tokens("abcdefgh")
    tool_tokens = estimate_text_tokens(_safe_json_stringify([tool]))
    assert estimate.tokens == message_tokens + system_tokens + tool_tokens
    assert estimate.trailing_tokens == estimate.tokens
    assert estimate.usage_tokens == 0
    assert estimate.last_usage_index is None


def test_estimate_context_tokens_adds_only_newly_added_tool_tokens_after_usage():
    usage = Usage(input=10, total_tokens=10)
    tool_a = Tool(name="a", description="d", parameters={"type": "object", "properties": {}})
    tool_b = Tool(name="b", description="d", parameters={"type": "object", "properties": {}})
    assistant = AssistantMessage(usage=usage, stop_reason="stop", timestamp=0)
    tool_result = ToolResultMessage(added_tool_names=["b"], timestamp=1)
    context = Context(messages=[assistant, tool_result], tools=[tool_a, tool_b])

    estimate = estimate_context_tokens(context)

    tool_result_tokens = estimate_message_tokens(tool_result)
    added_tool_tokens = estimate_text_tokens(_safe_json_stringify([tool_b]))
    # Only tool "b" was newly added after the usage checkpoint, so tool "a" must
    # not contribute tokens.
    assert estimate.usage_tokens == 10
    assert estimate.trailing_tokens == tool_result_tokens + added_tool_tokens
    assert estimate.tokens == estimate.usage_tokens + estimate.trailing_tokens


def test_estimate_context_tokens_does_not_add_tool_tokens_when_no_new_tools_added():
    usage = Usage(input=10, total_tokens=10)
    tool_a = Tool(name="a", description="d", parameters={"type": "object", "properties": {}})
    assistant = AssistantMessage(usage=usage, stop_reason="stop", timestamp=0)
    tool_result = ToolResultMessage(added_tool_names=[], timestamp=1)
    context = Context(messages=[assistant, tool_result], tools=[tool_a])

    estimate = estimate_context_tokens(context)

    assert estimate.trailing_tokens == estimate_message_tokens(tool_result)
