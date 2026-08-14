"""Text-content extraction tests.

Includes the Python port of `packages/ai/test/text.test.ts`.
"""

from pi_ai.types import ImageContent, TextContent, ThinkingContent, ToolCall
from pi_ai.utils.text import content_text


def test_content_text_string_passthrough():
    assert content_text("hello world") == "hello world"


def test_content_text_joins_only_text_blocks_with_default_separator():
    content = [
        TextContent(text="first"),
        TextContent(text="second"),
    ]
    assert content_text(content) == "first\nsecond"


def test_content_text_skips_image_thinking_and_tool_call_blocks():
    content = [
        TextContent(text="before"),
        ImageContent(data="base64data", mime_type="image/png"),
        ThinkingContent(thinking="reasoning..."),
        ToolCall(id="1", name="tool", arguments={}),
        TextContent(text="after"),
    ]
    assert content_text(content) == "before\nafter"


def test_content_text_custom_separator():
    content = [TextContent(text="a"), TextContent(text="b"), TextContent(text="c")]
    assert content_text(content, separator=", ") == "a, b, c"


def test_content_text_empty_list_returns_empty_string():
    assert content_text([]) == ""


def test_content_text_empty_string_passthrough():
    assert content_text("") == ""


# --------------------------------------------------------------------------
# Ported from `packages/ai/test/text.test.ts`
# --------------------------------------------------------------------------

_TS_CONTENT = [
    ThinkingContent(thinking="reasoning"),
    TextContent(text="first"),
    ToolCall(id="1", name="read", arguments={}),
    TextContent(text="second"),
]


def test_extracts_assistant_text_blocks():
    assert content_text(_TS_CONTENT) == "first\nsecond"


def test_supports_custom_separators():
    assert content_text(_TS_CONTENT, "") == "firstsecond"


def test_passes_string_content_through():
    assert content_text("hello") == "hello"


def test_extracts_text_from_tool_result_content():
    tool_result_content = [
        TextContent(text="first"),
        ImageContent(data="...", mime_type="image/png"),
        TextContent(text="second"),
    ]
    assert content_text(tool_result_content, "") == "firstsecond"
