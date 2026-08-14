"""Python port of `packages/ai/test/anthropic-sse-parsing.test.ts`.

TypeScript hands `streamAnthropic` a fake `@anthropic-ai/sdk` client whose
`messages.create().asResponse()` returns a canned SSE `Response`. This port has
no SDK client, so the same canned bodies are served by an
`httpx.MockTransport`.
"""

from __future__ import annotations

import json

import httpx
from pi_ai.api.anthropic_messages import AnthropicOptions
from pi_ai.api.anthropic_messages import stream as stream_anthropic
from pi_ai.providers.all import get_builtin_model
from pi_ai.types import Context, TextContent, ThinkingContent, Tool, UserMessage

MINIMAL_ANTHROPIC_EVENTS: list[tuple[str, str]] = [
    (
        "message_start",
        json.dumps(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_test",
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            }
        ),
    ),
    (
        "content_block_start",
        json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
    ),
    (
        "content_block_delta",
        json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}),
    ),
    ("content_block_stop", json.dumps({"type": "content_block_stop", "index": 0})),
    (
        "message_delta",
        json.dumps(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            }
        ),
    ),
    ("message_stop", json.dumps({"type": "message_stop"})),
]


def sse_body(events: list[tuple[str, str]]) -> str:
    return "\n".join(f"event: {name}\ndata: {data}\n" for name, data in events)


def make_client(events: list[tuple[str, str]]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse_body(events), headers={"content-type": "text/event-stream"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_repairs_malformed_sse_json_and_malformed_streamed_tool_json():
    model = get_builtin_model("anthropic", "claude-haiku-4-5")
    context = Context(
        messages=[UserMessage(content="Use the edit tool.")],
        tools=[
            Tool(
                name="edit",
                description="Edit a file.",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
                    "required": ["path", "text"],
                },
            )
        ],
    )

    # Raw (unescaped) `\H` and a literal tab: both the outer SSE JSON and the
    # inner streamed tool JSON are malformed and must be repaired.
    malformed_tool_json_delta = (
        r'{"type":"content_block_delta","index":0,"delta":'
        r'{"type":"input_json_delta","partial_json":"{\"path\":\"A\H\",\"text\":\"col1'
        "\t"
        r'col2\"}"}}'
    )

    events = [
        (
            "message_start",
            json.dumps(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_test",
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        },
                    },
                }
            ),
        ),
        (
            "content_block_start",
            json.dumps(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "tool_use", "id": "toolu_test", "name": "edit", "input": {}},
                }
            ),
        ),
        ("content_block_delta", malformed_tool_json_delta),
        ("content_block_stop", json.dumps({"type": "content_block_stop", "index": 0})),
        (
            "message_delta",
            json.dumps(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                }
            ),
        ),
        ("message_stop", json.dumps({"type": "message_stop"})),
    ]

    result = await stream_anthropic(
        model, context, AnthropicOptions(api_key="test-key"), client=make_client(events)
    ).result()

    assert result.stop_reason == "toolUse"
    assert result.error_message is None

    tool_call = next(block for block in result.content if block.type == "toolCall")
    assert tool_call.arguments == {"path": "A\\H", "text": "col1\tcol2"}


async def test_preserves_content_from_content_block_start_events():
    model = get_builtin_model("anthropic", "claude-haiku-4-5")
    context = Context(messages=[UserMessage(content="Say hello.")])
    events = [
        (
            "message_start",
            json.dumps(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_initial_content",
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        },
                    },
                }
            ),
        ),
        (
            "content_block_start",
            json.dumps(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": "Initial text"},
                }
            ),
        ),
        (
            "content_block_delta",
            json.dumps(
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " plus delta"}}
            ),
        ),
        ("content_block_stop", json.dumps({"type": "content_block_stop", "index": 0})),
        (
            "content_block_start",
            json.dumps(
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "thinking",
                        "thinking": "Initial thinking",
                        "signature": "initial signature",
                    },
                }
            ),
        ),
        (
            "content_block_delta",
            json.dumps(
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "thinking_delta", "thinking": " plus delta"},
                }
            ),
        ),
        (
            "content_block_delta",
            json.dumps(
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "signature_delta", "signature": " plus delta"},
                }
            ),
        ),
        ("content_block_stop", json.dumps({"type": "content_block_stop", "index": 1})),
        (
            "message_delta",
            json.dumps(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                }
            ),
        ),
        ("message_stop", json.dumps({"type": "message_stop"})),
    ]

    result = await stream_anthropic(
        model, context, AnthropicOptions(api_key="test-key"), client=make_client(events)
    ).result()

    # `toEqual` in TS checks the whole object shape (undefined props aside), so
    # compare full dataclass equality rather than field-by-field to also catch
    # unexpected extra fields (e.g. a stray `redacted`/`text_signature` value).
    assert result.content == [
        TextContent(text="Initial text plus delta"),
        ThinkingContent(thinking="Initial thinking plus delta", thinking_signature="initial signature plus delta"),
    ]


async def test_preserves_refusal_stop_details_from_message_delta():
    model = get_builtin_model("anthropic", "claude-fable-5")
    context = Context(messages=[UserMessage(content="blocked request")])
    explanation = (
        "This request triggered restrictions on violative cyber content and was blocked under "
        "Anthropic's Usage Policy. To learn more, provide feedback, or request an exemption based "
        "on how you use Claude, visit our help center: "
        "https://support.claude.com/en/articles/14604842-real-time-cyber-safeguards-on-claude."
    )
    events = [
        (
            "message_start",
            json.dumps(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_01XFUDYJgAACzvnptvVoYEL",
                        "usage": {
                            "input_tokens": 412,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        },
                    },
                }
            ),
        ),
        (
            "message_delta",
            json.dumps(
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "refusal",
                        "stop_details": {
                            "type": "refusal",
                            "category": "cyber",
                            "explanation": explanation,
                        },
                    },
                    "usage": {
                        "input_tokens": 412,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                }
            ),
        ),
        ("message_stop", json.dumps({"type": "message_stop"})),
    ]

    result = await stream_anthropic(
        model, context, AnthropicOptions(api_key="test-key"), client=make_client(events)
    ).result()

    assert result.stop_reason == "error"
    assert result.raw_stop_reason == "refusal"
    assert result.error_message == explanation


async def test_preserves_sensitive_stop_reasons_with_a_descriptive_error_message():
    model = get_builtin_model("anthropic", "claude-haiku-4-5")
    context = Context(messages=[UserMessage(content="blocked request")])
    events = [
        (
            "message_start",
            json.dumps(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_sensitive",
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        },
                    },
                }
            ),
        ),
        (
            "message_delta",
            json.dumps(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "sensitive"},
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                }
            ),
        ),
        ("message_stop", json.dumps({"type": "message_stop"})),
    ]

    result = await stream_anthropic(
        model, context, AnthropicOptions(api_key="test-key"), client=make_client(events)
    ).result()

    assert result.stop_reason == "error"
    assert result.raw_stop_reason == "sensitive"
    assert result.error_message == "Provider stopped with: sensitive"


async def test_treats_message_delta_without_usage_as_a_no_op_for_usage_accumulation():
    model = get_builtin_model("anthropic", "claude-haiku-4-5")
    context = Context(messages=[UserMessage(content="Say hello.")])
    events = [
        (
            ("message_delta", json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn"}}))
            if name == "message_delta"
            else (name, data)
        )
        for name, data in MINIMAL_ANTHROPIC_EVENTS
    ]

    result = await stream_anthropic(
        model, context, AnthropicOptions(api_key="test-key"), client=make_client(events)
    ).result()

    assert result.stop_reason == "stop"
    assert result.error_message is None
    assert result.content == [TextContent(text="Hello")]
    assert result.usage.input == 12
    assert result.usage.total_tokens == 12


async def test_ignores_unknown_sse_events_after_message_stop():
    model = get_builtin_model("anthropic", "claude-haiku-4-5")
    context = Context(messages=[UserMessage(content="Say hello.")])
    events = [*MINIMAL_ANTHROPIC_EVENTS, ("done", "[DONE]"), ("proxy.stats", "not json")]

    result = await stream_anthropic(
        model, context, AnthropicOptions(api_key="test-key"), client=make_client(events)
    ).result()

    assert result.stop_reason == "stop"
    assert result.error_message is None
    assert result.content == [TextContent(text="Hello")]
