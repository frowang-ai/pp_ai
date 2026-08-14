"""Python port of `packages/ai/test/constrained-sampling.test.ts`.

Named with a `_ported` suffix because `tests/test_constrained_sampling.py`
already exists in this repo and covers the `constrained_sampling` helpers
directly; this file is the port of the TypeScript test of the same name.
"""

from __future__ import annotations

import copy
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pi_ai.api.constrained_sampling import (
    GrammarToolInputJsonBuffer,
    append_grammar_tool_input_json_delta,
)
from pi_ai.api.openai_responses_shared import (
    ConvertResponsesMessagesOptions,
    ConvertResponsesToolsOptions,
    OpenAIResponsesStreamOptions,
    convert_responses_messages,
    convert_responses_tools,
    process_responses_stream,
)
from pi_ai.types import (
    AssistantMessage,
    Context,
    GrammarConstrainedSampling,
    JsonSchemaConstrainedSampling,
    Model,
    ModelCost,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
)
from pi_ai.utils.event_stream import AssistantMessageEventStream


def make_model() -> Model:
    return Model(
        id="gpt-test",
        name="GPT Test",
        api="openai-responses",
        provider="openai",
        base_url="https://api.openai.com/v1",
        reasoning=False,
        input=["text", "image"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128000,
        max_tokens=4096,
    )


def make_output() -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api="openai-responses",
        provider="openai",
        model="gpt-test",
        usage=Usage(),
        stop_reason="pending",
    )


async def iterate_events(events: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    for event in events:
        yield event


def make_tool(**overrides: Any) -> Tool:
    kwargs: dict[str, Any] = {
        "name": "sample_tool",
        "description": "Sample tool",
        "parameters": {
            "type": "object",
            "properties": {"payload": {"type": "string"}},
            "required": ["payload"],
            "additionalProperties": False,
        },
    }
    kwargs.update(overrides)
    return Tool(**kwargs)


def capture_tool_call_deltas(stream: AssistantMessageEventStream) -> list[str]:
    deltas: list[str] = []
    original_push = stream.push

    def push(event: Any) -> None:
        if getattr(event, "type", None) == "toolcall_delta":
            deltas.append(event.delta)
        original_push(event)

    stream.push = push  # type: ignore[method-assign]
    return deltas


def assert_matches(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Equivalent of vitest `toMatchObject` for flat/nested dict subsets."""
    for key, value in expected.items():
        assert key in actual, f"missing key {key!r} in {actual!r}"
        if isinstance(value, dict):
            assert_matches(actual[key], value)
        else:
            assert actual[key] == value


def test_converts_supported_constraints_and_falls_back_when_unsupported():
    prefer = convert_responses_tools([make_tool(constrained_sampling=JsonSchemaConstrainedSampling(strict="prefer"))])[
        0
    ]
    assert_matches(prefer, {"type": "function", "name": "sample_tool", "strict": True})

    with pytest.raises(Exception, match='Tool "sample_tool" requires JSON-schema constrained sampling'):
        convert_responses_tools(
            [make_tool(constrained_sampling=JsonSchemaConstrainedSampling(strict="require"))],
            ConvertResponsesToolsOptions(supports_strict_mode=False),
        )

    grammar_tool = make_tool(
        constrained_sampling=GrammarConstrainedSampling(variants={"openai_lark": "start: /[a-z]+/"})
    )
    grammar_converted = convert_responses_tools(
        [grammar_tool], ConvertResponsesToolsOptions(supports_openai_grammar_tools=True)
    )[0]
    assert_matches(
        grammar_converted,
        {
            "type": "custom",
            "name": "sample_tool",
            "format": {"type": "grammar", "syntax": "lark", "definition": "start: /[a-z]+/"},
        },
    )

    with pytest.raises(
        Exception,
        match=('Tool "sample_tool" cannot use grammar constrained sampling: no supported grammar variant was provided'),
    ):
        convert_responses_tools(
            [make_tool(constrained_sampling=GrammarConstrainedSampling(variants={}))],
            ConvertResponsesToolsOptions(supports_openai_grammar_tools=True),
        )

    fallback = convert_responses_tools(
        [grammar_tool],
        ConvertResponsesToolsOptions(supports_openai_grammar_tools=False, supports_strict_mode=False),
    )[0]
    assert_matches(fallback, {"type": "function", "name": "sample_tool"})
    assert "strict" not in fallback

    assert convert_responses_tools([make_tool(constrained_sampling=False)]) == convert_responses_tools([make_tool()])


def make_grammar_replay_context(arguments: dict[str, Any]) -> Context:
    return Context(
        messages=[
            AssistantMessage(
                api="openai-responses",
                provider="openai",
                model="gpt-test",
                content=[
                    ToolCall(
                        id="call_1|ctc_1",
                        name="sample_tool",
                        arguments=copy.deepcopy(arguments),
                    )
                ],
                usage=Usage(),
                stop_reason="toolUse",
            ),
            ToolResultMessage(
                tool_call_id="call_1|ctc_1",
                tool_name="sample_tool",
                content=[TextContent(text="done")],
                is_error=False,
            ),
        ]
    )


@pytest.mark.parametrize("invalid_arguments", [{}, {"payload": 42}])
def test_replay_rejects_non_string_grammar_tool_input(invalid_arguments: dict[str, Any]):
    with pytest.raises(Exception, match='Grammar tool call "sample_tool" requires argument "payload" to be a string'):
        convert_responses_messages(
            make_model(),
            make_grammar_replay_context(invalid_arguments),
            {"openai"},
            ConvertResponsesMessagesOptions(grammar_tool_input_properties={"sample_tool": "payload"}),
        )


def test_replays_grammar_calls_as_custom_responses_items():
    messages = convert_responses_messages(
        make_model(),
        make_grammar_replay_context({"payload": "abc"}),
        {"openai"},
        ConvertResponsesMessagesOptions(grammar_tool_input_properties={"sample_tool": "payload"}),
    )

    assert {
        "type": "custom_tool_call",
        "id": "ctc_1",
        "call_id": "call_1",
        "name": "sample_tool",
        "input": "abc",
    } in messages
    assert {
        "type": "custom_tool_call_output",
        "call_id": "call_1",
        "output": "done",
    } in messages


def test_keeps_grammar_input_json_deltas_append_only():
    buffer = GrammarToolInputJsonBuffer(input="", started=False, closed=False)
    first = append_grammar_tool_input_json_delta(buffer, "payload", 'a"', False)
    second = append_grammar_tool_input_json_delta(buffer, "payload", 'a"\nb', True)

    assert json.loads(f"{first}{second}") == {"payload": 'a"\nb'}
    assert append_grammar_tool_input_json_delta(buffer, "payload", 'a"\nb', True) is None
    with pytest.raises(
        Exception,
        match='grammar tool input for property "payload" changed after it was closed',
    ):
        append_grammar_tool_input_json_delta(buffer, "payload", "changed", True)


async def test_streams_custom_responses_tool_calls_as_string_arguments():
    output = make_output()
    stream = AssistantMessageEventStream()
    deltas = capture_tool_call_deltas(stream)
    events: list[dict[str, Any]] = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "custom_tool_call",
                "call_id": "call_1",
                "id": "ctc_1",
                "name": "sample_tool",
                "input": "",
            },
        },
        {
            "type": "response.custom_tool_call_input.delta",
            "output_index": 0,
            "item_id": "ctc_1",
            "delta": "ab",
        },
        {
            "type": "response.custom_tool_call_input.done",
            "output_index": 0,
            "item_id": "ctc_1",
            "input": "abc",
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "custom_tool_call",
                "call_id": "call_1",
                "id": "ctc_1",
                "name": "sample_tool",
                "input": "abc",
            },
        },
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        },
    ]

    await process_responses_stream(
        iterate_events(events),
        output,
        stream,
        make_model(),
        OpenAIResponsesStreamOptions(grammar_tool_input_properties={"sample_tool": "payload"}),
    )

    assert output.stop_reason == "toolUse"
    assert output.content == [ToolCall(id="call_1|ctc_1", name="sample_tool", arguments={"payload": "abc"})]
    assert json.loads("".join(deltas)) == {"payload": "abc"}
