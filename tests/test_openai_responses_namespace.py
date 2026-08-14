"""Python port of `packages/ai/test/openai-responses-namespace.test.ts`."""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator
from typing import Any

from pi_ai.api.openai_responses_shared import (
    ConvertResponsesMessagesOptions,
    OpenAIResponsesStreamOptions,
    convert_responses_messages,
    process_responses_stream,
)
from pi_ai.types import (
    AssistantMessage,
    Context,
    Cost,
    Model,
    ModelCost,
    ToolCall,
    Usage,
    now_ms,
)
from pi_ai.utils.event_stream import AssistantMessageEventStream

MODEL = Model(
    id="gpt-5.4",
    name="GPT-5.4",
    api="openai-responses",
    provider="openai",
    base_url="https://api.openai.com/v1",
    reasoning=True,
    input=["text"],
    cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    context_window=400000,
    max_tokens=128000,
)


def create_output() -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api=MODEL.api,
        provider=MODEL.provider,
        model=MODEL.id,
        usage=Usage(
            input=0,
            output=0,
            cache_read=0,
            cache_write=0,
            total_tokens=0,
            cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
        ),
        stop_reason="pending",
        timestamp=now_ms(),
    )


async def function_call_events() -> AsyncIterator[dict[str, Any]]:
    yield {
        "type": "response.output_item.added",
        "sequence_number": 0,
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": "fc_test",
            "call_id": "call_test",
            "name": "lookup",
            "arguments": "",
        },
    }
    yield {
        "type": "response.output_item.done",
        "sequence_number": 1,
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": "fc_test",
            "call_id": "call_test",
            "name": "lookup",
            "arguments": '{"value":"hello"}',
            "namespace": "dynamic_tools",
        },
    }
    yield {
        "type": "response.completed",
        "sequence_number": 2,
        "response": {"id": "resp_test", "status": "completed"},
    }


async def custom_tool_call_events() -> AsyncIterator[dict[str, Any]]:
    yield {
        "type": "response.output_item.added",
        "sequence_number": 0,
        "output_index": 0,
        "item": {
            "type": "custom_tool_call",
            "id": "ctc_test",
            "call_id": "call_test",
            "name": "query",
            "input": "",
        },
    }
    yield {
        "type": "response.output_item.done",
        "sequence_number": 1,
        "output_index": 0,
        "item": {
            "type": "custom_tool_call",
            "id": "ctc_test",
            "call_id": "call_test",
            "name": "query",
            "input": "hello",
            "namespace": "dynamic_tools",
        },
    }
    yield {
        "type": "response.completed",
        "sequence_number": 2,
        "response": {"id": "resp_test", "status": "completed"},
    }


def get_tool_call(output: AssistantMessage) -> ToolCall:
    block = output.content[0] if output.content else None
    if block is None or block.type != "toolCall":
        raise AssertionError("Expected toolCall block")
    return block


def find_item(items: list[dict[str, Any]], item_type: str) -> dict[str, Any] | None:
    return next((item for item in items if item.get("type") == item_type), None)


async def test_round_trips_a_function_namespace_received_only_on_output_item_done() -> None:
    output = create_output()
    await process_responses_stream(function_call_events(), output, AssistantMessageEventStream(), MODEL)

    tool_call = get_tool_call(output)
    assert tool_call.id == "call_test|fc_test"
    assert tool_call.name == "lookup"
    assert tool_call.arguments == {"value": "hello"}
    assert tool_call.namespace == "dynamic_tools"

    replayed = find_item(
        convert_responses_messages(MODEL, Context(messages=[output]), {"openai"}),
        "function_call",
    )
    assert replayed is not None
    assert replayed["id"] == "fc_test"
    assert replayed["call_id"] == "call_test"
    assert replayed["name"] == "lookup"
    assert replayed["arguments"] == '{"value":"hello"}'
    assert replayed["namespace"] == "dynamic_tools"


async def test_round_trips_a_custom_tool_namespace_received_only_on_output_item_done() -> None:
    output = create_output()
    grammar_tool_input_properties = {"query": "input"}
    await process_responses_stream(
        custom_tool_call_events(),
        output,
        AssistantMessageEventStream(),
        MODEL,
        OpenAIResponsesStreamOptions(grammar_tool_input_properties=grammar_tool_input_properties),
    )

    tool_call = get_tool_call(output)
    assert tool_call.id == "call_test|ctc_test"
    assert tool_call.name == "query"
    assert tool_call.arguments == {"input": "hello"}
    assert tool_call.namespace == "dynamic_tools"

    replayed = find_item(
        convert_responses_messages(
            MODEL,
            Context(messages=[output]),
            {"openai"},
            ConvertResponsesMessagesOptions(grammar_tool_input_properties=grammar_tool_input_properties),
        ),
        "custom_tool_call",
    )
    assert replayed is not None
    assert replayed["id"] == "ctc_test"
    assert replayed["call_id"] == "call_test"
    assert replayed["name"] == "query"
    assert replayed["input"] == "hello"
    assert replayed["namespace"] == "dynamic_tools"


def test_drops_namespaces_when_the_target_cannot_replay_their_load_items() -> None:
    output = create_output()
    output.content.append(
        ToolCall(
            id="call_function|fc_test",
            name="lookup",
            arguments={"value": "hello"},
            namespace="dynamic_tools",
        )
    )
    output.content.append(
        ToolCall(
            id="call_custom|ctc_test",
            name="query",
            arguments={"input": "hello"},
            namespace="dynamic_tools",
        )
    )

    target_models = [
        dataclasses.replace(MODEL, id="gpt-5.2", name="GPT-5.2"),
        dataclasses.replace(MODEL, provider="azure-openai-responses"),
        dataclasses.replace(
            MODEL,
            api="openai-codex-responses",
            provider="openai-codex",
            id="gpt-5.3-codex-spark",
            name="GPT-5.3 Codex Spark",
        ),
    ]

    for target_model in target_models:
        replayed = convert_responses_messages(
            target_model,
            Context(messages=[output]),
            {"openai"},
            ConvertResponsesMessagesOptions(grammar_tool_input_properties={"query": "input"}),
        )
        function_call = find_item(replayed, "function_call")
        custom_tool_call = find_item(replayed, "custom_tool_call")
        assert function_call is not None
        assert "namespace" not in function_call
        assert custom_tool_call is not None
        assert "namespace" not in custom_tool_call


def test_does_not_add_a_namespace_to_ordinary_function_calls() -> None:
    output = create_output()
    output.content.append(ToolCall(id="call_test|fc_test", name="lookup", arguments={"value": "hello"}))

    replayed = find_item(
        convert_responses_messages(MODEL, Context(messages=[output]), {"openai"}),
        "function_call",
    )
    assert replayed is not None
    assert "namespace" not in replayed
