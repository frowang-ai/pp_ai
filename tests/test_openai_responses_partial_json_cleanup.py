"""Python port of `packages/ai/test/openai-responses-partial-json-cleanup.test.ts`.

The TypeScript assertion is `"partialJson" in persistedToolCall === false`: the
scratch buffer used while streaming arguments must not survive onto the block
that is persisted in the assistant message. The port keeps that buffer on a
separate `_StreamingToolCall` entry rather than on the `ToolCall` dataclass, so
the equivalent assertion is that the persisted block has no `partial_json`
attribute at all.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pi_ai.api.openai_responses_shared import process_responses_stream
from pi_ai.types import AssistantMessage, AssistantMessageEvent, Model, ModelCost, ToolCall, now_ms
from pi_ai.utils.event_stream import AssistantMessageEventStream

MODEL = Model(
    id="gpt-5-mini",
    name="GPT-5 Mini",
    api="openai-responses",
    provider="openai",
    base_url="https://api.openai.com/v1",
    reasoning=True,
    input=["text"],
    cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    context_window=400000,
    max_tokens=128000,
)


def create_output(model: Model) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        stop_reason="pending",
        timestamp=now_ms(),
    )


async def create_function_call_events(arguments_json: str) -> AsyncIterator[dict[str, Any]]:
    yield {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": "fc_test",
            "call_id": "call_test",
            "name": "edit",
            "arguments": "",
        },
    }
    yield {
        "type": "response.function_call_arguments.delta",
        "output_index": 0,
        "delta": '{"path":"README.md"',
    }
    yield {
        "type": "response.function_call_arguments.delta",
        "output_index": 0,
        "delta": ',"content":"updated"}',
    }
    yield {
        "type": "response.function_call_arguments.done",
        "output_index": 0,
        "arguments": arguments_json,
    }
    yield {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": "fc_test",
            "call_id": "call_test",
            "name": "edit",
            "arguments": arguments_json,
        },
    }
    yield {
        "type": "response.completed",
        "sequence_number": 5,
        "response": {"id": "resp_test", "status": "completed"},
    }


async def test_removes_partial_json_from_persisted_tool_call_blocks_at_output_item_done() -> None:
    output = create_output(MODEL)
    stream = AssistantMessageEventStream()
    pushed: list[AssistantMessageEvent] = []
    original_push = stream.push

    def recording_push(event: AssistantMessageEvent) -> None:
        pushed.append(event)
        original_push(event)

    stream.push = recording_push  # type: ignore[method-assign]
    arguments_json = '{"path":"README.md","content":"updated"}'

    await process_responses_stream(create_function_call_events(arguments_json), output, stream, MODEL)

    assert len(output.content) == 1
    persisted_tool_call = output.content[0]
    assert persisted_tool_call.type == "toolCall"
    assert isinstance(persisted_tool_call, ToolCall)
    assert persisted_tool_call.arguments == {"path": "README.md", "content": "updated"}
    assert not hasattr(persisted_tool_call, "partial_json")

    tool_call_end = next((event for event in pushed if event.type == "toolcall_end"), None)
    assert tool_call_end is not None
    assert tool_call_end.tool_call is persisted_tool_call
    assert not hasattr(tool_call_end.tool_call, "partial_json")
