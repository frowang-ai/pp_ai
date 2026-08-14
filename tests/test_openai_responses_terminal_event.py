"""Python port of `packages/ai/test/openai-responses-terminal-event.test.ts`.

TypeScript mocks the `openai` SDK to yield a wrapper stream that ends without a
terminal `response.*` event; this port feeds the same event sequence as an SSE
body through `httpx.MockTransport`, which is where the port's adapter reads its
events from.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from pi_ai.api.openai_responses import stream as stream_openai_responses
from pi_ai.api.openai_responses_shared import process_responses_stream
from pi_ai.types import (
    AssistantMessage,
    Context,
    Model,
    ModelCost,
    SimpleStreamOptions,
    TextContent,
    UserMessage,
    now_ms,
)
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


EARLY_EOF_EVENTS: list[dict[str, Any]] = [
    {"type": "response.created", "sequence_number": 0, "response": {"id": "resp_early_eof"}},
    {
        "type": "response.output_item.added",
        "sequence_number": 1,
        "output_index": 0,
        "item": {"type": "reasoning", "id": "rs_early_eof", "summary": []},
    },
    {
        "type": "response.reasoning_text.delta",
        "sequence_number": 2,
        "output_index": 0,
        "content_index": 0,
        "item_id": "rs_early_eof",
        "delta": "partial reasoning before the stream ends",
    },
]


async def events_of(events: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    for event in events:
        yield event


def completed_events() -> list[dict[str, Any]]:
    return [
        {
            "type": "response.completed",
            "sequence_number": 0,
            "response": {
                "id": "resp_completed",
                "status": "completed",
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 7,
                    "total_tokens": 27,
                    "input_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 3},
                },
            },
        }
    ]


def incomplete_events(reason: str = "max_output_tokens") -> list[dict[str, Any]]:
    return [
        {
            "type": "response.incomplete",
            "sequence_number": 0,
            "response": {
                "id": "resp_incomplete",
                "status": "incomplete",
                "incomplete_details": {"reason": reason},
                "usage": {
                    "input_tokens": 30,
                    "output_tokens": 12,
                    "total_tokens": 42,
                    "input_tokens_details": {"cached_tokens": 5},
                },
            },
        }
    ]


def failed_events() -> list[dict[str, Any]]:
    return [
        {
            "type": "response.failed",
            "sequence_number": 0,
            "response": {
                "id": "resp_failed",
                "status": "failed",
                "error": {"code": "server_error", "message": "boom"},
            },
        }
    ]


def phased_message_events(phases: tuple[str, str], terminal_status: str = "completed") -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [
        {
            "type": "response.output_item.added",
            "sequence_number": 0,
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "msg_phase",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
                "phase": phases[0],
            },
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 1,
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "msg_phase",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "answer", "annotations": []}],
                "phase": phases[1],
            },
        },
    ]
    if terminal_status == "incomplete":
        events.append(
            {
                "type": "response.incomplete",
                "sequence_number": 2,
                "response": {
                    "id": "resp_phase",
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                },
            }
        )
        return events
    events.append(
        {
            "type": "response.completed",
            "sequence_number": 2,
            "response": {"id": "resp_phase", "status": "completed"},
        }
    )
    return events


class StopReasonRecorder(AssistantMessageEventStream):
    """Records `partial.stopReason` at every push, like the TypeScript spy."""

    def __init__(self) -> None:
        super().__init__()
        self.observed: list[str] = []

    def push(self, event: object) -> None:
        partial = getattr(event, "partial", None)
        if partial is not None:
            self.observed.append(partial.stop_reason)
        super().push(event)  # type: ignore[arg-type]


async def test_rejects_streams_that_end_before_a_terminal_response_event() -> None:
    output = create_output(MODEL)
    stream = AssistantMessageEventStream()

    with pytest.raises(Exception, match=r"OpenAI Responses stream ended before a terminal response event"):
        await process_responses_stream(events_of(EARLY_EOF_EVENTS), output, stream, MODEL)


async def test_emits_an_error_final_result_when_the_wrapper_stream_ends_early() -> None:
    async def sse_body() -> AsyncIterator[bytes]:
        # Yield one event per chunk with a scheduling point in between, the way a
        # real socket delivers them. A single buffered body would let the adapter
        # run to completion before the consumer reads its first event, and every
        # event shares one mutable `partial` message, so the `start` event would
        # already show the final `error` stop reason.
        for event in EARLY_EOF_EVENTS:
            yield f"data: {json.dumps(event)}\n\n".encode()
            await asyncio.sleep(0)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse_body(), headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    context = Context(
        system_prompt="",
        messages=[UserMessage(content=[TextContent(text="hi")], timestamp=0)],
        tools=[],
    )
    stream = stream_openai_responses(MODEL, context, SimpleStreamOptions(api_key="test"), client=client)

    events = []
    initial_stop_reason: str | None = None
    async for event in stream:
        if event.type == "start":
            initial_stop_reason = event.partial.stop_reason
        events.append(event)

    result = await stream.result()
    assert initial_stop_reason == "pending"
    assert events[-1].type == "error"
    assert result.stop_reason == "error"
    assert result.error_message == "OpenAI Responses stream ended before a terminal response event"


@pytest.mark.parametrize(
    ("phases", "expected"),
    [
        (("commentary", "commentary"), ["pending", "pending"]),
        (("final_answer", "final_answer"), ["stop", "stop"]),
        (("commentary", "final_answer"), ["pending", "stop"]),
    ],
)
async def test_tracks_message_phases(phases: tuple[str, str], expected: list[str]) -> None:
    output = create_output(MODEL)
    stream = StopReasonRecorder()

    await process_responses_stream(events_of(phased_message_events(phases)), output, stream, MODEL)

    assert stream.observed == expected
    assert output.stop_reason == "stop"


async def test_replaces_a_provisional_final_answer_stop_with_an_incomplete_terminal_reason() -> None:
    output = create_output(MODEL)
    stream = StopReasonRecorder()

    await process_responses_stream(
        events_of(phased_message_events(("final_answer", "final_answer"), "incomplete")),
        output,
        stream,
        MODEL,
    )

    assert stream.observed == ["stop", "stop"]
    assert output.stop_reason == "length"


async def test_finalizes_completed_terminal_events_as_stop() -> None:
    output = create_output(MODEL)

    await process_responses_stream(events_of(completed_events()), output, AssistantMessageEventStream(), MODEL)

    assert output.response_id == "resp_completed"
    assert output.stop_reason == "stop"
    assert output.raw_stop_reason == "completed"
    assert output.usage.input == 15
    assert output.usage.output == 7
    assert output.usage.cache_read == 2
    assert output.usage.cache_write == 3
    assert output.usage.total_tokens == 27


async def test_finalizes_incomplete_terminal_events_as_length_stops() -> None:
    output = create_output(MODEL)

    await process_responses_stream(events_of(incomplete_events()), output, AssistantMessageEventStream(), MODEL)

    assert output.response_id == "resp_incomplete"
    assert output.stop_reason == "length"
    assert output.raw_stop_reason == "incomplete.max_output_tokens"
    assert output.usage.input == 25
    assert output.usage.output == 12
    assert output.usage.cache_read == 5
    assert output.usage.cache_write == 0
    assert output.usage.total_tokens == 42


async def test_finalizes_content_filtered_incomplete_responses_as_non_retryable_errors() -> None:
    output = create_output(MODEL)

    await process_responses_stream(
        events_of(incomplete_events("content_filter")), output, AssistantMessageEventStream(), MODEL
    )

    assert output.stop_reason == "error"
    assert output.raw_stop_reason == "incomplete.content_filter"
    assert output.error_message == "Response incomplete: content_filter"


async def test_preserves_unknown_provider_incomplete_reasons_as_non_retryable_errors() -> None:
    output = create_output(MODEL)

    await process_responses_stream(
        events_of(incomplete_events("max_time_limit")), output, AssistantMessageEventStream(), MODEL
    )

    assert output.stop_reason == "error"
    assert output.raw_stop_reason == "incomplete.max_time_limit"
    assert output.error_message == "Response incomplete: max_time_limit"


async def test_rejects_failed_terminal_events_with_the_provider_error() -> None:
    output = create_output(MODEL)

    with pytest.raises(Exception, match=r"server_error: boom"):
        await process_responses_stream(events_of(failed_events()), output, AssistantMessageEventStream(), MODEL)
    assert output.raw_stop_reason == "failed"
