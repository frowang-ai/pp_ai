"""Python port of `packages/ai/test/azure-openai-responses-reasoning-replay.test.ts`."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pi_ai.api.openai_responses_shared import convert_responses_messages, process_responses_stream
from pi_ai.types import (
    AssistantMessage,
    Context,
    Model,
    ModelCost,
    Usage,
    UserMessage,
)
from pi_ai.utils.event_stream import AssistantMessageEventStream


def create_model() -> Model:
    return Model(
        id="gpt-5-mini",
        name="GPT-5 Mini",
        api="azure-openai-responses",
        provider="azure-openai-responses",
        base_url="https://example.invalid",
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
        usage=Usage(),
        stop_reason="pending",
    )


async def create_events(done_item: dict[str, Any], completed_item: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    yield {
        "type": "response.output_item.added",
        "output_index": 0,
        "sequence_number": 0,
        "item": {"type": "reasoning", "id": done_item["id"], "summary": []},
    }
    yield {
        "type": "response.output_item.done",
        "output_index": 0,
        "sequence_number": 1,
        "item": done_item,
    }
    yield {
        "type": "response.completed",
        "sequence_number": 2,
        "response": {"id": "resp_test", "status": "completed", "output": [completed_item]},
    }


def get_replayed_reasoning(model: Model, assistant: AssistantMessage) -> dict[str, Any] | None:
    context = Context(
        messages=[
            UserMessage(content="first"),
            assistant,
            UserMessage(content="follow-up"),
        ]
    )
    items = convert_responses_messages(model, context, {"azure-openai-responses"})
    return next((item for item in items if item.get("type") == "reasoning"), None)


async def test_preserves_existing_encrypted_content_from_output_item_done():
    model = create_model()
    output = create_output(model)
    done_item = {
        "type": "reasoning",
        "id": "rs_done",
        "summary": [],
        "encrypted_content": "from-output-item-done",
    }
    completed_item = {**done_item, "encrypted_content": "from-response-completed"}

    await process_responses_stream(
        create_events(done_item, completed_item), output, AssistantMessageEventStream(), model
    )

    replayed = get_replayed_reasoning(model, output)
    assert replayed is not None
    assert replayed["type"] == "reasoning"
    assert replayed["id"] == "rs_done"
    assert replayed["encrypted_content"] == "from-output-item-done"


async def test_fills_encrypted_content_when_output_item_done_omitted_it():
    model = create_model()
    output = create_output(model)
    done_item = {"type": "reasoning", "id": "rs_missing", "summary": []}
    completed_item = {**done_item, "encrypted_content": "from-response-completed"}

    await process_responses_stream(
        create_events(done_item, completed_item), output, AssistantMessageEventStream(), model
    )

    replayed = get_replayed_reasoning(model, output)
    assert replayed is not None
    assert replayed["type"] == "reasoning"
    assert replayed["id"] == "rs_missing"
    assert replayed["encrypted_content"] == "from-response-completed"
