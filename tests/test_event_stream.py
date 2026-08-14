import asyncio

import pytest
from pi_ai import (
    AssistantMessage,
    AssistantMessageEventStream,
    DoneEvent,
    ErrorEvent,
    EventStream,
    StartEvent,
    StreamEndedWithoutResult,
    TextDeltaEvent,
)
from pi_ai.utils.event_stream import _assistant_result, create_assistant_message_event_stream


def make_stream() -> EventStream[str, str]:
    return EventStream(lambda event: event == "END", lambda event: f"result:{event}")


async def test_queued_events_are_delivered_in_order():
    stream = make_stream()
    stream.push("a")
    stream.push("b")
    stream.push("END")

    received = [event async for event in stream]

    assert received == ["a", "b", "END"]
    assert await stream.result() == "result:END"


async def test_waiting_consumer_receives_pushed_event():
    stream = make_stream()

    async def produce():
        await asyncio.sleep(0)
        stream.push("a")
        stream.push("END")

    task = asyncio.create_task(produce())
    received = [event async for event in stream]
    await task

    assert received == ["a", "END"]


async def test_push_after_completion_is_ignored():
    stream = make_stream()
    stream.push("END")
    stream.push("late")

    received = [event async for event in stream]

    assert received == ["END"]


async def test_end_with_result_resolves_pending_waiters():
    stream = make_stream()
    result_task = asyncio.create_task(stream.result())
    await asyncio.sleep(0)
    stream.end("manual")

    assert await result_task == "manual"
    assert [event async for event in stream] == []


async def test_end_without_result_raises():
    stream = make_stream()
    stream.end()

    with pytest.raises(StreamEndedWithoutResult):
        await stream.result()


async def test_end_unblocks_iterating_consumer():
    stream = make_stream()

    async def consume():
        return [event async for event in stream]

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    stream.push("a")
    stream.end("done")

    assert await task == ["a"]


async def test_fail_propagates_error_to_result():
    stream = make_stream()
    stream.fail(RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        await stream.result()


async def test_assistant_stream_resolves_done_message():
    stream = AssistantMessageEventStream()
    message = AssistantMessage(model="m", stop_reason="stop")
    stream.push(StartEvent(partial=message))
    stream.push(TextDeltaEvent(content_index=0, delta="hi", partial=message))
    stream.push(DoneEvent(reason="stop", message=message))

    events = [event async for event in stream]

    assert [event.type for event in events] == ["start", "text_delta", "done"]
    assert await stream.result() is message


async def test_assistant_stream_resolves_error_message():
    stream = AssistantMessageEventStream()
    message = AssistantMessage(model="m", stop_reason="error", error_message="nope")
    stream.push(ErrorEvent(reason="error", error=message))

    assert await stream.result() is message
    assert stream.done is True


async def test_fail_after_result_already_set_is_ignored():
    stream = make_stream()
    stream.push("END")

    stream.fail(RuntimeError("late failure"))

    assert await stream.result() == "result:END"


async def test_end_called_twice_keeps_first_result():
    stream = make_stream()
    stream.end("first")
    stream.end("second")

    assert await stream.result() == "first"


async def test_end_called_twice_without_result_keeps_first_error():
    stream = make_stream()
    stream.end()
    stream.end()

    with pytest.raises(StreamEndedWithoutResult):
        await stream.result()


async def test_push_to_stream_with_cancelled_waiter_queues_for_next_consumer():
    stream = make_stream()

    async def consume_one():
        async for event in stream:
            return event
        return None

    task = asyncio.create_task(consume_one())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The cancelled waiter is still queued internally; push must skip over it
    # rather than delivering to a future nobody will read.
    stream.push("a")
    stream.push("END")

    received = [event async for event in stream]
    assert received == ["a", "END"]


async def test_result_awaited_by_several_consumers_concurrently():
    stream = make_stream()

    task1 = asyncio.create_task(stream.result())
    task2 = asyncio.create_task(stream.result())
    task3 = asyncio.create_task(stream.result())
    await asyncio.sleep(0)

    stream.push("END")

    results = await asyncio.gather(task1, task2, task3)
    assert results == ["result:END", "result:END", "result:END"]
    # A late call also observes the already-resolved result.
    assert await stream.result() == "result:END"


async def test_iterating_stream_that_ends_with_queued_events_still_pending():
    stream = make_stream()
    stream.push("a")
    stream.push("b")
    stream.end("manual-result")

    received = [event async for event in stream]

    assert received == ["a", "b"]
    assert await stream.result() == "manual-result"


async def test_end_wakes_pending_iterating_waiter_with_end_sentinel():
    stream = make_stream()

    async def consume():
        return [event async for event in stream]

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)  # let the consumer suspend on an empty-queue waiter
    stream.end("done")

    assert await task == []
    assert await stream.result() == "done"


async def test_fail_wakes_pending_iterating_waiter_with_end_sentinel():
    stream = make_stream()

    async def consume():
        return [event async for event in stream]

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)  # let the consumer suspend on an empty-queue waiter
    stream.fail(RuntimeError("boom"))

    assert await task == []
    with pytest.raises(RuntimeError, match="boom"):
        await stream.result()


async def test_set_result_skips_already_cancelled_result_waiter():
    stream = make_stream()

    task = asyncio.create_task(stream.result())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The cancelled future is still recorded as a result waiter; resolving the
    # result must skip it instead of raising InvalidStateError.
    stream.push("END")

    assert await stream.result() == "result:END"


async def test_set_error_directly_is_noop_when_result_already_set():
    stream = make_stream()
    stream.push("END")

    stream._set_error(RuntimeError("should be ignored"))

    assert await stream.result() == "result:END"


async def test_fail_propagates_error_to_pending_result_waiter():
    stream = make_stream()
    task = asyncio.create_task(stream.result())
    await asyncio.sleep(0)

    stream.fail(RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        await task


async def test_create_assistant_message_event_stream_returns_usable_stream():
    stream = create_assistant_message_event_stream()
    message = AssistantMessage(model="m", stop_reason="stop")
    stream.push(DoneEvent(reason="stop", message=message))

    assert await stream.result() is message


async def test_assistant_result_raises_for_unexpected_event_type():
    class _BogusEvent:
        type = "bogus"

    with pytest.raises(ValueError, match="Unexpected event type"):
        _assistant_result(_BogusEvent())
