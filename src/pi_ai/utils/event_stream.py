"""Async event streams.

Python port of `packages/ai/src/utils/event-stream.ts`. The TypeScript class is a
push-based queue that consumers drain with ``for await``; here consumers use
``async for`` and ``await stream.result()``.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable
from typing import Generic, TypeVar

from ..types import AssistantMessage, AssistantMessageEvent, ErrorEvent, Model, Usage, now_ms

T = TypeVar("T")
R = TypeVar("R")

_END = object()


class StreamEndedWithoutResult(RuntimeError):
    """Raised by :meth:`EventStream.result` when a stream ends without a result.

    The TypeScript implementation leaves the promise pending forever in this
    case; failing loudly makes the same bug debuggable in Python.
    """


class EventStream(Generic[T, R]):
    """A push-based async stream with a single terminal result."""

    def __init__(self, is_complete: Callable[[T], bool], extract_result: Callable[[T], R]) -> None:
        self._is_complete = is_complete
        self._extract_result = extract_result
        self._queue: deque[T] = deque()
        self._waiters: deque[asyncio.Future[object]] = deque()
        self._done = False
        self._result: R | None = None
        self._result_set = False
        self._result_error: BaseException | None = None
        self._result_waiters: list[asyncio.Future[R]] = []

    @property
    def done(self) -> bool:
        return self._done

    def push(self, event: T) -> None:
        if self._done:
            return

        if self._is_complete(event):
            self._done = True
            self._set_result(self._extract_result(event))

        self._deliver(event)

    def end(self, result: R | None = None) -> None:
        self._done = True
        if result is not None:
            self._set_result(result)
        elif not self._result_set:
            self._set_error(StreamEndedWithoutResult("stream ended before producing a result"))
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.set_result(_END)

    def fail(self, error: BaseException) -> None:
        """End the stream by failing the pending result. No TypeScript equivalent."""
        self._done = True
        if not self._result_set:
            self._set_error(error)
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.set_result(_END)

    async def result(self) -> R:
        if self._result_set:
            if self._result_error is not None:
                raise self._result_error
            return self._result  # type: ignore[return-value]
        future: asyncio.Future[R] = asyncio.get_running_loop().create_future()
        self._result_waiters.append(future)
        return await future

    def __aiter__(self) -> AsyncIterator[T]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[T]:
        while True:
            if self._queue:
                yield self._queue.popleft()
                continue
            if self._done:
                return
            future: asyncio.Future[object] = asyncio.get_running_loop().create_future()
            self._waiters.append(future)
            value = await future
            if value is _END:
                return
            yield value  # type: ignore[misc]

    def _deliver(self, event: T) -> None:
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.set_result(event)
                return
        self._queue.append(event)

    def _set_result(self, result: R) -> None:
        if self._result_set:
            return
        self._result_set = True
        self._result = result
        for future in self._result_waiters:
            if not future.done():
                future.set_result(result)
        self._result_waiters.clear()

    def _set_error(self, error: BaseException) -> None:
        if self._result_set:
            return
        self._result_set = True
        self._result_error = error
        for future in self._result_waiters:
            if not future.done():
                future.set_exception(error)
        self._result_waiters.clear()


def _assistant_is_complete(event: AssistantMessageEvent) -> bool:
    return event.type in ("done", "error")


def _assistant_result(event: AssistantMessageEvent) -> AssistantMessage:
    if event.type == "done":
        return event.message  # type: ignore[union-attr]
    if event.type == "error":
        return event.error  # type: ignore[union-attr]
    raise ValueError("Unexpected event type for final result")


class AssistantMessageEventStream(EventStream[AssistantMessageEvent, AssistantMessage]):
    def __init__(self) -> None:
        super().__init__(_assistant_is_complete, _assistant_result)


def create_assistant_message_event_stream() -> AssistantMessageEventStream:
    return AssistantMessageEventStream()


def setup_error_stream(model: Model, error: BaseException) -> AssistantMessageEventStream:
    """A finished stream that reports a setup failure in-band.

    Port of the failure path of `lazyStream` in `packages/ai/src/api/lazy.ts`:
    a stream-returning entry point never throws synchronously, it hands back a
    stream whose only event is an error and whose result is an
    ``AssistantMessage`` with ``stop_reason="error"``.
    """
    message = AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="error",
        error_message=str(error) or type(error).__name__,
        timestamp=now_ms(),
    )
    stream = AssistantMessageEventStream()
    stream.push(ErrorEvent(reason="error", error=message))
    stream.end(message)
    return stream
