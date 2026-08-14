"""Cooperative cancellation signal.

Python port of `packages/ai/src/utils/abort.ts` and
`packages/ai/src/utils/abort-signals.ts`. JavaScript's `AbortSignal` is
modelled here as an `asyncio.Event`-backed class: `abort()` sets the reason and
wakes anything awaiting `wait()`; `throw_if_aborted()` mirrors the DOM method of
the same name.

Also covers `packages/coding-agent/src/utils/abort.ts`: that file re-declares
the same `operationSignal`/`raceWithAbortSignal` pair against Node's native
`AbortSignal` (the coding-agent package does not import from `packages/ai`
for this in TypeScript), but `pi_coding_agent` has no reason to duplicate the
class here -- it imports `AbortSignal`/`operation_signal`/
`race_with_abort_signal` straight from this module (see e.g.
`pi_coding_agent.modes.interactive.interactive_mode`,
`pi_coding_agent.core.bash_executor`). One behavioral difference from the
coding-agent original: TypeScript's `raceWithAbortSignal` accepts an
*optional* signal and returns `operation` unchanged when it is undefined;
this module's `race_with_abort_signal` requires a signal, so callers with an
optional one check `signal is None` themselves before calling it (matching
what TypeScript's own callers effectively get for free from the `?? `
early-return).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from .tasks import spawn

T = TypeVar("T")


class AbortError(Exception):
    """Raised when an aborted signal is thrown or awaited."""


class AbortSignal:
    """A cooperative cancellation signal, analogous to DOM `AbortSignal`."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: BaseException | None = None

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> BaseException | None:
        return self._reason

    def abort(self, reason: BaseException | None = None) -> None:
        """Mark the signal as aborted. Subsequent calls are no-ops."""
        if self._event.is_set():
            return
        self._reason = reason if reason is not None else _abort_error()
        self._event.set()

    def throw_if_aborted(self) -> None:
        if self.aborted:
            raise self._abort_reason()

    async def wait(self) -> None:
        """Wait until the signal is aborted."""
        await self._event.wait()

    def _abort_reason(self) -> BaseException:
        return self._reason if self._reason is not None else _abort_error()


def _abort_error() -> AbortError:
    return AbortError("The operation was aborted")


def operation_signal(signal: AbortSignal | None = None) -> AbortSignal:
    """Create an operation-local signal for public APIs whose signal is optional."""
    return signal if signal is not None else AbortSignal()


async def race_with_abort_signal(operation: Awaitable[T], signal: AbortSignal) -> T:
    """Stop waiting for ``operation`` when ``signal`` aborts.

    The abandoned operation keeps running; if it later raises, the exception is
    swallowed since nothing is awaiting it anymore (mirrors the TypeScript
    `void operation.catch(() => {})` no-op handler for the abandoned promise).
    """
    if signal.aborted:
        _fire_and_forget_ignore_errors(operation)
        raise signal._abort_reason()

    operation_task: asyncio.Task[T] = asyncio.ensure_future(operation)
    abort_task: asyncio.Task[None] = asyncio.ensure_future(signal.wait())
    try:
        done, _pending = await asyncio.wait({operation_task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
        if operation_task in done:
            abort_task.cancel()
            return operation_task.result()

        # Signal aborted first: stop waiting but keep observing the operation.
        _fire_and_forget_ignore_errors(operation_task)
        raise signal._abort_reason()
    finally:
        if not abort_task.done():
            abort_task.cancel()


def _fire_and_forget_ignore_errors(awaitable: Awaitable[Any] | asyncio.Task[Any]) -> None:
    if isinstance(awaitable, asyncio.Task):
        task = awaitable
    else:

        async def _wrap() -> Any:
            return await awaitable

        task = spawn(_wrap())

    def _ignore(completed: asyncio.Task[Any]) -> None:
        if completed.cancelled():
            return
        completed.exception()

    task.add_done_callback(_ignore)


class AbortController:
    """Owns an :class:`AbortSignal` and the right to abort it.

    Mirrors the DOM `AbortController`: the owner of a run holds the controller
    and hands out only ``controller.signal`` to the work it starts.
    """

    def __init__(self) -> None:
        self.signal = AbortSignal()

    def abort(self, reason: BaseException | None = None) -> None:
        self.signal.abort(reason)


@dataclass
class CombinedAbortSignal:
    """The result of :func:`combine_abort_signals`.

    Port of `CombinedAbortSignal` in `packages/ai/src/utils/abort-signals.ts`.
    ``cleanup`` detaches the forwarding listeners; it is a no-op when no
    listener was installed.
    """

    signal: AbortSignal | None
    cleanup: Callable[[], None]


def combine_abort_signals(signals: Sequence[AbortSignal | None]) -> CombinedAbortSignal:
    """Abort as soon as any of ``signals`` aborts.

    Python port of `combineAbortSignals` in
    `packages/ai/src/utils/abort-signals.ts`. Zero or one active signal needs no
    forwarding, so the input signal (or ``None``) is returned directly.
    """
    active = [signal for signal in signals if signal is not None]
    if not active:
        return CombinedAbortSignal(signal=None, cleanup=lambda: None)
    if len(active) == 1:
        return CombinedAbortSignal(signal=active[0], cleanup=lambda: None)

    controller = AbortController()
    forwards: list[asyncio.Task[None]] = []

    def abort_from(source: AbortSignal) -> None:
        if not controller.signal.aborted:
            controller.abort(source.reason)

    for signal in active:
        if signal.aborted:
            abort_from(signal)
            break

        async def forward(source: AbortSignal = signal) -> None:
            await source.wait()
            abort_from(source)

        forwards.append(spawn(forward()))

    def cleanup() -> None:
        for task in forwards:
            task.cancel()

    return CombinedAbortSignal(signal=controller.signal, cleanup=cleanup)
