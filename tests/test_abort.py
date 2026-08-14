import asyncio

import pytest
from pi_ai.utils.abort import (
    AbortError,
    AbortSignal,
    combine_abort_signals,
    operation_signal,
    race_with_abort_signal,
)


def test_signal_starts_not_aborted():
    signal = AbortSignal()
    assert signal.aborted is False
    assert signal.reason is None


def test_abort_sets_aborted_and_default_reason():
    signal = AbortSignal()
    signal.abort()
    assert signal.aborted is True
    assert isinstance(signal.reason, AbortError)


def test_abort_with_custom_reason():
    signal = AbortSignal()
    reason = RuntimeError("custom reason")
    signal.abort(reason)
    assert signal.reason is reason


def test_abort_is_idempotent_keeps_first_reason():
    signal = AbortSignal()
    signal.abort(RuntimeError("first"))
    signal.abort(RuntimeError("second"))
    assert str(signal.reason) == "first"


def test_throw_if_aborted_raises_when_aborted():
    signal = AbortSignal()
    signal.abort(ValueError("nope"))
    with pytest.raises(ValueError, match="nope"):
        signal.throw_if_aborted()


def test_throw_if_aborted_is_noop_when_not_aborted():
    signal = AbortSignal()
    signal.throw_if_aborted()  # must not raise


async def test_wait_blocks_until_aborted():
    signal = AbortSignal()

    async def waiter():
        await signal.wait()
        return "done"

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0)
    assert not task.done()
    signal.abort()
    assert await task == "done"


def test_operation_signal_returns_given_signal():
    signal = AbortSignal()
    assert operation_signal(signal) is signal


def test_operation_signal_creates_new_signal_when_none_given():
    signal = operation_signal(None)
    assert isinstance(signal, AbortSignal)
    assert signal.aborted is False


async def test_race_with_abort_signal_returns_operation_result_when_it_finishes_first():
    signal = AbortSignal()

    async def op():
        await asyncio.sleep(0)
        return 42

    result = await race_with_abort_signal(op(), signal)
    assert result == 42


async def test_race_with_abort_signal_raises_reason_when_signal_aborts_first():
    signal = AbortSignal()
    reason = RuntimeError("aborted-first")

    async def op():
        await asyncio.sleep(10)
        return 1

    task = asyncio.ensure_future(race_with_abort_signal(op(), signal))
    await asyncio.sleep(0)
    signal.abort(reason)

    with pytest.raises(RuntimeError, match="aborted-first"):
        await task


async def test_race_with_abort_signal_raises_immediately_when_already_aborted():
    signal = AbortSignal()
    signal.abort(RuntimeError("already gone"))

    async def op():
        await asyncio.sleep(0)
        return 1

    with pytest.raises(RuntimeError, match="already gone"):
        await race_with_abort_signal(op(), signal)


async def test_race_with_abort_signal_propagates_operation_error_when_it_fails_first():
    signal = AbortSignal()

    async def op():
        await asyncio.sleep(0)
        raise ValueError("op failed")

    with pytest.raises(ValueError, match="op failed"):
        await race_with_abort_signal(op(), signal)


async def test_combine_abort_signals_returns_no_signal_for_an_empty_list():
    combined = combine_abort_signals([])
    assert combined.signal is None
    combined.cleanup()


async def test_combine_abort_signals_passes_a_lone_signal_through():
    signal = AbortSignal()
    combined = combine_abort_signals([None, signal, None])
    assert combined.signal is signal
    combined.cleanup()


async def test_combine_abort_signals_aborts_when_any_source_aborts():
    first = AbortSignal()
    second = AbortSignal()
    combined = combine_abort_signals([first, second])
    assert combined.signal is not None
    assert combined.signal.aborted is False

    reason = RuntimeError("second went first")
    second.abort(reason)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert combined.signal.aborted is True
    assert combined.signal.reason is reason
    combined.cleanup()


async def test_combine_abort_signals_short_circuits_an_already_aborted_source():
    first = AbortSignal()
    first.abort(RuntimeError("already gone"))
    second = AbortSignal()

    combined = combine_abort_signals([first, second])
    assert combined.signal is not None
    assert combined.signal.aborted is True
    assert str(combined.signal.reason) == "already gone"
    combined.cleanup()


async def test_combine_abort_signals_cleanup_stops_forwarding():
    first = AbortSignal()
    second = AbortSignal()
    combined = combine_abort_signals([first, second])
    assert combined.signal is not None

    combined.cleanup()
    await asyncio.sleep(0)
    first.abort()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert combined.signal.aborted is False
