"""Tests for `pi_ai.session_resources` — the session-scoped resource cleanup
registry.

Ported concept from `packages/ai/src/session-resources.ts`. TypeScript
aggregates cleanup failures into an `AggregateError`; this port uses Python's
native `ExceptionGroup` instead (see the module docstring).
"""

from __future__ import annotations

import pytest
from pi_ai.session_resources import cleanup_session_resources, register_session_resource_cleanup


def test_cleanup_runs_every_registered_callback() -> None:
    calls: list[str | None] = []
    unregister1 = register_session_resource_cleanup(lambda session_id: calls.append(session_id))
    unregister2 = register_session_resource_cleanup(lambda session_id: calls.append(session_id))
    try:
        cleanup_session_resources("session-1")
        assert calls == ["session-1", "session-1"]
    finally:
        unregister1()
        unregister2()


def test_unregister_stops_the_callback_from_running() -> None:
    calls: list[str | None] = []
    unregister = register_session_resource_cleanup(lambda session_id: calls.append(session_id))
    unregister()

    cleanup_session_resources("session-1")

    assert calls == []


def test_cleanup_collects_failures_into_an_exception_group_without_stopping_other_callbacks() -> None:
    calls: list[str] = []

    def failing(_session_id: str | None) -> None:
        raise ValueError("boom")

    def succeeding(session_id: str | None) -> None:
        calls.append("ran")

    unregister1 = register_session_resource_cleanup(failing)
    unregister2 = register_session_resource_cleanup(succeeding)
    try:
        with pytest.raises(ExceptionGroup) as exc_info:
            cleanup_session_resources("session-1")

        assert calls == ["ran"]
        assert len(exc_info.value.exceptions) == 1
        assert isinstance(exc_info.value.exceptions[0], ValueError)
    finally:
        unregister1()
        unregister2()


def test_cleanup_with_no_registered_callbacks_does_not_raise() -> None:
    cleanup_session_resources("session-1")
