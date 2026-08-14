"""Session-scoped resource cleanup registry.

Python port of `packages/ai/src/session-resources.ts`. Long-lived transports
(websocket connections, background pollers) register a cleanup callback here;
callers that tear down a session invoke `cleanup_session_resources` once to
run every registered callback, collecting failures instead of letting one
callback's exception stop the others.
"""

from __future__ import annotations

from collections.abc import Callable

SessionResourceCleanup = Callable[[str | None], None]

_session_resource_cleanups: set[SessionResourceCleanup] = set()


def register_session_resource_cleanup(cleanup: SessionResourceCleanup) -> Callable[[], None]:
    """Register a cleanup callback and return a function that unregisters it."""
    _session_resource_cleanups.add(cleanup)

    def unregister() -> None:
        _session_resource_cleanups.discard(cleanup)

    return unregister


def cleanup_session_resources(session_id: str | None = None) -> None:
    """Run every registered cleanup for `session_id`.

    Mirrors the TypeScript `AggregateError`: failures from individual cleanups
    are collected and re-raised together as an `ExceptionGroup` rather than
    aborting after the first one.
    """
    errors: list[BaseException] = []
    for cleanup in list(_session_resource_cleanups):
        try:
            cleanup(session_id)
        except Exception as error:
            errors.append(error)
    if errors:
        raise ExceptionGroup("Failed to cleanup session resources", errors)
