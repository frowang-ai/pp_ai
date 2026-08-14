"""Background task ownership.

`asyncio.ensure_future` returns a task that the event loop only holds weakly,
so a fire-and-forget producer can be garbage collected mid-flight and its
stream would stall. Every producer task started by this package is registered
here until it finishes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

_background_tasks: set[asyncio.Task[Any]] = set()


def spawn(coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
    """Start ``coro`` and keep a strong reference until it completes."""
    task = asyncio.ensure_future(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task
