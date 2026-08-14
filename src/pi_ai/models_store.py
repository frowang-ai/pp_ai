"""Persistent model catalog storage.

Python port of `packages/ai/src/models-store.ts`. Stores the model catalog a
provider last fetched from a remote source, plus enough HTTP validator state
(`etag`, `last_modified`) to make a conditional re-fetch.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Protocol

from .types import Model
from .utils.abort import AbortSignal


@dataclass
class ModelsStoreEntry:
    models: list[Model] = field(default_factory=list)
    last_modified: int | None = None
    """Unix timestamp from the remote catalog's Last-Modified header."""
    checked_at: int | None = None
    """Unix timestamp of the last completed remote check."""
    etag: str | None = None
    """Opaque validator from the remote catalog's ETag header, stored verbatim
    (quotes included) and echoed back as If-None-Match."""


@dataclass
class ModelsStoreOperationOptions:
    signal: AbortSignal | None = None


class ModelsStore(Protocol):
    """Persistent model catalogs keyed by provider ID."""

    async def read(
        self, provider_id: str, options: ModelsStoreOperationOptions | None = None
    ) -> ModelsStoreEntry | None: ...

    async def write(
        self, provider_id: str, entry: ModelsStoreEntry, options: ModelsStoreOperationOptions | None = None
    ) -> None: ...

    async def delete(self, provider_id: str, options: ModelsStoreOperationOptions | None = None) -> None: ...


class InMemoryModelsStore:
    def __init__(self) -> None:
        self._entries: dict[str, ModelsStoreEntry] = {}

    async def read(
        self, provider_id: str, options: ModelsStoreOperationOptions | None = None
    ) -> ModelsStoreEntry | None:
        if options and options.signal:
            options.signal.throw_if_aborted()
        entry = self._entries.get(provider_id)
        return copy.deepcopy(entry) if entry is not None else None

    async def write(
        self, provider_id: str, entry: ModelsStoreEntry, options: ModelsStoreOperationOptions | None = None
    ) -> None:
        if options and options.signal:
            options.signal.throw_if_aborted()
        self._entries[provider_id] = copy.deepcopy(entry)

    async def delete(self, provider_id: str, options: ModelsStoreOperationOptions | None = None) -> None:
        if options and options.signal:
            options.signal.throw_if_aborted()
        self._entries.pop(provider_id, None)
