"""Tests for `pi_ai.models_store` — the persisted model-catalog store.

Ported concept from `packages/ai/src/models-store.ts`. There's no dedicated
TypeScript test file for this module either; it's a small enough surface
(read/write/delete plus abort-signal checks) that a direct unit test covers
it fully.
"""

from __future__ import annotations

import pytest
from pi_ai.models_store import InMemoryModelsStore, ModelsStoreEntry, ModelsStoreOperationOptions
from pi_ai.types import Model
from pi_ai.utils.abort import AbortController, AbortError


async def test_read_returns_none_for_an_unknown_provider() -> None:
    store = InMemoryModelsStore()
    assert await store.read("unknown") is None


async def test_write_then_read_round_trips_the_entry() -> None:
    store = InMemoryModelsStore()
    entry = ModelsStoreEntry(models=[Model(id="m1")], etag='"abc"', last_modified=123, checked_at=456)

    await store.write("provider-a", entry)
    read_back = await store.read("provider-a")

    assert read_back == entry


async def test_write_deep_copies_so_later_mutation_does_not_leak() -> None:
    store = InMemoryModelsStore()
    entry = ModelsStoreEntry(models=[Model(id="m1")])

    await store.write("provider-a", entry)
    entry.models.append(Model(id="m2"))

    read_back = await store.read("provider-a")
    assert len(read_back.models) == 1


async def test_read_deep_copies_so_caller_mutation_does_not_leak() -> None:
    store = InMemoryModelsStore()
    entry = ModelsStoreEntry(models=[Model(id="m1")])
    await store.write("provider-a", entry)

    read_back = await store.read("provider-a")
    read_back.models.append(Model(id="m2"))

    read_again = await store.read("provider-a")
    assert len(read_again.models) == 1


async def test_delete_removes_the_entry() -> None:
    store = InMemoryModelsStore()
    await store.write("provider-a", ModelsStoreEntry(models=[Model(id="m1")]))

    await store.delete("provider-a")

    assert await store.read("provider-a") is None


async def test_delete_of_an_unknown_provider_is_a_no_op() -> None:
    store = InMemoryModelsStore()
    await store.delete("unknown")


async def test_read_write_delete_honour_an_already_aborted_signal() -> None:
    store = InMemoryModelsStore()
    controller = AbortController()
    controller.abort()
    options = ModelsStoreOperationOptions(signal=controller.signal)

    with pytest.raises(AbortError):
        await store.read("provider-a", options)
    with pytest.raises(AbortError):
        await store.write("provider-a", ModelsStoreEntry(), options)
    with pytest.raises(AbortError):
        await store.delete("provider-a", options)
