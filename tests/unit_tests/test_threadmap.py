"""Behavioral tests for the thread-map stores (D14) + the v0.3 consumed journal (D20/D21).

Both :class:`InMemoryThreadMap` and :class:`SqliteThreadMap` are exercised through the
:class:`ThreadMapStore` public surface (get/put/delete/items + is/mark/drop_consumed + the async
twins). The key property the production default must hold is **persistence across a process
restart** — proved here by closing a ``SqliteThreadMap`` and reopening a fresh one on the same db
file (for both the binding AND the consumed journal that makes a resume exactly-once).
"""

from __future__ import annotations

import pytest

from langchain_rine._threadmap import (
    InMemoryThreadMap,
    SqliteThreadMap,
    ThreadMapStore,
)

H = "peer@acme.rine.network"
C = "cccccccc-dddd-eeee-ffff-000000000002"
T = "thread-1"
M1 = "cccccccc-dddd-eeee-ffff-000000000001"
M2 = "cccccccc-dddd-eeee-ffff-000000000009"


class TestInMemoryThreadMap:
    def test_put_then_get_returns_thread_id(self):
        store = InMemoryThreadMap()
        store.put(H, C, T)
        assert store.get(H, C) == T

    def test_get_unknown_key_returns_none(self):
        assert InMemoryThreadMap().get(H, C) is None

    def test_put_same_key_newest_wins(self):
        store = InMemoryThreadMap()
        store.put(H, C, "old")
        store.put(H, C, "new")
        assert store.get(H, C) == "new"

    def test_delete_removes_binding(self):
        store = InMemoryThreadMap()
        store.put(H, C, T)
        store.delete(H, C)
        assert store.get(H, C) is None

    def test_delete_missing_key_is_noop(self):
        store = InMemoryThreadMap()
        store.delete(H, C)  # must not raise
        assert store.get(H, C) is None

    def test_items_iterates_all_bindings(self):
        store = InMemoryThreadMap()
        store.put(H, C, T)
        store.put("other@org", "conv2", "thread-2")
        assert set(store.items()) == {((H, C), T), (("other@org", "conv2"), "thread-2")}

    def test_satisfies_thread_map_store_protocol(self):
        assert isinstance(InMemoryThreadMap(), ThreadMapStore)


class TestSqliteThreadMap:
    def test_put_then_get_roundtrips(self, tmp_path):
        store = SqliteThreadMap(str(tmp_path / "tm.sqlite"))
        store.put(H, C, T)
        assert store.get(H, C) == T

    def test_get_unknown_key_returns_none(self, tmp_path):
        store = SqliteThreadMap(str(tmp_path / "tm.sqlite"))
        assert store.get(H, C) is None

    def test_put_same_key_upserts_newest_wins(self, tmp_path):
        store = SqliteThreadMap(str(tmp_path / "tm.sqlite"))
        store.put(H, C, "old")
        store.put(H, C, "new")
        assert store.get(H, C) == "new"
        # exactly one row for the key (the upsert did not duplicate)
        assert len(list(store.items())) == 1

    def test_delete_removes_binding(self, tmp_path):
        store = SqliteThreadMap(str(tmp_path / "tm.sqlite"))
        store.put(H, C, T)
        store.delete(H, C)
        assert store.get(H, C) is None

    def test_items_iterates_all_bindings(self, tmp_path):
        store = SqliteThreadMap(str(tmp_path / "tm.sqlite"))
        store.put(H, C, T)
        store.put("other@org", "conv2", "thread-2")
        assert set(store.items()) == {((H, C), T), (("other@org", "conv2"), "thread-2")}

    def test_binding_persists_across_reopen(self, tmp_path):
        """The production default must survive a 'process restart' (close + reopen same file)."""
        db = str(tmp_path / "tm.sqlite")
        first = SqliteThreadMap(db)
        first.put(H, C, T)
        first.close()

        reopened = SqliteThreadMap(db)
        assert reopened.get(H, C) == T  # the binding survived the restart

    def test_satisfies_thread_map_store_protocol(self, tmp_path):
        assert isinstance(SqliteThreadMap(str(tmp_path / "tm.sqlite")), ThreadMapStore)

    def test_setup_first_yields_cross_thread_shareable_connection(self, tmp_path):
        """tm-2: calling setup() before any op opens the SAME check_same_thread=False connection.

        A put/get issued from a *different* thread must succeed (it would raise a ProgrammingError
        if setup() had opened the default check_same_thread=True connection).
        """
        import threading

        store = SqliteThreadMap(str(tmp_path / "tm.sqlite"))
        store.setup()  # open the connection up-front

        errors: list[Exception] = []

        def _cross_thread_put() -> None:
            try:
                store.put(H, C, T)  # reuses the setup() connection from another thread
            except Exception as exc:  # noqa: BLE001 - capture to assert in the main thread
                errors.append(exc)

        worker = threading.Thread(target=_cross_thread_put)
        worker.start()
        worker.join()

        assert errors == []  # no sqlite cross-thread ProgrammingError
        assert store.get(H, C) == T


# =====================================================================
# D20 — the consumed-message-id journal (exactly-once resume)
# =====================================================================


def _both_stores(tmp_path) -> list[ThreadMapStore]:
    """One of each store kind so a journal test asserts identical behavior on both."""
    return [InMemoryThreadMap(), SqliteThreadMap(str(tmp_path / "tm.sqlite"))]


class TestConsumedJournal:
    @pytest.mark.parametrize("store_idx", [0, 1])
    def test_mark_then_is_consumed_true(self, tmp_path, store_idx):
        store = _both_stores(tmp_path)[store_idx]
        assert store.is_consumed(H, C, M1) is False
        store.mark_consumed(H, C, M1)
        assert store.is_consumed(H, C, M1) is True

    @pytest.mark.parametrize("store_idx", [0, 1])
    def test_mark_is_idempotent(self, tmp_path, store_idx):
        """A duplicate mark_consumed never raises (idempotent upsert)."""
        store = _both_stores(tmp_path)[store_idx]
        store.mark_consumed(H, C, M1)
        store.mark_consumed(H, C, M1)  # must not raise / must not double-count
        assert store.is_consumed(H, C, M1) is True

    @pytest.mark.parametrize("store_idx", [0, 1])
    def test_drop_consumed_clears_entry(self, tmp_path, store_idx):
        store = _both_stores(tmp_path)[store_idx]
        store.mark_consumed(H, C, M1)
        store.drop_consumed(H, C, M1)
        assert store.is_consumed(H, C, M1) is False

    @pytest.mark.parametrize("store_idx", [0, 1])
    def test_drop_missing_entry_is_noop(self, tmp_path, store_idx):
        store = _both_stores(tmp_path)[store_idx]
        store.drop_consumed(H, C, M1)  # never marked — must not raise
        assert store.is_consumed(H, C, M1) is False

    @pytest.mark.parametrize("store_idx", [0, 1])
    def test_journal_is_scoped_per_message_id(self, tmp_path, store_idx):
        """A distinct message id in the same conversation is unaffected by another's mark/drop."""
        store = _both_stores(tmp_path)[store_idx]
        store.mark_consumed(H, C, M1)
        assert store.is_consumed(H, C, M2) is False  # a sibling id is not marked
        store.drop_consumed(H, C, M1)
        store.mark_consumed(H, C, M2)
        assert store.is_consumed(H, C, M2) is True
        assert store.is_consumed(H, C, M1) is False

    def test_consumed_journal_persists_across_reopen(self, tmp_path):
        """D20 durable: the journal must survive a process restart (close + reopen same file).

        This is what makes the exactly-once guarantee hold across a crash, not just an ack failure.
        """
        db = str(tmp_path / "tm.sqlite")
        first = SqliteThreadMap(db)
        first.mark_consumed(H, C, M1)
        first.close()

        reopened = SqliteThreadMap(db)
        assert reopened.is_consumed(H, C, M1) is True  # the journal survived the restart

    def test_consumed_journal_independent_of_binding(self, tmp_path):
        """Deleting the thread binding leaves the consumed journal intact (separate tables).

        After a terminal-unregister the binding is dropped, but a redelivered already-consumed
        message must still be recognized — so ``delete`` must not touch the journal.
        """
        store = SqliteThreadMap(str(tmp_path / "tm.sqlite"))
        store.put(H, C, T)
        store.mark_consumed(H, C, M1)
        store.delete(H, C)
        assert store.get(H, C) is None  # binding gone
        assert store.is_consumed(H, C, M1) is True  # journal entry survives


# =====================================================================
# D21 — async twins (aget/aput/adelete + ais/amark/adrop_consumed)
# =====================================================================


async def _aclose(store: ThreadMapStore) -> None:
    """Close a store's aiosqlite connection (a no-op for the in-memory store).

    Keeps the aiosqlite worker thread from outliving the test's event loop (which would surface as
    a stray ``Event loop is closed`` thread warning); ``InMemoryThreadMap`` has no async conn.
    """
    aclose = getattr(store, "aclose", None)
    if aclose is not None:
        await aclose()


class TestAsyncStore:
    @pytest.mark.parametrize("store_idx", [0, 1])
    async def test_async_binding_roundtrip(self, tmp_path, store_idx):
        store = _both_stores(tmp_path)[store_idx]
        await store.aput(H, C, T)
        assert await store.aget(H, C) == T
        await store.adelete(H, C)
        assert await store.aget(H, C) is None
        await _aclose(store)

    @pytest.mark.parametrize("store_idx", [0, 1])
    async def test_async_consumed_roundtrip(self, tmp_path, store_idx):
        store = _both_stores(tmp_path)[store_idx]
        assert await store.ais_consumed(H, C, M1) is False
        await store.amark_consumed(H, C, M1)
        assert await store.ais_consumed(H, C, M1) is True
        await store.adrop_consumed(H, C, M1)
        assert await store.ais_consumed(H, C, M1) is False
        await _aclose(store)

    async def test_async_writes_are_visible_to_the_sync_path(self, tmp_path):
        """The aiosqlite + stdlib-sqlite3 conns share the file: an async write is sync-visible.

        (WAL is set on setup so the two connections coexist; this proves they back the same data,
        not two divergent stores.)
        """
        store = SqliteThreadMap(str(tmp_path / "tm.sqlite"))
        await store.aput(H, C, T)
        await store.amark_consumed(H, C, M1)
        assert store.get(H, C) == T  # the sync connection reads the async write
        assert store.is_consumed(H, C, M1) is True
        await store.aclose()
