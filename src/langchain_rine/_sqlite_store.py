"""``SqliteThreadMap`` — the durable production :class:`ThreadMapStore` (D14/D20/D21).

The thread-map binding + the v0.3 **consumed-message-id journal** (exactly-once resume,
D20), co-located in one sqlite db file (alongside the LangGraph ``SqliteSaver`` or a sibling
``*.sqlite``). Two tables:

- ``thread_map(handle, conversation_id, thread_id)`` — the ``(handle, conversation_id) ->
  thread_id`` binding; survives a process restart so cross-process handoffs work.
- ``consumed_message(handle, conversation_id, message_id)`` — the durable journal of message
  ids already resumed-and-recorded. ``is_consumed`` is the v0.3 exactly-once guard;
  ``mark_consumed`` records (act-then-record: after a successful resume, before the ack);
  ``drop_consumed`` self-cleans the entry after a successful ack so the journal stays bounded
  to in-flight ids (D20). The journal stores **only ``message.id``**, never payload (I-2).

Two connections share the file: a sync stdlib ``sqlite3`` connection (the sync resume path)
and a **lazily-imported ``aiosqlite``** connection (the async ``_aresolve`` path — D21). The
``import aiosqlite`` is lazy (inside :meth:`_aconnection`), so a sync-only user never needs
aiosqlite. ``PRAGMA journal_mode=WAL`` lets the sync conn, the aiosqlite conn, and the
caller's ``SqliteSaver`` coexist without lock contention. SQL is shared between the two paths
via the module-level ``_SQL_*`` constants — no duplicated SQL (I-1 import has no aiosqlite,
no network).
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    import aiosqlite

__all__ = ["SqliteThreadMap"]

# --- Shared SQL (one source of truth for both the sync and aiosqlite paths) -----------------
_SQL_CREATE_THREAD_MAP = (
    "CREATE TABLE IF NOT EXISTS thread_map ("
    "handle TEXT NOT NULL, conversation_id TEXT NOT NULL, thread_id TEXT NOT NULL, "
    "PRIMARY KEY (handle, conversation_id))"
)
_SQL_CREATE_CONSUMED = (
    "CREATE TABLE IF NOT EXISTS consumed_message ("
    "handle TEXT NOT NULL, conversation_id TEXT NOT NULL, message_id TEXT NOT NULL, "
    "PRIMARY KEY (handle, conversation_id, message_id))"
)
_SQL_GET = "SELECT thread_id FROM thread_map WHERE handle = ? AND conversation_id = ?"
_SQL_PUT = (
    "INSERT INTO thread_map (handle, conversation_id, thread_id) VALUES (?, ?, ?) "
    "ON CONFLICT (handle, conversation_id) DO UPDATE SET thread_id = excluded.thread_id"
)
_SQL_DELETE = "DELETE FROM thread_map WHERE handle = ? AND conversation_id = ?"
_SQL_ITEMS = "SELECT handle, conversation_id, thread_id FROM thread_map"
_SQL_IS_CONSUMED = (
    "SELECT 1 FROM consumed_message "
    "WHERE handle = ? AND conversation_id = ? AND message_id = ? LIMIT 1"
)
_SQL_MARK_CONSUMED = (
    "INSERT INTO consumed_message (handle, conversation_id, message_id) VALUES (?, ?, ?) "
    "ON CONFLICT (handle, conversation_id, message_id) DO NOTHING"
)
_SQL_DROP_CONSUMED = (
    "DELETE FROM consumed_message WHERE handle = ? AND conversation_id = ? AND message_id = ?"
)


class SqliteThreadMap:
    """A stdlib-``sqlite3`` (+ lazy aiosqlite) :class:`ThreadMapStore` — production default.

    The documented production default (D14): the ``thread_map`` binding + the ``consumed_message``
    journal both survive a process restart, so cross-process handoffs AND the exactly-once
    guarantee (D20) hold across a crash. The sync surface (``get``/``put``/``delete``/``items`` +
    ``is_consumed``/``mark_consumed``/``drop_consumed``) uses the stdlib ``sqlite3`` conn; the
    async twins (``a*``) use a lazily-opened ``aiosqlite`` connection on the SAME db file so the
    async resume path never blocks the event loop on stdlib sqlite (D21, fixes ``tm-4``).

    The connection is opened lazily on first use and reused; ``check_same_thread=False`` lets the
    sync connection be shared by a single-threaded async poll loop and its event loop. ``WAL``
    journal mode is set on setup so the sync conn, the aiosqlite conn, and the caller's
    ``SqliteSaver`` coexist.
    """

    def __init__(self, db_path: str) -> None:
        """Record the db path; the connection + tables are created lazily on first use."""
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._aconn: aiosqlite.Connection | None = None

    # --- sync connection lifecycle ---------------------------------------------------------

    def _open(self) -> sqlite3.Connection:
        """Open (once) and cache the sqlite connection with ``check_same_thread=False`` (tm-2).

        The single opener both ``_connection`` and ``setup`` route through, so a user who calls
        ``setup()`` first gets the SAME cross-thread-shareable connection as a lazy first use
        (no flag divergence between the two entry points).
        """
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        return self._conn

    def _connection(self) -> sqlite3.Connection:
        """Open (once) the sqlite connection and ensure both tables exist."""
        first_use = self._conn is None
        conn = self._open()
        if first_use:
            self.setup()
        return conn

    def setup(self) -> None:
        """Create both tables + set WAL if they do not yet exist (idempotent)."""
        conn = self._open()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(_SQL_CREATE_THREAD_MAP)
        conn.execute(_SQL_CREATE_CONSUMED)
        conn.commit()

    # --- sync thread-map surface -----------------------------------------------------------

    def get(self, handle: str, conversation_id: str) -> str | None:
        """Return the thread_id mapped to ``(handle, conversation_id)``, or None."""
        row = self._connection().execute(_SQL_GET, (handle, conversation_id)).fetchone()
        return None if row is None else str(row[0])

    def put(self, handle: str, conversation_id: str, thread_id: str) -> None:
        """Upsert ``(handle, conversation_id) -> thread_id`` (newest-wins on conflict)."""
        conn = self._connection()
        conn.execute(_SQL_PUT, (handle, conversation_id, thread_id))
        conn.commit()

    def delete(self, handle: str, conversation_id: str) -> None:
        """Remove the binding for ``(handle, conversation_id)`` if present."""
        conn = self._connection()
        conn.execute(_SQL_DELETE, (handle, conversation_id))
        conn.commit()

    def items(self) -> Iterable[tuple[tuple[str, str], str]]:
        """Iterate ``((handle, conversation_id), thread_id)`` for every binding."""
        rows = self._connection().execute(_SQL_ITEMS).fetchall()
        return [((str(h), str(c)), str(t)) for h, c, t in rows]

    # --- sync consumed journal (D20 exactly-once) ------------------------------------------

    def is_consumed(self, handle: str, conversation_id: str, message_id: str) -> bool:
        """True iff ``message_id`` was already resumed-and-recorded for this conversation."""
        row = (
            self._connection()
            .execute(_SQL_IS_CONSUMED, (handle, conversation_id, message_id))
            .fetchone()
        )
        return row is not None

    def mark_consumed(self, handle: str, conversation_id: str, message_id: str) -> None:
        """Durably record ``message_id`` as consumed (idempotent upsert)."""
        conn = self._connection()
        conn.execute(_SQL_MARK_CONSUMED, (handle, conversation_id, message_id))
        conn.commit()

    def drop_consumed(self, handle: str, conversation_id: str, message_id: str) -> None:
        """Drop the journal entry for ``message_id`` (self-clean after a successful ack)."""
        conn = self._connection()
        conn.execute(_SQL_DROP_CONSUMED, (handle, conversation_id, message_id))
        conn.commit()

    # --- async connection lifecycle (lazy aiosqlite, D21) ----------------------------------

    async def _aconnection(self) -> aiosqlite.Connection:
        """Open (once) the aiosqlite connection on the same db file; ensure tables + WAL.

        ``import aiosqlite`` is LAZY (here, not at module level), so a sync-only user of
        ``SqliteThreadMap`` never needs aiosqlite installed (I-1). The first open creates the
        tables (idempotent) and sets WAL so the async conn coexists with the sync conn + the
        caller's ``SqliteSaver``.
        """
        if self._aconn is None:
            import aiosqlite  # lazy: sync-only users never import aiosqlite

            self._aconn = await aiosqlite.connect(self._db_path)
            await self._aconn.execute("PRAGMA journal_mode=WAL")
            await self._aconn.execute(_SQL_CREATE_THREAD_MAP)
            await self._aconn.execute(_SQL_CREATE_CONSUMED)
            await self._aconn.commit()
        return self._aconn

    async def _afetchone(self, sql: str, params: tuple[Any, ...]) -> Any:
        conn = await self._aconnection()
        async with conn.execute(sql, params) as cursor:
            return await cursor.fetchone()

    async def _aexecute(self, sql: str, params: tuple[Any, ...]) -> None:
        conn = await self._aconnection()
        await conn.execute(sql, params)
        await conn.commit()

    # --- async thread-map + consumed journal (D21; SQL shared with the sync path) ----------

    async def aget(self, handle: str, conversation_id: str) -> str | None:
        """Async twin of :meth:`get`."""
        row = await self._afetchone(_SQL_GET, (handle, conversation_id))
        return None if row is None else str(row[0])

    async def aput(self, handle: str, conversation_id: str, thread_id: str) -> None:
        """Async twin of :meth:`put`."""
        await self._aexecute(_SQL_PUT, (handle, conversation_id, thread_id))

    async def adelete(self, handle: str, conversation_id: str) -> None:
        """Async twin of :meth:`delete`."""
        await self._aexecute(_SQL_DELETE, (handle, conversation_id))

    async def ais_consumed(self, handle: str, conversation_id: str, message_id: str) -> bool:
        """Async twin of :meth:`is_consumed`."""
        row = await self._afetchone(_SQL_IS_CONSUMED, (handle, conversation_id, message_id))
        return row is not None

    async def amark_consumed(self, handle: str, conversation_id: str, message_id: str) -> None:
        """Async twin of :meth:`mark_consumed`."""
        await self._aexecute(_SQL_MARK_CONSUMED, (handle, conversation_id, message_id))

    async def adrop_consumed(self, handle: str, conversation_id: str, message_id: str) -> None:
        """Async twin of :meth:`drop_consumed`."""
        await self._aexecute(_SQL_DROP_CONSUMED, (handle, conversation_id, message_id))

    async def aclose(self) -> None:
        """Close the aiosqlite connection if it was opened."""
        if self._aconn is not None:
            await self._aconn.close()
            self._aconn = None

    def close(self) -> None:
        """Close the underlying sync sqlite connection if it was opened.

        The aiosqlite connection (if any) is closed via :meth:`aclose` from async code; a
        sync-only user never opens one.
        """
        if self._conn is not None:
            self._conn.close()
            self._conn = None
