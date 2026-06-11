"""Durable ``(handle, conversation_id) -> thread_id`` map for the Tier-3 resumer.

The Tier-3 idle-wake resumer (``inbound.py``) needs to know which paused LangGraph
thread an inbound rine message should wake. That binding is the
``(sender_handle, conversation_id) -> thread_id`` map this module's stores hold, plus the
v0.3 **consumed-message-id journal** that makes a resume exactly-once (D20).

Two implementations behind one :class:`ThreadMapStore` protocol (D14):

- :class:`InMemoryThreadMap` — dicts; the default for tests / ephemeral runs, pairs with
  ``InMemorySaver``.
- :class:`SqliteThreadMap` (re-exported from ``_sqlite_store``) — **stdlib ``sqlite3`` + a
  lazily-imported ``aiosqlite`` async path (D21)**, co-located with the checkpointer db. The
  **documented production default**: both the binding and the consumed journal must survive a
  process restart for cross-process handoffs + exactly-once to work (SPEC §11/§17).

The protocol carries the sync surface (``get``/``put``/``delete``/``items``), the consumed
journal (``is_consumed``/``mark_consumed``/``drop_consumed``), and the async twins the resumer's
async path awaits (``aget``/``aput``/``adelete``/``ais_consumed``/``amark_consumed``/
``adrop_consumed`` — D21).

The whole module is side-effect-free at import (invariant I-1): no network, no credentials, no
client construction, no aiosqlite (its import is lazy inside ``SqliteThreadMap``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from langchain_rine._sqlite_store import SqliteThreadMap

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["ThreadMapStore", "InMemoryThreadMap", "SqliteThreadMap"]


@runtime_checkable
class ThreadMapStore(Protocol):
    """The store contract the resumer depends on.

    Maps a ``(handle, conversation_id)`` key to the id of the LangGraph thread that is (or
    will be) paused awaiting that conversation's next inbound message, and journals the
    message ids already resumed-and-recorded (the D20 exactly-once guard). Both key fields are
    the real ``rine`` ``DecryptedMessage`` attributes (sender handle + ``conversation_id``).

    The async twins (``a*``) mirror the sync surface the resumer's async path uses (D21); an
    in-process store may delegate them straight to the sync methods, a sqlite store backs them
    with aiosqlite to avoid blocking the event loop.
    """

    # --- thread-map binding ---
    def get(self, handle: str, conversation_id: str) -> str | None:
        """Return the thread_id mapped to ``(handle, conversation_id)``, or None."""
        ...

    def put(self, handle: str, conversation_id: str, thread_id: str) -> None:
        """Bind ``(handle, conversation_id)`` to ``thread_id`` (newest-wins overwrite)."""
        ...

    def delete(self, handle: str, conversation_id: str) -> None:
        """Remove the binding for ``(handle, conversation_id)`` if present."""
        ...

    def items(self) -> Iterable[tuple[tuple[str, str], str]]:
        """Iterate ``((handle, conversation_id), thread_id)`` for every binding."""
        ...

    # --- consumed-message-id journal (D20 exactly-once) ---
    def is_consumed(self, handle: str, conversation_id: str, message_id: str) -> bool:
        """True iff ``message_id`` was already resumed-and-recorded for this conversation."""
        ...

    def mark_consumed(self, handle: str, conversation_id: str, message_id: str) -> None:
        """Record ``message_id`` as consumed for this conversation (idempotent)."""
        ...

    def drop_consumed(self, handle: str, conversation_id: str, message_id: str) -> None:
        """Drop the consumed journal entry for ``message_id`` (self-clean after ack)."""
        ...

    # --- async twins (D21) ---
    async def aget(self, handle: str, conversation_id: str) -> str | None:
        """Async twin of :meth:`get`."""
        ...

    async def aput(self, handle: str, conversation_id: str, thread_id: str) -> None:
        """Async twin of :meth:`put`."""
        ...

    async def adelete(self, handle: str, conversation_id: str) -> None:
        """Async twin of :meth:`delete`."""
        ...

    async def ais_consumed(self, handle: str, conversation_id: str, message_id: str) -> bool:
        """Async twin of :meth:`is_consumed`."""
        ...

    async def amark_consumed(self, handle: str, conversation_id: str, message_id: str) -> None:
        """Async twin of :meth:`mark_consumed`."""
        ...

    async def adrop_consumed(self, handle: str, conversation_id: str, message_id: str) -> None:
        """Async twin of :meth:`drop_consumed`."""
        ...


class InMemoryThreadMap:
    """A dict-backed :class:`ThreadMapStore` — the ephemeral / test default.

    State lives only in this process; pair it with ``InMemorySaver`` when threads do not
    need to survive a restart. The async twins delegate straight to the sync methods (no I/O
    to offload). Use :class:`SqliteThreadMap` for production.
    """

    def __init__(self) -> None:
        """Start with an empty binding map + consumed journal."""
        self._map: dict[tuple[str, str], str] = {}
        self._consumed: set[tuple[str, str, str]] = set()

    def get(self, handle: str, conversation_id: str) -> str | None:
        """Return the thread_id mapped to ``(handle, conversation_id)``, or None."""
        return self._map.get((handle, conversation_id))

    def put(self, handle: str, conversation_id: str, thread_id: str) -> None:
        """Bind ``(handle, conversation_id)`` to ``thread_id`` (newest-wins overwrite)."""
        self._map[(handle, conversation_id)] = thread_id

    def delete(self, handle: str, conversation_id: str) -> None:
        """Remove the binding for ``(handle, conversation_id)`` if present."""
        self._map.pop((handle, conversation_id), None)

    def items(self) -> Iterable[tuple[tuple[str, str], str]]:
        """Iterate ``((handle, conversation_id), thread_id)`` for every binding."""
        return list(self._map.items())

    def is_consumed(self, handle: str, conversation_id: str, message_id: str) -> bool:
        """True iff ``message_id`` was already resumed-and-recorded for this conversation."""
        return (handle, conversation_id, message_id) in self._consumed

    def mark_consumed(self, handle: str, conversation_id: str, message_id: str) -> None:
        """Record ``message_id`` as consumed for this conversation (idempotent)."""
        self._consumed.add((handle, conversation_id, message_id))

    def drop_consumed(self, handle: str, conversation_id: str, message_id: str) -> None:
        """Drop the consumed journal entry for ``message_id`` if present."""
        self._consumed.discard((handle, conversation_id, message_id))

    async def aget(self, handle: str, conversation_id: str) -> str | None:
        """Async twin of :meth:`get` (direct, non-blocking delegation)."""
        return self.get(handle, conversation_id)

    async def aput(self, handle: str, conversation_id: str, thread_id: str) -> None:
        """Async twin of :meth:`put` (direct, non-blocking delegation)."""
        self.put(handle, conversation_id, thread_id)

    async def adelete(self, handle: str, conversation_id: str) -> None:
        """Async twin of :meth:`delete` (direct, non-blocking delegation)."""
        self.delete(handle, conversation_id)

    async def ais_consumed(self, handle: str, conversation_id: str, message_id: str) -> bool:
        """Async twin of :meth:`is_consumed` (direct, non-blocking delegation)."""
        return self.is_consumed(handle, conversation_id, message_id)

    async def amark_consumed(self, handle: str, conversation_id: str, message_id: str) -> None:
        """Async twin of :meth:`mark_consumed` (direct, non-blocking delegation)."""
        self.mark_consumed(handle, conversation_id, message_id)

    async def adrop_consumed(self, handle: str, conversation_id: str, message_id: str) -> None:
        """Async twin of :meth:`drop_consumed` (direct, non-blocking delegation)."""
        self.drop_consumed(handle, conversation_id, message_id)
