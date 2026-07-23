"""Advisory single-consumer lease for the Tier-3 poll driver.

v0.2 *documented* "run exactly one driver per inbox (poll XOR webhook)" but did not enforce it:
two ``PollDriver``s on one inbox (an accidental 2nd process, or a restarted-not-dead old one)
both fetch the same ``status="new"`` mail and double-resume. This module
turns that rule into an **enforced advisory lease**.

- :class:`Lease` — the protocol the driver depends on (``acquire``/``release``).
- :class:`SqliteLease` — a ``consumer_lease(inbox_key, owner_id, expires_at)`` row in the SAME
  sqlite db file as the thread-map (a different table). ``acquire`` is a ``BEGIN IMMEDIATE``
  read-modify-write: it grants the lease iff no live *other* owner holds it (acquire-or-RENEW for
  this owner; ttl-expiry takeover otherwise). Two drivers coordinate only when they point a
  ``SqliteLease`` at the same db file + ``inbox_key`` (typically the rine agent handle they poll).

Side-effect-free at import: the connection opens lazily on first :meth:`SqliteLease.acquire`.
``time.time()`` + ``uuid`` are runtime-legal (this is library code, not a workflow script).
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from typing import Protocol, runtime_checkable

__all__ = ["Lease", "SqliteLease"]

_SQL_CREATE_LEASE = (
    "CREATE TABLE IF NOT EXISTS consumer_lease ("
    "inbox_key TEXT PRIMARY KEY, owner_id TEXT NOT NULL, expires_at REAL NOT NULL)"
)
_SQL_READ_LEASE = "SELECT owner_id, expires_at FROM consumer_lease WHERE inbox_key = ?"
_SQL_UPSERT_LEASE = (
    "INSERT INTO consumer_lease (inbox_key, owner_id, expires_at) VALUES (?, ?, ?) "
    "ON CONFLICT (inbox_key) DO UPDATE SET owner_id = excluded.owner_id, "
    "expires_at = excluded.expires_at"
)
_SQL_RELEASE_LEASE = "DELETE FROM consumer_lease WHERE inbox_key = ? AND owner_id = ?"


@runtime_checkable
class Lease(Protocol):
    """An advisory single-consumer lease: one live owner at a time per inbox."""

    def acquire(self, ttl_seconds: float) -> bool:
        """Grant-or-renew the lease for ``ttl_seconds``; False iff another live owner holds it."""
        ...

    def release(self) -> None:
        """Release the lease if this owner still holds it (a no-op otherwise)."""
        ...


class SqliteLease:
    """A sqlite-backed advisory :class:`Lease`, co-located with the thread-map db.

    One ``consumer_lease`` row per ``inbox_key``. ``owner_id`` defaults to a per-instance
    ``uuid4().hex`` so two ``SqliteLease`` objects (typically in different processes) are distinct
    owners. ``acquire`` grants the lease when there is no row, an expired row, or a row this owner
    already holds (renew); it refuses only when a *live other* owner holds it. The read-then-write
    runs inside ``BEGIN IMMEDIATE`` so two concurrent ``acquire`` calls cannot both win.
    """

    def __init__(self, db_path: str, inbox_key: str, *, owner_id: str | None = None) -> None:
        """Bind the db file + ``inbox_key``; ``owner_id`` defaults to a per-instance uuid."""
        self._db_path = db_path
        self._inbox_key = inbox_key
        self._owner = owner_id or uuid.uuid4().hex
        self._conn: sqlite3.Connection | None = None

    @property
    def owner_id(self) -> str:
        """The id this lease acquires/renews under (per-instance uuid unless overridden)."""
        return self._owner

    def _connection(self) -> sqlite3.Connection:
        """Open (once) the sqlite connection and ensure the ``consumer_lease`` table exists.

        A short ``busy_timeout`` (200 ms) bounds how long a contended ``BEGIN IMMEDIATE`` waits for
        the other acquirer's microsecond-scale transaction to commit: it almost always acquires on
        the retry, and yields a clean ``False`` quickly (not a 5 s stall) when it does not, instead
        of inheriting stdlib's 5 s default → ``database is locked`` raise.
        """
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=200")
            self._conn.execute(_SQL_CREATE_LEASE)
            self._conn.commit()
        return self._conn

    def acquire(self, ttl_seconds: float) -> bool:
        """Grant-or-renew the lease; return False iff another *live* owner holds it.

        A ``BEGIN IMMEDIATE`` read-modify-write: read the row, then with ``now = time.time()``
        upsert ``(self._owner, now + ttl)`` and return True when there is no row, the row is
        expired (``expires_at <= now``), or the row is already this owner's (renew); otherwise a
        live other owner holds it → roll back, return False.

        A genuine acquire RACE (another live owner is mid-transaction when this ``BEGIN IMMEDIATE``
        runs) surfaces as ``sqlite3.OperationalError: database is locked`` (after the short
        ``busy_timeout``). That IS contention, so it degrades to ``False`` (contention →
        return []), never a propagated raw exception out of the documented ``poll_once`` entry.
        """
        conn = self._connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(_SQL_READ_LEASE, (self._inbox_key,)).fetchone()
            now = time.time()
            held_by_other = row is not None and row[0] != self._owner and float(row[1]) > now
            if held_by_other:
                conn.rollback()
                return False
            conn.execute(_SQL_UPSERT_LEASE, (self._inbox_key, self._owner, now + ttl_seconds))
            conn.commit()
            return True
        except sqlite3.OperationalError:
            # A locked db at BEGIN IMMEDIATE means a live other owner is mid-acquire → contention.
            conn.rollback()
            return False
        except Exception:
            conn.rollback()
            raise

    def release(self) -> None:
        """Delete this owner's lease row (a different owner's row is left untouched)."""
        conn = self._connection()
        conn.execute(_SQL_RELEASE_LEASE, (self._inbox_key, self._owner))
        conn.commit()

    def close(self) -> None:
        """Close the underlying sqlite connection if it was opened."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
