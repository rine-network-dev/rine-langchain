"""Behavioral tests for the advisory single-consumer lease (``_lease.py``).

Exercised through the :class:`Lease` public surface (``acquire`` / ``release``). Two
:class:`SqliteLease` owners on the SAME db file + ``inbox_key`` model two ``PollDriver`` processes
racing for one inbox: only one live owner may hold the lease at a time, an expired lease can be
taken over, and a release frees it for the next owner.

Expiry is exercised **deterministically** with a ``ttl=0`` lease (it expires the instant it is
granted, since ``expires_at = now + 0`` and a strictly-later ``time.time()`` makes
``expires_at <= now`` true) — no real-time sleep that could flake; the only sleep is a sub-ms nudge
to guarantee the wall clock advances past the grant instant.
"""

from __future__ import annotations

import time

from langchain_rine._lease import Lease, SqliteLease

INBOX = "agent@acme.rine.network"
OTHER_INBOX = "other@acme.rine.network"


def _db(tmp_path) -> str:
    return str(tmp_path / "lease.sqlite")


class TestSqliteLease:
    def test_satisfies_lease_protocol(self, tmp_path):
        assert isinstance(SqliteLease(_db(tmp_path), INBOX), Lease)

    def test_single_owner_acquires(self, tmp_path):
        lease = SqliteLease(_db(tmp_path), INBOX)
        assert lease.acquire(30.0) is True

    def test_second_owner_is_refused_while_first_lives(self, tmp_path):
        """Two owners on the same db+key → first wins, second backs off (False)."""
        db = _db(tmp_path)
        first = SqliteLease(db, INBOX)
        second = SqliteLease(db, INBOX)

        assert first.acquire(30.0) is True
        assert second.acquire(30.0) is False  # a live other owner holds it

    def test_same_owner_renews(self, tmp_path):
        """An owner re-acquiring its own live lease renews it (True), never blocks itself."""
        lease = SqliteLease(_db(tmp_path), INBOX)
        assert lease.acquire(30.0) is True
        assert lease.acquire(30.0) is True  # renew, not refuse

    def test_distinct_inboxes_do_not_contend(self, tmp_path):
        """The lease is per ``inbox_key``: two owners on DIFFERENT inboxes both acquire."""
        db = _db(tmp_path)
        a = SqliteLease(db, INBOX)
        b = SqliteLease(db, OTHER_INBOX)
        assert a.acquire(30.0) is True
        assert b.acquire(30.0) is True  # different key → no contention

    def test_expired_lease_is_taken_over(self, tmp_path):
        """Once the holder's ttl elapses, another owner may take the lease over.

        Deterministic: the first lease is granted with ttl=0 (expires immediately); a sub-ms sleep
        guarantees ``time.time()`` advances past the grant instant so the takeover sees it expired.
        """
        db = _db(tmp_path)
        holder = SqliteLease(db, INBOX)
        taker = SqliteLease(db, INBOX)

        assert holder.acquire(0.0) is True  # granted, but already expired
        time.sleep(0.005)  # nudge the wall clock strictly past expires_at
        assert taker.acquire(30.0) is True  # the expired lease is taken over

    def test_release_frees_the_lease_for_another_owner(self, tmp_path):
        """A clean release lets the next owner acquire immediately (no ttl wait)."""
        db = _db(tmp_path)
        first = SqliteLease(db, INBOX)
        second = SqliteLease(db, INBOX)

        assert first.acquire(30.0) is True
        assert second.acquire(30.0) is False  # blocked while first holds it
        first.release()
        assert second.acquire(30.0) is True  # freed → second now wins

    def test_release_by_non_holder_does_not_steal(self, tmp_path):
        """A release only deletes THIS owner's row — it cannot evict a different live owner."""
        db = _db(tmp_path)
        holder = SqliteLease(db, INBOX)
        other = SqliteLease(db, INBOX)

        assert holder.acquire(30.0) is True
        other.release()  # other never held it → no-op, must not free the holder's lease
        assert other.acquire(30.0) is False  # holder still holds it

    def test_owner_id_defaults_to_per_instance_distinct(self, tmp_path):
        """Two default-constructed leases are distinct owners (per-instance uuid)."""
        db = _db(tmp_path)
        a = SqliteLease(db, INBOX)
        b = SqliteLease(db, INBOX)
        assert a.owner_id != b.owner_id

    def test_explicit_owner_id_renews_across_instances(self, tmp_path):
        """A restarted poller reusing the same owner_id renews (not blocks) its own lease.

        Models the "same logical consumer, new process object" case: passing the prior owner_id
        lets the new instance treat the existing live lease as its own and renew it.
        """
        db = _db(tmp_path)
        first = SqliteLease(db, INBOX, owner_id="poller-1")
        restarted = SqliteLease(db, INBOX, owner_id="poller-1")
        assert first.acquire(30.0) is True
        assert restarted.acquire(30.0) is True  # same owner_id → renew, not refuse

    def test_close_then_reacquire_does_not_strand_lease(self, tmp_path):
        """Closing the sqlite connection and re-acquiring (same owner) still renews cleanly."""
        db = _db(tmp_path)
        lease = SqliteLease(db, INBOX, owner_id="poller-1")
        assert lease.acquire(30.0) is True
        lease.close()
        assert lease.acquire(30.0) is True  # reopens the connection lazily and renews

    def test_acquire_race_against_held_write_lock_returns_false_not_raises(self, tmp_path):
        """A true acquire RACE degrades to ``False`` quickly, never raises.

        Models owner-B calling ``acquire`` while owner-A is mid-transaction (between its own
        ``BEGIN IMMEDIATE`` and ``commit``): B's ``BEGIN IMMEDIATE`` cannot grab the write lock.
        With the short ``busy_timeout`` it must return ``False`` (contention → return [])
        within ~the timeout, NOT raise ``database is locked`` after a 5 s default stall.
        """
        import sqlite3
        import time as _time

        db = _db(tmp_path)
        # owner-A holds an OPEN write transaction (a real reserved/write lock on the db file).
        blocker = sqlite3.connect(db)
        blocker.execute("PRAGMA journal_mode=WAL")
        blocker.execute(
            "CREATE TABLE IF NOT EXISTS consumer_lease ("
            "inbox_key TEXT PRIMARY KEY, owner_id TEXT NOT NULL, expires_at REAL NOT NULL)"
        )
        blocker.commit()
        blocker.execute("BEGIN IMMEDIATE")  # take + HOLD the write lock
        blocker.execute(
            "INSERT OR REPLACE INTO consumer_lease VALUES (?, ?, ?)", (INBOX, "owner-A", 0.0)
        )
        try:
            contender = SqliteLease(db, INBOX, owner_id="owner-B")
            started = _time.monotonic()
            result = contender.acquire(30.0)  # must NOT raise, must return promptly
            elapsed = _time.monotonic() - started
        finally:
            blocker.rollback()
            blocker.close()

        assert result is False  # contention surfaced as a clean False, not an exception
        assert elapsed < 2.0  # bounded by busy_timeout (~200 ms), not the 5 s stdlib default
