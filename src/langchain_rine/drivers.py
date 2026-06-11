"""Drivers that feed inbound rine messages to a :class:`RineThreadResumer` (D15/D22).

The resumer (``inbound.py``) is transport-agnostic — it only knows graphs + a thread-map. The
**driver** owns the rine client and the ingress shape:

- :class:`PollDriver` — the **launch default** (D8/D15): loops ``client.inbox(status="new")``,
  hands each message to the resumer, and **acks the messages whose result is
  :attr:`~langchain_rine._resume.ResumeResult.should_ack`** (``RESUMED`` +
  ``SKIPPED_ALREADY_CONSUMED`` — D20; every other skip leaves the message in the inbox for a later
  poll, D16). After a *successful* ack it calls ``resumer.confirm_delivered`` to self-clean the
  consumed journal (D20). An optional :class:`~langchain_rine._lease.Lease` makes it the **single
  poller** for an inbox (D22). Needs no public URL, runs anywhere.
- :func:`make_webhook_handler` — the **webhook path is a handler hook, NOT an ingress server**
  (D15/D19): the package ships no ports / no public URL. The user's route verifies the rine
  outbound-webhook signature, then passes the **message id**; the handler does the authenticated
  + E2EE ``client.read(message_id)`` fetch (never trusting a caller-supplied body), resumes, and
  acks + confirms exactly like one poll step.

The rine client is built lazily (``build_client`` / ``abuild_client`` from ``_client.py``,
threading the same ``config_dir`` / ``api_url`` / ``agent`` overrides as ``BaseRineTool``) so the
import stays side-effect-free (I-1).

**Exactly-once (D20):** the resumer records each resumed message in a durable consumed journal
*before* the ack is attempted, so an ack failure no longer double-resumes — a redelivery is
recognized as ``SKIPPED_ALREADY_CONSUMED`` and re-acked. Run **one PollDriver per inbox** (the
optional lease enforces it); webhook-vs-poll coordination stays documented, not enforced (D22).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from langchain_rine._client import abuild_client, build_client

if TYPE_CHECKING:
    from collections.abc import Callable

    from rine import DecryptedMessage, RineClient, SyncRineClient

    from langchain_rine._lease import Lease
    from langchain_rine._resume import ResumeResult
    from langchain_rine.inbound import RineThreadResumer

logger = logging.getLogger("langchain_rine.drivers")

__all__ = ["PollDriver", "make_webhook_handler"]


def _ackable(
    messages: list[DecryptedMessage], results: list[ResumeResult]
) -> list[DecryptedMessage]:
    """Return the messages whose result is :attr:`ResumeResult.should_ack` (D20 ack policy)."""
    return [m for m, r in zip(messages, results, strict=True) if r.should_ack]


class PollDriver:
    """Poll ``client.inbox(status="new")`` and resume matching paused graphs (D15/D22).

    Owns the rine client (lazy, side-effect-free import). ``poll_once`` / ``apoll_once`` do one
    fetch-resume-ack-confirm sweep and return the per-message :class:`ResumeResult`s — tests drive
    those directly (no infinite loop). ``run`` / ``arun`` wrap them in a thin sleep loop bounded by
    an optional ``max_iterations`` and release the lease in a ``finally``. With a
    :class:`~langchain_rine._lease.Lease`, a sweep that loses the lease to another live owner
    fetches nothing and returns ``[]`` (D22).
    """

    def __init__(
        self,
        resumer: RineThreadResumer,
        *,
        config_dir: str | None = None,
        api_url: str | None = None,
        agent: str | None = None,
        limit: int = 20,
        lease: Lease | None = None,
        lease_ttl: float = 30.0,
    ) -> None:
        """Bind the resumer + the SDK overrides; the client is built lazily on first poll.

        ``lease`` (D22) makes this the single poller for an inbox: each sweep acquires-or-renews it
        and backs off (fetches nothing) when another live owner holds it. ``lease_ttl`` must exceed
        **the poll ``interval`` AND the worst-case sweep/resume duration**: the lease gates sweep
        entry only (not the in-sweep fetch/resume/ack work), and its expiry is fixed at acquire
        time, so a single sweep that out-runs the ttl forfeits single-consumer protection (a second
        poller may take the expired lease over mid-sweep). Set the ttl comfortably above your
        slowest resumed node.
        """
        self._resumer = resumer
        self._config_dir = config_dir
        self._api_url = api_url
        self._agent = agent
        self._limit = limit
        self._lease = lease
        self._lease_ttl = lease_ttl
        self._client: SyncRineClient | None = None
        self._aclient: RineClient | None = None

    def _get_client(self) -> SyncRineClient:
        """Build (once) the sync rine client from the threaded overrides."""
        if self._client is None:
            self._client = build_client(
                config_dir=self._config_dir, api_url=self._api_url, agent=self._agent
            )
        return self._client

    def _aget_client(self) -> RineClient:
        """Build (once) the async rine client from the threaded overrides."""
        if self._aclient is None:
            self._aclient = abuild_client(
                config_dir=self._config_dir, api_url=self._api_url, agent=self._agent
            )
        return self._aclient

    def poll_once(self) -> list[ResumeResult]:
        """One sync sweep: acquire the lease, fetch, resume, ack the should_ack ids, confirm.

        If a lease is set and another live owner holds it, the sweep fetches nothing and returns
        ``[]`` (D22). After a *successful* ack each acked message's journal entry is dropped via
        ``confirm_delivered`` (D20 self-clean).
        """
        if self._lease is not None and not self._lease.acquire(self._lease_ttl):
            logger.info("PollDriver: another consumer holds the inbox lease; backing off")
            return []
        client = self._get_client()
        page = client.inbox(status="new", limit=self._limit, agent=self._agent)
        messages: list[DecryptedMessage] = list(page)  # CursorPage is sync-iterable
        results = [self._resumer.handle_inbound(m) for m in messages]
        acked = self._ack(_ackable(messages, results))
        for m in acked:
            self._resumer.confirm_delivered(m)
        return results

    async def apoll_once(self) -> list[ResumeResult]:
        """Async twin of :meth:`poll_once` (``ahandle_inbound`` + awaited ack + ``aconfirm``)."""
        if self._lease is not None and not await asyncio.to_thread(
            self._lease.acquire, self._lease_ttl
        ):
            logger.info("PollDriver: another consumer holds the inbox lease; backing off")
            return []
        client = self._aget_client()
        page = await client.inbox(status="new", limit=self._limit, agent=self._agent)
        messages: list[DecryptedMessage] = list(page)  # CursorPage stays sync-iterable
        results = [await self._resumer.ahandle_inbound(m) for m in messages]
        acked = await self._aack(_ackable(messages, results))
        for m in acked:
            await self._resumer.aconfirm_delivered(m)
        return results

    def _ack(self, to_ack: list[DecryptedMessage]) -> list[DecryptedMessage]:
        """Best-effort mark-delivered; return the acked messages (empty on failure or nothing).

        A successful ack returns ``to_ack`` so the caller can ``confirm_delivered`` them; a failure
        is logged and returns ``[]`` (the journal entry is kept so a redelivery is recognized).
        """
        if not to_ack:
            return []
        try:
            self._get_client().mark_delivered([str(m.id) for m in to_ack], agent=self._agent)
        except Exception:  # noqa: BLE001 - ack is best-effort; a failure must not lose the resume
            logger.warning(
                "PollDriver: mark_delivered failed for %d message(s); they may reappear",
                len(to_ack),
                exc_info=True,
            )
            return []
        return to_ack

    async def _aack(self, to_ack: list[DecryptedMessage]) -> list[DecryptedMessage]:
        """Async twin of :meth:`_ack` (awaited ``mark_delivered``; same best-effort contract)."""
        if not to_ack:
            return []
        try:
            await self._aget_client().mark_delivered(
                [str(m.id) for m in to_ack], agent=self._agent
            )
        except Exception:  # noqa: BLE001 - best-effort, mirrors the sync path
            logger.warning(
                "PollDriver: async mark_delivered failed for %d message(s); may reappear",
                len(to_ack),
                exc_info=True,
            )
            return []
        return to_ack

    def run(self, *, interval: float = 2.0, max_iterations: int | None = None) -> None:
        """Poll forever (or ``max_iterations`` times), sleeping ``interval`` s between sweeps.

        A sweep is wrapped (``I-3``): a transient ``client.inbox`` failure logs a warning and the
        loop continues (the ``interval`` sleep is the backoff). The lease (if any) is released in a
        ``finally`` so a clean shutdown frees the inbox for another poller immediately (D22).
        """
        count = 0
        try:
            while max_iterations is None or count < max_iterations:
                try:
                    self.poll_once()
                except Exception:  # noqa: BLE001 - a transient sweep failure must not kill the loop
                    logger.warning(
                        "PollDriver: poll sweep failed; retrying next interval", exc_info=True
                    )
                count += 1
                if max_iterations is None or count < max_iterations:
                    time.sleep(interval)
        finally:
            if self._lease is not None:
                self._lease.release()

    async def arun(self, *, interval: float = 2.0, max_iterations: int | None = None) -> None:
        """Async twin of :meth:`run` (``apoll_once`` + ``asyncio.sleep``); same guards/release."""
        count = 0
        try:
            while max_iterations is None or count < max_iterations:
                try:
                    await self.apoll_once()
                except Exception:  # noqa: BLE001 - a transient sweep failure must not kill the loop
                    logger.warning(
                        "PollDriver: async poll sweep failed; retrying next interval",
                        exc_info=True,
                    )
                count += 1
                if max_iterations is None or count < max_iterations:
                    await asyncio.sleep(interval)
        finally:
            if self._lease is not None:
                await asyncio.to_thread(self._lease.release)


def make_webhook_handler(
    resumer: RineThreadResumer,
    *,
    config_dir: str | None = None,
    api_url: str | None = None,
    agent: str | None = None,
) -> Callable[[str], ResumeResult]:
    """Build a one-**message-id** handler the user mounts on their OWN web route (D15/D19).

    The package ships **no** ingress server: it returns a plain ``Callable[[str], ResumeResult]``.
    Your route MUST first **verify the rine outbound-webhook signature** (HMAC), then pass the
    ``message_id`` — the handler does the authenticated + E2EE ``client.read(message_id)`` fetch
    and **never trusts a caller-supplied body** (a forged ``message.received`` POST cannot inject
    content into a paused graph, ``wh-1``). It then resumes the matching paused graph and acks +
    confirms the message when the result is :attr:`ResumeResult.should_ack` (mirroring one
    :meth:`PollDriver.poll_once` step). The rine client is built lazily from the same overrides.
    """
    client_box: dict[str, SyncRineClient] = {}

    def _client() -> SyncRineClient:
        if "client" not in client_box:
            client_box["client"] = build_client(
                config_dir=config_dir, api_url=api_url, agent=agent
            )
        return client_box["client"]

    def handle(message_id: str) -> ResumeResult:
        client = _client()
        message = client.read(
            message_id, agent=agent
        )  # authenticated + E2EE; never trust the POST
        result = resumer.handle_inbound(message)
        if result.should_ack:
            try:
                client.mark_delivered([message_id], agent=agent)
            except Exception:  # noqa: BLE001 - best-effort ack, never lose the resume
                logger.warning(
                    "webhook handler: mark_delivered failed for %s; may reappear",
                    message_id,
                    exc_info=True,
                )
            else:
                resumer.confirm_delivered(message)  # self-clean only after a successful ack
        return result

    return handle
