"""Tier-3 idle-wake resumer: wake a paused LangGraph thread on an inbound rine message.

The :class:`RineThreadResumer` is the v0.2 headline (SPEC §11 Tier-3, D14-D17), hardened in
v0.3 to **exactly-once** resume (D20). It keeps a ``(handle, conversation_id) -> thread_id``
:class:`~langchain_rine._threadmap.ThreadMapStore` (+ a durable consumed-message-id journal)
and, given an inbound decrypted rine message, resumes the matching paused graph by streaming
``Command(resume=<HumanResponse>)`` — speaking the Agent-Inbox ``HumanInterrupt`` /
``HumanResponse`` wire (PHASE0_v2_CONTRACT §7). ``ResumeStatus`` / ``ResumeResult`` /
``default_key_fn`` and the verification-gate + shared-guard helpers live in the sibling
``_resume.py`` (re-exported here so the public import surface stays stable).

**LangGraph is an OPTIONAL extra (D17).** This module imports langgraph at module level and is
**not** re-exported from ``langchain_rine/__init__.py``, so ``import langchain_rine`` (the 12
tools) never needs langgraph. Importing ``langchain_rine.inbound`` IS the opt-in; install it with
``pip install langchain-rine[inbound]``.

Invariants: **I-1** — import builds no client, no network, no creds (the resumer holds no rine
client; the ``drivers.py`` layer owns the rine I/O). **I-2** — a message is resumed with its
**decrypted plaintext** only; a ``decrypt_error`` message is never resumed, and the consumed
journal stores only ``message.id`` (never payload). **I-3** — guards fail readably (logged + a
typed :class:`ResumeStatus`), never a raw traceback — including a user node that raises on resume
(``ResumeStatus.FAILED``; the message is not acked, so the sweep continues).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

try:
    # `Command` is used at runtime (Command(resume=...)); importing it here is what makes a
    # missing langgraph fail fast with the friendly [inbound]-extra hint below.
    from langgraph.types import Command
except ModuleNotFoundError as exc:  # pragma: no cover - exercised via the [inbound] extra hint
    raise ModuleNotFoundError(
        "langchain_rine.inbound requires langgraph; install it with "
        "`pip install langchain-rine[inbound]`."
    ) from exc

from langchain_rine._resume import (
    ResumeResult,
    ResumeStatus,
    build_human_response,
    default_key_fn,
    pre_state_guard,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from rine import DecryptedMessage

    from langchain_rine._threadmap import ThreadMapStore

logger = logging.getLogger("langchain_rine.inbound")

# Re-exported so `from langchain_rine.inbound import RineThreadResumer, ResumeStatus,
# ResumeResult, default_key_fn` stays stable after the _resume.py split.
__all__ = ["ResumeStatus", "ResumeResult", "RineThreadResumer", "default_key_fn"]


class RineThreadResumer:
    """Resume a paused LangGraph thread when its conversation gets an inbound rine reply.

    Wraps an **already-compiled** graph (the caller binds the checkpointer — D17; the resumer does
    not own the saver lifecycle) and a :class:`~langchain_rine._threadmap.ThreadMapStore`. Holds no
    rine client: the ``drivers.py`` poll/webhook layer fetches messages and calls
    :meth:`handle_inbound` / :meth:`ahandle_inbound`, then :meth:`confirm_delivered` after a
    successful ack so the consumed journal self-cleans (D20).
    """

    def __init__(
        self,
        graph: Any,
        store: ThreadMapStore,
        *,
        key_fn: Callable[[DecryptedMessage], tuple[str, str]] | None = None,
        require_verified: bool = False,
    ) -> None:
        """Bind the compiled graph + store; ``key_fn`` defaults to :func:`default_key_fn`.

        ``require_verified`` (D23) makes the verification gate also block ``unverifiable`` senders
        (→ ``SKIPPED_UNVERIFIED``) instead of annotating-and-resuming; default False keeps v0.2.
        """
        self._graph = graph
        self._store = store
        self._key_fn = key_fn or default_key_fn
        self._require_verified = require_verified

    def register(self, handle: str, conversation_id: str, thread_id: str) -> None:
        """Bind ``(handle, conversation_id)`` to ``thread_id`` (newest-wins; D16).

        If a *different* thread_id is already mapped for this key, the previous binding is
        superseded and a WARNING is logged (the common cause is a legitimate restart / re-arm of
        the same conversation; rejecting would strand the new thread).
        """
        existing = self._store.get(handle, conversation_id)
        if existing is not None and existing != thread_id:
            logger.warning(
                "re-registering (%s, %s); previous thread %s superseded by %s",
                handle,
                conversation_id,
                existing,
                thread_id,
            )
        self._store.put(handle, conversation_id, thread_id)

    def unregister(self, handle: str, conversation_id: str) -> None:
        """Drop the ``(handle, conversation_id) -> thread_id`` binding if present."""
        self._store.delete(handle, conversation_id)

    def confirm_delivered(self, message: DecryptedMessage) -> None:
        """Self-clean the consumed-journal entry after a successful ack (D20).

        A successfully-acked message has left the server inbox and can never be redelivered, so its
        journal entry is dead — drop it (keeps the journal bounded to in-flight ids). The driver
        calls this for every message it acked.
        """
        handle, conversation_id = self._key_fn(message)
        self._store.drop_consumed(handle, conversation_id, str(message.id))

    async def aconfirm_delivered(self, message: DecryptedMessage) -> None:
        """Async twin of :meth:`confirm_delivered` (uses ``adrop_consumed``)."""
        handle, conversation_id = self._key_fn(message)
        await self._store.adrop_consumed(handle, conversation_id, str(message.id))

    # --- inbound handling ------------------------------------------------------------------

    def _resolve(self, message: DecryptedMessage) -> tuple[ResumeResult | None, str, str, str]:
        """Run the no-resume guards (sync); return ``(early, thread_id, handle, conversation_id)``.

        Shares the no-I/O guard chain (decrypt-error / signature / no-conversation) with the async
        path via :func:`pre_state_guard`, then adds the sync store lookups: the already-consumed
        guard (D20, BEFORE the no-thread check so a redelivered consumed message is acked even
        after its thread terminally unregistered), then the thread lookup.
        """
        early, handle, conversation_id = pre_state_guard(
            message, self._key_fn, require_verified=self._require_verified
        )
        if early is not None:
            self._log_pre_guard(early, message)
            return early, "", handle, conversation_id
        if self._store.is_consumed(handle, conversation_id, str(message.id)):
            logger.info("skip resume: message %s already consumed (exactly-once)", message.id)
            return ResumeResult(ResumeStatus.SKIPPED_ALREADY_CONSUMED), "", handle, conversation_id
        thread_id = self._store.get(handle, conversation_id)
        if thread_id is None:
            self._log_no_thread(handle, conversation_id)
            return ResumeResult(ResumeStatus.SKIPPED_NO_THREAD), "", handle, conversation_id
        return None, thread_id, handle, conversation_id

    async def _aresolve(
        self, message: DecryptedMessage
    ) -> tuple[ResumeResult | None, str, str, str]:
        """Async twin of :meth:`_resolve` (awaits ``ais_consumed`` / ``aget`` — D21)."""
        early, handle, conversation_id = pre_state_guard(
            message, self._key_fn, require_verified=self._require_verified
        )
        if early is not None:
            self._log_pre_guard(early, message)
            return early, "", handle, conversation_id
        if await self._store.ais_consumed(handle, conversation_id, str(message.id)):
            logger.info("skip resume: message %s already consumed (exactly-once)", message.id)
            return ResumeResult(ResumeStatus.SKIPPED_ALREADY_CONSUMED), "", handle, conversation_id
        thread_id = await self._store.aget(handle, conversation_id)
        if thread_id is None:
            self._log_no_thread(handle, conversation_id)
            return ResumeResult(ResumeStatus.SKIPPED_NO_THREAD), "", handle, conversation_id
        return None, thread_id, handle, conversation_id

    def _log_pre_guard(self, early: ResumeResult, message: DecryptedMessage) -> None:
        """Log the reason a no-I/O guard fired (shared by the sync + async resolve)."""
        if early.status is ResumeStatus.SKIPPED_DECRYPT_ERROR:
            logger.info("skip resume: message %s has decrypt_error (I-2)", message.id)
        elif early.status is ResumeStatus.SKIPPED_UNVERIFIED:
            logger.warning(
                "skip resume: message %s sig=%s blocks the resume (wh-2/D23); graph not woken",
                message.id,
                message.verification_status,
            )
        else:  # SKIPPED_NO_CONVERSATION
            logger.info("skip resume: message %s has no conversation_id", message.id)

    def _log_no_thread(self, handle: str, conversation_id: str) -> None:
        """Log the no-thread skip (shared by the sync + async resolve)."""
        logger.info(
            "skip resume: no thread mapped for (%s, %s); leaving in inbox", handle, conversation_id
        )

    def _check_state(self, state: Any, thread_id: str) -> ResumeResult | None:
        """Guard on the thread's graph state; return a SKIP result or None if resumable."""
        if not state.next:
            logger.info("skip resume: thread %s is not interrupted", thread_id)
            return ResumeResult(ResumeStatus.SKIPPED_NOT_INTERRUPTED, thread_id=thread_id)
        if len(state.interrupts) > 1:
            logger.warning(
                "skip resume: thread %s has %d pending interrupts; v0.2 resumes "
                "one-per-superstep only (langgraph#6533)",
                thread_id,
                len(state.interrupts),
            )
            return ResumeResult(ResumeStatus.SKIPPED_MULTIPLE_INTERRUPTS, thread_id=thread_id)
        return None

    @staticmethod
    def _is_terminal(post_state: Any) -> bool:
        """True iff a resumed run reached a terminal state (``not next`` — A/D18).

        A terminal run means the conversation is done: its binding would otherwise linger and
        shadow/strand a later message (``tm-1``/``resumes-inv-1/2``). A graph that **re-armed** at
        a new interrupt (``next`` truthy) is NOT terminal — it keeps its binding and resumes again.
        Shared by the sync :meth:`_unregister_if_terminal` and the async unregister so the only
        sync/async difference is ``delete`` vs ``await adelete`` (no duplicated predicate).
        """
        return not post_state.next

    def _unregister_if_terminal(self, post_state: Any, handle: str, conversation_id: str) -> None:
        """Drop the binding (sync ``delete``) when a resumed run reached a terminal state (A/D18).

        (The exactly-once guard now covers the re-arming + ack-failure redelivery case — D20.)
        """
        if self._is_terminal(post_state):
            self._store.delete(handle, conversation_id)
            logger.info("unregistered terminal binding (%s, %s)", handle, conversation_id)

    async def _aunregister_if_terminal(
        self, post_state: Any, handle: str, conversation_id: str
    ) -> None:
        """Async twin of :meth:`_unregister_if_terminal` — awaits ``adelete`` (D21).

        Keeps the async resume path fully non-blocking: the terminal-unregister no longer runs a
        synchronous stdlib-sqlite ``delete``+``commit`` on the event loop (which re-opened ``tm-4``
        on the dominant single-interrupt-to-completion case); it awaits the aiosqlite twin instead.
        """
        if self._is_terminal(post_state):
            await self._store.adelete(handle, conversation_id)
            logger.info("unregistered terminal binding (%s, %s)", handle, conversation_id)

    def _run_resume(
        self,
        thread_id: str,
        resume_value: Any,
        handle: str,
        conversation_id: str,
        message_id: str | None,
    ) -> ResumeResult:
        """Sync resume body shared by :meth:`handle_inbound` + :meth:`expire`.

        Guards on graph state, invokes ``Command(resume=resume_value)``, records the consumed id
        (when ``message_id`` is given — act-then-record, BEFORE the driver acks), then
        terminal-unregisters. Wrapped: a raising node OR a ``mark_consumed`` store error ->
        ``FAILED`` (logged, not acked; degrades to at-least-once, never a crash — I-3).
        """
        config = {"configurable": {"thread_id": thread_id}}
        try:
            skip = self._check_state(self._graph.get_state(config), thread_id)
            if skip is not None:
                return skip
            output = self._graph.invoke(Command(resume=resume_value), config)
            if message_id is not None:
                self._store.mark_consumed(handle, conversation_id, message_id)
            self._unregister_if_terminal(self._graph.get_state(config), handle, conversation_id)
        except Exception:
            logger.exception("resume of thread %s failed (message %s)", thread_id, message_id)
            return ResumeResult(ResumeStatus.FAILED, thread_id=thread_id)
        logger.info("resumed thread %s (message %s)", thread_id, message_id)
        return ResumeResult(ResumeStatus.RESUMED, thread_id=thread_id, output=output)

    def handle_inbound(self, message: DecryptedMessage) -> ResumeResult:
        """Wake the mapped paused thread with ``message``'s plaintext (sync path).

        Guard order (D20): decrypt-error -> invalid-sig (+require_verified) -> no-conversation ->
        **already-consumed** -> no-thread -> not-interrupted -> multiple-interrupts (#6533) ->
        resume -> **mark_consumed** -> terminal-unregister. ``mark_consumed`` records **after** the
        resume commits and **before** the driver's ack (act-then-record), so the ack-failure
        double-resume window is closed (D20). The resume runs arbitrary user code; see
        :meth:`_run_resume` for the failure wrapping.
        """
        early, thread_id, handle, conversation_id = self._resolve(message)
        if early is not None:
            return early
        return self._run_resume(
            thread_id, build_human_response(message), handle, conversation_id, str(message.id)
        )

    async def ahandle_inbound(self, message: DecryptedMessage) -> ResumeResult:
        """Async twin of :meth:`handle_inbound` (``_aresolve`` + ``aget_state`` / ``ainvoke``)."""
        early, thread_id, handle, conversation_id = await self._aresolve(message)
        if early is not None:
            return early
        config = {"configurable": {"thread_id": thread_id}}
        try:
            skip = self._check_state(await self._graph.aget_state(config), thread_id)
            if skip is not None:
                return skip
            output = await self._graph.ainvoke(
                Command(resume=build_human_response(message)), config
            )
            await self._store.amark_consumed(handle, conversation_id, str(message.id))
            await self._aunregister_if_terminal(
                await self._graph.aget_state(config), handle, conversation_id
            )
        except Exception:
            logger.exception("resume of thread %s failed for message %s", thread_id, message.id)
            return ResumeResult(ResumeStatus.FAILED, thread_id=thread_id)
        logger.info("resumed thread %s from message %s", thread_id, message.id)
        return ResumeResult(ResumeStatus.RESUMED, thread_id=thread_id, output=output)

    def expire(
        self, handle: str, conversation_id: str, *, note: str = "[reply timeout]"
    ) -> ResumeResult:
        """Resume a still-parked mapped thread with a synthetic timeout note, then unregister.

        D24. The CALLER owns timing (call this when its own deadline fires); v0.3 adds no
        scheduler. The synthetic ``HumanResponse`` carries only the literal ``note`` (never
        message content, I-2), in the same ``{"text": …}`` dict shape :func:`build_human_response`
        emits (no reply ids — a synthetic timeout has no real inbound message to reply to).
        Returns ``SKIPPED_NO_THREAD`` / ``SKIPPED_NOT_INTERRUPTED`` if not parked, ``FAILED`` if
        the resumed node raises, else ``RESUMED``.
        """
        thread_id = self._store.get(handle, conversation_id)
        if thread_id is None:
            return ResumeResult(ResumeStatus.SKIPPED_NO_THREAD)
        # message_id=None: a synthetic timeout is never journaled (no real message to dedupe).
        return self._run_resume(
            thread_id, {"type": "response", "args": {"text": note}}, handle, conversation_id, None
        )
