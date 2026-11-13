"""Tier-3 idle-wake resumer: wake a paused LangGraph thread on an inbound rine message.

:class:`RineThreadResumer` keeps a ``(handle, conversation_id) -> thread_id``
:class:`~langchain_rine._threadmap.ThreadMapStore` (plus a durable consumed-message-id journal)
and, given an inbound decrypted message, resumes the matching paused graph with
``Command(resume=<HumanResponse>)`` (the Agent-Inbox ``HumanInterrupt`` / ``HumanResponse``
wire), exactly once. The guard chain lives in ``_resolve.py``, the terminal-unregister in
``_terminal.py``, and ``ResumeStatus`` / ``ResumeResult`` / ``default_key_fn`` in ``_resume.py``
(re-exported here so the public import surface stays stable).

**LangGraph is an OPTIONAL extra.** This module imports langgraph at module level and is not
re-exported from ``langchain_rine/__init__.py``, so ``import langchain_rine`` never needs it.
Importing ``langchain_rine.inbound`` IS the opt-in: ``pip install langchain-rine[inbound]``.

Import builds no client and no network (``drivers.py`` owns the rine I/O). A message is resumed
with its decrypted plaintext only, and the journal stores only ``message.id``. Guards fail
readably (logged plus a typed :class:`ResumeStatus`), including a user node that raises on resume
(``ResumeStatus.FAILED``; not acked, so the sweep continues).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

try:
    # Fails fast with the [inbound]-extra hint when langgraph is missing.
    from langgraph.types import Command
except ModuleNotFoundError as exc:  # pragma: no cover - exercised via the [inbound] extra hint
    raise ModuleNotFoundError(
        "langchain_rine.inbound requires langgraph; install it with "
        "`pip install langchain-rine[inbound]`."
    ) from exc

from langchain_rine._resolve import aresolve, check_state, resolve
from langchain_rine._resume import (
    ResumeResult,
    ResumeStatus,
    build_human_response,
    default_key_fn,
)
from langchain_rine._terminal import aunregister_if_terminal, unregister_if_terminal

if TYPE_CHECKING:
    from collections.abc import Callable

    from rine import DecryptedMessage

    from langchain_rine._threadmap import ThreadMapStore

logger = logging.getLogger("langchain_rine.inbound")

# Re-exported so the public import surface stays stable after the module split.
__all__ = ["ResumeStatus", "ResumeResult", "RineThreadResumer", "default_key_fn"]


class RineThreadResumer:
    """Resume a paused LangGraph thread when its conversation gets an inbound rine reply.

    Wraps an already-compiled graph (the caller owns the checkpointer) and a
    :class:`~langchain_rine._threadmap.ThreadMapStore`. The ``drivers.py`` layer fetches messages,
    calls :meth:`handle_inbound` / :meth:`ahandle_inbound`, then :meth:`confirm_delivered` after a
    successful ack.
    """

    def __init__(
        self,
        graph: Any,
        store: ThreadMapStore,
        *,
        key_fn: Callable[[DecryptedMessage], tuple[str, str]] | None = None,
        require_verified: bool = False,
    ) -> None:
        """Bind the graph and store; ``require_verified`` also blocks ``unverifiable`` senders."""
        self._graph = graph
        self._store = store
        self._key_fn = key_fn or default_key_fn
        self._require_verified = require_verified

    def register(self, handle: str, conversation_id: str, thread_id: str) -> None:
        """Bind ``(handle, conversation_id)`` to ``thread_id``; newest wins, with a WARNING."""
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
        """Self-clean the consumed-journal entry after a successful ack.

        An acked message can never be redelivered, so its journal entry is dead; dropping it keeps
        the journal bounded to in-flight ids. The driver calls this for every message it acked.
        """
        handle, conversation_id = self._key_fn(message)
        self._store.drop_consumed(handle, conversation_id, str(message.id))

    async def aconfirm_delivered(self, message: DecryptedMessage) -> None:
        """Async twin of :meth:`confirm_delivered` (uses ``adrop_consumed``)."""
        handle, conversation_id = self._key_fn(message)
        await self._store.adrop_consumed(handle, conversation_id, str(message.id))

    def _run_resume(
        self,
        thread_id: str,
        resume_value: Any,
        handle: str,
        conversation_id: str,
        message_id: str | None,
    ) -> ResumeResult:
        """Sync resume body shared by :meth:`handle_inbound` + :meth:`expire`.

        Records the consumed id (when given) before the driver acks. A raising node or a
        ``mark_consumed`` store error gives ``FAILED`` (logged, not acked).
        """
        config = {"configurable": {"thread_id": thread_id}}
        try:
            skip = check_state(self._graph.get_state(config), thread_id)
            if skip is not None:
                return skip
            output = self._graph.invoke(Command(resume=resume_value), config)
            if message_id is not None:
                self._store.mark_consumed(handle, conversation_id, message_id)
            unregister_if_terminal(
                self._store, self._graph.get_state(config), handle, conversation_id
            )
        except Exception:
            logger.exception("resume of thread %s failed (message %s)", thread_id, message_id)
            return ResumeResult(ResumeStatus.FAILED, thread_id=thread_id)
        logger.info("resumed thread %s (message %s)", thread_id, message_id)
        return ResumeResult(ResumeStatus.RESUMED, thread_id=thread_id, output=output)

    def handle_inbound(self, message: DecryptedMessage) -> ResumeResult:
        """Wake the mapped paused thread with ``message``'s plaintext (sync path).

        Guard order: control frame (acked only on a parked conversation, else left ``new``) ->
        decrypt-error -> refused-sig (+require_verified) -> no-conversation ->
        **already-consumed** -> no-thread -> not-interrupted -> multiple-interrupts (#6533) ->
        resume -> **mark_consumed** -> terminal-unregister. ``mark_consumed`` records **after** the
        resume commits and **before** the driver's ack, closing the ack-failure double-resume
        window. See :meth:`_run_resume` for the failure wrapping.
        """
        early, thread_id, handle, conversation_id = resolve(
            message, self._store, self._key_fn, require_verified=self._require_verified
        )
        if early is not None:
            return early
        return self._run_resume(
            thread_id, build_human_response(message), handle, conversation_id, str(message.id)
        )

    async def ahandle_inbound(self, message: DecryptedMessage) -> ResumeResult:
        """Async twin of :meth:`handle_inbound` (``aresolve`` + ``aget_state`` / ``ainvoke``)."""
        early, thread_id, handle, conversation_id = await aresolve(
            message, self._store, self._key_fn, require_verified=self._require_verified
        )
        if early is not None:
            return early
        config = {"configurable": {"thread_id": thread_id}}
        try:
            skip = check_state(await self._graph.aget_state(config), thread_id)
            if skip is not None:
                return skip
            output = await self._graph.ainvoke(
                Command(resume=build_human_response(message)), config
            )
            await self._store.amark_consumed(handle, conversation_id, str(message.id))
            await aunregister_if_terminal(
                self._store, await self._graph.aget_state(config), handle, conversation_id
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

        The caller owns timing. The ``HumanResponse`` carries only the literal ``note``. Returns
        ``SKIPPED_NO_THREAD`` / ``SKIPPED_NOT_INTERRUPTED`` if not parked, ``FAILED`` if the
        resumed node raises, else ``RESUMED``.
        """
        thread_id = self._store.get(handle, conversation_id)
        if thread_id is None:
            return ResumeResult(ResumeStatus.SKIPPED_NO_THREAD)
        # message_id=None: a synthetic timeout is never journaled.
        return self._run_resume(
            thread_id, {"type": "response", "args": {"text": note}}, handle, conversation_id, None
        )
