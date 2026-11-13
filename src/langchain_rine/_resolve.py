"""The no-resume guard chain the Tier-3 resumer runs before it touches the graph.

``resolve`` (sync store) and ``aresolve`` (async store) answer one question for an inbound
message: is there an early result to return, or a thread to resume? Order, first match wins:

1. a control frame (key material or a group invite): skipped and acked only when its
   conversation is parked and the frame settled (opened, or refused for authenticity); an
   unopened one stays ``new``, as an unsettled row does on every other path (the sender-key
   recovery scan lists distributions whatever their status, so this is consistency, not the
   frame's last chance), and anywhere else it stays ``new`` so ``rine_inbox`` still shows it.
   A control frame with no conversation id cannot be parked, so it always stays ``new``;
2. the no-I/O chain in :func:`~langchain_rine._resume.pre_state_guard`: decrypt-error, refused
   signature (and unverifiable under ``require_verified``), no conversation;
3. already-consumed, BEFORE the thread lookup so a redelivered consumed message is acked even
   after its thread terminally unregistered;
4. no thread mapped.

``check_state`` is the last guard, run on the thread's graph state once a thread is found.
Side-effect-free at import: no network, no client, no langgraph.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from rine import is_settled

from langchain_rine._resume import (
    ResumeResult,
    ResumeStatus,
    is_control_frame,
    pre_state_guard,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from rine import DecryptedMessage

    from langchain_rine._threadmap import ThreadMapStore

logger = logging.getLogger("langchain_rine.inbound")

#: ``(early result or None, thread_id, handle, conversation_id)``.
Resolution = tuple[ResumeResult | None, str, str, str]


def _log_pre_guard(early: ResumeResult, message: DecryptedMessage) -> None:
    """Log the reason a no-I/O guard fired."""
    if early.status is ResumeStatus.SKIPPED_DECRYPT_ERROR:
        logger.info("skip resume: message %s has decrypt_error", message.id)
    elif early.status is ResumeStatus.SKIPPED_UNVERIFIED:
        logger.warning(
            "skip resume: message %s sig=%s blocks the resume; graph not woken",
            message.id,
            message.verification_status,
        )
    else:  # SKIPPED_NO_CONVERSATION
        logger.info("skip resume: message %s has no conversation_id", message.id)


def _log_no_thread(handle: str, conversation_id: str) -> None:
    logger.info(
        "skip resume: no thread mapped for (%s, %s); leaving in inbox", handle, conversation_id
    )


def _control_frame_early(
    message: DecryptedMessage, parked: bool, handle: str, conversation_id: str
) -> Resolution:
    """Settle a control frame: skipped and acked only when parked and settled."""
    if not parked:
        _log_no_thread(handle, conversation_id)
        return ResumeResult(ResumeStatus.SKIPPED_NO_THREAD), "", handle, conversation_id
    if not is_settled(message):
        logger.info("skip resume: control frame %s has not opened; leaving it new", message.id)
        return ResumeResult(ResumeStatus.SKIPPED_DECRYPT_ERROR), "", handle, conversation_id
    logger.info("skip resume: message %s is a control frame; acking", message.id)
    return ResumeResult(ResumeStatus.SKIPPED_CONTROL_FRAME), "", handle, conversation_id


def resolve(
    message: DecryptedMessage,
    store: ThreadMapStore,
    key_fn: Callable[[DecryptedMessage], tuple[str, str]],
    *,
    require_verified: bool,
) -> Resolution:
    """Run the guard chain against the sync store."""
    if is_control_frame(message):
        handle, conversation_id = key_fn(message)
        parked = bool(conversation_id) and store.get(handle, conversation_id) is not None
        return _control_frame_early(message, parked, handle, conversation_id)
    early, handle, conversation_id = pre_state_guard(
        message, key_fn, require_verified=require_verified
    )
    if early is not None:
        _log_pre_guard(early, message)
        return early, "", handle, conversation_id
    if store.is_consumed(handle, conversation_id, str(message.id)):
        logger.info("skip resume: message %s already consumed (exactly-once)", message.id)
        return ResumeResult(ResumeStatus.SKIPPED_ALREADY_CONSUMED), "", handle, conversation_id
    thread_id = store.get(handle, conversation_id)
    if thread_id is None:
        _log_no_thread(handle, conversation_id)
        return ResumeResult(ResumeStatus.SKIPPED_NO_THREAD), "", handle, conversation_id
    return None, thread_id, handle, conversation_id


async def aresolve(
    message: DecryptedMessage,
    store: ThreadMapStore,
    key_fn: Callable[[DecryptedMessage], tuple[str, str]],
    *,
    require_verified: bool,
) -> Resolution:
    """Async twin of :func:`resolve` (awaits ``aget`` / ``ais_consumed``)."""
    if is_control_frame(message):
        handle, conversation_id = key_fn(message)
        parked = bool(conversation_id) and await store.aget(handle, conversation_id) is not None
        return _control_frame_early(message, parked, handle, conversation_id)
    early, handle, conversation_id = pre_state_guard(
        message, key_fn, require_verified=require_verified
    )
    if early is not None:
        _log_pre_guard(early, message)
        return early, "", handle, conversation_id
    if await store.ais_consumed(handle, conversation_id, str(message.id)):
        logger.info("skip resume: message %s already consumed (exactly-once)", message.id)
        return ResumeResult(ResumeStatus.SKIPPED_ALREADY_CONSUMED), "", handle, conversation_id
    thread_id = await store.aget(handle, conversation_id)
    if thread_id is None:
        _log_no_thread(handle, conversation_id)
        return ResumeResult(ResumeStatus.SKIPPED_NO_THREAD), "", handle, conversation_id
    return None, thread_id, handle, conversation_id


def check_state(state: Any, thread_id: str) -> ResumeResult | None:
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
