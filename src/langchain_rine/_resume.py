"""Resume primitives shared by the Tier-3 resumer (``inbound.py``) and drivers.

Carved out of ``inbound.py`` so each file stays under the ~200-LOC cap (the v0.2
audit fixes — FAILED status, verification gating, terminal-unregister — pushed
``inbound.py`` over). This module holds the *transport-free* pieces:

- :class:`ResumeStatus` — the typed outcome of handling one inbound message (no
  stringly-typed status anywhere).
- :class:`ResumeResult` — the per-message result the driver acts on (its
  :attr:`~ResumeResult.should_ack` property is the v0.3 ack policy — D20).
- :func:`default_key_fn` — the ``(handle, conversation_id)`` store-key derivation.
- :func:`verification_blocks_resume` / :func:`build_human_response` — the ``wh-2`` /
  D19 / D23 signature-verification gate: a *tampered* (``invalid``) signature must
  never wake a graph; an *unverifiable* one resumes-but-annotated by default, or is
  *blocked* when the caller opts into ``require_verified`` (D23). Both reuse
  ``_format`` — never the ciphertext (I-2).
- :func:`pre_state_guard` — the **no-I/O** guard chain (decrypt-error → invalid-sig →
  no-conversation) shared verbatim by the sync ``_resolve`` and async ``_aresolve``
  paths so neither duplicates the guard logic (D21).

Side-effect-free at import (I-1): no network, no client, no langgraph. The
re-exports in ``inbound.py`` keep ``from langchain_rine.inbound import
ResumeStatus, ResumeResult, default_key_fn`` stable.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_rine._format import render_message_body

if TYPE_CHECKING:
    from collections.abc import Callable

    from langgraph.prebuilt.interrupt import HumanResponse
    from rine import DecryptedMessage

    KeyFn = Callable[[DecryptedMessage], tuple[str, str]]

__all__ = ["ResumeStatus", "ResumeResult", "default_key_fn"]


class ResumeStatus(enum.Enum):
    """The outcome of handling one inbound message (no stringly-typed status)."""

    RESUMED = "resumed"
    SKIPPED_NO_THREAD = "skipped_no_thread"
    SKIPPED_NO_CONVERSATION = "skipped_no_conversation"
    SKIPPED_NOT_INTERRUPTED = "skipped_not_interrupted"
    SKIPPED_MULTIPLE_INTERRUPTS = "skipped_multiple_interrupts"
    SKIPPED_DECRYPT_ERROR = "skipped_decrypt_error"
    SKIPPED_UNVERIFIED = "skipped_unverified"
    SKIPPED_ALREADY_CONSUMED = "skipped_already_consumed"
    FAILED = "failed"


# The statuses whose message the driver must mark-delivered (ack). ``RESUMED`` leaves the
# inbox because it was handled; ``SKIPPED_ALREADY_CONSUMED`` is a redelivery of a message we
# already resumed (its journal entry proves it) — re-ack it so it finally leaves the inbox
# without a second resume (D20). Every other status leaves the message ``new`` for a later
# sweep (D16).
_ACK_STATUSES = frozenset({ResumeStatus.RESUMED, ResumeStatus.SKIPPED_ALREADY_CONSUMED})


@dataclass
class ResumeResult:
    """The result of :meth:`RineThreadResumer.handle_inbound` for one message.

    ``thread_id`` is the mapped thread (when one was found); ``output`` is the resumed graph's
    return value on a ``RESUMED`` status, else None. The driver acks the
    :attr:`should_ack` results (``RESUMED`` + ``SKIPPED_ALREADY_CONSUMED`` — D20); every other
    ``SKIPPED_*`` / ``FAILED`` message stays in the rine inbox for a later sweep (D16).
    """

    status: ResumeStatus
    thread_id: str | None = None
    output: Any | None = None

    @property
    def should_ack(self) -> bool:
        """True iff the driver should mark this message delivered (D20 ack policy)."""
        return self.status in _ACK_STATUSES


def default_key_fn(message: DecryptedMessage) -> tuple[str, str]:
    """Derive the ``(handle, conversation_id)`` store key from a decrypted message.

    Uses the real SDK attributes: ``sender_handle`` (falling back to ``from_agent_id`` when the
    server did not include a handle) and the stringified ``conversation_id``. Both are stringified
    so the key matches what :meth:`RineThreadResumer.register` stores.
    """
    handle = message.sender_handle or (
        str(message.from_agent_id) if message.from_agent_id is not None else ""
    )
    conversation_id = str(message.conversation_id) if message.conversation_id is not None else ""
    return handle, conversation_id


def verification_blocks_resume(
    message: DecryptedMessage, *, require_verified: bool = False
) -> bool:
    """Return True iff the signature blocks the resume (never wakes the graph).

    The ``wh-2`` / D19 / D23 gate. ``invalid`` (tampered/forged) is **always** blocked — a
    tampered signature must never wake an autonomous graph. ``unverifiable`` (sender key
    unfetchable) is blocked **only when ``require_verified`` is True** (D23: a strict deployment
    skips it as ``SKIPPED_UNVERIFIED`` rather than annotate-and-resume); otherwise it is allowed
    through and :func:`build_human_response` annotates it. ``verified`` is always allowed.
    """
    if message.verification_status == "invalid":
        return True
    return require_verified and message.verification_status == "unverifiable"


def pre_state_guard(
    message: DecryptedMessage, key_fn: KeyFn, *, require_verified: bool = False
) -> tuple[ResumeResult, str, str] | tuple[None, str, str]:
    """Run the **no-I/O** inbound guards shared by the sync + async resume paths (D21).

    Returns ``(early, handle, conversation_id)``. A non-None ``early`` is a terminal
    :class:`ResumeResult` to return directly (decrypt-error / blocked-signature /
    no-conversation); when ``early`` is None the caller proceeds to the I/O guards
    (already-consumed, thread lookup) with the derived ``handle`` / ``conversation_id``.

    Factored here so ``_resolve`` and ``_aresolve`` do not duplicate the guard chain — each
    only adds its own (sync vs awaited) store lookups (D21). No store / network / langgraph
    touch happens here; logging is left to the caller so the message text can name the path.
    """
    if message.decrypt_error:
        return ResumeResult(ResumeStatus.SKIPPED_DECRYPT_ERROR), "", ""
    if verification_blocks_resume(message, require_verified=require_verified):
        return ResumeResult(ResumeStatus.SKIPPED_UNVERIFIED), "", ""
    handle, conversation_id = key_fn(message)
    if not conversation_id:
        return ResumeResult(ResumeStatus.SKIPPED_NO_CONVERSATION), "", ""
    return None, handle, conversation_id


def build_human_response(message: DecryptedMessage) -> HumanResponse:
    """Agent-Inbox ``HumanResponse`` carrying the (verification-annotated) plaintext.

    Reuses ``_format.render_message_body`` (the unwrapped plaintext — never the ciphertext, I-2).
    When the sender signature is ``unverifiable`` (D19), the content is prefixed with an
    ``[unverified sender]`` marker so the resumed graph/LLM sees the body is untrusted; a
    ``verified`` message is passed clean. (An ``invalid`` signature never reaches here — the
    resumer skips it via :func:`verification_blocks_resume`.)
    """
    body = render_message_body(message)
    if message.verification_status == "unverifiable":
        body = f"[unverified sender] {body}"
    return {"type": "response", "args": body}
