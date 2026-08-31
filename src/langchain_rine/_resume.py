"""Resume primitives shared by the Tier-3 resumer (``inbound.py``) and drivers.

Carved out of ``inbound.py`` so each file stays under the ~200-LOC cap (the v0.2
audit fixes — FAILED status, verification gating, terminal-unregister — pushed
``inbound.py`` over). This module holds the *transport-free* pieces:

- :class:`ResumeStatus` — the typed outcome of handling one inbound message (no
  stringly-typed status anywhere).
- :class:`ResumeResult` — the per-message result the driver acts on (its
  :attr:`~ResumeResult.should_ack` property is the v0.3 ack policy).
- :func:`default_key_fn` — the ``(handle, conversation_id)`` store-key derivation.
- :func:`verification_blocks_resume` / :func:`build_human_response` — the
  signature-verification gate: an *authenticity failure* (``invalid`` or
  ``sender-mismatch``) must never wake a graph; an *unverifiable* one
  resumes-but-annotated by default, or is *blocked* when the caller opts into
  ``require_verified``. Both reuse ``_format`` — never the ciphertext.
- :func:`pre_state_guard` — the **no-I/O** guard chain (decrypt-error → refused
  signature → no-conversation) shared verbatim by the sync ``_resolve`` and async
  ``_aresolve`` paths so neither duplicates the guard logic.

Side-effect-free at import: no network, no client, no langgraph. The
re-exports in ``inbound.py`` keep ``from langchain_rine.inbound import
ResumeStatus, ResumeResult, default_key_fn`` stable.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rine import is_authenticity_failure
from rine.sender_key_words import SenderKeyRefusal

from langchain_rine._format import render_message_body

if TYPE_CHECKING:
    from collections.abc import Callable

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
# without a second resume. Every other status leaves the message ``new`` for a later
# sweep.
_ACK_STATUSES = frozenset({ResumeStatus.RESUMED, ResumeStatus.SKIPPED_ALREADY_CONSUMED})


@dataclass
class ResumeResult:
    """The result of :meth:`RineThreadResumer.handle_inbound` for one message.

    ``thread_id`` is the mapped thread (when one was found); ``output`` is the resumed graph's
    return value on a ``RESUMED`` status, else None. The driver acks the
    :attr:`should_ack` results (``RESUMED`` + ``SKIPPED_ALREADY_CONSUMED``); every other
    ``SKIPPED_*`` / ``FAILED`` message stays in the rine inbox for a later sweep.
    """

    status: ResumeStatus
    thread_id: str | None = None
    output: Any | None = None

    @property
    def should_ack(self) -> bool:
        """True iff the driver should mark this message delivered (ack policy)."""
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

    The signature-verification gate. An **authenticity failure** is always blocked: the SDK
    stamps ``invalid`` on a signature that does not cover the bytes and ``sender-mismatch``
    on one that does but belongs to an agent other than the sender the server names, and
    neither may wake an autonomous graph. The set is the SDK's own
    :func:`~rine.is_authenticity_failure` rather than a literal here, so a verdict added to
    :data:`~rine.VerificationStatus` is blocked the day it ships instead of falling through
    this gate unnamed.

    ``unverifiable`` is not a finding against the message — no verdict was reached, because
    the sender's key could not be read — so it is blocked **only when ``require_verified`` is
    True** (a strict deployment skips it as ``SKIPPED_UNVERIFIED`` rather than
    annotate-and-resume); otherwise it is allowed through and :func:`build_human_response`
    annotates it. ``verified`` is always allowed.
    """
    if is_authenticity_failure(message.verification_status):
        return True
    return require_verified and message.verification_status == "unverifiable"


def pre_state_guard(
    message: DecryptedMessage, key_fn: KeyFn, *, require_verified: bool = False
) -> tuple[ResumeResult, str, str] | tuple[None, str, str]:
    """Run the **no-I/O** inbound guards shared by the sync + async resume paths.

    Returns ``(early, handle, conversation_id)``. A non-None ``early`` is a terminal
    :class:`ResumeResult` to return directly (decrypt-error / refused signature /
    no-conversation); when ``early`` is None the caller proceeds to the I/O guards
    (already-consumed, thread lookup) with the derived ``handle`` / ``conversation_id``.

    Factored here so ``_resolve`` and ``_aresolve`` do not duplicate the guard chain — each
    only adds its own (sync vs awaited) store lookups. No store / network / langgraph
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


def build_human_response(message: DecryptedMessage) -> dict[str, Any]:
    """Agent-Inbox resume payload carrying the (verification-annotated) plaintext + reply ids.

    Reuses ``_format.render_message_body`` (the unwrapped plaintext — never the ciphertext).
    Only a message the SDK actually attributed to its sender is passed clean; anything else
    is prefixed with an ``[unverified sender]`` marker so the resumed graph/LLM sees the body
    is untrusted. Keying that on the SDK's own ``verified`` flag rather than on the one
    ``unverifiable`` status means a verdict added later is marked, not silently trusted. (An
    authenticity failure never reaches here — the resumer refuses it in
    :func:`verification_blocks_resume`.)

    ``args`` is a dict so the graph's reply node can reply IN-PLACE: it carries
    ``rine_message_id`` (the inbound id, for ``rine_reply(message_id=…)`` which preserves the
    conversation instead of forking via ``rine_send``) and ``rine_conversation_id`` alongside the
    ``text``. Both ids are routing metadata, not ciphertext. Graph nodes read
    ``args["text"]`` / ``args["rine_message_id"]``. :meth:`RineThreadResumer.expire` emits the same
    ``{"text": …}`` shape (without ids — a synthetic note has no real message to reply to).
    """
    # One inbound message per resume payload — a single-message render, so a fresh
    # refusal explains a missing sender key in full rather than marking it.
    body = render_message_body(message, SenderKeyRefusal())
    if not message.verified:
        body = f"[unverified sender] {body}"
    args = {
        "text": body,
        "rine_message_id": str(message.id),
        "rine_conversation_id": str(message.conversation_id),
    }
    return {"type": "response", "args": args}
