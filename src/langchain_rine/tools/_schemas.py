"""Pydantic ``args_schema`` models for the messaging tools.

Kept separate from ``messaging.py`` so the tool logic stays under the per-file LOC
budget. Each model maps 1:1 to a tool's typed arguments; descriptions are the text
the LLM sees when it inspects the tool.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from rine import JOIN_REFERENCE_RULE, TRANSCRIPT_REFERENCE_REQUIRED
from rine.x402 import FEATURED_FACILITATOR

#: Inbox slices this tool layer offers. Three of them are the server's own
#: ``status`` values; ``all`` is this layer's word for *omit the parameter*,
#: because the route has no such value and would refuse it.
InboxSlice = Literal["new", "delivered", "read", "all"]

#: How a group is named, said once for every tool that takes one.
#:
#: Eleven schemas take a group reference and all eleven used to retype this
#: sentence, which is how nine of them came to omit the bare name the resolver
#: has always accepted: a model reading those nine is told the only spellings
#: are a handle and a UUID, so a model holding a bare group name builds a handle
#: out of it — and the handle it builds is the schema's own example with the
#: name swapped in. Both spellings are shown as worked values for the same
#: reason: a model shown the *shape* of an answer fills the shape in.
GROUP_REFERENCE = (
    "Group handle, bare name, or UUID. Examples: #logistics@acme.rine.network, or just logistics."
)


#: How ``rine_group_join`` names its subject, and the one group tool that is not
#: :data:`GROUP_REFERENCE`.
#:
#: 🔴 **The rule itself is the SDK's, imported and never retyped.** ``join``'s
#: reference rules are not the shared ones — a handle walks three rungs, and a
#: bare name is read against this agent's pending invitations and nowhere wider,
#: deliberately refusing a network-wide search that would feed an irreversible
#: verb. All five connectors advertise the same rule, so it lives in
#: :data:`rine.JOIN_REFERENCE_RULE` on the same terms as
#: :data:`rine.VOTE_ELECTORATE_RULE`; only the surface's own discovery verb is
#: added here, because which tool prints a directory row the SDK cannot know.
GROUP_JOIN_REFERENCE = (
    f"{JOIN_REFERENCE_RULE} An id is what a rine_discover_groups row prints beside a handle."
)


class RineSendInput(BaseModel):
    """Arguments for ``rine_send``."""

    to: str = Field(
        description=(
            "Who this goes to: an agent handle or UUID, a group handle, or a group's "
            "own name. Examples: kofi@acme.rine.network for one agent; "
            "#logistics@acme.rine.network, or just logistics, for a group."
        )
    )
    body: str = Field(description="Message text (sent as the payload's 'text' field).")
    message_type: str = Field(default="rine.v1.task_request", description="Rine message type.")
    idempotency_key: str | None = Field(
        default=None, description="Optional idempotency key to dedupe retried sends."
    )


class RineSendAndWaitInput(BaseModel):
    """Arguments for ``rine_send_and_wait``."""

    to: str = Field(description="Recipient handle/UUID (1:1 only — not a group).")
    body: str = Field(description="Message text (sent as the payload's 'text' field).")
    wait_seconds: int = Field(
        default=30, ge=1, le=300, description="Seconds to block for a reply (1-300)."
    )
    message_type: str = Field(default="rine.v1.task_request", description="Rine message type.")


class RineInboxInput(BaseModel):
    """Arguments for ``rine_inbox``."""

    status: InboxSlice = Field(
        default="new",
        description=(
            "Which slice of the inbox to return: 'new' (undelivered, the default), "
            "'delivered', 'read', or 'all' for every message regardless of state. Only "
            "'new' marks what it returns delivered."
        ),
    )
    limit: int = Field(default=20, ge=1, le=100, description="Max messages to return.")


class RineReadInput(BaseModel):
    """Arguments for ``rine_read``."""

    message_id: str = Field(description="UUID of the message to fetch and decrypt.")


class RineReplyInput(BaseModel):
    """Arguments for ``rine_reply``."""

    message_id: str = Field(description="UUID of the message to reply to.")
    body: str = Field(description="Reply text (sent as the payload's 'text' field).")
    message_type: str = Field(default="rine.v1.task_response", description="Rine message type.")


class RineThreadInput(BaseModel):
    """Arguments for ``rine_thread``.

    Neither reference is required on its own and exactly one of them is: a
    transcript is either a conversation's or a group's, and the rule is stated in
    the SDK's own words (``TRANSCRIPT_REFERENCE_REQUIRED``) because that is the
    sentence the call is refused with. ``conversation_id`` stays first, so a
    caller that named it positionally still names it.
    """

    conversation_id: str | None = Field(
        default=None,
        description=(
            f"UUID of the conversation to fetch the transcript of. {TRANSCRIPT_REFERENCE_REQUIRED}"
        ),
    )
    group: str | None = Field(default=None, description=GROUP_REFERENCE)
    limit: int | None = Field(
        default=None, ge=1, le=100, description="Max most-recent turns to return (1-100)."
    )


class RinePayInput(BaseModel):
    """Arguments for ``rine_pay``."""

    message_id: str = Field(
        description="UUID of the received rine.v1.x402_payment_required message to pay."
    )
    auto_pay: bool = Field(
        default=False,
        description=(
            "If true, pay ONLY when the quote is at/below the policy's autoPayThreshold; "
            "otherwise return `above-auto-pay-threshold` without paying. Default false — an "
            "explicit rine_pay call is the authorization, still bounded by the spend caps."
        ),
    )
    allow_repay: bool = Field(
        default=False,
        description=(
            "If true, re-pay a quote already paid today (a deliberate second, permanent "
            "debit). Default false — a repeat of the same message_id returns `already-paid`."
        ),
    )
    emit_marker: bool = Field(
        default=True,
        description=(
            "Attach the cleartext status marker (metadata['rine.x402']). Default true. The "
            "marker carries only coarse lifecycle state — never amounts/assets/addresses."
        ),
    )


class RineFulfillInput(BaseModel):
    """Arguments for ``rine_fulfill``."""

    message_id: str = Field(
        description="UUID of the received rine.v1.x402_payment message to fulfil."
    )
    facilitator: str = Field(
        default=FEATURED_FACILITATOR,
        description=(
            "Facilitator preset that verifies+settles the authorization off-rine: `cdp` "
            "(default), keyless `payai`, or self-hosted `x402-rs`. Ignored when "
            "facilitator_url is set."
        ),
    )
    facilitator_url: str | None = Field(
        default=None,
        description="Explicit facilitator base URL, overriding the preset (e.g. a self-host).",
    )
    add_marker: bool = Field(
        default=True,
        description=(
            "Attach the cleartext {status, rail} marker to the receipt. Default true; never "
            "carries amounts/addresses."
        ),
    )


__all__ = [
    "InboxSlice",
    "RineSendInput",
    "RineSendAndWaitInput",
    "RineInboxInput",
    "RineReadInput",
    "RineReplyInput",
    "RineThreadInput",
    "RinePayInput",
    "RineFulfillInput",
]
