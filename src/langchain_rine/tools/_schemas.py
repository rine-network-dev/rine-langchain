"""Pydantic ``args_schema`` models for the messaging tools (SPEC §6.1–6.5).

Kept separate from ``messaging.py`` so the tool logic stays under the per-file LOC
budget. Each model maps 1:1 to a tool's typed arguments; descriptions are the text
the LLM sees when it inspects the tool.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from rine.x402 import FEATURED_FACILITATOR


class RineSendInput(BaseModel):
    """Arguments for ``rine_send``."""

    to: str = Field(description="Recipient handle/UUID, or '#group@org' for a group.")
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


class RineCheckInboxInput(BaseModel):
    """Arguments for ``rine_check_inbox``."""

    limit: int = Field(default=20, ge=1, le=100, description="Max new messages to return.")


class RineReadInput(BaseModel):
    """Arguments for ``rine_read``."""

    message_id: str = Field(description="UUID of the message to fetch and decrypt.")


class RineReplyInput(BaseModel):
    """Arguments for ``rine_reply``."""

    message_id: str = Field(description="UUID of the message to reply to.")
    body: str = Field(description="Reply text (sent as the payload's 'text' field).")
    message_type: str = Field(default="rine.v1.task_response", description="Rine message type.")


class RineThreadInput(BaseModel):
    """Arguments for ``rine_thread``."""

    conversation_id: str = Field(
        description="UUID of the conversation to fetch the transcript of."
    )
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
    "RineSendInput",
    "RineSendAndWaitInput",
    "RineCheckInboxInput",
    "RineReadInput",
    "RineReplyInput",
    "RineThreadInput",
    "RinePayInput",
    "RineFulfillInput",
]
