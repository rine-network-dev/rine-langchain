"""``rine_pay`` — pay a received x402 quote in-thread (payer, Mode T).

A thin ``BaseTool`` over ``rine.x402`` — never a reimplementation of signing,
policy, or the spend journal. Reads a ``rine.v1.x402_payment_required``, runs
``prepare_payment`` (select under the deny-by-default policy → sign EIP-3009 →
reserve the daily budget as the LAST local act, a permanent debit), then replies
in-thread with the signed ``rine.v1.x402_payment``. Returns a typed status string
reusing the ``rine_pay`` vocabulary verbatim; ``auto_pay`` gates on
``autoPayThreshold`` BEFORE any reserve so a refusal never debits. No key material
is surfaced; the cleartext marker stays ``{status, rail}``-only. Payee side lives
in ``payments_payee.py``.
"""

from typing import TYPE_CHECKING

# BaseModel is a RUNTIME import: pydantic (langchain's BaseTool is a pydantic model)
# resolves the `args_schema: type[BaseModel]` field annotation at class definition, so
# this module intentionally does NOT use `from __future__ import annotations` (mirroring
# messaging.py). SDK types used only in annotations are quoted + kept under TYPE_CHECKING.
from pydantic import BaseModel
from rine._constants import MessageType
from rine.x402 import (
    PreparedPayment,
    X402Error,
    X402ErrorCode,
    is_auto_payable,
    load_policy,
    parse_x402_payload,
    prepare_payment,
    read_daily_spent,
    select_requirement,
    wallet_key_exists,
)

from langchain_rine._client import (
    AsyncCallbackManagerForToolRun,
    BaseRineTool,
    CallbackManagerForToolRun,
    _format_error,
)
from langchain_rine.tools._schemas import RinePayInput

if TYPE_CHECKING:
    from rine.types import DecryptedMessage, MessageRead


def _status_for_error(err: X402Error) -> str:
    """Map an ``X402Error`` from select/reserve to a typed ``rine_pay`` status string."""
    if err.code == X402ErrorCode.ALREADY_PAID:
        return f"status: already-paid — {err.message}"
    if err.code == X402ErrorCode.WALLET_BUSY:
        return f"status: wallet-busy — {err.message}"
    if err.code == X402ErrorCode.INAUTHENTIC_SENDER:
        return f"status: inauthentic-quote — {err.message}"
    if err.code == X402ErrorCode.SENDER_UNVERIFIABLE:
        return f"status: sender-unverifiable — {err.message}"
    reason = err.details.get("reason")
    detail = reason if isinstance(reason, str) else err.code
    return f"status: policy-refused — {detail}: {err.message}"


def _auto_pay_gate(config_dir: str, agent_id: str, pr: dict[str, object]) -> str | None:
    """Refuse (WITHOUT reserving) when ``auto_pay`` is set and the quote is above threshold.

    Returns a typed status string to short-circuit, or ``None`` to proceed to
    ``prepare_payment``. Runs a read-only ``select_requirement`` (no debit) purely
    to evaluate ``isAutoPayable`` — mirroring the MCP ordering (gate before reserve).
    """
    policy = load_policy(config_dir, agent_id)
    daily = read_daily_spent(config_dir, agent_id)
    try:
        requirement = select_requirement(pr, policy, daily)
    except X402Error as err:
        return _status_for_error(err)
    if policy is None or not is_auto_payable(policy, requirement):
        threshold = policy.auto_pay_threshold if policy else None
        return (
            f"status: above-auto-pay-threshold — quote exceeds autoPayThreshold "
            f"({threshold}); call without auto_pay to authorize explicitly."
        )
    return None


def _resolve_pay(
    config_dir: str,
    agent_id: str,
    quote: "DecryptedMessage",
    *,
    auto_pay: bool,
    allow_repay: bool,
    emit_marker: bool,
) -> "str | PreparedPayment":
    """Validate the quote and prepare (sign+reserve) the payment.

    Returns a typed status STRING on any refusal (nothing sent; the pre-reserve
    refusals never debit) or a signed :class:`PreparedPayment` ready to transmit.
    """
    if quote.type != MessageType.X402_PAYMENT_REQUIRED:
        return f"status: not-payment-required — message {quote.id} is type '{quote.type}'."
    if not wallet_key_exists(config_dir, agent_id):
        return f"status: no-wallet — agent {agent_id} has no wallet (run `rine wallet create`)."
    if quote.plaintext is None:
        return f"status: not-payment-required — message {quote.id} could not be decrypted."
    try:
        pr = parse_x402_payload(quote.plaintext)
    except Exception as err:  # noqa: BLE001 — malformed quote → typed refusal, never a crash
        return f"status: not-payment-required — no PaymentRequired in {quote.id}: {err}"
    if auto_pay and (gate := _auto_pay_gate(config_dir, agent_id, pr)) is not None:
        return gate
    try:
        return prepare_payment(
            config_dir,
            agent_id,
            pr,
            quote=quote,
            message_id=str(quote.id),
            force=allow_repay,
            add_marker=emit_marker,
        )
    except X402Error as err:
        return _status_for_error(err)


def _pay_success(sent: "MessageRead", prepared: PreparedPayment) -> str:
    """Render the ``payment-submitted`` status (amounts stay OUT of the marker plane)."""
    req = prepared.requirement
    return (
        f"status: payment-submitted — sent x402_payment {sent.id} in conversation "
        f"{sent.conversation_id} (network={req.get('network')}, asset={req.get('asset')}, "
        f"amount={req.get('amount')}, pay_to={req.get('payTo')})."
    )


class RinePayTool(BaseRineTool):
    """Pay a received x402 payment request in-thread (MUTATING — moves funds)."""

    name: str = "rine_pay"
    description: str = (
        "Pay a received x402 payment request (rine.v1.x402_payment_required) in-thread. "
        "MUTATING: signs a stablecoin authorization with the agent's wallet under its "
        "deny-by-default spend policy and sends it back in the same conversation. The "
        "spend is reserved as the last act before the send (permanent debit; no refund). "
        "Does NOT wait for settlement — the receipt arrives later as an ordinary inbox "
        "message (read it with rine_read / rine_inbox / rine_thread). Returns a typed "
        "`status:` line — payment-submitted / no-wallet / not-payment-required / "
        "policy-refused / above-auto-pay-threshold / already-paid / wallet-busy / "
        "inauthentic-quote (the quote was not signed by the agent that sent it) / "
        "sender-unverifiable (the sender check could not run: nothing was paid, retry later)."
    )
    args_schema: type[BaseModel] = RinePayInput

    def _run(
        self,
        message_id: str,
        auto_pay: bool = False,
        allow_repay: bool = False,
        emit_marker: bool = True,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            client = self._get_client()
            quote = client.read(message_id, agent=self.agent)
            agent_id = str(quote.to_agent_id) if quote.to_agent_id else None
            if agent_id is None:
                return f"status: not-payment-required — message {message_id} has no recipient."
            result = _resolve_pay(
                client._config_dir,
                agent_id,
                quote,
                auto_pay=auto_pay,
                allow_repay=allow_repay,
                emit_marker=emit_marker,
            )
            if isinstance(result, str):
                return result
            sent = client.reply(
                message_id,
                result.message.payload,
                message_type=result.message.message_type,
                metadata=result.message.metadata,
                content_type=result.message.content_type,
                agent=agent_id,
            )
        except Exception as err:  # noqa: BLE001 — surface as a tool message, never crash
            return _format_error(err)
        return _pay_success(sent, result)

    async def _arun(
        self,
        message_id: str,
        auto_pay: bool = False,
        allow_repay: bool = False,
        emit_marker: bool = True,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            client = self._aget_client()
            quote = await client.read(message_id, agent=self.agent)
            agent_id = str(quote.to_agent_id) if quote.to_agent_id else None
            if agent_id is None:
                return f"status: not-payment-required — message {message_id} has no recipient."
            # prepare_payment is synchronous (file IO + signing); only read/reply await.
            result = _resolve_pay(
                client._config_dir,
                agent_id,
                quote,
                auto_pay=auto_pay,
                allow_repay=allow_repay,
                emit_marker=emit_marker,
            )
            if isinstance(result, str):
                return result
            sent = await client.reply(
                message_id,
                result.message.payload,
                message_type=result.message.message_type,
                metadata=result.message.metadata,
                content_type=result.message.content_type,
                agent=agent_id,
            )
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return _pay_success(sent, result)


__all__ = ["RinePayTool"]
