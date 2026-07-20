"""Behavioral tests for the x402 payment tools (``rine_pay`` + ``rine_fulfill``).

Driven through the public LangChain interface (``tool.invoke`` / ``tool.ainvoke``);
only the HTTP boundary is mocked (respx) — the rine SDK's real crypto/signing/policy
runs (the wallet key + spend policy live on disk, EIP-3009 signing is real). The
facilitator verify/settle HTTP is mocked at an explicit base URL, matching how the
payee ladder (``rine.x402.fulfill``) reaches an external facilitator.

Invariants asserted here: the typed ``rine_pay`` status vocabulary is reused verbatim;
the auto-pay gate refuses WITHOUT debiting; a repeat pay is ``already-paid``; the
cleartext marker on the wire stays ``{status, rail}``-only (never the amount); a failed
verification is a business outcome (a failure receipt), not a crash.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import respx
from rine._constants import MessageType
from rine._crypto.keys import load_agent_keys
from rine._crypto.message import encrypt_message
from rine._crypto.wallet import save_wallet_key
from rine.x402 import PaymentPolicy, save_policy

from langchain_rine import RineFulfillTool, RinePayTool

from ..conftest import (
    AGENT_ID,
    API_URL,
    OTHER_AGENT_ID,
    REPLY_ID,
    message_payload,
    mock_agents,
    mock_recipient_keys,
    mock_token,
    sent_message_payload,
)


def _seal(config_dir: str, payload: dict[str, Any]) -> str:
    """HPKE-seal ``payload`` from OTHER_AGENT_ID (the peer) to AGENT_ID (us).

    Signed by the peer and sealed to us, so ``client.read`` both decrypts (our private
    key is on disk) AND verifies the sender signature (the peer's keys are mocked at
    ``/agents/{OTHER_AGENT_ID}/keys`` via ``mock_recipient_keys``). Requires the
    ``other_keys`` fixture (peer keys on disk).
    """
    own_pk = load_agent_keys(config_dir, AGENT_ID).encryption.public_key
    return encrypt_message(config_dir, OTHER_AGENT_ID, own_pk, payload).encrypted_payload


QUOTE_ID = "cccccccc-dddd-eeee-ffff-000000000001"  # == MESSAGE_ID (the fetched message)
FAC_URL = "https://fac.local"

BASE = "eip155:8453"
EURC = "0x60a3e35cc302bfa44cb288bc5a4f316fdb1adb42"
PAY_TO = "0x808456652fdb597867f38412077a9182bf77359f"
WALLET_KEY = bytes(31) + b"\x01"


def _accepts(amount: str = "500000") -> list[dict[str, Any]]:
    """One ``exact`` EURC-on-Base requirement (0.50 EURC by default)."""
    return [
        {
            "scheme": "exact",
            "network": BASE,
            "amount": amount,
            "asset": EURC,
            "payTo": PAY_TO,
            "maxTimeoutSeconds": 60,
            "extra": {"name": "EURC", "version": "2"},
        }
    ]


def _payment_required(amount: str = "500000") -> dict[str, Any]:
    """A verbatim x402 ``PaymentRequired`` object (what the quote body decrypts to)."""
    return {
        "x402Version": 2,
        "resource": {"description": "summarize"},
        "accepts": _accepts(amount),
    }


def _install_wallet(config_dir: str, *, policy: bool = True) -> None:
    """Put the payer's wallet key (and, by default, a spend policy) on disk."""
    save_wallet_key(config_dir, AGENT_ID, WALLET_KEY)
    if policy:
        save_policy(
            config_dir,
            AGENT_ID,
            PaymentPolicy(
                per_tx_cap="5.00",
                daily_cap="10.00",
                allowed_assets=["EURC"],
                allowed_networks=[BASE],
                auto_pay_threshold="1.00",
            ),
        )


def _mock_quote(config_dir: str, amount: str = "500000", *, msg_type: str | None = None) -> None:
    """Mock ``GET /messages/{id}`` → an x402 quote sealed to the payer (self-decryptable)."""
    ciphertext = _seal(config_dir, _payment_required(amount))
    respx.get(f"{API_URL}/messages/{QUOTE_ID}").mock(
        return_value=httpx.Response(
            200,
            json=message_payload(
                encryption_version="hpke-v1",
                encrypted_payload=ciphertext,
                msg_id=QUOTE_ID,
                from_agent_id=OTHER_AGENT_ID,  # payee = quote sender
                to_agent_id=AGENT_ID,  # payer = us
                msg_type=msg_type or MessageType.X402_PAYMENT_REQUIRED,
            ),
        )
    )


def _mock_reply() -> respx.Route:
    """Mock ``POST /messages/{id}/reply`` → a sent message (id == REPLY_ID)."""
    return respx.post(f"{API_URL}/messages/{QUOTE_ID}/reply").mock(
        return_value=httpx.Response(200, json={**sent_message_payload(), "id": REPLY_ID})
    )


def _pay_tool(config_dir: str) -> RinePayTool:
    return RinePayTool(config_dir=config_dir, api_url=API_URL)


# =====================================================================
# RinePayTool — happy path + marker discipline
# =====================================================================


class TestRinePayHappyPath:
    @respx.mock
    def test_pay_valid_quote_signs_and_submits(self, config_dir, other_keys):
        """A payable quote → payment-submitted, the signed x402_payment posted to reply."""
        mock_token()
        mock_agents()
        mock_recipient_keys(config_dir)  # payee (OTHER_AGENT_ID) keys, to encrypt the payment
        _install_wallet(config_dir)
        _mock_quote(config_dir)
        route = _mock_reply()

        out = _pay_tool(config_dir).invoke({"message_id": QUOTE_ID})

        assert route.called
        assert "status: payment-submitted" in out
        assert REPLY_ID in out
        assert BASE in out  # network surfaced to the agent

    @respx.mock
    def test_pay_wire_body_type_is_x402_payment(self, config_dir, other_keys):
        """The posted reply is a rine.v1.x402_payment carrying opaque ciphertext."""
        mock_token()
        mock_agents()
        mock_recipient_keys(config_dir)
        _install_wallet(config_dir)
        _mock_quote(config_dir)
        route = _mock_reply()

        _pay_tool(config_dir).invoke({"message_id": QUOTE_ID})

        body = json.loads(route.calls[0].request.content)
        assert body["type"] == MessageType.X402_PAYMENT
        assert "encrypted_payload" in body

    @respx.mock
    def test_pay_marker_is_status_rail_only_never_amount(self, config_dir, other_keys):
        """The cleartext marker carries {status, rail} only — the amount never leaks (inv.2)."""
        mock_token()
        mock_agents()
        mock_recipient_keys(config_dir)
        _install_wallet(config_dir)
        _mock_quote(config_dir, amount="500000")
        route = _mock_reply()

        _pay_tool(config_dir).invoke({"message_id": QUOTE_ID})

        body = json.loads(route.calls[0].request.content)
        marker = body["metadata"]["rine.x402"]
        assert set(marker) == {"status", "rail"}
        assert marker["rail"] == "x402"
        assert "500000" not in json.dumps(body["metadata"])  # no amount in the marker plane

    @respx.mock
    def test_pay_emit_marker_false_sends_sealed(self, config_dir, other_keys):
        """emit_marker=False sends a fully-sealed payment (no cleartext marker)."""
        mock_token()
        mock_agents()
        mock_recipient_keys(config_dir)
        _install_wallet(config_dir)
        _mock_quote(config_dir)
        route = _mock_reply()

        _pay_tool(config_dir).invoke({"message_id": QUOTE_ID, "emit_marker": False})

        body = json.loads(route.calls[0].request.content)
        assert "metadata" not in body or body["metadata"] in ({}, None)

    @respx.mock
    async def test_pay_arun_matches_run(self, config_dir, other_keys):
        """AC-8: the async path submits the payment identically."""
        mock_token()
        mock_agents()
        mock_recipient_keys(config_dir)
        _install_wallet(config_dir)
        _mock_quote(config_dir)
        route = _mock_reply()

        out = await _pay_tool(config_dir).ainvoke({"message_id": QUOTE_ID})

        assert route.called
        assert "status: payment-submitted" in out
        assert REPLY_ID in out


# =====================================================================
# RinePayTool — typed refusals (no debit / no send)
# =====================================================================


class TestRinePayRefusals:
    @respx.mock
    def test_pay_non_payment_message_returns_not_payment_required(self, config_dir, other_keys):
        """A non-x402 message → not-payment-required, nothing signed or sent."""
        mock_token()
        mock_agents()
        mock_recipient_keys(config_dir)
        _install_wallet(config_dir)
        _mock_quote(config_dir, msg_type=MessageType.TASK_REQUEST)
        route = _mock_reply()

        out = _pay_tool(config_dir).invoke({"message_id": QUOTE_ID})

        assert "status: not-payment-required" in out
        assert not route.called

    @respx.mock
    def test_pay_without_wallet_returns_no_wallet(self, config_dir, other_keys):
        """A valid quote but no wallet on disk → no-wallet (never touches the network)."""
        mock_token()
        mock_agents()
        mock_recipient_keys(config_dir)
        _install_wallet(config_dir, policy=False)
        # remove the wallet: install policy only
        import os

        os.remove(os.path.join(config_dir, "keys", AGENT_ID, "wallet.key"))
        _mock_quote(config_dir)
        route = _mock_reply()

        out = _pay_tool(config_dir).invoke({"message_id": QUOTE_ID})

        assert "status: no-wallet" in out
        assert not route.called

    @respx.mock
    def test_pay_without_policy_returns_policy_refused(self, config_dir, other_keys):
        """Deny-by-default: a wallet but no spend policy → policy-refused, no send."""
        mock_token()
        mock_agents()
        mock_recipient_keys(config_dir)
        _install_wallet(config_dir, policy=False)  # wallet but NO policy
        _mock_quote(config_dir)
        route = _mock_reply()

        out = _pay_tool(config_dir).invoke({"message_id": QUOTE_ID})

        assert "status: policy-refused" in out
        assert not route.called

    @respx.mock
    def test_pay_auto_pay_above_threshold_refuses_without_debiting(self, config_dir, other_keys):
        """auto_pay + above autoPayThreshold → above-auto-pay-threshold; no send, no debit."""
        mock_token()
        mock_agents()
        mock_recipient_keys(config_dir)
        _install_wallet(config_dir)  # threshold 1.00
        _mock_quote(config_dir, amount="2000000")  # 2.00 EURC > 1.00 threshold
        route = _mock_reply()

        out = _pay_tool(config_dir).invoke({"message_id": QUOTE_ID, "auto_pay": True})

        assert "status: above-auto-pay-threshold" in out
        assert not route.called
        # No debit occurred: a subsequent EXPLICIT pay of the same quote still succeeds
        # (the refusal never reserved, so this is not blocked as already-paid).
        out2 = _pay_tool(config_dir).invoke({"message_id": QUOTE_ID})
        assert "status: payment-submitted" in out2

    @respx.mock
    def test_pay_repeat_same_quote_returns_already_paid(self, config_dir, other_keys):
        """Paying the same quote twice → already-paid on the second call (once-per-message)."""
        mock_token()
        mock_agents()
        mock_recipient_keys(config_dir)
        _install_wallet(config_dir)
        _mock_quote(config_dir)
        _mock_reply()

        first = _pay_tool(config_dir).invoke({"message_id": QUOTE_ID})
        second = _pay_tool(config_dir).invoke({"message_id": QUOTE_ID})

        assert "status: payment-submitted" in first
        assert "status: already-paid" in second

    @respx.mock
    def test_pay_repeat_with_allow_repay_pays_again(self, config_dir, other_keys):
        """allow_repay bypasses the once-per-message guard (a deliberate second debit)."""
        mock_token()
        mock_agents()
        mock_recipient_keys(config_dir)
        _install_wallet(config_dir)
        _mock_quote(config_dir)
        _mock_reply()

        _pay_tool(config_dir).invoke({"message_id": QUOTE_ID})
        again = _pay_tool(config_dir).invoke({"message_id": QUOTE_ID, "allow_repay": True})

        assert "status: payment-submitted" in again

    @respx.mock
    def test_pay_never_raises_on_error(self, config_dir):
        """A transport error is returned as a string, never raised out of the tool."""
        mock_token()
        mock_agents()
        _install_wallet(config_dir)
        respx.get(f"{API_URL}/messages/{QUOTE_ID}").mock(
            return_value=httpx.Response(404, json={"detail": "message not found"})
        )

        out = _pay_tool(config_dir).invoke({"message_id": QUOTE_ID})

        assert isinstance(out, str)
        assert "Not found" in out


# =====================================================================
# RineFulfillTool — payee verify → settle → receipt
# =====================================================================

SIGNED_PAYER = "0xSignerFromAuthorization"
PAYMENT_PAYLOAD = {
    "x402Version": 2,
    "accepted": _accepts()[0],
    "payload": {
        "signature": "0xsig",
        "authorization": {"from": SIGNED_PAYER, "to": PAY_TO, "value": "500000"},
    },
}
SETTLED = {
    "success": True,
    "transaction": "0xhash",
    "network": "base-sepolia",  # PayAI echoes the v1 slug — stored verbatim
    "payer": SIGNED_PAYER,
    "amount": "500000",
}


def _mock_payment(config_dir: str, *, msg_type: str | None = None) -> None:
    """Mock ``GET /messages/{id}`` → an x402_payment sealed to the payee (self-decryptable)."""
    ciphertext = _seal(config_dir, PAYMENT_PAYLOAD)
    respx.get(f"{API_URL}/messages/{QUOTE_ID}").mock(
        return_value=httpx.Response(
            200,
            json=message_payload(
                encryption_version="hpke-v1",
                encrypted_payload=ciphertext,
                msg_id=QUOTE_ID,
                from_agent_id=OTHER_AGENT_ID,  # payer = message sender
                to_agent_id=AGENT_ID,  # payee = us
                msg_type=msg_type or MessageType.X402_PAYMENT,
            ),
        )
    )


def _fulfill_tool(config_dir: str) -> RineFulfillTool:
    return RineFulfillTool(config_dir=config_dir, api_url=API_URL)


class TestRineFulfill:
    @respx.mock
    def test_fulfill_valid_settles_and_sends_completed_receipt(self, config_dir, other_keys):
        """Valid verify → settle → payment-completed receipt threaded to the payer."""
        mock_token()
        mock_agents()
        mock_recipient_keys(config_dir)  # payer keys, to encrypt the receipt
        respx.post(f"{FAC_URL}/verify").mock(
            return_value=httpx.Response(200, json={"isValid": True, "payer": SIGNED_PAYER})
        )
        settle = respx.post(f"{FAC_URL}/settle").mock(
            return_value=httpx.Response(200, json=SETTLED)
        )
        _mock_payment(config_dir)
        reply = _mock_reply()

        out = _fulfill_tool(config_dir).invoke(
            {"message_id": QUOTE_ID, "facilitator_url": FAC_URL}
        )

        assert settle.called and reply.called
        assert "status: payment-completed" in out
        assert "0xhash" in out
        assert "base-sepolia" in out  # v1 slug surfaced verbatim

    @respx.mock
    def test_fulfill_receipt_marker_is_status_rail_only(self, config_dir, other_keys):
        """The receipt's cleartext marker is {status, rail}-only — no economics leak."""
        mock_token()
        mock_agents()
        mock_recipient_keys(config_dir)
        respx.post(f"{FAC_URL}/verify").mock(
            return_value=httpx.Response(200, json={"isValid": True})
        )
        respx.post(f"{FAC_URL}/settle").mock(return_value=httpx.Response(200, json=SETTLED))
        _mock_payment(config_dir)
        reply = _mock_reply()

        _fulfill_tool(config_dir).invoke({"message_id": QUOTE_ID, "facilitator_url": FAC_URL})

        body = json.loads(reply.calls[0].request.content)
        marker = body["metadata"]["rine.x402"]
        assert marker == {"status": "payment-completed", "rail": "x402"}
        assert "500000" not in json.dumps(body["metadata"])

    @respx.mock
    def test_fulfill_invalid_verification_sends_failure_receipt(self, config_dir, other_keys):
        """A failed verification is a business outcome: settle skipped, payment-failed receipt."""
        mock_token()
        mock_agents()
        mock_recipient_keys(config_dir)
        respx.post(f"{FAC_URL}/verify").mock(
            return_value=httpx.Response(
                200, json={"isValid": False, "invalidReason": "insufficient_funds"}
            )
        )
        settle = respx.post(f"{FAC_URL}/settle").mock(return_value=httpx.Response(200, json={}))
        _mock_payment(config_dir)
        reply = _mock_reply()

        out = _fulfill_tool(config_dir).invoke(
            {"message_id": QUOTE_ID, "facilitator_url": FAC_URL}
        )

        assert not settle.called  # settle skipped on invalid
        assert reply.called
        assert "status: payment-failed" in out
        assert "insufficient_funds" in out

    @respx.mock
    def test_fulfill_wrong_message_type_returns_error_string_no_facilitator(
        self, config_dir, other_keys
    ):
        """A non-x402_payment message → a clear error string; the facilitator is never hit."""
        mock_token()
        mock_agents()
        mock_recipient_keys(config_dir)
        verify = respx.post(f"{FAC_URL}/verify")
        _mock_payment(config_dir, msg_type=MessageType.TASK_REQUEST)

        out = _fulfill_tool(config_dir).invoke(
            {"message_id": QUOTE_ID, "facilitator_url": FAC_URL}
        )

        assert not verify.called
        assert "Cannot fulfil" in out

    @respx.mock
    async def test_fulfill_arun_matches_run(self, config_dir, other_keys):
        """AC-8: the async path settles and reports payment-completed identically."""
        mock_token()
        mock_agents()
        mock_recipient_keys(config_dir)
        respx.post(f"{FAC_URL}/verify").mock(
            return_value=httpx.Response(200, json={"isValid": True})
        )
        respx.post(f"{FAC_URL}/settle").mock(return_value=httpx.Response(200, json=SETTLED))
        _mock_payment(config_dir)
        reply = _mock_reply()

        out = await _fulfill_tool(config_dir).ainvoke(
            {"message_id": QUOTE_ID, "facilitator_url": FAC_URL}
        )

        assert reply.called
        assert "status: payment-completed" in out
