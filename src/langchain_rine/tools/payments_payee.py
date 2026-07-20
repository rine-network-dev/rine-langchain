"""``rine_fulfill`` — fulfil a received x402 payment as the payee.

A thin ``BaseTool`` over the ONE ``rine.x402.fulfill`` orchestrator (verify →
settle-or-synthetic-failure → receipt in-thread) — never a hand-composed ladder.
verify/settle are plain external HTTP through a caller-owned ``FacilitatorClient``,
never a rine route (the pure-metadata-relay posture). A failed verification is a
business outcome (a failure receipt is threaded so the payer reaches a terminal
state), not an exception. The cleartext receipt marker stays ``{status, rail}``-only.

``fulfill`` is async-only, so the sync ``_run`` opens a fresh async client inside
``asyncio.run`` (the sync path can't reuse a cached async client bound to another
loop); ``_arun`` uses the tool's async client directly.
"""

import asyncio
import os
from typing import TYPE_CHECKING

# BaseModel is a RUNTIME import — pydantic resolves the `args_schema: type[BaseModel]`
# field annotation at class definition, so this module does NOT use future annotations
# (mirroring messaging.py). SDK types used only in annotations are quoted + TYPE_CHECKING.
from pydantic import BaseModel
from rine.x402 import FacilitatorClient, X402Error
from rine.x402 import fulfill as _fulfill

from langchain_rine._client import (
    AsyncCallbackManagerForToolRun,
    BaseRineTool,
    CallbackManagerForToolRun,
    _format_error,
    abuild_client,
)
from langchain_rine.tools._schemas import RineFulfillInput

if TYPE_CHECKING:
    from rine import RineClient
    from rine.x402 import FulfillResult

_FACILITATOR_API_KEY_ENV = "RINE_X402_FACILITATOR_API_KEY"


def _render_fulfill(result: "FulfillResult") -> str:
    """Render a ``FulfillResult`` as a receipt-lifecycle status (no amounts)."""
    receipt = result.receipt
    if result.settlement is not None:
        completed = bool(result.settlement.get("success"))
        status = "payment-completed" if completed else "payment-failed"
        return (
            f"status: {status} — settled receipt {receipt.id} in conversation "
            f"{receipt.conversation_id} (network={result.settlement.get('network')}, "
            f"tx={result.settlement.get('transaction')})."
        )
    reason = result.verification.get("invalidReason") or "verification_failed"
    return (
        f"status: payment-failed — verification rejected ({reason}); sent failure receipt "
        f"{receipt.id} in conversation {receipt.conversation_id}."
    )


async def _do_fulfill(
    client: "RineClient",
    message_id: str,
    agent: str | None,
    add_marker: bool,
    fac: FacilitatorClient,
) -> "FulfillResult":
    """Read the payment and run the one ``fulfill`` orchestrator (verify→settle→receipt)."""
    payment = await client.read(message_id, agent=agent)
    return await _fulfill(client, payment, facilitator=fac, agent=agent, add_marker=add_marker)


class RineFulfillTool(BaseRineTool):
    """Fulfil a received x402 payment: verify → settle → receipt in-thread."""

    name: str = "rine_fulfill"
    description: str = (
        "Fulfil a received x402 payment (rine.v1.x402_payment) as the payee. Verifies the "
        "signed authorization and settles it on-chain through your configured facilitator "
        "(verify/settle are external HTTP, never a rine route), then sends the settlement "
        "receipt back in-thread. A failed verification is a business outcome — a failure "
        "receipt is threaded so the payer reaches a terminal state, not an error. Returns a "
        "typed `status:` line — payment-completed on a settled receipt, else payment-failed. "
        "Set facilitator_url for a self-hosted facilitator; an API key (if the preset needs "
        f"one) comes from the {_FACILITATOR_API_KEY_ENV} env var."
    )
    args_schema: type[BaseModel] = RineFulfillInput

    def _facilitator(self, facilitator: str, facilitator_url: str | None) -> FacilitatorClient:
        return FacilitatorClient(
            facilitator,
            base_url=facilitator_url,
            api_key=os.environ.get(_FACILITATOR_API_KEY_ENV),
        )

    def _run(
        self,
        message_id: str,
        facilitator: str = "cdp",
        facilitator_url: str | None = None,
        add_marker: bool = True,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        async def _go() -> "FulfillResult":
            async with (
                abuild_client(
                    config_dir=self.config_dir, api_url=self.api_url, agent=self.agent
                ) as client,
                self._facilitator(facilitator, facilitator_url) as fac,
            ):
                return await _do_fulfill(client, message_id, self.agent, add_marker, fac)

        try:
            result = asyncio.run(_go())
        except X402Error as err:
            return f"Cannot fulfil {message_id}: {err.message}"
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return _render_fulfill(result)

    async def _arun(
        self,
        message_id: str,
        facilitator: str = "cdp",
        facilitator_url: str | None = None,
        add_marker: bool = True,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            client = self._aget_client()
            async with self._facilitator(facilitator, facilitator_url) as fac:
                result = await _do_fulfill(client, message_id, self.agent, add_marker, fac)
        except X402Error as err:
            return f"Cannot fulfil {message_id}: {err.message}"
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return _render_fulfill(result)


__all__ = ["RineFulfillTool"]
