"""Opt-in agent/chain-lifecycle callback handler (SPEC §8).

``RineCallbackHandler`` is a ``langchain_core.callbacks.BaseCallbackHandler`` that
sends a best-effort rine message on selected LangChain lifecycle events. It is the
proof that *native* beats *MCP*: a callback wires into the Python process; an MCP
server physically cannot.

Activation is opt-in by instantiation — the developer constructs the handler
(``RineCallbackHandler(to="ops@acme")``) and threads it through
``config={"callbacks": [handler]}``. The ``SyncRineClient`` is built lazily on the
first fired event (never at import/``__init__``), keeping import side-effect-free
(invariant I-1).

A notification failure must NEVER crash an agent run: ``_notify`` swallows its own
exceptions and logs at debug (AC-13). Each overridden lifecycle method fires only
when its selector is present in ``on``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import BaseCallbackHandler

from langchain_rine._client import build_client

if TYPE_CHECKING:
    from uuid import UUID

    from rine import SyncRineClient

logger = logging.getLogger("langchain_rine.callbacks")

_MAX_SUMMARY = 500


class RineCallbackHandler(BaseCallbackHandler):
    """Send a rine message on selected LangChain lifecycle events (SPEC §8).

    Args:
        to: rine handle/UUID to notify.
        config_dir: Override the SDK config dir (else the SDK resolution chain).
        api_url: Override the API base URL.
        agent: Agent name/handle/UUID to send as (multi-agent orgs).
        on: Which lifecycle events fire — any of ``"agent_finish"``, ``"chain_end"``,
            ``"chain_error"``, ``"agent_action"``. Defaults to agent-finish + chain-error.
    """

    def __init__(
        self,
        to: str,
        *,
        config_dir: str | None = None,
        api_url: str | None = None,
        agent: str | None = None,
        on: tuple[str, ...] = ("agent_finish", "chain_error"),
    ) -> None:
        """Configure the handler. No client is built here (I-1)."""
        self._to = to
        self._config_dir = config_dir
        self._api_url = api_url
        self._agent = agent
        self._on = on
        self._client: SyncRineClient | None = None

    def _get_client(self) -> SyncRineClient:
        """Build (once) the shared ``SyncRineClient`` from the same chain as tools."""
        if self._client is None:
            self._client = build_client(
                config_dir=self._config_dir, api_url=self._api_url, agent=self._agent
            )
        return self._client

    def _notify(self, text: str) -> None:
        """Best-effort send. A notification failure must not crash the run (AC-13)."""
        try:
            self._get_client().send(
                self._to, {"text": text[:_MAX_SUMMARY]}, message_type="rine.v1.text"
            )
        except Exception:  # noqa: BLE001 — never propagate out of a callback
            logger.debug("rine notification send failed", exc_info=True)

    # --- selector → LC handler-method map (SPEC §8; signatures verified PHASE0 §6) ---
    def on_agent_finish(
        self, finish: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> None:
        """Notify on agent finish, carrying its final output."""
        if "agent_finish" in self._on:
            output = str(finish.return_values.get("output", ""))
            self._notify(f"agent finished: {output}")

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Notify on chain end, carrying its outputs."""
        if "chain_end" in self._on:
            self._notify(f"chain ended: {outputs}")

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Notify on chain error, carrying the error string."""
        if "chain_error" in self._on:
            self._notify(f"chain error: {error}")

    def on_agent_action(
        self, action: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> None:
        """Notify on agent action, naming the tool invoked."""
        if "agent_action" in self._on:
            self._notify(f"agent action: {getattr(action, 'tool', '')}")


__all__ = ["RineCallbackHandler"]
