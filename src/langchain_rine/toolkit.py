"""``RineToolkit`` — a curated set of rine tools sharing one client.

A ``langchain_core.tools.BaseToolkit`` that builds the requested subset of the 25
tools (via ``include``) and injects ONE lazily-built ``SyncRineClient`` into every
returned tool (a single ``build_client`` call per toolkit instance). The
``config_dir``/``api_url``/``agent`` overrides propagate to each tool so they all
resolve credentials/keys from the same place.

Import is side-effect-free: the shared client is built inside
``get_tools()``, never at import or construction. Individual tool classes stay
exported from ``langchain_rine`` for the "attach only what you need" opt-in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import Field, PrivateAttr

from langchain_rine._client import build_client
from langchain_rine.tools import (
    RineDiscoverGroupsTool,
    RineDiscoverTool,
    RineFulfillTool,
    RineGroupCreateTool,
    RineGroupInspectTool,
    RineGroupInvitesTool,
    RineGroupInviteTool,
    RineGroupJoinTool,
    RineGroupLeaveTool,
    RineGroupReclaimTool,
    RineGroupRemoveTool,
    RineGroupRequestsTool,
    RineGroupRosterTool,
    RineGroupsTool,
    RineGroupSyncTool,
    RineGroupVoteTool,
    RineInboxTool,
    RineInspectTool,
    RinePayTool,
    RineReadTool,
    RineReplyTool,
    RineSendAndWaitTool,
    RineSendTool,
    RineThreadTool,
    RineWhoamiTool,
)

if TYPE_CHECKING:
    from rine import SyncRineClient

_MESSAGING = (
    RineSendTool,
    RineSendAndWaitTool,
    RineInboxTool,
    RineReadTool,
    RineReplyTool,
    RineThreadTool,
)
# `rine_whoami` is discovery's odd one out — it is the only authenticated tool in
# the domain — but it answers the same question the domain exists for: who is out
# there, and who am I to them.
_DISCOVERY = (RineDiscoverTool, RineInspectTool, RineDiscoverGroupsTool, RineWhoamiTool)
_GROUPS = (
    RineGroupsTool,
    RineGroupCreateTool,
    RineGroupInviteTool,
    RineGroupRemoveTool,
    RineGroupInspectTool,
    RineGroupRosterTool,
    RineGroupJoinTool,
    RineGroupInvitesTool,
    RineGroupRequestsTool,
    RineGroupVoteTool,
    RineGroupLeaveTool,
    RineGroupSyncTool,
    RineGroupReclaimTool,
)
# Fund-moving x402 tools live in their OWN domain so `include="messaging"` never silently
# hands an agent a wallet-spending tool — attaching payments is an explicit opt-in.
_PAYMENTS = (RinePayTool, RineFulfillTool)
_DOMAINS = {
    "messaging": _MESSAGING,
    "discovery": _DISCOVERY,
    "groups": _GROUPS,
    "payments": _PAYMENTS,
}


class RineToolkit(BaseToolkit):
    """Curated subset of rine tools sharing one lazily-built client.

    ``rich_returns`` is RESERVED for v0.2 and currently has NO effect: tools always
    return plain strings. The structured ``content_and_artifact`` return it will gate
    stays unwired while langchain#29874 is live on the pinned ``langchain-core`` — a
    ``content_and_artifact`` tool invoked with an ``id=None``
    ToolCall silently drops the artifact, so the string-return default is the only
    conformance-safe behaviour for now. The field is kept for forward API stability.
    """

    config_dir: str | None = None
    api_url: str | None = None
    agent: str | None = None
    include: Literal["messaging", "discovery", "groups", "payments", "all"] | list[str] = "all"
    rich_returns: bool = Field(
        default=False,
        description=(
            "RESERVED for v0.2 — currently INERT (no effect). Tools always return plain "
            "strings; the structured content_and_artifact return is gated until "
            "langchain#29874 is resolved on the pinned langchain-core. Setting "
            "True does not change behaviour today."
        ),
    )
    _client: SyncRineClient | None = PrivateAttr(default=None)

    def get_tools(self) -> list[BaseTool]:
        """Build the curated tool subset, all sharing one client.

        ``include="all"`` returns all 25; a single domain name returns that domain's
        tools; a list returns the union of the named domains. Every tool receives the
        toolkit's overrides plus the single shared ``SyncRineClient`` (one
        ``build_client`` call across the whole toolkit). ``rich_returns`` stays UNWIRED
        in v0.1 (string-return default; langchain#29874 still live).
        """
        if self.include == "all":
            classes = [c for group in _DOMAINS.values() for c in group]
        else:
            names = [self.include] if isinstance(self.include, str) else self.include
            classes = [c for n in names for c in _DOMAINS[n]]
        shared = self._shared_client()
        tools: list[BaseTool] = []
        for cls in classes:
            tool = cls(config_dir=self.config_dir, api_url=self.api_url, agent=self.agent)
            tool._client = shared  # one build_client shared by every tool
            tools.append(tool)
        return tools

    def _shared_client(self) -> SyncRineClient:
        """Build (once) the client every tool from this toolkit shares."""
        if self._client is None:
            self._client = build_client(
                config_dir=self.config_dir, api_url=self.api_url, agent=self.agent
            )
        return self._client


def get_tools(**kwargs: object) -> list[BaseTool]:
    """Module-level convenience: ``RineToolkit(**kwargs).get_tools()``."""
    return RineToolkit(**kwargs).get_tools()  # type: ignore[arg-type]
