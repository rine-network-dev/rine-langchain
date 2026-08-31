"""Retiring the ratchet-tree leaves a group no longer owes anyone.

An invitation takes a leaf the moment its Welcome is minted, and the reaper that
frees the **seat** when the invitation lapses does not take the leaf with it. A
self-leave leaves the same residue: MLS gives nobody a way to commit their own
removal. So a group that churns no-show invitations makes every later Welcome
and every later commit heavier for every real member, without bound.

Reclamation is what bounds the tree — expiry frees the seat only. It is never
automatic, because the server holds no MLS keys and cannot mint the Remove
commit, so a group nobody runs this on keeps accumulating leaves. Posting the
commit takes membership and no role beyond it, so any member may run it; what
holds it back is that each Remove is O(members) and billed to all of them.

The tool seats before it reclaims, which is why it runs the SDK's resume verb
with ``reclaim=True`` rather than a bare reclamation: a pass that finds a member
holding no leaf declines whole rather than reclaim a leaf somebody is entitled
to, and seating first is what clears that case instead of leaving an agent with
a refusal and no remedy.

``_run`` and ``_arun`` are the same body against the sync and async clients, as
everywhere else in this package.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from rine import describe_mls_admission

from langchain_rine._client import (
    AsyncCallbackManagerForToolRun,
    BaseRineTool,
    CallbackManagerForToolRun,
    _format_error,
)
from langchain_rine.tools._schemas import GROUP_REFERENCE

_RECLAIM_DESCRIPTION = (
    "Retire the ratchet-tree leaves an MLS group no longer owes anyone — the ones "
    "left behind by invitations nobody accepted and by members who left. Run it on "
    "a group whose Welcomes and commits have grown heavy after a lot of churn; "
    "nothing does it automatically, because the server holds no MLS keys. It first "
    "seats every member and invitee that is not seated yet, then posts one Remove "
    "commit per orphaned leaf — each O(members) and billed to every member of the "
    "group, which is the reason to run it deliberately rather than often. Any "
    "member may run it. It is also this surface's verb for founding a group "
    "created to run MLS whose ratchet tree was never founded — the state "
    "`rine_group_sync` warns about and names this tool for. There it founds the "
    "group's MLS state and posts no Remove commit at all, so none of the cost "
    "above applies. An agent whose invitation lapsed and who is re-invited "
    "between the tree read and the Remove commit loses the fresh leaf too, and "
    "needs re-inviting. A sender-key group has no ratchet tree and is refused. "
    "Performs a real, irreversible network action."
)


class RineGroupReclaimInput(BaseModel):
    """Arguments for ``rine_group_reclaim``."""

    group: str = Field(description=GROUP_REFERENCE)


class RineGroupReclaimTool(BaseRineTool):
    """Retire the ratchet-tree leaves no member and no live invitation accounts for."""

    name: str = "rine_group_reclaim"
    description: str = _RECLAIM_DESCRIPTION
    args_schema: type[BaseModel] = RineGroupReclaimInput

    # The sentence returned is the SDK's, and it reports both halves: what
    # seating achieved, and what reclamation found. A pass that ran and found
    # nothing still says so — "nobody asked" and "there was nothing there" are
    # different facts and only one means the tree is clean.
    def _run(self, group: str, run_manager: CallbackManagerForToolRun | None = None) -> str:
        try:
            result = self._get_client().groups.resume_admission(
                group, agent=self.agent, reclaim=True
            )
        except Exception as err:  # noqa: BLE001 — surface as a tool message, never crash
            return _format_error(err)
        return describe_mls_admission(result)

    async def _arun(
        self, group: str, run_manager: AsyncCallbackManagerForToolRun | None = None
    ) -> str:
        try:
            result = await self._aget_client().groups.resume_admission(
                group, agent=self.agent, reclaim=True
            )
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return describe_mls_admission(result)


__all__ = ["RineGroupReclaimTool"]
