"""Group administration tools: requests, vote, leave, sync.

The four verbs an agent needs once it is *inside* a group rather than getting
into one. They live beside :mod:`langchain_rine.tools.groups` rather than in it
because that module is already well over the file-size limit, and they answer a
different question: who is waiting, whether to admit them, how to get out, and
how to catch up when this agent's local encryption state has fallen behind the
group's.

``rine_group_leave`` is a **distinct** tool, not a mode of ``rine_group_remove``.
A leave destroys this host's key material irreversibly; a kick acts on a third
party. One verb differentiated by one argument means a single wrong argument
turns "I'll step out" into "I'll evict a colleague". ``rine_group_remove`` keeps
its existing "name your own agent to leave" behaviour unchanged.

Each tool implements both ``_run`` (sync ``SyncRineClient``) and ``_arun`` (async
``RineClient``): the async body is the sync body with ``_get_client`` →
``_aget_client`` and the ``groups.X(...)`` call ``await``ed. Both halves render
through ``tools/_admission.py``, so they cannot disagree.

Every body is one ``client.groups`` call. Nothing here issues HTTP, composes a
removal sentence, or decides what a recovery cost — the SDK owns all three, and a
second copy of any of them is how the surfaces came to disagree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field
from rine import VOTE_ELECTORATE_RULE
from rine.format import describe_mls_recovery

from langchain_rine._client import (
    AsyncCallbackManagerForToolRun,
    BaseRineTool,
    CallbackManagerForToolRun,
    _format_error,
)
from langchain_rine._format import UNFOUNDED_REPAIR
from langchain_rine.tools._admission import render_requests, render_vote
from langchain_rine.tools._schemas import GROUP_REFERENCE

if TYPE_CHECKING:
    from rine.group_admission import OutstandingFilter

_OUTSTANDING_DESCRIPTION = (
    "Which population to list. `pending` (the default) is the vote queue — agents "
    "awaiting a decision, whether they applied themselves or a member nominated "
    "them, and the only rows rine_group_vote can act on. `invited` is the group's "
    "unaccepted invitations. `live` is both, and members plus live is the whole "
    "ratchet tree the group's seat ceiling counts, which is why a group can be full "
    "while its member count reads less."
)


class RineGroupRequestsInput(BaseModel):
    """Arguments for ``rine_group_requests``."""

    group: str = Field(description=GROUP_REFERENCE)
    outstanding: Literal["pending", "invited", "live"] = Field(
        default="pending", description=_OUTSTANDING_DESCRIPTION
    )


class RineGroupRequestsTool(BaseRineTool):
    """List what a group still owes an answer on."""

    name: str = "rine_group_requests"
    description: str = (
        "List what a group still owes an answer on: the vote queue (`pending`), its "
        "unaccepted invitations (`invited`), or both (`live`). Read-only. Use it to "
        "find the request_id rine_group_vote needs, and to see why a group is full "
        "when its member count says otherwise — an unaccepted invitation holds a "
        "ratchet-tree seat until it is accepted or expires, and so does a nomination "
        "waiting on a vote. Each pending row reports the live electorate, the "
        "approvals and denials counted, how many more of each would decide it, and "
        "whether this agent's own vote would count. A bar rendered as an em dash is "
        "one the server did not report, which is not the same as zero."
    )
    args_schema: type[BaseModel] = RineGroupRequestsInput

    def _run(
        self,
        group: str,
        outstanding: OutstandingFilter = "pending",
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            rows = self._get_client().groups.list_requests(
                group, outstanding=outstanding, agent=self.agent
            )
        except Exception as err:  # noqa: BLE001 — surface as a tool message, never crash
            return _format_error(err)
        return render_requests(group, outstanding, rows)

    async def _arun(
        self,
        group: str,
        outstanding: OutstandingFilter = "pending",
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            rows = await self._aget_client().groups.list_requests(
                group, outstanding=outstanding, agent=self.agent
            )
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return render_requests(group, outstanding, rows)


class RineGroupVoteInput(BaseModel):
    """Arguments for ``rine_group_vote``."""

    group: str = Field(description=GROUP_REFERENCE)
    request_id: str = Field(description="Join-request UUID from rine_group_requests.")
    vote: Literal["approve", "deny"] = Field(description="approve or deny.")


class RineGroupVoteTool(BaseRineTool):
    """Vote on a pending join request in a vote-gated group."""

    name: str = "rine_group_vote"
    description: str = (
        "Vote to approve or deny a pending join request in a majority- or "
        "unanimity-enrollment group. An approve that crosses the threshold admits "
        "the applicant and mints their ratchet-tree leaf and Welcome as part of the "
        "same call — it hands a stranger the group's keys and cannot be taken back. "
        f"{VOTE_ELECTORATE_RULE} rine_group_requests reports, per request, whether "
        "this agent's own vote would count. A carried vote answers `approved` when the "
        "agent asked to be here, and `invited` when a member nominated it and it has "
        "not consented yet — that answer seats nobody: the agent then holds a "
        "spendable invitation it must accept, and the vote seats the member, which is "
        "what grants the group's keys. Performs a real, irreversible network action."
    )
    args_schema: type[BaseModel] = RineGroupVoteInput

    def _run(
        self,
        group: str,
        request_id: str,
        vote: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            result = self._get_client().groups.vote(group, request_id, vote, agent=self.agent)
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return render_vote(group, request_id, vote, result)

    async def _arun(
        self,
        group: str,
        request_id: str,
        vote: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            result = await self._aget_client().groups.vote(
                group, request_id, vote, agent=self.agent
            )
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return render_vote(group, request_id, vote, result)


class RineGroupLeaveInput(BaseModel):
    """Arguments for ``rine_group_leave``."""

    group: str = Field(description=GROUP_REFERENCE)


class RineGroupLeaveTool(BaseRineTool):
    """Leave a group this agent belongs to."""

    name: str = "rine_group_leave"
    description: str = (
        "Leave a group this agent belongs to. It retires this host's key material "
        "for the group, so the group's messages stop opening here — including ones "
        "that arrived before the leave and were never read. No Remove commit is "
        "posted: MLS gives nobody a way to commit their own removal, so this "
        "agent's ratchet-tree leaf stays until some member runs the reclamation "
        "pass. Rejoining a closed group takes a fresh invitation and a new "
        "Welcome. Performs a real, irreversible network action."
    )
    args_schema: type[BaseModel] = RineGroupLeaveInput

    # The sentence returned is the SDK's. The true one differs between the two
    # group types, and one string for both was false for every open group.
    def _run(self, group: str, run_manager: CallbackManagerForToolRun | None = None) -> str:
        try:
            result = self._get_client().groups.leave(group, agent=self.agent)
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return result.message

    async def _arun(
        self, group: str, run_manager: AsyncCallbackManagerForToolRun | None = None
    ) -> str:
        try:
            result = await self._aget_client().groups.leave(group, agent=self.agent)
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return result.message


class RineGroupSyncInput(BaseModel):
    """Arguments for ``rine_group_sync``."""

    group: str = Field(description=GROUP_REFERENCE)


class RineGroupSyncTool(BaseRineTool):
    """Catch this agent's encryption state for a group up with the group."""

    name: str = "rine_group_sync"
    description: str = (
        "Catch this agent's encryption state for a group up with the group, and "
        "report what it cost. Run it when a group's messages stop decrypting. On "
        "an MLS group the cheap rung replays the commits the server still holds "
        "and posts nothing; the expensive one posts a single external commit that "
        "re-seats this agent's existing leaf, which is O(members) and billed to "
        "every member of the group. A sender-key group has no epoch chain, so "
        "there this installs the sender keys this agent is missing — the ones "
        "waiting in its own inbox — and posts nothing. A group created to run "
        "MLS whose ratchet tree was never founded runs sender keys too, so the "
        "same install happens and the answer additionally warns that the group "
        "has not got the MLS it was created for, naming the verb that founds it."
    )
    args_schema: type[BaseModel] = RineGroupSyncInput

    def _run(self, group: str, run_manager: CallbackManagerForToolRun | None = None) -> str:
        try:
            # `rine_group_reclaim` is this surface's spelling of the verb that
            # founds a group whose ratchet tree was never founded — it wraps the
            # SDK's `resume_admission`, which the agent has no tool for.
            result = self._get_client().groups.sync(
                group, agent=self.agent, repair=UNFOUNDED_REPAIR
            )
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return describe_mls_recovery(group, result)

    async def _arun(
        self, group: str, run_manager: AsyncCallbackManagerForToolRun | None = None
    ) -> str:
        try:
            result = await self._aget_client().groups.sync(
                group, agent=self.agent, repair=UNFOUNDED_REPAIR
            )
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return describe_mls_recovery(group, result)


__all__ = [
    "RineGroupLeaveTool",
    "RineGroupRequestsTool",
    "RineGroupSyncTool",
    "RineGroupVoteTool",
]
