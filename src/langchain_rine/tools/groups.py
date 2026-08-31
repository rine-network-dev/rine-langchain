"""Group tools.

Six ``BaseTool``s over ``client.groups``: create, invite, remove, inspect, join,
invites. Group **send/read are NOT here** — they ride ``#``-prefix routing through
the messaging tools. New groups default to post-quantum MLS (X-Wing) and
``enable_mls`` on create is how an agent asks for a sender-key group instead;
``RineGroupInspectTool`` reports which scheme a group uses. Both MLS and sender-key
groups are end-to-end encrypted and readable/postable from here.
``RineGroupJoinTool`` is how an agent accepts an
invite it discovered via ``RineGroupInvitesTool`` (agent-scoped, across all groups);
joining installs the agent's local MLS state so it can immediately read and post
(see ``SyncGroups.join``/``AsyncGroups.join``).

Each tool implements both ``_run`` (sync ``SyncRineClient``) and ``_arun`` (async
``RineClient``): the async body is the sync body with ``_get_client`` →
``_aget_client`` and every ``groups.X(...)`` call ``await``ed. 🔴 A migration that
moves one half and not the other reads green under a suite that drives only one,
so the two halves are moved together and both are asserted.

A group reference is **not** matched here. This module used to carry its own
UUID/handle/bare-name matcher over ``groups.list()``; ``resolve_group`` is the SDK's
one ladder and answers with the group's whole record, so ``rine_group_inspect``
renders the row the resolve already read instead of re-reading it, and no local copy
of the rungs is left to drift from the SDK's.

``visibility`` carries no default and never will. Every surface used to invent
one and they disagreed, so the same call produced a group listed at
dir.rine.network from one and an unlisted one from another — and the listing also
flips the member-joined signal, so the two differed in who gets told about a join
as well as in who can find the group. It is required in the schema, so the model
is told at the point of choice, and the refusal when it is omitted anyway is the
SDK's own sentence rather than a second copy of it.

Admission is plural: ``members`` on create and ``agents_to_invite`` on invite each
take a batch and report one outcome per requested agent. Both tool bodies render
through ``tools/_admission.py``, so the sync and async halves cannot disagree.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator
from pydantic_core import PydanticCustomError
from rine import (
    VISIBILITY_REQUIRED_MESSAGE,
    VOTE_DURATION_MAX_HOURS,
    VOTE_DURATION_MIN_HOURS,
    VOTE_DURATION_RULE,
    GroupRead,
    GroupTarget,
    JoinRequestRead,
    JoinResult,
    NotFoundError,
)

# 🔴 Private module, deliberately. Labelling a group's seats with the org's agent
# handles is the SDK's own step — ``groups.list``/``get``/``update``/``create`` all
# run it on the rows they validate — and a row taken off ``GroupTarget`` is minted
# outside those four, so the connector runs it at the fifth site rather than
# letting ``your_agents`` fall back to raw UUIDs. ``rine`` exposes no public seam
# for it; ``@rine-network/sdk`` does (``client.agentHandles()``), and the two
# stacks should agree — recorded as an escalation, not worked around twice.
from rine._agent_handles import label_own_agents_async, label_own_agents_sync

from langchain_rine._client import (
    _GROUP_NOT_FOUND_LEAD,
    AsyncCallbackManagerForToolRun,
    BaseRineTool,
    CallbackManagerForToolRun,
    _format_error,
)
from langchain_rine._format import render_group
from langchain_rine.tools._admission import (
    INVITE_TARGET_REQUIRED,
    render_admission,
    render_created,
    render_invite_one,
    render_mls,
)
from langchain_rine.tools._schemas import GROUP_JOIN_REFERENCE, GROUP_REFERENCE


class RineGroupCreateInput(BaseModel):
    """Arguments for ``rine_group_create``."""

    name: str = Field(description="Group name (1-63 chars, DNS-safe).")
    enrollment: str = Field(
        default="closed",
        description="Enrollment policy: open, closed, majority, or unanimity.",
    )
    visibility: str = Field(
        description=(
            "Required, no default. `public` lists the group at dir.rine.network for "
            "anyone to find, and turns the member-joined signal off. `private` does "
            "not list it, and members are signalled when someone joins."
        )
    )
    description: str | None = Field(
        default=None,
        description=(
            "Optional durable house rules, readable by every member and every later "
            "arrival. NOT end-to-end encrypted — unlike a message body it is a plain "
            "column on the group row, readable by the rine server and by any agent the "
            "group is visible to. Put the rules here; send the secret."
        ),
    )
    # Both bounds and the sentence come from the SDK's ``vote_words``, the one
    # place that holds them: this copy called the window the TTL of a
    # nomination, and the server stamps it on a self-filed join request too.
    vote_duration_hours: int | None = Field(
        default=None,
        ge=VOTE_DURATION_MIN_HOURS,
        le=VOTE_DURATION_MAX_HOURS,
        description=VOTE_DURATION_RULE,
    )
    members: list[str] | None = Field(
        default=None,
        description=(
            "Agents to invite as the group is created: handles or UUIDs, reported one "
            "outcome per entry. A roster INVITES, it never seats — the new group has "
            "one member until each invitee joins — and an unaccepted invitation holds "
            "a ratchet-tree seat against the group's seat ceiling until it expires. A "
            "founding roster mints real invitations under every enrollment policy, "
            "including majority and unanimity where a later invite instead nominates: "
            "at founding your agent is the only member, so the vote would be a "
            "formality it casts against itself."
        ),
    )
    enable_mls: bool | None = Field(
        default=None,
        description=(
            "Whether the new group runs post-quantum MLS. Omit it for the default: the "
            "group is an MLS group and the SDK founds its state as part of this call. "
            "Set it false for a sender-key broadcast group instead — the value reaches "
            "the server, so the group is recorded as a sender-key group rather than as "
            "an MLS group that was never founded. An open-enrollment group is sender-key "
            "whatever this says; the server refuses MLS there."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _require_visibility(cls, data: Any) -> Any:
        """State the visibility refusal in rine's words, not pydantic's.

        This framework validates ``args_schema`` BEFORE the handler runs, so the
        SDK's own refusal can never be reached through the public entry point —
        the model would see ``1 validation error ... visibility Field required``,
        which is the dump this refusal exists to replace. ``mode="before"`` runs
        ahead of field validation, so the field stays REQUIRED in the JSON schema
        the model reads AND the refusal is still rine's sentence. Same shape as the
        server's own ``PydanticCustomError`` in ``group_schemas.py``.
        """
        if isinstance(data, dict) and not data.get("visibility"):
            raise PydanticCustomError("visibility_required", VISIBILITY_REQUIRED_MESSAGE)
        return data


class RineGroupCreateTool(BaseRineTool):
    """Create a new end-to-end-encrypted coordination group (post-quantum MLS by default)."""

    name: str = "rine_group_create"
    description: str = (
        "Create a new end-to-end-encrypted coordination group. Performs a real, "
        "irreversible network action. Your agent owns and administers the group. New "
        "groups are post-quantum MLS (X-Wing) by default; open-enrollment groups use "
        "sender keys. Either way your agent can read and post here. `visibility` is "
        "required and has no default: `public` lists the group at dir.rine.network and "
        "turns the member-joined signal off, `private` does neither. Name `members` to "
        "invite agents as the group is created — a founding roster mints real "
        "invitations whatever the enrollment policy, because at founding your agent is "
        "the only member and a vote would be a formality it casts against itself."
    )
    args_schema: type[BaseModel] = RineGroupCreateInput

    def _run(
        self,
        name: str,
        visibility: str | None = None,
        enrollment: str = "closed",
        description: str | None = None,
        vote_duration_hours: int | None = None,
        members: list[str] | None = None,
        enable_mls: bool | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        # `visibility` is required in the schema and passed straight through. The
        # SDK owns the refusal — restating it here would be a second copy
        # of a sentence with no gate keeping the two in step.
        try:
            group = self._get_client().groups.create(
                name,
                description=description,
                enrollment=enrollment,
                visibility=visibility,
                members=members,
                vote_duration_hours=vote_duration_hours,
                enable_mls=enable_mls,
                agent=self.agent,
            )
        except Exception as err:  # noqa: BLE001 — surface as a tool message, never crash
            return _format_error(err)
        return render_created(group)

    async def _arun(
        self,
        name: str,
        visibility: str | None = None,
        enrollment: str = "closed",
        description: str | None = None,
        vote_duration_hours: int | None = None,
        members: list[str] | None = None,
        enable_mls: bool | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            group = await self._aget_client().groups.create(
                name,
                description=description,
                enrollment=enrollment,
                visibility=visibility,
                members=members,
                vote_duration_hours=vote_duration_hours,
                enable_mls=enable_mls,
                agent=self.agent,
            )
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return render_created(group)


class RineGroupInviteInput(BaseModel):
    """Arguments for ``rine_group_invite``."""

    group: str = Field(description=GROUP_REFERENCE)
    agent_to_invite: str | None = Field(
        default=None,
        description=(
            "One agent to invite: handle or UUID. On a majority- or "
            "unanimity-enrollment group this nominates rather than invites. Name "
            "exactly one of agent_to_invite or agents_to_invite."
        ),
    )
    agents_to_invite: list[str] | None = Field(
        default=None,
        description=(
            "Several agents to invite in one call: handles or UUIDs, reported one "
            "outcome per entry in the order given. A batch drops what it cannot "
            "admit — a revoked agent, one already a member — instead of refusing "
            "everyone. On a majority- or unanimity-enrollment group each outcome is "
            "`nominated`, not `invited`. Name exactly one of agent_to_invite or "
            "agents_to_invite."
        ),
    )
    message: str | None = Field(default=None, description="Optional invitation message.")


class RineGroupInviteTool(BaseRineTool):
    """Invite one agent, or a batch, into a group your agent administers."""

    name: str = "rine_group_invite"
    description: str = (
        "Invite one agent (agent_to_invite) or several at once (agents_to_invite) "
        "into a group your agent administers. On a closed group this mints an "
        "invitation the agent spends by joining. On a majority- or "
        "unanimity-enrollment group it NOMINATES instead: it files a join request "
        "the group's electorate decides, your own invite counts as one approval, and "
        "nobody is seated until the vote carries. A batch reports one outcome per "
        "agent and drops what it cannot admit rather than refusing everyone. Performs "
        "a real, irreversible network action."
    )
    args_schema: type[BaseModel] = RineGroupInviteInput

    def _run(
        self,
        group: str,
        agent_to_invite: str | None = None,
        agents_to_invite: list[str] | None = None,
        message: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        batch = agents_to_invite or []
        if (agent_to_invite is None) == (not batch):
            return _format_error(ValueError(INVITE_TARGET_REQUIRED))
        try:
            if agent_to_invite is not None:
                result = self._get_client().groups.invite(
                    group, agent_to_invite, message=message, agent=self.agent
                )
                return render_invite_one(group, agent_to_invite, result)
            report = self._get_client().groups.invite_many(
                group, batch, message=message, agent=self.agent
            )
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return f"Invited into {group}:\n{render_admission(report)}{render_mls(report.mls)}"

    async def _arun(
        self,
        group: str,
        agent_to_invite: str | None = None,
        agents_to_invite: list[str] | None = None,
        message: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        batch = agents_to_invite or []
        if (agent_to_invite is None) == (not batch):
            return _format_error(ValueError(INVITE_TARGET_REQUIRED))
        try:
            if agent_to_invite is not None:
                result = await self._aget_client().groups.invite(
                    group, agent_to_invite, message=message, agent=self.agent
                )
                return render_invite_one(group, agent_to_invite, result)
            report = await self._aget_client().groups.invite_many(
                group, batch, message=message, agent=self.agent
            )
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return f"Invited into {group}:\n{render_admission(report)}{render_mls(report.mls)}"


class RineGroupRemoveInput(BaseModel):
    """Arguments for ``rine_group_remove``."""

    group: str = Field(description=GROUP_REFERENCE)
    agent_id: str = Field(description="UUID of the member to remove.")


class RineGroupRemoveTool(BaseRineTool):
    """Remove a member from a group your agent administers, or leave it."""

    name: str = "rine_group_remove"
    description: str = (
        "Remove a member from a group your agent administers, or leave a group by "
        "naming your own agent. On an MLS group this posts a real Remove commit "
        "that takes the member's ratchet-tree leaf with it, so it is an O(members) "
        "operation that can fail; an open group has no cryptographic eviction — the "
        "server stops delivering to them, and what bounds their reach into later "
        "traffic is each remaining member rotating their sender key on the next "
        "send. A kick takes nothing back: messages already delivered still open "
        "for everyone, the removed member included. A leave is different — it "
        "retires this host's key material for the group, so the group's messages "
        "stop opening here, including ones that arrived before the leave and were "
        "never read. Performs a real, irreversible network action."
    )
    args_schema: type[BaseModel] = RineGroupRemoveInput

    def _run(
        self,
        group: str,
        agent_id: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        # The sentence is the SDK's, not this surface's. The true one differs
        # between the two group types AND between the two verbs — a leave says
        # what local key material is now gone, a kick names the epoch a Remove
        # commit landed at — and the one string this tool used to print for all
        # four cases was false for every open group.
        try:
            result = self._get_client().groups.remove_member(group, agent_id, agent=self.agent)
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return result.message

    async def _arun(
        self,
        group: str,
        agent_id: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            result = await self._aget_client().groups.remove_member(
                group, agent_id, agent=self.agent
            )
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return result.message


def _group_from_row(target: GroupTarget) -> GroupRead:
    """The group the resolve already read — never a second read of it.

    ``GroupTarget.row`` is the row whichever rung answered had in hand, so
    ``rine_group_inspect`` pays exactly the reads the resolve paid and no more.
    Handing the id back to ``groups.get`` would turn the one read this
    verb has always cost into two.

    ``row`` is nullable for the JOIN ladder alone, whose target is by definition
    a group ``GET /groups`` cannot describe. Every rung of the seated ladder
    :meth:`rine.SyncRineClient.resolve_group` walks sets it, so a ``None`` here
    is a broken SDK contract rather than a caller's mistake — refused as one,
    rather than papered over with a second read.

    🔴 A group minted here is one **the SDK did not mint**, so its seats carry no
    handle map and ``your_agents`` would degrade to raw UUIDs. The caller labels
    it (``label_own_agents_*``), which is what ``groups.list``/``get`` do for the
    rows they validate; the map is memoised on the client, so a labelled row
    costs the same ``GET /agents`` the list path already paid.
    """
    if target.row is None:
        # 🔴 The lead is the CONSTANT, never a second literal: ``_format_error``
        # routes a group 404 by ``detail.startswith(_GROUP_NOT_FOUND_LEAD)``, and
        # the contract gate pins exactly one copy of the phrase per package — the
        # one in ``_client``. A retyped literal here would survive the next reword
        # untouched by the gate, and this refusal would then be routed to the
        # AGENT directory, which cannot resolve a group handle.
        raise NotFoundError(f"{_GROUP_NOT_FOUND_LEAD}: {target.spelling}")
    return GroupRead.model_validate(target.row)


class RineGroupInspectInput(BaseModel):
    """Arguments for ``rine_group_inspect``."""

    group: str = Field(description=GROUP_REFERENCE)


class RineGroupInspectTool(BaseRineTool):
    """Show a group's details + its encryption mode."""

    name: str = "rine_group_inspect"
    description: str = (
        "Show a group's details and encryption mode: post-quantum MLS, an MLS group "
        "still initialising, or sender-key. Every mode is end-to-end encrypted and "
        "readable/postable from here."
    )
    args_schema: type[BaseModel] = RineGroupInspectInput

    def _run(self, group: str, run_manager: CallbackManagerForToolRun | None = None) -> str:
        try:
            client = self._get_client()
            found = _group_from_row(client.resolve_group(group, agent=self.agent))
            label_own_agents_sync(client, [found])
            return render_group(found)
        except Exception as err:  # noqa: BLE001
            return _format_error(err)

    async def _arun(
        self, group: str, run_manager: AsyncCallbackManagerForToolRun | None = None
    ) -> str:
        try:
            client = self._aget_client()
            found = _group_from_row(await client.resolve_group(group, agent=self.agent))
            await label_own_agents_async(client, [found])
            return render_group(found)
        except Exception as err:  # noqa: BLE001
            return _format_error(err)


class RineGroupJoinInput(BaseModel):
    """Arguments for ``rine_group_join``."""

    group: str = Field(description=GROUP_JOIN_REFERENCE)
    message: str | None = Field(default=None, description="Optional message for join request.")


class RineGroupJoinTool(BaseRineTool):
    """Join a group, or submit a join request if the group requires a vote."""

    name: str = "rine_group_join"
    description: str = (
        "Join a group. Open groups: instant join. Approval-gated groups: submits a request "
        "that members vote on. Called on a nomination a member filed for your agent, it "
        "records your agent's consent to that request and answers the row — it does not "
        "join the group, because the electorate still has to carry the vote. Performs a "
        "real, irreversible network action."
    )
    args_schema: type[BaseModel] = RineGroupJoinInput

    def _run(
        self,
        group: str,
        message: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            result = self._get_client().groups.join(group, message=message, agent=self.agent)
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return _render_join_result(group, result)

    async def _arun(
        self,
        group: str,
        message: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            result = await self._aget_client().groups.join(
                group, message=message, agent=self.agent
            )
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return _render_join_result(group, result)


def _render_join_result(group: str, result: JoinResult) -> str:
    """Render a ``JoinedResult``/``PendingJoinResult``, discriminated on ``.status``.

    A pending answer is one of two different events, told apart by ``invited_by``
    on the request row: this agent applied, or a member nominated it and this
    call recorded its consent. Both wait on the same vote, but only the
    second was somebody else's idea, and an operator that cannot tell them apart
    cannot tell why its agent is waiting.
    """
    if result.status == "joined":
        return f"Joined {group} (member id {result.member.agent_id})."
    if result.request.invited_by is not None:
        return (
            f"Consent recorded for the nomination into {group} (request "
            f"{result.request.id}). A member nominated your agent; the group's "
            f"electorate still has to carry the vote."
        )
    return (
        f"Requested to join {group} (request {result.request.id}, "
        f"status {result.request.status}); awaiting a member vote."
    )


class RineGroupInvitesInput(BaseModel):
    """Arguments for ``rine_group_invites`` (none required)."""


class RineGroupInvitesTool(BaseRineTool):
    """List open group invites addressed to your agent, across all groups."""

    name: str = "rine_group_invites"
    description: str = (
        "List the open offers addressed to your agent across all groups. Two kinds, "
        "told apart by status: an `invited` row is an invitation your agent spends "
        "with rine_group_join, and a `pending` row is a nomination a member filed on "
        "its behalf, which rine_group_join consents to and the group's electorate "
        "decides. This is the only place a nomination is visible to the agent it names."
    )
    args_schema: type[BaseModel] = RineGroupInvitesInput

    def _run(self, run_manager: CallbackManagerForToolRun | None = None) -> str:
        try:
            invites = self._get_client().groups.list_invites(agent=self.agent)
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return _render_invites(invites)

    async def _arun(self, run_manager: AsyncCallbackManagerForToolRun | None = None) -> str:
        try:
            invites = await self._aget_client().groups.list_invites(agent=self.agent)
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return _render_invites(invites)


def _render_invites(invites: list[JoinRequestRead]) -> str:
    """One line per open offer, naming which of the two kinds it is.

    ``status`` is the whole discriminator: ``invited`` is a voucher to spend,
    ``pending`` is a nomination to consent to. The clause spells that out because
    a bare token under the head "Pending invites" reads as one kind of thing.
    """
    if not invites:
        return "No pending group invites."
    lines = [
        f"- {inv.group_handle or inv.group_id} (request {inv.id}, status {inv.status}"
        f", {_offer_kind(inv.status)})"
        for inv in invites
    ]
    return "Pending invites:\n" + "\n".join(lines)


def _offer_kind(status: str) -> str:
    """What the agent named by this row has to do about it."""
    if status == "pending":
        return "a nomination to consent to; the electorate decides it"
    return "an invitation to spend"


__all__ = [
    "RineGroupCreateTool",
    "RineGroupInviteTool",
    "RineGroupRemoveTool",
    "RineGroupInspectTool",
    "RineGroupJoinTool",
    "RineGroupInvitesTool",
]
