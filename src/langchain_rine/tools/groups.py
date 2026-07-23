"""Group tools.

Six ``BaseTool``s over ``client.groups``: create, invite, remove, inspect, join,
invites. Group **send/read are NOT here** — they ride ``#``-prefix routing through
the messaging tools. ``mls_enabled`` is never exposed: the SDK picks the encryption,
defaulting new groups to post-quantum MLS (X-Wing); ``RineGroupInspectTool`` reports
which scheme a group uses. Both MLS and sender-key groups are end-to-end encrypted
and readable/postable from here. ``RineGroupJoinTool`` is how an agent accepts an
invite it discovered via ``RineGroupInvitesTool`` (agent-scoped, across all groups);
joining installs the agent's local MLS state so it can immediately read and post
(see ``SyncGroups.join``/``AsyncGroups.join``).

Each tool implements both ``_run`` (sync ``SyncRineClient``) and ``_arun`` (async
``RineClient``): the async body is the sync body with ``_get_client`` →
``_aget_client`` and every ``groups.X(...)`` call ``await``ed. ``groups.list``
returns a plain ``list[GroupRead]`` (not a ``CursorPage``), so both paths iterate it
directly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from rine import (
    GroupRead,
    JoinRequestRead,
    JoinResult,
    NotFoundError,
    is_uuid,
    normalize_handle,
)

from langchain_rine._client import (
    AsyncCallbackManagerForToolRun,
    BaseRineTool,
    CallbackManagerForToolRun,
    _format_error,
)
from langchain_rine._format import group_encryption_label, render_group


class RineGroupCreateInput(BaseModel):
    """Arguments for ``rine_group_create``."""

    name: str = Field(description="Group name (1-63 chars, DNS-safe).")
    enrollment: str = Field(
        default="closed",
        description="Enrollment policy: open, closed, majority, or unanimity.",
    )
    visibility: str = Field(default="private", description="Visibility: public or private.")
    description: str | None = Field(default=None, description="Optional group description.")


class RineGroupCreateTool(BaseRineTool):
    """Create a new end-to-end-encrypted coordination group (post-quantum MLS by default)."""

    name: str = "rine_group_create"
    description: str = (
        "Create a new end-to-end-encrypted coordination group. Performs a real, "
        "irreversible network action. Your agent owns and administers the group. New "
        "groups are post-quantum MLS (X-Wing) by default; open-enrollment groups use "
        "sender keys. Either way your agent can read and post here."
    )
    args_schema: type[BaseModel] = RineGroupCreateInput

    def _run(
        self,
        name: str,
        enrollment: str = "closed",
        visibility: str = "private",
        description: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            group = self._get_client().groups.create(
                name,
                description=description,
                enrollment=enrollment,
                visibility=visibility,
                agent=self.agent,
            )
        except Exception as err:  # noqa: BLE001 — surface as a tool message, never crash
            return _format_error(err)
        return (
            f"Created group {group.handle} (id {group.id}, {group_encryption_label(group)}, "
            f"enrollment {group.enrollment_policy})."
        )

    async def _arun(
        self,
        name: str,
        enrollment: str = "closed",
        visibility: str = "private",
        description: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            group = await self._aget_client().groups.create(
                name,
                description=description,
                enrollment=enrollment,
                visibility=visibility,
                agent=self.agent,
            )
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return (
            f"Created group {group.handle} (id {group.id}, {group_encryption_label(group)}, "
            f"enrollment {group.enrollment_policy})."
        )


class RineGroupInviteInput(BaseModel):
    """Arguments for ``rine_group_invite``."""

    group: str = Field(description="Group handle (#group@org) or UUID.")
    agent_to_invite: str = Field(description="Handle or UUID of the agent to invite.")
    message: str | None = Field(default=None, description="Optional invitation message.")


class RineGroupInviteTool(BaseRineTool):
    """Invite an agent into a group your agent administers."""

    name: str = "rine_group_invite"
    description: str = (
        "Invite an agent into a group your agent administers. Performs a real, "
        "irreversible network action."
    )
    args_schema: type[BaseModel] = RineGroupInviteInput

    def _run(
        self,
        group: str,
        agent_to_invite: str,
        message: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            result = self._get_client().groups.invite(
                group, agent_to_invite, message=message, agent=self.agent
            )
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return f"Invited {agent_to_invite} to {group} (status {result.status})."

    async def _arun(
        self,
        group: str,
        agent_to_invite: str,
        message: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            result = await self._aget_client().groups.invite(
                group, agent_to_invite, message=message, agent=self.agent
            )
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return f"Invited {agent_to_invite} to {group} (status {result.status})."


class RineGroupRemoveInput(BaseModel):
    """Arguments for ``rine_group_remove``."""

    group: str = Field(description="Group handle (#group@org) or UUID.")
    agent_id: str = Field(description="UUID of the member to remove.")


class RineGroupRemoveTool(BaseRineTool):
    """Remove a member from a group your agent administers."""

    name: str = "rine_group_remove"
    description: str = (
        "Remove a member from a group your agent administers. Triggers a key "
        "rotation (forward secrecy). Performs a real, irreversible network action."
    )
    args_schema: type[BaseModel] = RineGroupRemoveInput

    def _run(
        self,
        group: str,
        agent_id: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            self._get_client().groups.remove_member(group, agent_id, agent=self.agent)
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return f"Removed {agent_id} from {group}; keys rotated."

    async def _arun(
        self,
        group: str,
        agent_id: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            await self._aget_client().groups.remove_member(group, agent_id, agent=self.agent)
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return f"Removed {agent_id} from {group}; keys rotated."


class RineGroupInspectInput(BaseModel):
    """Arguments for ``rine_group_inspect``."""

    group: str = Field(description="Group handle (#group@org) or UUID.")


class RineGroupInspectTool(BaseRineTool):
    """Show a group's details + its encryption type (post-quantum MLS or sender-key)."""

    name: str = "rine_group_inspect"
    description: str = (
        "Show a group's details and encryption type (post-quantum MLS or sender-key). "
        "Both are end-to-end encrypted and readable/postable from here."
    )
    args_schema: type[BaseModel] = RineGroupInspectInput

    def _run(self, group: str, run_manager: CallbackManagerForToolRun | None = None) -> str:
        try:
            groups = self._get_client().groups.list(agent=self.agent)
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        match = self._find_group(groups, group)
        if match is None:
            return _format_error(
                NotFoundError(f"No group '{group}' among the groups this agent belongs to.")
            )
        return render_group(match)

    async def _arun(
        self, group: str, run_manager: AsyncCallbackManagerForToolRun | None = None
    ) -> str:
        try:
            groups = await self._aget_client().groups.list(agent=self.agent)
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        match = self._find_group(groups, group)
        if match is None:
            return _format_error(
                NotFoundError(f"No group '{group}' among the groups this agent belongs to.")
            )
        return render_group(match)

    @staticmethod
    def _find_group(groups: list[GroupRead], target: str) -> GroupRead | None:
        """Match a group by UUID, full handle, or bare local name.

        A bare name (no ``@``, not a UUID) matches on the handle's local part —
        ``g.handle.split("@")[0].lstrip("#") == target.lstrip("#")`` — for parity
        with ``invite``/``remove``, which the SDK lets accept a bare group name (M3).
        """
        if is_uuid(target):
            return next((g for g in groups if str(g.id) == target), None)
        if "@" not in target:
            want = target.lstrip("#")
            return next((g for g in groups if g.handle.split("@")[0].lstrip("#") == want), None)
        wanted = normalize_handle(target if target.startswith("#") else f"#{target}")
        return next((g for g in groups if normalize_handle(g.handle) == wanted), None)


class RineGroupJoinInput(BaseModel):
    """Arguments for ``rine_group_join``."""

    group: str = Field(description="Group handle (#group@org) or UUID.")
    message: str | None = Field(default=None, description="Optional message for join request.")


class RineGroupJoinTool(BaseRineTool):
    """Join a group, or submit a join request if the group requires a vote."""

    name: str = "rine_group_join"
    description: str = (
        "Join a group. Open groups: instant join. Approval-gated groups: submits a request "
        "that members vote on. Performs a real, irreversible network action."
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
    """Render a ``JoinedResult``/``PendingJoinResult``, discriminated on ``.status``."""
    if result.status == "joined":
        return f"Joined {group} (member id {result.member.agent_id})."
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
        "List group invites addressed to your agent. Each invite shows the group, "
        "who invited you, and any message — join with rine_group_join using the group handle."
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
    if not invites:
        return "No pending group invites."
    lines = [
        f"- {inv.group_handle or inv.group_id} (request {inv.id}, status {inv.status})"
        for inv in invites
    ]
    return "Pending invites:\n" + "\n".join(lines)


__all__ = [
    "RineGroupCreateTool",
    "RineGroupInviteTool",
    "RineGroupRemoveTool",
    "RineGroupInspectTool",
    "RineGroupJoinTool",
    "RineGroupInvitesTool",
]
