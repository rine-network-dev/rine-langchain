"""Sender-key group tools (SPEC §6.8, §6.10-6.12).

Four ``BaseTool``s over ``client.groups``: create, invite, remove, inspect.
Group **send/read are NOT here** — they ride ``#``-prefix routing through the
messaging tools (SPEC §6.9). ``mls_enabled`` is never exposed: the SDK hard-codes
``mls_enabled=False`` on create, so a LangChain agent only ever owns sender-key
groups. ``RineGroupInspectTool`` surfaces the MLS latches so an operator can
self-diagnose an unreadable (TS-created MLS) group.

Each tool implements both ``_run`` (sync ``SyncRineClient``) and ``_arun`` (async
``RineClient``): the async body is the sync body with ``_get_client`` →
``_aget_client`` and every ``groups.X(...)`` call ``await``ed. ``groups.list``
returns a plain ``list[GroupRead]`` (not a ``CursorPage``), so both paths iterate it
directly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from rine import GroupRead, NotFoundError, is_uuid, normalize_handle

from langchain_rine._client import (
    AsyncCallbackManagerForToolRun,
    BaseRineTool,
    CallbackManagerForToolRun,
    _format_error,
)
from langchain_rine._format import render_group


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
    """Create a new sender-key-encrypted coordination group."""

    name: str = "rine_group_create"
    description: str = (
        "Create a new encrypted coordination group (sender-key E2EE). Performs a real, "
        "irreversible network action. Your agent owns and administers the group. The group "
        "is always sender-key (MLS is not exposed; the Python side cannot operate MLS groups)."
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
            f"Created group {group.handle} (id {group.id}, sender-key E2EE, "
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
            f"Created group {group.handle} (id {group.id}, sender-key E2EE, "
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
        "Remove a member from a group your agent administers. Triggers a sender-key "
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
    """Show a group's details + sender-key-vs-MLS self-diagnosis."""

    name: str = "rine_group_inspect"
    description: str = (
        "Show a group's details and encryption status, so you can tell whether your agent "
        "can read/post it (sender-key) or not (MLS)."
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


__all__ = [
    "RineGroupCreateTool",
    "RineGroupInviteTool",
    "RineGroupRemoveTool",
    "RineGroupInspectTool",
]
