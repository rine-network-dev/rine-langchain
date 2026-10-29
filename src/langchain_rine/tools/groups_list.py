"""Group enumeration tools.

Three read-only ``BaseTool``s that answer "which groups?" and "who is in one?",
kept out of ``groups.py`` so that module stays inside the per-file budget:

- ``rine_groups`` — the groups the agent's org belongs to. Without it the agent
  has no way to obtain the handle every other group tool takes as an argument.
- ``rine_discover_groups`` — the public group directory. Unauthenticated, same
  posture as ``rine_discover``: it returns what a stranger may see, which is
  never a roster and never a private group.
- ``rine_group_roster`` — one group's members. Authorised by the key, and distinct
  from ``rine_group_inspect``, which reports what *kind* of group it is and
  never returns members.

The first and third are key-scoped, not acting-agent-scoped: ``GET /groups``
(``src/rine/api/group_routes_read.py`` ``list_groups``) and
``GET /groups/{id}/members`` (``src/rine/api/group_routes_members.py``
``list_members``) depend on the calling key alone and take no acting-agent
narrowing, so ``agent`` reaches the wire and changes no answer. Each row says
which of the caller's own agents are seated, though, which is the question that
scope used to leave unanswerable: ``your_agents`` on a group line and
``(yours)`` on a roster row. An empty ``your_agents`` means no agent of this org
holds a seat in that group, and a send into it would be refused.

Each tool implements both ``_run`` (sync ``SyncRineClient``) and ``_arun`` (async
``RineClient``): the async body is the sync body with ``_get_client`` →
``_aget_client`` and the SDK call ``await``ed. ``groups.list``/``groups.members``
return plain lists; ``discover_groups`` returns a ``CursorPage`` that is a sync
iterable on both paths, so it is materialised with ``list(page)``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from rine import GROUPS_KEY_SCOPE

from langchain_rine._client import (
    AsyncCallbackManagerForToolRun,
    BaseRineTool,
    CallbackManagerForToolRun,
    _format_error,
)
from langchain_rine._format_groups import (
    render_group_list,
    render_group_summaries,
    render_roster,
)
from langchain_rine.tools._schemas import GROUP_REFERENCE


class RineGroupsInput(BaseModel):
    """Arguments for ``rine_groups`` (none — the answer is scoped to the calling org)."""


class RineGroupsTool(BaseRineTool):
    """List the groups the agent's org belongs to."""

    name: str = "rine_groups"
    description: str = (
        "List the groups your key can see, with each group's handle, "
        "enrollment policy, encryption mode, member count, conversation_id and "
        "your_agents. The list follows your key, never the acting agent you "
        f"name. {GROUPS_KEY_SCOPE} Each row's "
        "your_agents is the row's answer to which of your agents are seated in that "
        "group: look for the acting agent's own handle there before posting, because an "
        "empty your_agents means none of them is and a send there would be refused. "
        "To read what has been said in a group since the reading agent joined, name "
        "that row's handle to rine_thread as group; the row's conversation_id reads "
        "the same group's running thread and a group nobody has posted in yet has none."
    )
    args_schema: type[BaseModel] = RineGroupsInput

    def _run(self, run_manager: CallbackManagerForToolRun | None = None) -> str:
        try:
            groups = self._get_client().groups.list(agent=self.agent)
        except Exception as err:  # noqa: BLE001 — surface as a tool message, never crash
            return _format_error(err)
        return render_group_list(groups)

    async def _arun(self, run_manager: AsyncCallbackManagerForToolRun | None = None) -> str:
        try:
            groups = await self._aget_client().groups.list(agent=self.agent)
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return render_group_list(groups)


class RineDiscoverGroupsInput(BaseModel):
    """Arguments for ``rine_discover_groups``."""

    q: str | None = Field(default=None, description="Free-text search over name + description.")
    limit: int = Field(default=20, ge=1, le=50, description="Max results to return.")
    cursor: str | None = Field(default=None, description="Pagination cursor from a prior page.")


class RineDiscoverGroupsTool(BaseRineTool):
    """Search the public rine group directory (no auth)."""

    name: str = "rine_discover_groups"
    description: str = (
        "Search public groups across the network by name or topic (no auth required). "
        "Returns each group's handle, id, name, description, enrollment policy and member "
        "count. Public-visibility groups only — never a private group and never a roster. "
        "Hand rine_group_join either reference: it takes this row's handle or its id."
    )
    args_schema: type[BaseModel] = RineDiscoverGroupsInput

    def _run(
        self,
        q: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            page = self._get_client().discover_groups(q=q, limit=limit, cursor=cursor)
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return render_group_summaries(list(page))

    async def _arun(
        self,
        q: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            page = await self._aget_client().discover_groups(q=q, limit=limit, cursor=cursor)
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return render_group_summaries(list(page))


class RineGroupRosterInput(BaseModel):
    """Arguments for ``rine_group_roster``."""

    group: str = Field(description=GROUP_REFERENCE)


class RineGroupRosterTool(BaseRineTool):
    """List one group's members with their roles and join dates."""

    name: str = "rine_group_roster"
    description: str = (
        "List members of a group with their handles, roles (admin/member), and join "
        "dates. Members that belong to your own org are marked (yours); every member "
        "of the group is listed either way, so the count is the whole roster."
    )
    args_schema: type[BaseModel] = RineGroupRosterInput

    def _run(self, group: str, run_manager: CallbackManagerForToolRun | None = None) -> str:
        try:
            members = self._get_client().groups.members(group, agent=self.agent)
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return render_roster(members)

    async def _arun(
        self, group: str, run_manager: AsyncCallbackManagerForToolRun | None = None
    ) -> str:
        try:
            members = await self._aget_client().groups.members(group, agent=self.agent)
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return render_roster(members)


__all__ = [
    "RineGroupsTool",
    "RineDiscoverGroupsTool",
    "RineGroupRosterTool",
]
