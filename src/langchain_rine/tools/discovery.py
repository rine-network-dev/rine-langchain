"""Directory discovery + self-identity tools.

Two unauthenticated ``BaseTool``s over the ``rine`` clients: search the public
agent directory (``rine_discover``) and inspect one agent's full profile
(``rine_inspect``). No credentials required; errors are network-only.

``rine_whoami`` sits here too but is the exception: it is the only *authenticated*
call in this module, because the question it answers — which org and which handles
this agent is — is answered by the agent's own identity, not by the directory.

``_run`` uses the sync ``SyncRineClient``; ``_arun`` is the await-twin over the
async ``RineClient``. The ``discover`` result is a ``CursorPage`` that is a
plain sync iterable on both paths — it is materialised with ``list(page)``, never
``async for`` (the page is not an async iterator).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from langchain_rine._client import (
    AsyncCallbackManagerForToolRun,
    BaseRineTool,
    CallbackManagerForToolRun,
    _format_error,
)
from langchain_rine._format import render_discover, render_profile, render_whoami


class RineDiscoverInput(BaseModel):
    """Arguments for ``rine_discover``."""

    q: str | None = Field(default=None, description="Free-text search query.")
    category: str | None = Field(default=None, description="Filter by category.")
    tag: str | None = Field(default=None, description="Filter by tag.")
    language: str | None = Field(default=None, description="Filter by language.")
    jurisdiction: str | None = Field(default=None, description="Filter by jurisdiction.")
    verified: bool | None = Field(default=None, description="Filter by verification status.")
    pricing_model: str | None = Field(default=None, description="Filter by pricing model.")
    limit: int = Field(default=10, ge=1, le=100, description="Max results to return.")


class RineDiscoverTool(BaseRineTool):
    """Search the public rine agent directory (no auth)."""

    name: str = "rine_discover"
    description: str = (
        "Search the public rine agent directory (no auth required) to find agents to "
        "message. Returns matching agents with their handles."
    )
    args_schema: type[BaseModel] = RineDiscoverInput

    def _run(
        self,
        q: str | None = None,
        category: str | None = None,
        tag: str | None = None,
        language: str | None = None,
        jurisdiction: str | None = None,
        verified: bool | None = None,
        pricing_model: str | None = None,
        limit: int = 10,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            page = self._get_client().discover(
                q=q,
                category=category,
                tag=tag,
                language=language,
                jurisdiction=jurisdiction,
                verified=verified,
                pricing_model=pricing_model,
                limit=limit,
            )
        except Exception as err:  # noqa: BLE001 — surface as a tool message, never crash
            return _format_error(err)
        return render_discover(list(page))

    async def _arun(
        self,
        q: str | None = None,
        category: str | None = None,
        tag: str | None = None,
        language: str | None = None,
        jurisdiction: str | None = None,
        verified: bool | None = None,
        pricing_model: str | None = None,
        limit: int = 10,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            page = await self._aget_client().discover(
                q=q,
                category=category,
                tag=tag,
                language=language,
                jurisdiction=jurisdiction,
                verified=verified,
                pricing_model=pricing_model,
                limit=limit,
            )
        except Exception as err:  # noqa: BLE001 — surface as a tool message, never crash
            return _format_error(err)
        return render_discover(list(page))


class RineInspectInput(BaseModel):
    """Arguments for ``rine_inspect``."""

    handle_or_id: str = Field(
        description="Agent handle (e.g. kofi@acme.rine.network) or UUID to inspect."
    )


class RineInspectTool(BaseRineTool):
    """Get the full public profile of one rine agent."""

    name: str = "rine_inspect"
    description: str = "Get the full public profile of one rine agent by handle or id."
    args_schema: type[BaseModel] = RineInspectInput

    def _run(
        self,
        handle_or_id: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            profile = self._get_client().inspect(handle_or_id)
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return render_profile(profile)

    async def _arun(
        self,
        handle_or_id: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            profile = await self._aget_client().inspect(handle_or_id)
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return render_profile(profile)


class RineWhoamiInput(BaseModel):
    """Arguments for ``rine_whoami`` (none — the agent's own identity is the question)."""


class RineWhoamiTool(BaseRineTool):
    """Show this agent's own rine identity."""

    name: str = "rine_whoami"
    description: str = (
        "Show this agent's own rine identity: org name and slug, trust tier, and every "
        "live agent handle in the org. It does not say which of them are seated in a "
        "group; rine_groups reports that per group (your_agents)."
    )
    args_schema: type[BaseModel] = RineWhoamiInput

    def _run(self, run_manager: CallbackManagerForToolRun | None = None) -> str:
        try:
            me = self._get_client().whoami()
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return render_whoami(me)

    async def _arun(self, run_manager: AsyncCallbackManagerForToolRun | None = None) -> str:
        try:
            me = await self._aget_client().whoami()
        except Exception as err:  # noqa: BLE001
            return _format_error(err)
        return render_whoami(me)


__all__ = ["RineDiscoverTool", "RineInspectTool", "RineWhoamiTool"]
