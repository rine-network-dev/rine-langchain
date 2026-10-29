"""Shared config/auth resolver and tool base for langchain-rine.

One place every tool gets its ``rine`` client from. The resolution chain is the
SDK's, untouched: ``RINE_CONFIG_DIR`` env → ``~/.config/rine`` → ``cwd/.rine``, the first
directory holding a ``profiles.json``. We do **not** reimplement
that chain — the SDK's clients resolve it when ``config_dir``/``api_url``/``agent`` are
left at their defaults. This module only:

- builds a ``SyncRineClient`` (``_run``) / async ``RineClient`` (``_arun``) lazily on
  first use (never at import/``__init__``);
- maps typed ``RineError``s to clear LLM-facing strings (``_format_error``);
- gives every tool a ``BaseRineTool`` base carrying the override fields + lazy clients.

Side-effect-free import: nothing here touches the network, reads
credentials, or constructs a client at import time.
"""

from __future__ import annotations

from typing import Any

# Run-manager types are re-exported here so the tool modules can pull them from one
# place (`from langchain_rine._client import CallbackManagerForToolRun, ...`). The
# `as`-alias form marks these as intentional re-exports (they are referenced only in the
# `_run`/`_arun` signatures of the tool modules, never inside this file).
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun as AsyncCallbackManagerForToolRun,
)
from langchain_core.callbacks import (
    CallbackManagerForToolRun as CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr
from rine import (
    GROUP_ROSTER_LEAD,
    NO_GROUPS_IN_ORG,
    AuthenticationError,
    ConfigError,
    NotFoundError,
    RateLimitError,
    RineApiError,
    RineClient,
    RineError,
    SyncRineClient,
    is_not_a_member,
    not_a_member_message,
)

# The package's exported surface (the run-manager re-exports above stay out of __all__).
__all__ = ["BaseRineTool", "RineError", "_format_error", "build_client", "abuild_client"]


def build_client(
    *, config_dir: str | None, api_url: str | None, agent: str | None
) -> SyncRineClient:
    """Build a ``SyncRineClient``, passing only the non-``None`` overrides.

    Any override left ``None`` is omitted so the SDK resolves it via its own chain
    (``RINE_CONFIG_DIR`` → ``~/.config/rine`` → ``cwd/.rine``). Shared by
    ``BaseRineTool._get_client`` and the callback handler's ``_get_client`` (same contract).
    """
    kwargs: dict[str, Any] = {"agent_only": True}
    if config_dir is not None:
        kwargs["config_dir"] = config_dir
    if api_url is not None:
        kwargs["api_url"] = api_url
    if agent is not None:
        kwargs["agent"] = agent
    return SyncRineClient(**kwargs)


def abuild_client(*, config_dir: str | None, api_url: str | None, agent: str | None) -> RineClient:
    """Build an async ``RineClient`` with only the non-``None`` overrides (mirror of build_client).

    The async twin of :func:`build_client`: same omit-``None`` semantics so the SDK
    resolves any unset override via its own chain. Used by ``_arun`` paths.
    """
    kwargs: dict[str, Any] = {"agent_only": True}
    if config_dir is not None:
        kwargs["config_dir"] = config_dir
    if api_url is not None:
        kwargs["api_url"] = api_url
    if agent is not None:
        kwargs["agent"] = agent
    return RineClient(**kwargs)


#: The lead sentences ``rine`` opens an unknown **acting agent** refusal with
#: (``rine._actor.unknown_actor_refusal`` and ``unknown_handle_refusal`` — the
#: only two, and reached from the actor rung alone).
_UNKNOWN_ACTOR_LEADS = ("No agent named '", "No agent with handle '")

#: The lead the server opens EVERY group 404 with, restated here to branch on.
#:
#: Hand-typed on both sides — the server raises it as a bare literal from four
#: modules and no constant — so ``tests/test_client-server-contract_group0.py``
#: pins this copy against those, the way ``NOT_A_MEMBER_CLAUSE`` is pinned. A
#: reword there with no gate here silently sends every group 404 on five
#: published surfaces back to the AGENT directory, with every suite green.
_GROUP_NOT_FOUND_LEAD = "Group not found"

#: The SECOND lead, and the one a BARE recipient name now carries.
#:
#: The bare-name miss stopped leading with a group verdict — by the time the SDK
#: refuses, the agent namespace has declined the spelling on syntax and no agent
#: of this org carries the name, so both halves have answered no and the
#: sentence now says so. The REMEDY below is unchanged, and that is the whole
#: reason this constant exists here: dropping the first lead without adding this
#: one sends every bare-name miss down the final ``return`` of
#: :func:`_not_found_message`, which names ``rine_discover`` — the public AGENT
#: directory, which cannot resolve a group handle. That is the old misdirection
#: restored, with every suite green.
#:
#: 🔴 Matched by CONTAINMENT, not ``startswith``: the sentence OPENS with the
#: caller's own spelling (``'kofi' names neither …``), which is the point of the
#: reword. Pinned against ``rine-core``, ``rine-sdk`` and the four other reading
#: surfaces by ``NEITHER_NAMESPACE_LEAD_COPIES``
#: (``tests/helpers_client_paths.py``).
_NEITHER_NAMESPACE_LEAD = "names neither an agent nor a group you can reach"

#: What ``rine_send_and_wait`` answers a group target with, on both arms.
#:
#: The SDK raises ``UnsupportedTargetError`` before any request — ``POST
#: /messages/sync`` is 1:1 only — so this sentence is the whole result of the
#: turn and the server never sees the send. The sync ``_run`` and the async
#: ``_arun`` return the same constant, which is the local reason it is one:
#: this surface is the only Python wrapper that says it twice.
#:
#: 🔴 It names ``rine_send_and_wait``, the id this surface registers, and
#: ``rine_send``, the one that can carry the message. A model reads a refusal
#: literally and calls the verb the sentence names, so a bare ``send_and_wait``
#: here spends the next turn on a tool-not-found.
#:
#: The cross-package reason is the one the two leads above carry: five published
#: surfaces say this sentence, nothing on the wire carries it, and a reword on
#: one of them fails in no suite. Pinned against the other four by
#: ``GROUP_ON_WAIT_MESSAGE_COPIES`` (``tests/helpers_client_paths.py``).
_GROUP_ON_WAIT_MESSAGE = "rine_send_and_wait is 1:1 only; use rine_send for groups."


def _not_found_message(detail: str) -> str:
    """A 404, plus the tool that can answer it — and no verb when none can.

    An unknown **acting agent** needs no search: the refusal already lists this
    org's agents by name, and that list is the answer. ``rine_discover`` reads
    the unauthenticated, org-agnostic public directory, so it cannot say which of
    the caller's own agents it may act as. A missing **group** is handed to
    ``rine_groups`` first: it lists every group the key can see, public
    and private alike, and a mistyped handle on a send is almost always a group
    the caller is already in. ``rine_discover_groups`` is named second because it
    searches the public directory only — the one place that finds a public group
    this org has never joined, and the one place a private group can never
    appear. Anything else names somebody else's agent, which is what
    ``rine_discover`` is for.

    ``rine_groups`` is dropped when the refusal already carries that verb's own
    answer. The SDK lists the groups the key can see inside its group refusals —
    :data:`rine.GROUP_ROSTER_LEAD` introduces the roster, and
    :data:`rine.NO_GROUPS_IN_ORG` is the arm for an org that holds a seat
    nowhere — so naming ``rine_groups`` underneath one of those tells a caller
    to go and fetch what it is already reading. That is the ``rine_discover``
    defect inverted, and it costs more here than a wasted call: models were
    measured parroting this hint back as their answer instead of acting on it.
    ``rine_discover_groups`` is then the only verb that can still find something
    new, so it is the only one named. A **server** group 404 carries no roster
    and keeps both verbs. The clause is recognised by the SDK's own exported
    constants rather than a retyped copy, so a reword there cannot leave a stale
    literal here quietly restoring the loop.

    The lead is punctuated once, not twice: the zero-agent arm of an actor
    refusal already ends in a full stop (*"No active agents in this org."*), and
    a second one is what a caller reads at the very end of the sentence. The
    roster arm is the one detail that is punctuated **not at all**: it ends on a
    list row, and a full stop there belongs to that row rather than to the
    sentence.
    """
    lead = f"Not found: {detail.rstrip('.')}."
    if detail.startswith(_UNKNOWN_ACTOR_LEADS):
        return lead
    if detail.startswith(_GROUP_NOT_FOUND_LEAD) or _NEITHER_NAMESPACE_LEAD in detail:
        search_the_directory = "Try rine_discover_groups to search the public directory."
        if GROUP_ROSTER_LEAD in detail:
            # The roster arm ends on a LIST ROW, not on a sentence. A clause
            # appended with a space reads as part of that last group's entry —
            # and so does ``lead``'s own full stop, which ``rstrip`` cannot take
            # back off a row ending in ``)``. ``#eng@acme.rine.network (Eng).``
            # is a group nobody can name back, so neither is appended here.
            return f"Not found: {detail}\n{search_the_directory}"
        if NO_GROUPS_IN_ORG in detail:
            return f"{lead} {search_the_directory}"
        return (
            f"{lead} Try rine_groups to find the right group handle, "
            "or rine_discover_groups to search the public directory."
        )
    return f"{lead} Try rine_discover to find the right handle."


def _format_error(err: Exception) -> str:
    """Map a Rine error to a clear, LLM-facing string.

    Tool ``_run``/``_arun`` bodies return these strings; LangChain surfaces them to the
    agent so it retries/continues rather than crashing. Never leak a raw traceback to a
    tool return. Most-specific-first ordering matters: the typed subclasses extend
    ``RineError`` directly, not ``RineApiError`` — keep them ahead of the
    ``RineApiError`` catch-all. Anything without a branch falls through to ``str(err)``,
    which is where a crypto failure such as ``NoMlsGroupStateError`` lands: the SDK's
    own message already tells the agent what to do about it, so restating it here would
    only give the two places to drift apart. The 404 arm defers to
    :func:`_not_found_message`, which reads *what* was not found before naming a remedy.

    The one refusal that is not forwarded whole is the server's "this agent holds
    no seat in that group": it names the sibling to act as (which nothing here
    could work out) and then tells the caller to set ``X-Rine-Agent`` (which
    nothing here exposes — the SDK writes that header itself). The SDK's own
    sentence is appended rather than substituted, so the half only the server
    knows survives and the half only this stack knows gets said.
    """
    if isinstance(err, AuthenticationError):
        return (
            "Rine auth failed — onboard (`python -m langchain_rine.onboard`). "
            "A host clock more than 30 s off also fails sign-in."
        )
    if isinstance(err, NotFoundError):
        return _not_found_message(err.detail)
    if isinstance(err, RateLimitError):
        return f"Rate-limited; retry after {err.retry_after}s."
    if isinstance(err, ConfigError):
        return str(err)
    if isinstance(err, RineApiError) and is_not_a_member(err.detail):
        return f"{err.status}: {not_a_member_message(err.detail)}"
    if isinstance(err, RineApiError):
        return f"{err.status}: {err.detail}"
    return str(err)


class BaseRineTool(BaseTool):
    """Shared base for every langchain_rine tool (LangChain ``BaseTool``).

    Declares the SDK config overrides as constructor fields (per-tool, optional) and
    lazily builds the sync/async ``rine`` clients on first use. Building a client is
    deferred to ``_get_client``/``_aget_client`` (called from ``_run``/``_arun``) so
    import and ``__init__`` stay side-effect-free.
    """

    config_dir: str | None = None
    api_url: str | None = None
    agent: str | None = None

    _client: SyncRineClient | None = PrivateAttr(default=None)
    _aclient: RineClient | None = PrivateAttr(default=None)

    def _get_client(self) -> SyncRineClient:
        """Build (once) and return the shared ``SyncRineClient``.

        Passes only the override fields through; when they are ``None`` the SDK resolves
        them via its own chain (``RINE_CONFIG_DIR`` → ``~/.config/rine`` →
        ``cwd/.rine``). We never reimplement that resolution here.
        """
        if self._client is None:
            self._client = build_client(
                config_dir=self.config_dir, api_url=self.api_url, agent=self.agent
            )
        return self._client

    def _aget_client(self) -> RineClient:
        """Lazily build the async client (one per tool instance)."""
        if self._aclient is None:
            self._aclient = abuild_client(
                config_dir=self.config_dir, api_url=self.api_url, agent=self.agent
            )
        return self._aclient
