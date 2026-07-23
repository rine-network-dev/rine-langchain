"""Shared config/auth resolver and tool base for langchain-rine.

One place every tool gets its ``rine`` client from. The resolution chain is the
SDK's, untouched: ``RINE_CLIENT_ID``+``RINE_CLIENT_SECRET`` env →
``RINE_CONFIG_DIR`` env → ``~/.config/rine`` → ``cwd/.rine``. We do **not** reimplement
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
    AuthenticationError,
    ConfigError,
    NotFoundError,
    RateLimitError,
    RineApiError,
    RineClient,
    RineError,
    SyncRineClient,
)

# The package's exported surface (the run-manager re-exports above stay out of __all__).
__all__ = ["BaseRineTool", "RineError", "_format_error", "build_client", "abuild_client"]


def build_client(
    *, config_dir: str | None, api_url: str | None, agent: str | None
) -> SyncRineClient:
    """Build a ``SyncRineClient``, passing only the non-``None`` overrides.

    Any override left ``None`` is omitted so the SDK resolves it via its own chain
    (env creds → ``RINE_CONFIG_DIR`` → ``~/.config/rine`` → ``cwd/.rine``). Shared by
    ``BaseRineTool._get_client`` and the callback handler's ``_get_client`` (same contract).
    """
    kwargs: dict[str, Any] = {}
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
    kwargs: dict[str, Any] = {}
    if config_dir is not None:
        kwargs["config_dir"] = config_dir
    if api_url is not None:
        kwargs["api_url"] = api_url
    if agent is not None:
        kwargs["agent"] = agent
    return RineClient(**kwargs)


def _format_error(err: Exception) -> str:
    """Map a Rine error to a clear, LLM-facing string.

    Tool ``_run``/``_arun`` bodies return these strings; LangChain surfaces them to the
    agent so it retries/continues rather than crashing. Never leak a raw traceback to a
    tool return. Most-specific-first ordering matters: the typed subclasses extend
    ``RineError`` directly, not ``RineApiError`` — keep them ahead of the
    ``RineApiError`` catch-all. Anything without a branch falls through to ``str(err)``,
    which is where a crypto failure such as ``NoMlsGroupStateError`` lands: the SDK's
    own message already tells the agent what to do about it, so restating it here would
    only give the two places to drift apart.
    """
    if isinstance(err, AuthenticationError):
        return (
            "Rine auth failed — set RINE_CLIENT_ID/RINE_CLIENT_SECRET or onboard "
            "(`python -m langchain_rine.onboard`)."
        )
    if isinstance(err, NotFoundError):
        return f"Not found: {err.detail}. Try rine_discover to find the right handle."
    if isinstance(err, RateLimitError):
        return f"Rate-limited; retry after {err.retry_after}s."
    if isinstance(err, ConfigError):
        return str(err)
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
        them via its own chain (env creds → ``RINE_CONFIG_DIR`` → ``~/.config/rine`` →
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
