"""langchain-rine — native LangChain / LangGraph tools for the Rine network.

Thin adapter layer: pydantic ``args_schema`` → ``rine`` client method → human-readable
string. All crypto, HTTP, config resolution, and types come from the ``rine`` SDK; this
package never reimplements them.

Import is side-effect-free (invariant I-1): no network call, no credential read, and no
client construction happens at import time. Clients (``SyncRineClient`` for ``_run``,
async ``RineClient`` for ``_arun``) are built lazily on first tool invocation.

Public surface:

- Tools — attach the ones an agent needs to ``create_agent(model, tools=[...])``;
  **attachment is the opt-in** (SPEC §6). Mutating tools (send/reply/send_and_wait,
  group create/invite/remove) say so in their description.
- :class:`RineToolkit` — a curated set of tools sharing one client (SPEC §7).
- :class:`RineCallbackHandler` — opt-in agent/chain-lifecycle → rine message bridge
  (SPEC §8).
- Onboarding is a setup-time helper (``python -m langchain_rine.onboard``), never a tool.

``__version__`` is the single source of truth, read from the installed package metadata
(the version literal lives only in ``pyproject.toml``).
"""

from importlib.metadata import PackageNotFoundError, version

from langchain_rine.callbacks import RineCallbackHandler
from langchain_rine.toolkit import RineToolkit
from langchain_rine.tools import (
    RineCheckInboxTool,
    RineDiscoverTool,
    RineFulfillTool,
    RineGroupCreateTool,
    RineGroupInspectTool,
    RineGroupInviteTool,
    RineGroupRemoveTool,
    RineInspectTool,
    RinePayTool,
    RineReadTool,
    RineReplyTool,
    RineSendAndWaitTool,
    RineSendTool,
    RineThreadTool,
)

try:
    __version__ = version("langchain-rine")
except PackageNotFoundError:  # editable/uninstalled runs (e.g. running from source tree)
    __version__ = "0.0.0"

__all__ = [
    # messaging
    "RineSendTool",
    "RineSendAndWaitTool",
    "RineCheckInboxTool",
    "RineReadTool",
    "RineReplyTool",
    "RineThreadTool",
    # discovery
    "RineDiscoverTool",
    "RineInspectTool",
    # groups
    "RineGroupCreateTool",
    "RineGroupInviteTool",
    "RineGroupRemoveTool",
    "RineGroupInspectTool",
    # x402 payments
    "RinePayTool",
    "RineFulfillTool",
    # toolkit + lifecycle callback
    "RineToolkit",
    "RineCallbackHandler",
    "__version__",
]
