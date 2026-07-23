"""Tool classes for langchain-rine, split by domain.

Re-exports the ``langchain_core.tools.BaseTool`` subclasses from ``messaging``,
``discovery``, and ``groups``. Group send/read are not separate tools — they ride
``#``-prefix routing through the messaging tools.
"""

from langchain_rine.tools.discovery import RineDiscoverTool, RineInspectTool
from langchain_rine.tools.groups import (
    RineGroupCreateTool,
    RineGroupInspectTool,
    RineGroupInvitesTool,
    RineGroupInviteTool,
    RineGroupJoinTool,
    RineGroupRemoveTool,
)
from langchain_rine.tools.messaging import (
    RineCheckInboxTool,
    RineReadTool,
    RineReplyTool,
    RineSendAndWaitTool,
    RineSendTool,
    RineThreadTool,
)
from langchain_rine.tools.payments import RinePayTool
from langchain_rine.tools.payments_payee import RineFulfillTool

__all__ = [
    # messaging (1:1 + #-prefixed group send/read)
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
    "RineGroupJoinTool",
    "RineGroupInvitesTool",
    # x402 payments (pay a quote / fulfil a payment)
    "RinePayTool",
    "RineFulfillTool",
]
