"""Tool classes for langchain-rine, split by domain.

Re-exports the ``langchain_core.tools.BaseTool`` subclasses from ``messaging``,
``discovery``, ``groups``, ``groups_list``, ``groups_admin`` and
``groups_reclaim``. Group send/read are not separate tools — they ride
``#``-prefix routing through the messaging tools.
"""

from langchain_rine.tools.discovery import (
    RineDiscoverTool,
    RineInspectTool,
    RineWhoamiTool,
)
from langchain_rine.tools.groups import (
    RineGroupCreateTool,
    RineGroupInspectTool,
    RineGroupInvitesTool,
    RineGroupInviteTool,
    RineGroupJoinTool,
    RineGroupRemoveTool,
)
from langchain_rine.tools.groups_admin import (
    RineGroupLeaveTool,
    RineGroupRequestsTool,
    RineGroupSyncTool,
    RineGroupVoteTool,
)
from langchain_rine.tools.groups_list import (
    RineDiscoverGroupsTool,
    RineGroupRosterTool,
    RineGroupsTool,
)
from langchain_rine.tools.groups_reclaim import RineGroupReclaimTool
from langchain_rine.tools.messaging import (
    RineInboxTool,
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
    "RineInboxTool",
    "RineReadTool",
    "RineReplyTool",
    "RineThreadTool",
    # discovery
    "RineDiscoverTool",
    "RineInspectTool",
    "RineDiscoverGroupsTool",
    "RineWhoamiTool",
    # groups
    "RineGroupsTool",
    "RineGroupCreateTool",
    "RineGroupInviteTool",
    "RineGroupRemoveTool",
    "RineGroupInspectTool",
    "RineGroupRosterTool",
    "RineGroupJoinTool",
    "RineGroupInvitesTool",
    "RineGroupRequestsTool",
    "RineGroupVoteTool",
    "RineGroupLeaveTool",
    "RineGroupSyncTool",
    "RineGroupReclaimTool",
    # x402 payments (pay a quote / fulfil a payment)
    "RinePayTool",
    "RineFulfillTool",
]
