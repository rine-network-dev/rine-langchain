"""Drop a thread binding once a resumed run reached a terminal state.

A terminal run (``not next``) means the conversation is done: its binding would otherwise linger
and shadow or strand a later message. A graph that re-armed at a new interrupt is NOT terminal and
keeps its binding. The sync and async twins share one predicate; the only difference is
``delete`` versus ``await adelete``, which keeps the async resume path off a blocking sqlite call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_rine._threadmap import ThreadMapStore

logger = logging.getLogger("langchain_rine.inbound")


def _is_terminal(post_state: Any) -> bool:
    return not post_state.next


def unregister_if_terminal(
    store: ThreadMapStore, post_state: Any, handle: str, conversation_id: str
) -> None:
    """Drop the binding when a resumed run reached a terminal state."""
    if _is_terminal(post_state):
        store.delete(handle, conversation_id)
        logger.info("unregistered terminal binding (%s, %s)", handle, conversation_id)


async def aunregister_if_terminal(
    store: ThreadMapStore, post_state: Any, handle: str, conversation_id: str
) -> None:
    """Async twin of :func:`unregister_if_terminal` (awaits ``adelete``)."""
    if _is_terminal(post_state):
        await store.adelete(handle, conversation_id)
        logger.info("unregistered terminal binding (%s, %s)", handle, conversation_id)
