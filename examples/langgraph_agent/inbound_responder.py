"""Tier-3 idle-wake demo: an inbound rine message wakes a paused LangGraph thread.

This is the runnable, clonable cross-process handoff demo (SPEC §11 Tier-3, D14-D17). It shows
the full loop:

1. A small ``StateGraph`` has a node that calls ``interrupt(HumanInterrupt{...})`` to *pause and
   wait* for an inbound rine reply — the graph is durably checkpointed, so the pause survives a
   process restart.
2. The graph sends a question out on rine and then sits idle (its thread is parked at the
   ``interrupt``).
3. A :class:`~langchain_rine.drivers.PollDriver` polls the rine inbox; when the peer's reply
   arrives, the :class:`~langchain_rine.inbound.RineThreadResumer` matches it to the parked thread
   by ``(sender_handle, conversation_id)`` and streams ``Command(resume=<HumanResponse>)`` — the
   ``interrupt()`` call site receives the decrypted reply text and the graph runs to completion.

**Caller owns the saver lifecycle (D17).** ``SqliteSaver.from_conn_string(...)`` is opened in a
``with``-block here; the resumer takes the *already-compiled* graph. The thread-map
(:class:`~langchain_rine._threadmap.SqliteThreadMap`) is co-located in a sibling sqlite file so the
``(handle, conversation_id) -> thread_id`` binding also survives a restart.

Install the optional extra and onboard once, then run::

    pip install "langchain-rine[inbound]"
    python -m langchain_rine.onboard --email you@example.com \
        --org-slug acme --org-name Acme --agent-name worker
    python examples/langgraph_agent/inbound_responder.py "peer@other.rine.network" "what's the ETA?"

NOT run in CI (it drives a real network + a real peer). Heavily commented as a starting point.
"""

from __future__ import annotations

import sys
from typing import Any, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt.interrupt import HumanInterrupt
from langgraph.types import interrupt

from langchain_rine._client import build_client
from langchain_rine._threadmap import SqliteThreadMap
from langchain_rine.drivers import PollDriver
from langchain_rine.inbound import RineThreadResumer

CHECKPOINT_DB = "rine_threads.sqlite"
THREADMAP_DB = "rine_threadmap.sqlite"


class State(TypedDict):
    """The graph's working state: who we asked, what we asked, and the reply we waited for."""

    to: str
    question: str
    answer: str


def await_reply(state: State) -> dict[str, Any]:
    """Pause the graph until an inbound rine reply is delivered to the ``interrupt`` call.

    The ``HumanInterrupt`` payload follows the Agent-Inbox schema (PHASE0_v2_CONTRACT §7). On the
    first pass this raises/pauses; when the resumer streams ``Command(resume=<HumanResponse>)`` the
    call returns that response and we read the peer's free-text reply out of ``args``.
    """
    request: HumanInterrupt = {
        "action_request": {
            "action": "reply_to_message",
            "args": {"to": state["to"], "question": state["question"]},
        },
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": False,
            "allow_accept": False,
        },
        "description": f"Awaiting a rine reply from {state['to']}",
    }
    human_response = interrupt(request)
    return {"answer": str(human_response.get("args", ""))}


def build_graph(saver: SqliteSaver) -> Any:
    """Compile the one-node await-reply graph against a durable ``SqliteSaver`` (caller-owned)."""
    builder: StateGraph = StateGraph(State)
    builder.add_node("await_reply", await_reply)
    builder.add_edge(START, "await_reply")
    builder.add_edge("await_reply", END)
    return builder.compile(checkpointer=saver)


def main() -> None:
    """Send a question on rine, park the thread at the interrupt, then poll until the reply wakes it."""
    to = sys.argv[1] if len(sys.argv) > 1 else "peer@other.rine.network"
    question = sys.argv[2] if len(sys.argv) > 2 else "What's the ETA on the task?"

    # The caller owns the checkpointer context (D17): everything runs inside this with-block.
    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as saver:
        graph = build_graph(saver)
        store = SqliteThreadMap(THREADMAP_DB)
        resumer = RineThreadResumer(graph, store)

        # 1. Send the question on rine and learn the conversation_id it belongs to.
        client = build_client(config_dir=None, api_url=None, agent=None)
        sent = client.send(to, {"text": question})
        conversation_id = str(sent.conversation_id)
        thread_id = f"thread-{conversation_id}"

        # 2. Bind (peer handle, conversation) -> this thread, then start the graph so it parks at
        #    the interrupt awaiting the peer's reply.
        resumer.register(to, conversation_id, thread_id)
        config = {"configurable": {"thread_id": thread_id}}
        graph.invoke({"to": to, "question": question, "answer": ""}, config)
        print(f"Parked thread {thread_id}; awaiting a reply from {to} ...")

        # 3. Poll the rine inbox. When the peer's reply lands, the resumer wakes the parked thread;
        #    bound to a few iterations for the demo (use driver.run() for an unbounded loop).
        driver = PollDriver(resumer)
        driver.run(interval=3.0, max_iterations=20)

        final = graph.get_state(config)
        if not final.next:
            print(f"Reply received; graph completed: {final.values.get('answer')!r}")
        else:
            print(
                "No reply within the demo window; thread is still parked (durable — rerun to resume)."
            )


if __name__ == "__main__":
    main()
