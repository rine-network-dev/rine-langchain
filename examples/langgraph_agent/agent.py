"""A runnable LangGraph agent wired with the full rine toolkit + a checkpointer.

This is the clonable starting point: ``RineToolkit().get_tools()`` +
``create_agent(model, tools)`` + an in-memory checkpointer so the conversation (and the
agent's place in a multi-turn rine coordination) survives across turns under one
``thread_id``.

Run it (see README.md for setup):

    export OPENAI_API_KEY=sk-...
    python -m langchain_rine.onboard --email you@example.com \
        --org-slug acme --org-name Acme --agent-name worker     # once
    python examples/langgraph_agent/agent.py "check my rine inbox and summarize it"

The model id, the toolkit's ``include`` curation, and the checkpointer are the three
knobs you will most often change. Swap ``InMemorySaver`` for a persistent
``langgraph.checkpoint`` backend (sqlite/postgres) to keep threads across process
restarts.
"""

from __future__ import annotations

import asyncio
import sys

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from langchain_rine import RineToolkit

SYSTEM_PROMPT = (
    "You are an agent on the rine network with access to encrypted messaging, directory "
    "discovery, and coordination-group tools. Use rine_discover to find peers, "
    "rine_send / rine_send_and_wait / rine_reply to talk to them (every send is a real, "
    "irreversible, end-to-end-encrypted network message), rine_inbox / rine_read to "
    "read mail, and the rine_group_* tools to run sender-key groups. Be explicit about "
    "any irreversible action before you take it."
)


def build_agent() -> object:
    """Build the LangGraph agent: full rine toolkit + an in-memory checkpointer.

    ``RineToolkit().get_tools()`` returns all 25 tools sharing one rine client.
    ``create_agent`` (LangChain 1.0) compiles them into a LangGraph agent; the
    ``checkpointer`` persists state per ``thread_id`` so multi-turn coordination resumes.
    """
    tools = RineToolkit().get_tools()
    return create_agent(
        "openai:gpt-4o-mini",
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )


async def run_once(prompt: str, *, thread_id: str = "demo") -> str:
    """Drive one turn through the checkpointed agent and return its final text.

    The ``thread_id`` selects the checkpoint thread; reuse it across calls to continue
    the same conversation (the checkpointer replays prior messages). Async-native, so the
    rine tools take their ``_arun`` (async client) path.
    """
    agent = build_agent()
    config = {"configurable": {"thread_id": thread_id}}
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config=config,
    )
    final = result["messages"][-1]
    return str(getattr(final, "content", final))


def main() -> None:
    """CLI entrypoint: ``python examples/langgraph_agent/agent.py "<prompt>"``."""
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Check my rine inbox and summarize it."
    print(asyncio.run(run_once(prompt)))


if __name__ == "__main__":
    main()
