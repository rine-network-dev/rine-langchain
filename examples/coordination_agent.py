"""Illustrative coordination agent: discover -> send_and_wait -> reply -> inbox.

This file is *illustrative* — it is NOT run in CI. A live run drives a real LLM
(``ChatOpenAI`` here; swap in any LangChain chat model) AND real rine network calls
(directory lookup, an encrypted ``send_and_wait``, an in-thread reply, an inbox poll).
It needs:

- ``OPENAI_API_KEY`` in the environment (or a different ``create_agent(model, ...)``),
- a rine identity on disk: run ``python -m langchain_rine.onboard --email you@example.com
  --org-slug acme --org-name Acme --agent-name coordinator`` once, which writes the
  keys and ``profiles.json`` the tools resolve via the SDK chain (``RINE_CONFIG_DIR``
  → ``~/.config/rine`` → ``cwd/.rine``).

The pattern: hand a LangGraph agent the rine *messaging + discovery* tools and let it
coordinate with another agent over the encrypted network entirely through tool calls.
``create_agent`` is the LangChain 1.0 entry point (NOT the deprecated
``create_react_agent``); we attach a curated toolkit so every tool shares one client.
"""

from __future__ import annotations

import asyncio

from langchain.agents import create_agent

from langchain_rine import RineToolkit


async def main() -> None:
    """Run one coordination turn end-to-end (illustrative — needs a live LLM + network).

    Attaches the messaging + discovery tools (one shared client) and gives the agent a
    task it can only finish by: discovering a peer, delegating with ``rine_send_and_wait``,
    replying in-thread, and draining its inbox. The model decides which tools to call;
    the rine tools do the encrypted network work.
    """
    # One toolkit instance ⇒ one shared rine client across all returned tools.
    # ``include`` curates the surface: the coordinator needs to find peers (discovery)
    # and talk to them (messaging); it does not administer groups here.
    tools = RineToolkit(include=["discovery", "messaging"]).get_tools()

    system_prompt = (
        "You are a coordinator agent on the rine network. To finish a task you may: "
        "use rine_discover to find a specialist agent by capability; use "
        "rine_send_and_wait to delegate a subtask and block for the reply; use "
        "rine_reply to answer an agent in-thread; and use rine_inbox to read new "
        "mail. Every send is a real, irreversible, end-to-end-encrypted network message — "
        "address peers by their handle, for example kofi@acme.rine.network. "
        "Summarize what each peer told you."
    )

    # ``create_agent`` (LangChain 1.0) builds a LangGraph ReAct-style agent. The model
    # may be a string id (resolved to a provider) or any BaseChatModel instance.
    agent = create_agent("openai:gpt-4o-mini", tools=tools, system_prompt=system_prompt)

    task = (
        "Find a translation specialist on the rine directory, ask them (and wait for the "
        "reply) to translate 'Hello, partner' into French, then report what they said."
    )

    # Async-native: ``ainvoke`` drives the tools' ``_arun`` path (the async rine client).
    result = await agent.ainvoke({"messages": [{"role": "user", "content": task}]})

    # The final assistant turn is the agent's summary of the coordination round-trip.
    final = result["messages"][-1]
    print(getattr(final, "content", final))


if __name__ == "__main__":
    asyncio.run(main())
