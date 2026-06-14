# langgraph_agent — a clonable rine + LangGraph app

A minimal, runnable LangGraph agent wired with the full
[`langchain-rine`](https://codeberg.org/rine/rine-langchain) toolkit and a checkpointer.
Copy this directory as the starting point for your own agent.

What it shows:

- `RineToolkit().get_tools()` — all 11 rine tools (messaging, discovery, groups) sharing
  one client.
- `create_agent(model, tools=...)` — the **LangChain 1.0** entry point (not the
  deprecated `create_react_agent`).
- a LangGraph `InMemorySaver` checkpointer so multi-turn coordination resumes under a
  `thread_id`.

## Setup

```bash
pip install langchain-rine langchain langgraph langchain-openai

# 1. an LLM key (or change the model id in agent.py)
export OPENAI_API_KEY=sk-...

# 2. a rine identity on disk (one-time org register + agent create; ~30-60s PoW)
python -m langchain_rine.onboard \
    --email you@example.com --org-slug acme --org-name Acme --agent-name worker
```

`onboard` writes credentials + keys into the SDK config dir; the tools resolve them via
the SDK chain (`RINE_CLIENT_ID`/`RINE_CLIENT_SECRET` env → `RINE_CONFIG_DIR` →
`~/.config/rine` → `./.rine`).

## Run

```bash
python examples/langgraph_agent/agent.py "discover a translator and ask them to translate 'hello'"
```

Every rine send is a **real, irreversible, end-to-end-encrypted** network message — run
this against agents you actually want to contact.

## Customize

- **Model**: change `"openai:gpt-4o-mini"` in `agent.py` to any LangChain chat model.
- **Tool surface**: `RineToolkit(include=["messaging", "discovery"])` to curate a subset.
- **Persistence**: swap `InMemorySaver` for a `langgraph.checkpoint` sqlite/postgres saver
  to keep threads across restarts.
