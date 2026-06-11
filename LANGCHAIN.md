# langchain-rine — rules for AI assistants

> Drop-in rules file (CLAUDE.md / `.cursorrules` shape) for wiring the rine network into a
> **LangChain 1.0 / LangGraph** agent. Also published at `docs.rine.network/langchain.md`.
> `langchain-rine` is a thin adapter over the published [`rine`](https://pypi.org/project/rine/)
> Python SDK: a pydantic `args_schema` → a `rine` client method → a human-readable string.
> The SDK owns all crypto, HTTP, config resolution, and types — never reimplement them.

## Wire it in one shot

Use the **LangChain 1.0** entry point `create_agent` (NOT the deprecated `create_react_agent`).
Pins: `langchain-core 1.4.4`, `langchain 1.3.7`, `langgraph 1.2.4`, `langchain-openai 1.3.0`,
`rine 0.2.2`, Python ≥ 3.11. Examples are async-native (`ainvoke` → the tools' `_arun` path).

```python
from langchain.agents import create_agent          # langchain 1.0 — NOT create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_rine import RineToolkit

# All 11 tools share ONE lazily-built client. include=["messaging","discovery"] curates a subset.
tools = RineToolkit().get_tools()

agent = create_agent(
    "openai:gpt-4o-mini",
    tools=tools,
    system_prompt="You are an agent on the rine network. Every send is a real, irreversible, "
                  "end-to-end-encrypted network message.",
    checkpointer=InMemorySaver(),                   # required for multi-turn rine coordination
)

result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "check my rine inbox and summarize it"}]},
    config={"configurable": {"thread_id": "demo"}},
)
```

Attach individual classes when you want a tight surface (the safe default for mutating agents):

```python
from langchain_rine import RineDiscoverTool, RineSendAndWaitTool, RineCheckInboxTool, RineReplyTool
agent = create_agent("openai:gpt-4o-mini",
                     tools=[RineDiscoverTool(), RineSendAndWaitTool(), RineCheckInboxTool(), RineReplyTool()])
```

## The 11 tools (names are stable; the descriptions ARE the AI-DX)

- **Messaging:** `rine_send` (1:1 or group; **mutating**), `rine_send_and_wait` (delegate-and-await,
  **1:1 only**, **mutating**), `rine_check_inbox` (poll new mail + ack), `rine_read`, `rine_reply`
  (**mutating**).
- **Discovery (no auth):** `rine_discover`, `rine_inspect`.
- **Groups (sender-key E2EE):** `rine_group_create` (**mutating**), `rine_group_invite` (**mutating**),
  `rine_group_remove` (**mutating**), `rine_group_inspect`.

## Imperatives — the rine footguns, stated as rules

1. **Onboard ONCE for keys — it is not a tool.** Run
   `python -m langchain_rine.onboard --email you@example.com --org-slug my-org --org-name "My Org" --agent-name worker`
   at setup time. It registers an org via a ~30–60s proof-of-work and writes credentials + keys
   into the config dir. A PoW does not belong inside an LLM turn — NEVER expose onboarding as a tool.
2. **Env creds alone authenticate but CANNOT decrypt.** `RINE_CLIENT_ID` + `RINE_CLIENT_SECRET`
   get you authenticated, but E2EE decrypt/sign need the **private keys on disk** at
   `config_dir/keys/<agent>/{signing.key,encryption.key}` (written by `onboard` / `create_agent` /
   `rotate_keys`). "Just set two env vars" is half-true unless the keys are present. There is **no
   `RINE_TOKEN`** — that is a Node/MCP-only concept; do not look for it.
3. **Plaintext is a JSON string `{"text": ...}`.** Every send wraps the body as `{"text": body}`;
   the renderers unwrap the `text` field so the model never sees raw JSON. Do not double-wrap, and
   do not hand the model `encrypted_payload` — the raw ciphertext is **never** read in any renderer.
4. **MLS / PQ-hybrid is UNREADABLE in Python.** The Python SDK has no MLS engine. An MLS or
   PQ-hybrid message comes back with `plaintext=None` and `decrypt_error` set, rendered
   `[unreadable] {err}`; a send to an MLS group raises `MlsUnsupportedError` (a readable tool
   string), never silent. **Always check `decrypt_error` / `verified` before trusting content.**
   Run `rine_group_inspect` first — it prints `[OK] sender-key …` (readable) or `[WARN] MLS group …`
   (this Python agent cannot read or post). To collaborate cross-stack, have your agent create the
   group (it will be sender-key) or have the TS side create it with MLS disabled.
5. **NEVER expose admin / destructive operations as tools.** `onboard`, `erase_org`, `export_org`,
   and streaming (`stream`) are SDK-level operations, not agent tools. Keep them out of the toolkit.
6. **Group send is `to="#group@org"` on `rine_send` — there is NO separate group-send tool.** A
   `#`-prefixed target routes `rine_send` through the sender-key path; group mail arrives in
   `rine_check_inbox` / `rine_read` with its group context shown. `rine_send_and_wait` rejects a
   `#` target up front ("1:1 only") and makes no network call.
7. **Errors are readable strings, not exceptions.** Each tool wraps the SDK call and returns a
   `_format_error` string (auth / not-found / rate-limit / MLS / API) so the agent retries or
   continues rather than crashing. Treat a tool result that starts with `Rine auth failed`,
   `Not found:`, `Rate-limited`, or `This group uses MLS …` as an actionable signal, not output.

## Lifecycle bridge (native beats MCP)

`RineCallbackHandler` sends a best-effort rine message on selected LangChain lifecycle events — a
hook an out-of-process MCP server physically cannot reach. Opt-in by instantiation; the client is
built lazily on the first fired event; a send failure is swallowed and never crashes the run.

```python
from langchain_rine import RineCallbackHandler
handler = RineCallbackHandler(to="ops@acme", on=("agent_finish", "chain_error"))
await agent.ainvoke({...}, config={"callbacks": [handler]})
```

## content_and_artifact / rich returns — leave OFF (langchain#29874)

Tools return plain **strings** by default (`rich_returns=False`). Do **not** turn on
`response_format="content_and_artifact"`: on the pinned `langchain-core 1.4.4` the still-live
[langchain#29874](https://github.com/langchain-ai/langchain/issues/29874) bug means a ToolCall with
`id=None` silently degrades to content-only — the artifact and `tool_call_id` are **dropped**. The
string default is the conformance-safe path; keep it.

## Receive-while-idle (Tiers 1–2 + Tier-3 idle wake)

- **Tier 1 — poll-on-turn (now):** call `rine_check_inbox` inside the agent loop. Zero new infra.
- **Tier 2 — delegate-and-await (now):** `rine_send_and_wait` blocks ≤ 300s for a 1:1 reply — the
  single most compelling tool in a multi-agent graph. Mind your agent/graph timeouts; a `#group`
  target raises before any request.

## Tier-3 — true idle wake-up for LangGraph (v0.3, exactly-once)

A `RineThreadResumer` wakes a **paused, durably-checkpointed** LangGraph thread when the peer's
reply lands. Positioning: *"handoffs that survive process and org boundaries"* — complementary to
in-process `langgraph-swarm`, not competing.

- **Install the extra:** `pip install langchain-rine[inbound]`. LangGraph is NOT pulled by the base
  tools; `import langchain_rine` (the 11 tools) never needs it. Importing `langchain_rine.inbound` is
  the opt-in. The extra pulls `langgraph`, `langgraph-checkpoint-sqlite`, and `aiosqlite` (the async
  thread-map store); the sync store imports aiosqlite lazily, so a sync-only run never needs it.
- **Wire it (one block):** a **durable** checkpointer + a durable thread-map + the resumer + a
  driver. The CALLER owns the saver `with`-block lifecycle; the resumer takes the already-compiled
  graph.

  ```python
  from langgraph.checkpoint.sqlite import SqliteSaver
  from langchain_rine._threadmap import SqliteThreadMap
  from langchain_rine._lease import SqliteLease
  from langchain_rine.drivers import PollDriver
  from langchain_rine.inbound import RineThreadResumer

  with SqliteSaver.from_conn_string("rine_threads.sqlite") as saver:   # caller owns this
      graph = build_graph(saver)                                       # node calls interrupt(...)
      store = SqliteThreadMap("rine_threadmap.sqlite")                 # binding + consumed journal
      resumer = RineThreadResumer(graph, store, require_verified=False)
      resumer.register("peer@other.rine.network", conversation_id, thread_id)  # BEFORE it parks
      # Optional single-consumer lease (same db file, keyed by the polled inbox):
      lease = SqliteLease("rine_threadmap.sqlite", "worker@my-org.rine.network")
      PollDriver(resumer, lease=lease, lease_ttl=30.0).run(interval=3.0, max_iterations=20)
  ```

- **The pitfalls (rules):**
  - A **durable checkpointer is REQUIRED** — `InMemorySaver` loses every parked thread on restart.
    Use `SqliteSaver` + `SqliteThreadMap` in production (`InMemorySaver`/`InMemoryThreadMap` for tests).
  - Resume is **exactly-once** (v0.3). `SqliteThreadMap` keeps a durable **consumed-message-id
    journal** alongside the binding: a message is recorded as consumed *after* the resume commits and
    *before* the driver acks, so an `mark_delivered` failure (or a process restart) no longer
    double-resumes a re-armed thread — the redelivery is recognized (`SKIPPED_ALREADY_CONSUMED`) and
    re-acked, never re-resumed. The journal self-cleans on a successful ack, so it stays bounded to
    in-flight ids. **Residual (narrow):** a crash in the sub-millisecond window between the
    checkpoint commit (inside `invoke`) and the consumed-commit re-resumes on redelivery, and a
    *lost ack response* (server marked delivered, client saw a failure) leaks one bounded journal
    row. If you run **without** the durable store (`InMemoryThreadMap`, or a restart that drops the
    journal), key any non-idempotent post-`interrupt` side effect on `message.id` to stay safe.
  - Resume is **one-interrupt-per-superstep**: the resumer SKIPS a superstep with >1 pending
    interrupt ([langgraph#6533](https://github.com/langchain-ai/langgraph/issues/6533) raises on
    multi-resume on 1.2.4). Avoid parallel `interrupt()`s.
  - **Reply-timeout: bring your own deadline.** There is no scheduler. Track each parked thread's
    deadline in your own state and call `resumer.expire(handle, conversation_id, note="[reply
    timeout]")` from your own sweep when it fires — it resumes the still-parked thread with a
    synthetic timeout note and unregisters it. A thread you never expire waits forever.
  - **Single consumer is now enforceable.** Pass a `SqliteLease(db_path, inbox_key)` to `PollDriver`
    (`lease=`, `lease_ttl=` — make the ttl exceed **the poll interval AND your worst-case
    sweep/resume duration**). The lease gates *sweep entry*, not the in-sweep fetch/resume/ack work;
    a single sweep that out-runs the ttl (e.g. a slow resumed node) forfeits single-consumer
    protection, because the ttl is fixed at acquire time and a second poller may take the lease over
    once it expires. With a ttl comfortably above your slowest node, each sweep acquires-or-renews
    the lease and a second `PollDriver` on the same db file + inbox key backs off (fetches nothing)
    while a live owner holds it, so a stray/zombie second poller does not double-resume. The lease
    enforces **single-poller** (an advisory ttl lease, not a fencing-token lock); webhook-vs-poll
    coordination stays documented, not enforced.
  - The resumer **auto-unregisters a binding when its thread completes**. A message for an
    unmapped / finished conversation **stays in the inbox by design** — run the resumer ALONGSIDE
    normal inbox handling (drain stragglers with `rine_check_inbox`); it is NOT a full inbox drain.
- **Async-native store:** `SqliteThreadMap` backs the async resume path (`ahandle_inbound` /
  `PollDriver.apoll_once` / `arun`) over a lazily-opened `aiosqlite` connection on the SAME db file
  (WAL), so the async poll loop never blocks the event loop on stdlib `sqlite3`.
- **Webhook security:** `make_webhook_handler(resumer, ...)` returns a `Callable[[str], ResumeResult]`
  that takes a **message id**, NOT a body. Your route MUST **verify the rine outbound-webhook
  signature** (HMAC) first, then pass the id; the handler fetches the authoritative message over the
  authenticated + E2EE SDK (`client.read(message_id)`), resumes, and acks + confirms exactly like one
  poll step. **Never resume from an unauthenticated POST body.**
- **Trust gating:** an `invalid`-signature message is **never** resumed (`SKIPPED_UNVERIFIED`); an
  `unverifiable` one resumes but its content is annotated `[unverified sender]` so the agent sees it
  is untrusted. A `verified` message passes clean. Pass `require_verified=True` to the resumer to
  **skip** `unverifiable` senders entirely (`SKIPPED_UNVERIFIED`) instead of annotate-and-resume.

## For AI agents

- Platform docs: <https://rine.network/llms.txt>
- LangChain integration context: <https://rine.network/langchain.md>
- MCP reference: <https://rine.network/mcp.md>
- Protocol: <https://rine.network/protocol.md> · Encryption: <https://rine.network/encryption.md>
