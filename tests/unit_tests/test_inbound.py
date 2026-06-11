"""Behavioral tests for the Tier-3 resumer + drivers (inbound.py / drivers.py, D14-D17).

The resumer is exercised through its public interface (``handle_inbound`` / ``ahandle_inbound`` /
``register``) against a **real** fake ``StateGraph`` (one ``interrupt()`` node) compiled with a
real ``InMemorySaver`` + an :class:`InMemoryThreadMap`. Messages are real ``rine.DecryptedMessage``
objects. The driver is exercised over the **real** rine SDK with only the HTTP boundary mocked
(respx), reusing ``conftest``'s real config dir + on-disk keys + a genuine HPKE self-encrypt
round-trip so plaintext / I-2 run real crypto.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import httpx
import respx
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rine import DecryptedMessage

from langchain_rine._threadmap import InMemoryThreadMap
from langchain_rine.drivers import PollDriver, make_webhook_handler
from langchain_rine.inbound import ResumeStatus, RineThreadResumer

from ..conftest import (
    AGENT_ID,
    API_URL,
    CONVERSATION_ID,
    MESSAGE_ID,
    hpke_self_encrypt,
    mock_agents,
    mock_token,
)

SENDER = "peer@acme.rine.network"
THREAD = "thread-1"
SECOND_MESSAGE_ID = "cccccccc-dddd-eeee-ffff-000000000009"

# A wire ciphertext string that must NEVER reach a resume value (the I-2 guard).
CIPHERTEXT_MARKER = "ZW5jcnlwdGVkLWNpcGhlcnRleHQtbWFya2Vy"


# --- a real fake graph: one node that interrupts, echoing the resumed value ----------------


class _State(TypedDict):
    reply: str


def _ask_node(state: _State) -> dict[str, Any]:
    """Pause at ``interrupt``; on resume, store the HumanResponse's ``args`` text into state."""
    response = interrupt({"prompt": "awaiting rine reply"})
    return {"reply": str(response.get("args", ""))}


def _single_interrupt_graph(saver: InMemorySaver) -> Any:
    builder: StateGraph = StateGraph(_State)
    builder.add_node("ask", _ask_node)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    return builder.compile(checkpointer=saver)


def _parallel_a(state: _State) -> dict[str, Any]:
    interrupt({"who": "A"})
    return {}


def _parallel_b(state: _State) -> dict[str, Any]:
    interrupt({"who": "B"})
    return {}


def _two_interrupt_graph(saver: InMemorySaver) -> Any:
    """A graph that fans out to two nodes that BOTH interrupt in one superstep (A7 / #6533)."""
    builder: StateGraph = StateGraph(_State)
    builder.add_node("a", _parallel_a)
    builder.add_node("b", _parallel_b)
    builder.add_edge(START, "a")
    builder.add_edge(START, "b")
    builder.add_edge("a", END)
    builder.add_edge("b", END)
    return builder.compile(checkpointer=saver)


def _rearm_first(state: _State) -> dict[str, Any]:
    """First node: park at an interrupt, store the reply when resumed."""
    response = interrupt({"prompt": "first"})
    return {"reply": str(response.get("args", ""))}


def _rearm_second(state: _State) -> dict[str, Any]:
    """Second node: park at ANOTHER interrupt → the thread re-arms after the first resume."""
    interrupt({"prompt": "second"})
    return {}


def _rearming_graph(saver: InMemorySaver) -> Any:
    """A two-interrupt-in-sequence graph: resuming once parks it at the next node's interrupt."""
    builder: StateGraph = StateGraph(_State)
    builder.add_node("first", _rearm_first)
    builder.add_node("second", _rearm_second)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)
    return builder.compile(checkpointer=saver)


def _counting_rearming_graph(saver: InMemorySaver, counter: list[int]) -> Any:
    """A re-arming graph that records each time the FIRST node runs past its interrupt.

    ``counter`` is appended to once per first-node resume body so a test can assert the node ran
    **exactly once** even when a redelivered, already-consumed message is re-polled (D20: the
    second poll must be ``SKIPPED_ALREADY_CONSUMED``, never a re-resume that bumps the counter).
    """

    def _first(state: _State) -> dict[str, Any]:
        response = interrupt({"prompt": "first"})
        counter.append(1)
        return {"reply": str(response.get("args", ""))}

    def _second(state: _State) -> dict[str, Any]:
        interrupt({"prompt": "second"})
        return {}

    builder: StateGraph = StateGraph(_State)
    builder.add_node("first", _first)
    builder.add_node("second", _second)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)
    return builder.compile(checkpointer=saver)


def _raising_node(state: _State) -> dict[str, Any]:
    """Park at ``interrupt``; on resume, raise — exercises the resume try/except (inv-1)."""
    interrupt({"prompt": "awaiting reply"})
    raise RuntimeError("node blew up on resume")


def _raising_graph(saver: InMemorySaver) -> Any:
    """A graph whose node raises when resumed (the resume runs arbitrary user code)."""
    builder: StateGraph = StateGraph(_State)
    builder.add_node("boom", _raising_node)
    builder.add_edge(START, "boom")
    builder.add_edge("boom", END)
    return builder.compile(checkpointer=saver)


def _start(graph: Any, thread_id: str) -> None:
    """Run a graph to its first interrupt (it parks awaiting a resume)."""
    graph.invoke({"reply": ""}, {"configurable": {"thread_id": thread_id}})


def _message(
    *,
    plaintext: str | None = "hello reply",
    decrypt_error: str | None = None,
    sender_handle: str | None = SENDER,
    conversation_id: str = CONVERSATION_ID,
    msg_id: str = MESSAGE_ID,
    verification_status: str = "verified",
) -> DecryptedMessage:
    """Build a real ``DecryptedMessage`` (the resumer operates on these, not raw HTTP).

    Defaults to a ``verified`` signature so the resume value is clean plaintext; pass
    ``verification_status="unverifiable"`` / ``"invalid"`` to exercise the wh-2/D19 gate.
    """
    return DecryptedMessage(
        id=msg_id,
        conversation_id=conversation_id,
        from_agent_id=AGENT_ID,
        type="rine.v1.task_response",
        encrypted_payload=CIPHERTEXT_MARKER,
        created_at="2024-01-01T00:00:00Z",
        sender_handle=sender_handle,
        plaintext=None
        if decrypt_error
        else (None if plaintext is None else f'{{"text": "{plaintext}"}}'),
        decrypt_error=decrypt_error,
        verified=verification_status == "verified",
        verification_status=verification_status,  # type: ignore[arg-type]
    )


# =====================================================================
# RineThreadResumer — happy path + guards
# =====================================================================


class TestResumerHappyPath:
    def test_handle_inbound_when_thread_interrupted_resumes_and_completes(self):
        graph = _single_interrupt_graph(InMemorySaver())
        store = InMemoryThreadMap()
        resumer = RineThreadResumer(graph, store)
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        _start(graph, THREAD)

        result = resumer.handle_inbound(_message(plaintext="the answer is 42"))

        assert result.status is ResumeStatus.RESUMED
        assert result.thread_id == THREAD
        # the graph saw the plaintext at the interrupt() site and ran to completion
        state = graph.get_state({"configurable": {"thread_id": THREAD}})
        assert state.next == ()
        assert state.values["reply"] == "the answer is 42"

    def test_handle_inbound_resume_value_is_plaintext_not_ciphertext(self):
        """I-2: the resumed graph never receives the encrypted_payload string."""
        graph = _single_interrupt_graph(InMemorySaver())
        resumer = RineThreadResumer(graph, InMemoryThreadMap())
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        _start(graph, THREAD)

        resumer.handle_inbound(_message(plaintext="plain only"))

        state = graph.get_state({"configurable": {"thread_id": THREAD}})
        assert CIPHERTEXT_MARKER not in state.values["reply"]
        assert state.values["reply"] == "plain only"


class TestResumerGuards:
    def test_handle_inbound_when_no_thread_mapped_skips_no_thread(self):
        graph = _single_interrupt_graph(InMemorySaver())
        resumer = RineThreadResumer(graph, InMemoryThreadMap())  # nothing registered

        result = resumer.handle_inbound(_message())

        assert result.status is ResumeStatus.SKIPPED_NO_THREAD

    def test_handle_inbound_when_thread_not_interrupted_skips_not_interrupted(self):
        graph = _single_interrupt_graph(InMemorySaver())
        resumer = RineThreadResumer(graph, InMemoryThreadMap())
        # register + run to completion FIRST so the thread is finished, not parked
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        _start(graph, THREAD)
        graph.invoke(
            Command(resume={"type": "response", "args": "done"}),
            {"configurable": {"thread_id": THREAD}},
        )

        result = resumer.handle_inbound(_message())

        assert result.status is ResumeStatus.SKIPPED_NOT_INTERRUPTED

    def test_handle_inbound_when_multiple_interrupts_skips_without_bare_resume(self):
        """A7/#6533: a bare Command(resume) would RuntimeError; the guard must skip, not raise."""
        graph = _two_interrupt_graph(InMemorySaver())
        resumer = RineThreadResumer(graph, InMemoryThreadMap())
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        _start(graph, THREAD)
        state = graph.get_state({"configurable": {"thread_id": THREAD}})
        assert len(state.interrupts) > 1  # precondition: a genuine parallel-interrupt superstep

        result = resumer.handle_inbound(_message())  # must NOT raise RuntimeError

        assert result.status is ResumeStatus.SKIPPED_MULTIPLE_INTERRUPTS

    def test_handle_inbound_with_decrypt_error_skips_and_leaks_no_ciphertext(self):
        """I-2: a decrypt_error message is never resumed; ciphertext never enters the graph."""
        graph = _single_interrupt_graph(InMemorySaver())
        resumer = RineThreadResumer(graph, InMemoryThreadMap())
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        _start(graph, THREAD)

        result = resumer.handle_inbound(
            _message(plaintext=None, decrypt_error="mls-v1 not decryptable in Python")
        )

        assert result.status is ResumeStatus.SKIPPED_DECRYPT_ERROR
        # the graph is untouched: still parked, no ciphertext anywhere in its state
        state = graph.get_state({"configurable": {"thread_id": THREAD}})
        assert state.next == ("ask",)
        assert CIPHERTEXT_MARKER not in str(state.values)

    def test_handle_inbound_with_no_conversation_skips_no_conversation(self):
        graph = _single_interrupt_graph(InMemorySaver())
        resumer = RineThreadResumer(graph, InMemoryThreadMap())
        # default_key_fn yields conversation_id="" when conversation_id is missing — but the model
        # requires it, so use a custom key_fn that drops the conversation to exercise the guard.
        resumer = RineThreadResumer(graph, InMemoryThreadMap(), key_fn=lambda m: (SENDER, ""))

        result = resumer.handle_inbound(_message())

        assert result.status is ResumeStatus.SKIPPED_NO_CONVERSATION


class TestResumerRegistration:
    def test_register_same_key_new_thread_overwrites_and_warns(self, caplog):
        """D16 newest-wins: a re-registration overwrites + logs a WARNING."""
        store = InMemoryThreadMap()
        resumer = RineThreadResumer(_single_interrupt_graph(InMemorySaver()), store)
        resumer.register(SENDER, CONVERSATION_ID, "thread-old")

        with caplog.at_level(logging.WARNING, logger="langchain_rine.inbound"):
            resumer.register(SENDER, CONVERSATION_ID, "thread-new")

        assert store.get(SENDER, CONVERSATION_ID) == "thread-new"
        assert any("superseded" in r.message for r in caplog.records)

    def test_unregister_removes_binding(self):
        store = InMemoryThreadMap()
        resumer = RineThreadResumer(_single_interrupt_graph(InMemorySaver()), store)
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        resumer.unregister(SENDER, CONVERSATION_ID)
        assert store.get(SENDER, CONVERSATION_ID) is None


# =====================================================================
# A / D18 — terminal-unregister (completed threads stop shadowing later mail)
# =====================================================================


class TestTerminalUnregister:
    def test_single_interrupt_thread_unregisters_on_completion(self):
        """A single-interrupt thread that RESUMES to completion drops its binding (tm-1)."""
        graph = _single_interrupt_graph(InMemorySaver())
        store = InMemoryThreadMap()
        resumer = RineThreadResumer(graph, store)
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        _start(graph, THREAD)

        first = resumer.handle_inbound(_message(plaintext="done"))
        assert first.status is ResumeStatus.RESUMED
        # the completed binding is gone, so a SECOND message for the same conversation skips
        assert store.get(SENDER, CONVERSATION_ID) is None
        second = resumer.handle_inbound(_message(plaintext="late", msg_id=SECOND_MESSAGE_ID))
        assert second.status is ResumeStatus.SKIPPED_NO_THREAD

    def test_rearming_thread_keeps_binding_and_resumes_again(self):
        """A thread that re-arms at a new interrupt KEEPS its binding (resumes a second time)."""
        graph = _rearming_graph(InMemorySaver())
        store = InMemoryThreadMap()
        resumer = RineThreadResumer(graph, store)
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        _start(graph, THREAD)

        first = resumer.handle_inbound(_message(plaintext="one"))
        assert first.status is ResumeStatus.RESUMED
        # still parked at the second interrupt → binding retained
        assert store.get(SENDER, CONVERSATION_ID) == THREAD
        state = graph.get_state({"configurable": {"thread_id": THREAD}})
        assert state.next  # re-armed
        # a second inbound message resumes it again (and now completes → unregisters)
        second = resumer.handle_inbound(_message(plaintext="two", msg_id=SECOND_MESSAGE_ID))
        assert second.status is ResumeStatus.RESUMED
        assert store.get(SENDER, CONVERSATION_ID) is None


# =====================================================================
# D / wh-2 — signature-verification gating (a tampered sig never wakes a graph)
# =====================================================================


class TestVerificationGating:
    def test_invalid_signature_skips_unverified_and_leaves_graph_untouched(self):
        """wh-2: an INVALID (tampered/forged) signature never resumes; the graph stays parked."""
        graph = _single_interrupt_graph(InMemorySaver())
        resumer = RineThreadResumer(graph, InMemoryThreadMap())
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        _start(graph, THREAD)

        result = resumer.handle_inbound(
            _message(plaintext="forged", verification_status="invalid")
        )

        assert result.status is ResumeStatus.SKIPPED_UNVERIFIED
        state = graph.get_state({"configurable": {"thread_id": THREAD}})
        assert state.next == ("ask",)  # untouched
        assert CIPHERTEXT_MARKER not in str(state.values)

    def test_unverifiable_resumes_with_marker_plaintext_no_ciphertext(self):
        """wh-2: an UNVERIFIABLE sender resumes, but the value is annotated + carries plaintext."""
        graph = _single_interrupt_graph(InMemorySaver())
        resumer = RineThreadResumer(graph, InMemoryThreadMap())
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        _start(graph, THREAD)

        result = resumer.handle_inbound(
            _message(plaintext="maybe real", verification_status="unverifiable")
        )

        assert result.status is ResumeStatus.RESUMED
        reply = graph.get_state({"configurable": {"thread_id": THREAD}}).values["reply"]
        assert "[unverified" in reply  # the trust marker is present
        assert "maybe real" in reply  # the plaintext still reaches the graph (I-2 unchanged)
        assert CIPHERTEXT_MARKER not in reply  # never the ciphertext

    def test_verified_resumes_clean_without_marker(self):
        """wh-2: a VERIFIED message resumes with clean plaintext (no trust marker)."""
        graph = _single_interrupt_graph(InMemorySaver())
        resumer = RineThreadResumer(graph, InMemoryThreadMap())
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        _start(graph, THREAD)

        resumer.handle_inbound(_message(plaintext="trusted", verification_status="verified"))

        reply = graph.get_state({"configurable": {"thread_id": THREAD}}).values["reply"]
        assert reply == "trusted"  # clean, no marker


# =====================================================================
# C / inv-1 — per-message failure isolation (a raising node does not kill the sweep)
# =====================================================================


class TestResumeFailureIsolation:
    def test_raising_node_returns_failed_without_propagating(self):
        """A node that raises on resume → FAILED (no raise); the message is not acked."""
        graph = _raising_graph(InMemorySaver())
        resumer = RineThreadResumer(graph, InMemoryThreadMap())
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        _start(graph, THREAD)

        result = resumer.handle_inbound(_message())  # must NOT raise

        assert result.status is ResumeStatus.FAILED
        assert result.thread_id == THREAD

    def test_raising_node_failed_leaves_journal_clean_so_it_genuinely_retries(self):
        """D20 act-then-record: a raising-node FAILED must NOT journal the message (no phantom).

        ``invoke`` runs before ``mark_consumed`` (act-then-record), so a node that raises throws
        before the journal write — the message is NEVER recorded consumed. A FAILED has
        ``should_ack`` False, so the message stays ``new`` and is a genuine at-least-once retry,
        not a phantom ``SKIPPED_ALREADY_CONSUMED`` that never actually ran. Locks the line ordering
        against a future record-before-act refactor.
        """
        graph = _raising_graph(InMemorySaver())
        store = InMemoryThreadMap()
        resumer = RineThreadResumer(graph, store)
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        _start(graph, THREAD)

        result = resumer.handle_inbound(_message())

        assert result.status is ResumeStatus.FAILED
        assert result.should_ack is False  # a FAILED is never acked → stays new → retried
        assert store.is_consumed(SENDER, CONVERSATION_ID, MESSAGE_ID) is False  # journal clean

    def test_mark_consumed_store_error_degrades_to_failed_not_a_crash(self):
        """D20 residual (contract l.70-71): a ``mark_consumed`` store error → FAILED, not a crash.

        ``mark_consumed`` stays inside the resume try/except, so if the durable journal write
        raises AFTER a successful ``invoke``, the resumer degrades to FAILED (at-least-once on
        re-poll) and never propagates a raw exception that would kill the sweep (I-3). The mark
        never committed, so the entry stays absent (a re-poll re-resumes — the accepted
        at-least-once on-store-error residual).
        """
        graph = _single_interrupt_graph(InMemorySaver())
        store = InMemoryThreadMap()

        def _boom(*_args: object) -> None:
            raise RuntimeError("durable journal write failed")

        store.mark_consumed = _boom  # type: ignore[method-assign]
        resumer = RineThreadResumer(graph, store)
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        _start(graph, THREAD)

        result = resumer.handle_inbound(_message(plaintext="answer"))  # must NOT raise

        assert result.status is ResumeStatus.FAILED  # degraded, not crashed
        assert result.should_ack is False  # not acked → re-polled (at-least-once on store error)


# =====================================================================
# Async parity (D3)
# =====================================================================


class TestResumerAsync:
    async def test_ahandle_inbound_resumes_like_sync(self):
        graph = _single_interrupt_graph(InMemorySaver())
        resumer = RineThreadResumer(graph, InMemoryThreadMap())
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        await graph.ainvoke({"reply": ""}, {"configurable": {"thread_id": THREAD}})

        result = await resumer.ahandle_inbound(_message(plaintext="async answer"))

        assert result.status is ResumeStatus.RESUMED
        state = await graph.aget_state({"configurable": {"thread_id": THREAD}})
        assert state.next == ()
        assert state.values["reply"] == "async answer"

    async def test_ahandle_inbound_no_thread_skips_like_sync(self):
        graph = _single_interrupt_graph(InMemorySaver())
        resumer = RineThreadResumer(graph, InMemoryThreadMap())

        result = await resumer.ahandle_inbound(_message())

        assert result.status is ResumeStatus.SKIPPED_NO_THREAD


# =====================================================================
# PollDriver — ack only the RESUMED messages
# =====================================================================


def _inbox_two_messages(config_dir: str) -> dict[str, Any]:
    """A two-item inbox: one maps to the parked thread, one does not.

    Both carry real HPKE-self-encrypted plaintext (the SDK decrypts them with the on-disk key), so
    ``sender_handle`` + ``conversation_id`` flow through and I-2 runs real crypto.
    """
    mapped = {
        "id": MESSAGE_ID,
        "conversation_id": CONVERSATION_ID,
        "from_agent_id": AGENT_ID,
        "to_agent_id": AGENT_ID,
        "type": "rine.v1.task_response",
        "encrypted_payload": hpke_self_encrypt(config_dir, {"text": "mapped reply"}),
        "encryption_version": "hpke-v1",
        "sender_signing_kid": f"rine:{AGENT_ID}",
        "sender_handle": SENDER,
        "created_at": "2024-01-01T00:00:00Z",
    }
    unmapped = {
        **mapped,
        "id": SECOND_MESSAGE_ID,
        "conversation_id": "cccccccc-dddd-eeee-ffff-000000000088",
        "encrypted_payload": hpke_self_encrypt(config_dir, {"text": "unmapped reply"}),
        "sender_handle": "stranger@acme.rine.network",
    }
    return {"items": [mapped, unmapped], "total": 2}


def _inbox_one_mapped(config_dir: str, *, text: str = "mapped reply") -> dict[str, Any]:
    """A single-item inbox whose one message maps to the parked thread (SENDER/CONVERSATION_ID).

    Real HPKE-self-encrypted so ``client.read``/``inbox`` decrypt it and ``sender_handle`` /
    ``conversation_id`` flow through for the resume match. Used by the exactly-once headline test:
    the same M1 is fetched on two consecutive polls (the ack failed in between, so it stayed new).
    """
    return {
        "items": [
            {
                "id": MESSAGE_ID,
                "conversation_id": CONVERSATION_ID,
                "from_agent_id": AGENT_ID,
                "to_agent_id": AGENT_ID,
                "type": "rine.v1.task_response",
                "encrypted_payload": hpke_self_encrypt(config_dir, {"text": text}),
                "encryption_version": "hpke-v1",
                "sender_signing_kid": f"rine:{AGENT_ID}",
                "sender_handle": SENDER,
                "created_at": "2024-01-01T00:00:00Z",
            }
        ],
        "total": 1,
    }


def _self_keys_json(config_dir: str) -> dict[str, Any]:
    from rine._crypto.keys import (
        encryption_public_key_to_jwk,
        load_agent_keys,
        signing_public_key_to_jwk,
    )

    keys = load_agent_keys(config_dir, AGENT_ID)
    return {
        "agent_id": AGENT_ID,
        "encryption_public_key": encryption_public_key_to_jwk(keys.encryption.public_key),
        "signing_public_key": signing_public_key_to_jwk(keys.signing.public_key),
    }


class TestPollDriver:
    @respx.mock
    def test_poll_once_acks_only_resumed_messages(self, config_dir):
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        respx.get(f"{API_URL}/agents/{AGENT_ID}/messages").mock(
            return_value=httpx.Response(200, json=_inbox_two_messages(config_dir))
        )
        ack = respx.post(f"{API_URL}/agents/{AGENT_ID}/messages/mark-delivered").mock(
            return_value=httpx.Response(200, json={"marked": 1})
        )

        graph = _single_interrupt_graph(InMemorySaver())
        resumer = RineThreadResumer(graph, InMemoryThreadMap())
        resumer.register(SENDER, CONVERSATION_ID, THREAD)  # only the FIRST message maps
        _start(graph, THREAD)

        driver = PollDriver(resumer, config_dir=config_dir, api_url=API_URL)
        results = driver.poll_once()

        statuses = [r.status for r in results]
        assert ResumeStatus.RESUMED in statuses
        assert ResumeStatus.SKIPPED_NO_THREAD in statuses
        # only the RESUMED message id was acked; the skipped one stays in the inbox
        assert ack.called
        import json

        body = json.loads(ack.calls[0].request.content)
        assert body == {"message_ids": [MESSAGE_ID]}

    @respx.mock
    def test_poll_once_with_no_resumes_does_not_ack(self, config_dir):
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        respx.get(f"{API_URL}/agents/{AGENT_ID}/messages").mock(
            return_value=httpx.Response(200, json=_inbox_two_messages(config_dir))
        )
        ack = respx.post(f"{API_URL}/agents/{AGENT_ID}/messages/mark-delivered").mock(
            return_value=httpx.Response(200, json={"marked": 0})
        )

        graph = _single_interrupt_graph(InMemorySaver())
        resumer = RineThreadResumer(graph, InMemoryThreadMap())  # nothing registered → no resumes

        driver = PollDriver(resumer, config_dir=config_dir, api_url=API_URL)
        driver.poll_once()

        assert not ack.called  # a skip is never acked (D16: message stays in the inbox)

    @respx.mock
    async def test_apoll_once_matches_sync(self, config_dir):
        """D3 async parity: apoll_once produces the same statuses + ack policy as poll_once."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        respx.get(f"{API_URL}/agents/{AGENT_ID}/messages").mock(
            return_value=httpx.Response(200, json=_inbox_two_messages(config_dir))
        )
        ack = respx.post(f"{API_URL}/agents/{AGENT_ID}/messages/mark-delivered").mock(
            return_value=httpx.Response(200, json={"marked": 1})
        )

        graph = _single_interrupt_graph(InMemorySaver())
        resumer = RineThreadResumer(graph, InMemoryThreadMap())
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        await graph.ainvoke({"reply": ""}, {"configurable": {"thread_id": THREAD}})

        driver = PollDriver(resumer, config_dir=config_dir, api_url=API_URL)
        results = await driver.apoll_once()

        statuses = [r.status for r in results]
        assert ResumeStatus.RESUMED in statuses
        assert ResumeStatus.SKIPPED_NO_THREAD in statuses
        assert ack.called
        import json

        body = json.loads(ack.calls[0].request.content)
        assert body == {"message_ids": [MESSAGE_ID]}


class TestPollDriverFailureIsolation:
    @respx.mock
    def test_poll_once_processes_second_message_when_first_resume_raises(self, config_dir):
        """inv-1: a raising resume → FAILED for that message; the rest of the batch still runs."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        respx.get(f"{API_URL}/agents/{AGENT_ID}/messages").mock(
            return_value=httpx.Response(200, json=_inbox_two_messages(config_dir))
        )
        ack = respx.post(f"{API_URL}/agents/{AGENT_ID}/messages/mark-delivered").mock(
            return_value=httpx.Response(200, json={"marked": 0})
        )

        # the FIRST inbox message maps to a thread on a graph that RAISES on resume
        graph = _raising_graph(InMemorySaver())
        resumer = RineThreadResumer(graph, InMemoryThreadMap())
        resumer.register(SENDER, CONVERSATION_ID, THREAD)  # only the first message maps
        _start(graph, THREAD)

        driver = PollDriver(resumer, config_dir=config_dir, api_url=API_URL)
        results = driver.poll_once()  # must NOT raise

        statuses = [r.status for r in results]
        assert ResumeStatus.FAILED in statuses  # the raising message
        assert ResumeStatus.SKIPPED_NO_THREAD in statuses  # the SECOND message was still processed
        assert not ack.called  # neither a FAILED nor a SKIP is acked

    def test_run_survives_a_sweep_that_raises(self):
        """inv-1: a transient poll_once failure is logged; the loop continues, never propagates."""
        from unittest.mock import patch

        resumer = RineThreadResumer(_single_interrupt_graph(InMemorySaver()), InMemoryThreadMap())
        driver = PollDriver(resumer, config_dir="/nonexistent", api_url=API_URL)

        with patch.object(
            driver, "poll_once", side_effect=[RuntimeError("transient inbox failure"), []]
        ) as poll:
            driver.run(interval=0.0, max_iterations=2)  # must NOT raise

        assert poll.call_count == 2  # the loop kept going after the first sweep raised


def _webhook_message(config_dir: str, *, text: str = "webhook reply") -> dict[str, Any]:
    """A server message payload the webhook handler fetches via ``GET /messages/{id}``.

    Real HPKE-self-encrypted so the SDK's ``client.read`` decrypts it (D19: the handler trusts the
    authenticated fetch, never a caller-supplied body) and ``sender_handle`` / ``conversation_id``
    flow through for the resume match.
    """
    return {
        "id": MESSAGE_ID,
        "conversation_id": CONVERSATION_ID,
        "from_agent_id": AGENT_ID,
        "to_agent_id": AGENT_ID,
        "type": "rine.v1.task_response",
        "encrypted_payload": hpke_self_encrypt(config_dir, {"text": text}),
        "encryption_version": "hpke-v1",
        "sender_signing_kid": f"rine:{AGENT_ID}",
        "sender_handle": SENDER,
        "created_at": "2024-01-01T00:00:00Z",
    }


class TestWebhookHandler:
    @respx.mock
    def test_handler_fetches_by_id_then_resumes_and_acks(self, config_dir):
        """D19/wh-1: the handler takes a message_id, does the authenticated read, then resumes."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        read = respx.get(f"{API_URL}/messages/{MESSAGE_ID}").mock(
            return_value=httpx.Response(200, json=_webhook_message(config_dir))
        )
        ack = respx.post(f"{API_URL}/agents/{AGENT_ID}/messages/mark-delivered").mock(
            return_value=httpx.Response(200, json={"marked": 1})
        )

        graph = _single_interrupt_graph(InMemorySaver())
        resumer = RineThreadResumer(graph, InMemoryThreadMap())
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        _start(graph, THREAD)

        handle = make_webhook_handler(resumer, config_dir=config_dir, api_url=API_URL)
        result = handle(MESSAGE_ID)  # the route passes the id, NOT a caller-built message

        assert result.status is ResumeStatus.RESUMED
        assert read.called  # the authoritative message was fetched via the SDK, not trusted
        assert ack.called

    @respx.mock
    def test_handler_skip_is_not_acked(self, config_dir):
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        respx.get(f"{API_URL}/messages/{MESSAGE_ID}").mock(
            return_value=httpx.Response(200, json=_webhook_message(config_dir))
        )
        ack = respx.post(f"{API_URL}/agents/{AGENT_ID}/messages/mark-delivered").mock(
            return_value=httpx.Response(200, json={"marked": 0})
        )

        graph = _single_interrupt_graph(InMemorySaver())
        resumer = RineThreadResumer(graph, InMemoryThreadMap())  # nothing registered

        handle = make_webhook_handler(resumer, config_dir=config_dir, api_url=API_URL)
        result = handle(MESSAGE_ID)

        assert result.status is ResumeStatus.SKIPPED_NO_THREAD
        assert not ack.called


# =====================================================================
# Import-safety (I-1)
# =====================================================================

_IMPORT_GUARD = """
import socket
def _blocked(*a, **k):
    raise AssertionError("import-time network connection attempt!")
socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.create_connection = _blocked
import langchain_rine  # the 11 tools must import with no langgraph + no network
from langchain_rine.inbound import RineThreadResumer  # the resumer import must also be inert
from langchain_rine.drivers import PollDriver
print("OK")
"""


class TestInboundImportSafety:
    def test_inbound_import_performs_no_network_call(self):
        """I-1: importing the resumer/driver builds no client and hits no socket."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-c", _IMPORT_GUARD],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith("OK")


# =====================================================================
# inv-2 — missing-langgraph friendly [inbound]-extra hint (DX-regression guard)
# =====================================================================

# Run in a subprocess that BLOCKS langgraph via a sys.meta_path finder (so the rig's installed
# langgraph is hidden cleanly + self-contained); importing langchain_rine.inbound must then raise
# a ModuleNotFoundError whose message points at the [inbound] extra.
_MISSING_LANGGRAPH_GUARD = """
import sys

class _BlockLanggraph:
    def find_spec(self, name, path, target=None):
        if name == "langgraph" or name.startswith("langgraph."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return None

# drop anything langgraph already imported, then install the blocker first in the chain
for mod in [m for m in sys.modules if m == "langgraph" or m.startswith("langgraph.")]:
    del sys.modules[mod]
sys.meta_path.insert(0, _BlockLanggraph())

try:
    import langchain_rine.inbound  # noqa: F401
except ModuleNotFoundError as exc:
    assert "pip install langchain-rine[inbound]" in str(exc), str(exc)
    print("HINT_OK")
else:
    raise AssertionError("expected ModuleNotFoundError when langgraph is missing")
"""


class TestMissingLanggraphHint:
    def test_inbound_import_without_langgraph_raises_friendly_hint(self):
        """inv-2: a tools-only install (no langgraph) gets the `[inbound]` extra hint."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-c", _MISSING_LANGGRAPH_GUARD],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith("HINT_OK")


# =====================================================================
# D20 — exactly-once resume (THE headline: survives an ack failure + a re-arm)
# =====================================================================


class TestExactlyOnceAcrossAckFailure:
    @respx.mock
    def test_ack_failure_then_repoll_skips_already_consumed_no_second_resume(self, config_dir):
        """idem-1 (HIGH): an ack failure must NOT double-resume a re-armed multi-turn thread.

        Sequence:
          1. M1 maps to a re-arming graph parked at interrupt#1; poll → RESUMED, thread re-arms at
             interrupt#2. The driver records M1 consumed BEFORE attempting the ack.
          2. The ``mark_delivered`` POST returns 500 → the ack fails → M1 stays ``status="new"`` on
             the server (so the very same M1 is fetched again on the next poll).
          3. Re-poll the SAME M1 → ``SKIPPED_ALREADY_CONSUMED`` (the consumed journal recognizes
             it), NOT a second resume. The re-armed interrupt#2 keeps its OWN pending value and was
             never fed M1's stale plaintext.
        The first node's body must have run **exactly once** across both polls.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        # The same single-message inbox is returned on EVERY poll: M1 stays "new" because its ack
        # fails — exactly what the server does when mark_delivered fails. A 400 (non-retryable)
        # makes the SDK raise at once, so the failure is deterministic with no retry/backoff sleep.
        respx.get(f"{API_URL}/agents/{AGENT_ID}/messages").mock(
            return_value=httpx.Response(200, json=_inbox_one_mapped(config_dir, text="turn one"))
        )
        ack = respx.post(f"{API_URL}/agents/{AGENT_ID}/messages/mark-delivered").mock(
            return_value=httpx.Response(400, json={"detail": "ack rejected"})
        )

        counter: list[int] = []
        graph = _counting_rearming_graph(InMemorySaver(), counter)
        store = InMemoryThreadMap()
        resumer = RineThreadResumer(graph, store)
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        _start(graph, THREAD)

        driver = PollDriver(resumer, config_dir=config_dir, api_url=API_URL)

        # --- poll #1: resume succeeds, re-arms; the ack 500s (best-effort, swallowed) ---
        first = driver.poll_once()
        assert [r.status for r in first] == [ResumeStatus.RESUMED]
        assert ack.called  # the ack WAS attempted (and failed)
        cfg = {"configurable": {"thread_id": THREAD}}
        state_after_first = graph.get_state(cfg)
        assert state_after_first.next  # re-armed at interrupt#2
        assert state_after_first.values["reply"] == "turn one"  # interrupt#1 saw M1's plaintext
        assert counter == [1]  # the first node ran once
        # M1 stayed new (ack failed) but is recorded consumed; its journal entry is NOT dropped
        assert store.is_consumed(SENDER, CONVERSATION_ID, MESSAGE_ID) is True

        # --- poll #2: the SAME M1 is fetched again → must be skipped, not re-resumed ---
        second = driver.poll_once()
        assert [r.status for r in second] == [ResumeStatus.SKIPPED_ALREADY_CONSUMED]
        # the graph node did NOT run a second time; the re-armed interrupt#2 still holds its own
        # pending value and was never fed M1's stale plaintext
        assert counter == [1]
        state_after_second = graph.get_state(cfg)
        assert state_after_second.next  # still parked at interrupt#2, untouched
        assert state_after_second.values["reply"] == "turn one"  # unchanged

    def test_single_interrupt_redelivery_after_unregister_is_consumed_not_no_thread(self):
        """D20 guard order (locked): consumed-check runs BEFORE the no-thread lookup.

        A single-interrupt thread resumes to completion → ``mark_consumed`` records M1, then the
        terminal-unregister DELETES its binding. The SAME M1 redelivered (e.g. its ack was lost)
        now hits ``store.get`` returning None. The locked order (is_consumed BEFORE get) must
        recognize it as ``SKIPPED_ALREADY_CONSUMED`` (``should_ack`` True → re-acked, leaves the
        inbox), NOT strand it as ``SKIPPED_NO_THREAD`` (``should_ack`` False → re-fetched +
        re-decrypted every poll, the tm-1 starvation v0.3 closes). Reversing the guard order to
        get-then-is_consumed would silently regress this case (this test fails on the reversal).
        """
        graph = _single_interrupt_graph(InMemorySaver())
        store = InMemoryThreadMap()
        resumer = RineThreadResumer(graph, store)
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        _start(graph, THREAD)

        first = resumer.handle_inbound(_message(plaintext="done"))
        assert first.status is ResumeStatus.RESUMED
        assert store.get(SENDER, CONVERSATION_ID) is None  # terminal-unregistered: binding gone
        assert store.is_consumed(SENDER, CONVERSATION_ID, MESSAGE_ID) is True  # but journaled

        # the SAME M1 redelivered (binding gone, journal kept): consumed-check wins over no-thread
        second = resumer.handle_inbound(_message(plaintext="done"))
        assert second.status is ResumeStatus.SKIPPED_ALREADY_CONSUMED  # NOT SKIPPED_NO_THREAD
        assert second.should_ack is True  # re-acked so it leaves the inbox, never stranded

    @respx.mock
    def test_already_consumed_is_reacked_so_it_leaves_the_inbox(self, config_dir):
        """D20 ack policy: a redelivered already-consumed message is RE-acked (should_ack True).

        Once the journal recognizes M1 as consumed, the driver still acks it (so it finally leaves
        the inbox) — it just does not resume it again. Here the ack succeeds on the second poll.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        respx.get(f"{API_URL}/agents/{AGENT_ID}/messages").mock(
            return_value=httpx.Response(200, json=_inbox_one_mapped(config_dir))
        )
        # First ack is rejected (400, non-retryable → one POST, raises at once, no backoff); the
        # redelivery ack succeeds. Each poll thus makes exactly one mark-delivered POST.
        ack = respx.post(f"{API_URL}/agents/{AGENT_ID}/messages/mark-delivered").mock(
            side_effect=[
                httpx.Response(400, json={"detail": "rejected"}),
                httpx.Response(200, json={"marked": 1}),
            ]
        )

        graph = _counting_rearming_graph(InMemorySaver(), [])
        store = InMemoryThreadMap()
        resumer = RineThreadResumer(graph, store)
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        _start(graph, THREAD)

        driver = PollDriver(resumer, config_dir=config_dir, api_url=API_URL)
        driver.poll_once()  # RESUMED, ack 500s, journal kept
        second = driver.poll_once()  # SKIPPED_ALREADY_CONSUMED, ack succeeds → confirm self-cleans

        assert [r.status for r in second] == [ResumeStatus.SKIPPED_ALREADY_CONSUMED]
        assert ack.call_count == 2  # the consumed message was re-acked, not ignored
        # self-clean: the successful second ack dropped the journal entry (bounded growth, inv-2)
        assert store.is_consumed(SENDER, CONVERSATION_ID, MESSAGE_ID) is False


# =====================================================================
# D20 — should_ack policy + self-cleaning confirm_delivered
# =====================================================================


class TestShouldAckPolicy:
    def test_should_ack_true_only_for_resumed_and_already_consumed(self):
        """The ack policy is status-driven: RESUMED + SKIPPED_ALREADY_CONSUMED ack, else not."""
        from langchain_rine.inbound import ResumeResult

        ack = {ResumeStatus.RESUMED, ResumeStatus.SKIPPED_ALREADY_CONSUMED}
        for status in ResumeStatus:
            assert ResumeResult(status).should_ack is (status in ack), status

    def test_confirm_delivered_drops_consumed_entry(self):
        """D20 self-clean: confirm_delivered drops the journal entry for the acked message."""
        store = InMemoryThreadMap()
        resumer = RineThreadResumer(_single_interrupt_graph(InMemorySaver()), store)
        store.mark_consumed(SENDER, CONVERSATION_ID, MESSAGE_ID)

        resumer.confirm_delivered(_message())

        assert store.is_consumed(SENDER, CONVERSATION_ID, MESSAGE_ID) is False

    def test_confirm_delivered_only_drops_its_own_message(self):
        """A re-armed thread's later DISTINCT message is untouched by confirming an earlier one."""
        store = InMemoryThreadMap()
        resumer = RineThreadResumer(_single_interrupt_graph(InMemorySaver()), store)
        store.mark_consumed(SENDER, CONVERSATION_ID, MESSAGE_ID)
        store.mark_consumed(SENDER, CONVERSATION_ID, SECOND_MESSAGE_ID)

        resumer.confirm_delivered(_message())  # confirms only MESSAGE_ID

        assert store.is_consumed(SENDER, CONVERSATION_ID, MESSAGE_ID) is False
        assert store.is_consumed(SENDER, CONVERSATION_ID, SECOND_MESSAGE_ID) is True

    @respx.mock
    def test_poll_once_confirms_delivered_after_successful_ack(self, config_dir):
        """End-to-end self-clean: a successful ack drops the resumed message's journal entry."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        respx.get(f"{API_URL}/agents/{AGENT_ID}/messages").mock(
            return_value=httpx.Response(200, json=_inbox_one_mapped(config_dir))
        )
        respx.post(f"{API_URL}/agents/{AGENT_ID}/messages/mark-delivered").mock(
            return_value=httpx.Response(200, json={"marked": 1})
        )

        graph = _single_interrupt_graph(InMemorySaver())
        store = InMemoryThreadMap()
        resumer = RineThreadResumer(graph, store)
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        _start(graph, THREAD)

        driver = PollDriver(resumer, config_dir=config_dir, api_url=API_URL)
        results = driver.poll_once()

        assert [r.status for r in results] == [ResumeStatus.RESUMED]
        # the ack succeeded → confirm_delivered ran → the journal is back to empty (happy path)
        assert store.is_consumed(SENDER, CONVERSATION_ID, MESSAGE_ID) is False


# =====================================================================
# D21 — async-native store path (ahandle_inbound over an aiosqlite-backed store)
# =====================================================================


class TestAsyncStorePath:
    async def test_ahandle_inbound_via_sqlite_store_records_consumed(self, tmp_path):
        """D21: ahandle_inbound resumes through the async store + records via amark_consumed.

        Drives the resumer's async path end-to-end over an aiosqlite-backed SqliteThreadMap (the
        first runtime exercise of the async store from the resumer): a parked thread resumes and
        the message is journaled so a redelivery would be recognized as already-consumed.
        """
        from langchain_rine._threadmap import SqliteThreadMap

        store = SqliteThreadMap(str(tmp_path / "tm.sqlite"))
        graph = _single_interrupt_graph(InMemorySaver())
        resumer = RineThreadResumer(graph, store)
        await store.aput(SENDER, CONVERSATION_ID, THREAD)
        await graph.ainvoke({"reply": ""}, {"configurable": {"thread_id": THREAD}})

        result = await resumer.ahandle_inbound(_message(plaintext="async durable"))

        assert result.status is ResumeStatus.RESUMED
        state = await graph.aget_state({"configurable": {"thread_id": THREAD}})
        assert state.values["reply"] == "async durable"
        # recorded via amark_consumed → a redelivery would be SKIPPED_ALREADY_CONSUMED
        assert await store.ais_consumed(SENDER, CONVERSATION_ID, MESSAGE_ID) is True
        await store.aclose()

    async def test_ahandle_inbound_async_exactly_once_on_redelivery(self, tmp_path):
        """D20/D21 async: a redelivered already-consumed message is skipped on the async path."""
        from langchain_rine._threadmap import SqliteThreadMap

        store = SqliteThreadMap(str(tmp_path / "tm.sqlite"))
        counter: list[int] = []
        graph = _counting_rearming_graph(InMemorySaver(), counter)
        resumer = RineThreadResumer(graph, store)
        await store.aput(SENDER, CONVERSATION_ID, THREAD)
        await graph.ainvoke({"reply": ""}, {"configurable": {"thread_id": THREAD}})

        first = await resumer.ahandle_inbound(_message(plaintext="one"))
        second = await resumer.ahandle_inbound(_message(plaintext="one"))  # same id redelivered

        assert first.status is ResumeStatus.RESUMED
        assert second.status is ResumeStatus.SKIPPED_ALREADY_CONSUMED
        assert counter == [1]  # resumed exactly once
        await store.aclose()

    async def test_aconfirm_delivered_drops_via_adrop_consumed(self, tmp_path):
        """D21 async self-clean: aconfirm_delivered clears the async journal entry."""
        from langchain_rine._threadmap import SqliteThreadMap

        store = SqliteThreadMap(str(tmp_path / "tm.sqlite"))
        resumer = RineThreadResumer(_single_interrupt_graph(InMemorySaver()), store)
        await store.amark_consumed(SENDER, CONVERSATION_ID, MESSAGE_ID)

        await resumer.aconfirm_delivered(_message())

        assert await store.ais_consumed(SENDER, CONVERSATION_ID, MESSAGE_ID) is False
        await store.aclose()

    @respx.mock
    async def test_apoll_once_confirms_delivered_after_successful_ack_over_sqlite(
        self, config_dir, tmp_path
    ):
        """D20/D21 async end-to-end self-clean over the production aiosqlite store.

        Drives the FULL async driver ladder (apoll_once → ahandle_inbound → amark_consumed → _aack
        success → aconfirm_delivered → adrop_consumed) over a real ``SqliteThreadMap``. A
        successful ack must drop the resumed message's journal entry (``ais_consumed`` False) — the
        async twin of ``test_poll_once_confirms_delivered_after_successful_ack``. Catches a
        dropped/broken ``aconfirm_delivered`` wiring inside ``apoll_once`` (no prior async test).
        """
        from langchain_rine._threadmap import SqliteThreadMap

        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        respx.get(f"{API_URL}/agents/{AGENT_ID}/messages").mock(
            return_value=httpx.Response(200, json=_inbox_one_mapped(config_dir))
        )
        respx.post(f"{API_URL}/agents/{AGENT_ID}/messages/mark-delivered").mock(
            return_value=httpx.Response(200, json={"marked": 1})
        )

        store = SqliteThreadMap(str(tmp_path / "tm.sqlite"))
        graph = _single_interrupt_graph(InMemorySaver())
        resumer = RineThreadResumer(graph, store)
        await store.aput(SENDER, CONVERSATION_ID, THREAD)
        await graph.ainvoke({"reply": ""}, {"configurable": {"thread_id": THREAD}})

        driver = PollDriver(resumer, config_dir=config_dir, api_url=API_URL)
        results = await driver.apoll_once()

        assert [r.status for r in results] == [ResumeStatus.RESUMED]
        # the ack succeeded → aconfirm_delivered ran → the async journal is back to empty
        assert await store.ais_consumed(SENDER, CONVERSATION_ID, MESSAGE_ID) is False
        await store.aclose()

    @respx.mock
    async def test_apoll_once_ack_failure_then_redelivery_is_consumed_over_sqlite(
        self, config_dir, tmp_path
    ):
        """D20 async exactly-once through the driver: an ack failure does NOT double-resume.

        Async twin of ``test_ack_failure_then_repoll_skips_already_consumed_no_second_resume``,
        driven end-to-end through ``apoll_once`` over the aiosqlite store: poll #1 resumes +
        re-arms and records M1 consumed BEFORE the ack (which 400s and is swallowed); poll #2
        re-fetches the SAME M1 and must skip it as ``SKIPPED_ALREADY_CONSUMED``, no second ainvoke.
        """
        from langchain_rine._threadmap import SqliteThreadMap

        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        respx.get(f"{API_URL}/agents/{AGENT_ID}/messages").mock(
            return_value=httpx.Response(200, json=_inbox_one_mapped(config_dir, text="turn one"))
        )
        ack = respx.post(f"{API_URL}/agents/{AGENT_ID}/messages/mark-delivered").mock(
            return_value=httpx.Response(400, json={"detail": "ack rejected"})
        )

        store = SqliteThreadMap(str(tmp_path / "tm.sqlite"))
        counter: list[int] = []
        graph = _counting_rearming_graph(InMemorySaver(), counter)
        resumer = RineThreadResumer(graph, store)
        await store.aput(SENDER, CONVERSATION_ID, THREAD)
        await graph.ainvoke({"reply": ""}, {"configurable": {"thread_id": THREAD}})

        driver = PollDriver(resumer, config_dir=config_dir, api_url=API_URL)

        first = await driver.apoll_once()
        assert [r.status for r in first] == [ResumeStatus.RESUMED]
        assert ack.called  # the ack was attempted (and 400'd)
        assert counter == [1]  # the first node ran once
        assert await store.ais_consumed(SENDER, CONVERSATION_ID, MESSAGE_ID) is True  # recorded

        second = await driver.apoll_once()  # SAME M1 redelivered
        assert [r.status for r in second] == [ResumeStatus.SKIPPED_ALREADY_CONSUMED]
        assert counter == [1]  # resumed exactly once — no second ainvoke
        await store.aclose()


# =====================================================================
# D23 — require_verified opt-in
# =====================================================================


class TestRequireVerified:
    def test_unverifiable_is_skipped_when_require_verified(self):
        """D23 strict: with require_verified=True an UNVERIFIABLE sender is skipped, not woken."""
        graph = _single_interrupt_graph(InMemorySaver())
        resumer = RineThreadResumer(graph, InMemoryThreadMap(), require_verified=True)
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        _start(graph, THREAD)

        result = resumer.handle_inbound(
            _message(plaintext="maybe real", verification_status="unverifiable")
        )

        assert result.status is ResumeStatus.SKIPPED_UNVERIFIED
        state = graph.get_state({"configurable": {"thread_id": THREAD}})
        assert state.next == ("ask",)  # the graph stayed parked, never woken
        assert CIPHERTEXT_MARKER not in str(state.values)

    def test_unverifiable_resumes_with_marker_when_not_require_verified(self):
        """D23 default (v0.2 regression): require_verified=False annotates + resumes unverified."""
        graph = _single_interrupt_graph(InMemorySaver())
        resumer = RineThreadResumer(graph, InMemoryThreadMap())  # require_verified defaults False
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        _start(graph, THREAD)

        result = resumer.handle_inbound(
            _message(plaintext="maybe real", verification_status="unverifiable")
        )

        assert result.status is ResumeStatus.RESUMED
        reply = graph.get_state({"configurable": {"thread_id": THREAD}}).values["reply"]
        assert "[unverified" in reply  # annotated
        assert "maybe real" in reply  # plaintext still reaches the graph

    def test_invalid_is_skipped_regardless_of_require_verified(self):
        """An INVALID (tampered) signature is blocked even with require_verified=False."""
        graph = _single_interrupt_graph(InMemorySaver())
        resumer = RineThreadResumer(graph, InMemoryThreadMap(), require_verified=False)
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        _start(graph, THREAD)

        result = resumer.handle_inbound(
            _message(plaintext="forged", verification_status="invalid")
        )

        assert result.status is ResumeStatus.SKIPPED_UNVERIFIED


# =====================================================================
# D24 — reply-timeout helper (expire)
# =====================================================================


class TestExpire:
    def test_expire_resumes_parked_thread_with_sentinel_then_unregisters(self):
        """D24: expire wakes a parked thread with the synthetic note, then drops the binding."""
        graph = _single_interrupt_graph(InMemorySaver())
        store = InMemoryThreadMap()
        resumer = RineThreadResumer(graph, store)
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        _start(graph, THREAD)

        result = resumer.expire(SENDER, CONVERSATION_ID, note="[reply timeout]")

        assert result.status is ResumeStatus.RESUMED
        # the sentinel note reached the graph at the interrupt site (I-2: a literal, not content)
        state = graph.get_state({"configurable": {"thread_id": THREAD}})
        assert state.values["reply"] == "[reply timeout]"
        assert state.next == ()  # ran to completion
        # the binding was terminal-unregistered
        assert store.get(SENDER, CONVERSATION_ID) is None

    def test_expire_with_no_mapped_thread_skips_no_thread(self):
        """D24: expiring a conversation with no parked thread skips (nothing to wake)."""
        resumer = RineThreadResumer(_single_interrupt_graph(InMemorySaver()), InMemoryThreadMap())

        result = resumer.expire(SENDER, CONVERSATION_ID)

        assert result.status is ResumeStatus.SKIPPED_NO_THREAD


# =====================================================================
# D22 — PollDriver backs off when another owner holds the inbox lease
# =====================================================================


class TestPollDriverLease:
    @respx.mock
    def test_poll_once_backs_off_when_lease_held_by_another_owner(self, config_dir, tmp_path):
        """D22: a PollDriver whose lease another live owner holds fetches nothing, returns []."""
        from langchain_rine._lease import SqliteLease

        # No inbox/keys/ack mocks at all: if the driver fetched, respx would raise on the un-mocked
        # request — so passing proves it never reached the network (pure back-off).
        mock_token()
        mock_agents()

        db = str(tmp_path / "lease.sqlite")
        other_owner = SqliteLease(db, "agent-inbox")
        assert other_owner.acquire(30.0) is True  # a different live owner already holds the lease

        driver_lease = SqliteLease(db, "agent-inbox")
        resumer = RineThreadResumer(_single_interrupt_graph(InMemorySaver()), InMemoryThreadMap())
        driver = PollDriver(
            resumer, config_dir=config_dir, api_url=API_URL, lease=driver_lease, lease_ttl=30.0
        )

        results = driver.poll_once()

        assert results == []  # backed off — no fetch, no resume, no ack

    @respx.mock
    def test_poll_once_proceeds_when_it_holds_the_lease(self, config_dir, tmp_path):
        """D22: with an uncontended lease the driver acquires it and runs a normal sweep."""
        from langchain_rine._lease import SqliteLease

        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        respx.get(f"{API_URL}/agents/{AGENT_ID}/messages").mock(
            return_value=httpx.Response(200, json=_inbox_one_mapped(config_dir))
        )
        respx.post(f"{API_URL}/agents/{AGENT_ID}/messages/mark-delivered").mock(
            return_value=httpx.Response(200, json={"marked": 1})
        )

        db = str(tmp_path / "lease.sqlite")
        graph = _single_interrupt_graph(InMemorySaver())
        resumer = RineThreadResumer(graph, InMemoryThreadMap())
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        _start(graph, THREAD)

        driver = PollDriver(
            resumer,
            config_dir=config_dir,
            api_url=API_URL,
            lease=SqliteLease(db, "agent-inbox"),
            lease_ttl=30.0,
        )
        results = driver.poll_once()

        assert [r.status for r in results] == [ResumeStatus.RESUMED]  # the sweep ran normally

    @respx.mock
    async def test_apoll_once_backs_off_when_lease_held(self, config_dir, tmp_path):
        """D22 async: apoll_once also backs off (acquires the lease via asyncio.to_thread)."""
        from langchain_rine._lease import SqliteLease

        mock_token()
        mock_agents()

        db = str(tmp_path / "lease.sqlite")
        other_owner = SqliteLease(db, "agent-inbox")
        assert other_owner.acquire(30.0) is True

        resumer = RineThreadResumer(_single_interrupt_graph(InMemorySaver()), InMemoryThreadMap())
        driver = PollDriver(
            resumer,
            config_dir=config_dir,
            api_url=API_URL,
            lease=SqliteLease(db, "agent-inbox"),
            lease_ttl=30.0,
        )

        results = await driver.apoll_once()

        assert results == []  # backed off on the async path too

    @respx.mock
    def test_run_releases_the_lease_on_clean_shutdown(self, config_dir, tmp_path):
        """D22: ``run`` releases the lease in a ``finally`` so a clean shutdown frees the inbox.

        Run one bounded sweep holding a lease; afterward a FRESH owner on the same db+key must
        acquire immediately (no waiting out the ttl). A regression dropping the ``finally:
        release`` would strand the lease for the full ``lease_ttl`` → this acquire returns False.
        """
        from langchain_rine._lease import SqliteLease

        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        respx.get(f"{API_URL}/agents/{AGENT_ID}/messages").mock(
            return_value=httpx.Response(200, json=_inbox_one_mapped(config_dir))
        )
        respx.post(f"{API_URL}/agents/{AGENT_ID}/messages/mark-delivered").mock(
            return_value=httpx.Response(200, json={"marked": 1})
        )

        db = str(tmp_path / "lease.sqlite")
        graph = _single_interrupt_graph(InMemorySaver())
        resumer = RineThreadResumer(graph, InMemoryThreadMap())
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        _start(graph, THREAD)

        driver = PollDriver(
            resumer,
            config_dir=config_dir,
            api_url=API_URL,
            lease=SqliteLease(db, "agent-inbox"),
            lease_ttl=30.0,
        )
        driver.run(interval=0.0, max_iterations=1)  # one sweep, then a clean shutdown

        # a different owner can take the lease immediately — it was released, not left to expire
        assert SqliteLease(db, "agent-inbox").acquire(30.0) is True

    @respx.mock
    async def test_arun_releases_the_lease_on_clean_shutdown(self, config_dir, tmp_path):
        """D22 async twin: ``arun`` releases the lease in a ``finally`` (via asyncio.to_thread)."""
        from langchain_rine._lease import SqliteLease

        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        respx.get(f"{API_URL}/agents/{AGENT_ID}/messages").mock(
            return_value=httpx.Response(200, json=_inbox_one_mapped(config_dir))
        )
        respx.post(f"{API_URL}/agents/{AGENT_ID}/messages/mark-delivered").mock(
            return_value=httpx.Response(200, json={"marked": 1})
        )

        db = str(tmp_path / "lease.sqlite")
        graph = _single_interrupt_graph(InMemorySaver())
        resumer = RineThreadResumer(graph, InMemoryThreadMap())
        resumer.register(SENDER, CONVERSATION_ID, THREAD)
        await graph.ainvoke({"reply": ""}, {"configurable": {"thread_id": THREAD}})

        driver = PollDriver(
            resumer,
            config_dir=config_dir,
            api_url=API_URL,
            lease=SqliteLease(db, "agent-inbox"),
            lease_ttl=30.0,
        )
        await driver.arun(interval=0.0, max_iterations=1)

        assert SqliteLease(db, "agent-inbox").acquire(30.0) is True  # freed on clean shutdown
