"""Behavioral tests for RineCallbackHandler.

The handler is driven through its real public surface: instantiate it, then call
the LangChain lifecycle method directly (``on_agent_finish`` / ``on_chain_error``
— the dispatch a runtime invokes) and assert the observable effect: a rine
``send`` on the HTTP boundary.

Unlike an event-bus integration there is no process-wide event bus and no
ThreadPoolExecutor — LC callbacks run synchronously inline, so the whole
``_drain_event_bus`` / ``_emit_and_wait`` / ``scoped_handlers`` determinism
machinery is dropped. The content tests still decrypt the *actual wire body*
(not a shared boolean), so an assertion can never be satisfied by a leaked
sibling send.

A notification failure must NEVER crash an agent run: the failure-path
test points the handler at a failing send and asserts the lifecycle method does
not raise AND that a send was actually attempted (the failing route was called)
— so the swallow is proven, not vacuously passed by a handler that never fired.
"""

from __future__ import annotations

import json
import logging
from uuid import uuid4

import httpx
import respx
from langchain_core.agents import AgentFinish

from langchain_rine import RineCallbackHandler

from ..conftest import (
    AGENT_ID,
    API_URL,
    OTHER_AGENT_ID,
    decrypt_self_wire,
    mock_agents,
    mock_recipient_keys,
    mock_token,
    mock_webfinger,
    sent_message_payload,
)

NOTIFY_HANDLE = "ops@acme.rine.network"


def _mock_send(config_dir):
    """Mock the full send path the handler uses (token, resolve, encrypt, POST)."""
    mock_token()
    mock_agents()
    mock_webfinger()
    mock_recipient_keys(config_dir)
    return respx.post(f"{API_URL}/messages").mock(
        return_value=httpx.Response(200, json=sent_message_payload())
    )


def _mock_send_to_self(config_dir):
    """Mock a send addressed to AGENT_ID so the wire ciphertext is self-decryptable.

    The handler is pointed at the caller agent's own UUID, so no WebFinger is
    needed and the posted ``encrypted_payload`` is HPKE-sealed to a key we hold —
    letting a test decrypt the *actual notification text* off the wire.
    """
    mock_token()
    mock_agents()
    mock_recipient_keys(config_dir, AGENT_ID)
    return respx.post(f"{API_URL}/messages").mock(
        return_value=httpx.Response(200, json=sent_message_payload())
    )


def _agent_finish(output: str) -> AgentFinish:
    """Build the real LangChain ``AgentFinish`` payload a runtime would pass."""
    return AgentFinish(return_values={"output": output}, log="")


class TestRineCallbackHandler:
    @respx.mock
    def test_agent_finish_sends_one_best_effort_message(self, config_dir, other_keys):
        """A fired agent_finish event ⇒ exactly one rine send."""
        route = _mock_send(config_dir)

        handler = RineCallbackHandler(
            NOTIFY_HANDLE, config_dir=config_dir, api_url=API_URL, on=("agent_finish",)
        )
        handler.on_agent_finish(_agent_finish("all done"), run_id=uuid4())

        assert route.call_count == 1
        body = json.loads(route.calls[0].request.content)
        assert body["type"] == "rine.v1.text"

    @respx.mock
    def test_agent_finish_wire_body_carries_the_summary(self, config_dir):
        """The decrypted wire body equals the expected ``agent finished:`` summary.

        Decrypt the *actual* ``encrypted_payload`` (not a boolean) to prove the
        real notification content reached the wire — exactly one send whose
        decrypted ``{"text"}`` is the formatted summary.
        """
        route = _mock_send_to_self(config_dir)

        handler = RineCallbackHandler(
            AGENT_ID, config_dir=config_dir, api_url=API_URL, on=("agent_finish",)
        )
        handler.on_agent_finish(_agent_finish("all done"), run_id=uuid4())

        assert route.call_count == 1
        wire = json.loads(route.calls[0].request.content)["encrypted_payload"]
        plaintext = decrypt_self_wire(config_dir, wire)
        assert json.loads(plaintext)["text"] == "agent finished: all done"

    @respx.mock
    def test_chain_error_wire_body_carries_the_error(self, config_dir):
        """The decrypted body equals the ``chain error:`` summary (error verb)."""
        route = _mock_send_to_self(config_dir)

        handler = RineCallbackHandler(
            AGENT_ID, config_dir=config_dir, api_url=API_URL, on=("chain_error",)
        )
        handler.on_chain_error(RuntimeError("boom"), run_id=uuid4())

        assert route.call_count == 1
        wire = json.loads(route.calls[0].request.content)["encrypted_payload"]
        plaintext = decrypt_self_wire(config_dir, wire)
        assert json.loads(plaintext)["text"] == "chain error: boom"

    @respx.mock
    def test_send_failure_is_swallowed_and_debug_logged(self, config_dir, caplog):
        """A send failure is swallowed + debug-logged, never raised — but it fired.

        The recipient-key fetch 500s → ``send()`` raises internally; the handler
        swallows it. Asserting the failing route WAS called proves a send was
        actually attempted (a handler that silently never fired would also "not
        raise" — that weaker pass is what this guards against).
        """
        mock_token()
        mock_agents()
        mock_webfinger()
        keys_route = respx.get(f"{API_URL}/agents/{OTHER_AGENT_ID}/keys").mock(
            return_value=httpx.Response(500, json={"detail": "kaboom"})
        )

        handler = RineCallbackHandler(
            NOTIFY_HANDLE, config_dir=config_dir, api_url=API_URL, on=("agent_finish",)
        )
        with caplog.at_level(logging.DEBUG, logger="langchain_rine.callbacks"):
            # Must not raise even though the send fails.
            handler.on_agent_finish(_agent_finish("done"), run_id=uuid4())

        # reaching here == no exception propagated; the route call proves a send was tried.
        assert keys_route.called
        assert any("rine notification send failed" in r.message for r in caplog.records), (
            "expected the handler's swallow-path debug log to fire"
        )

    @respx.mock
    def test_selector_fires_only_for_selected_event(self, config_dir, other_keys):
        """Selector mapping: on=("agent_finish",) fires on_agent_finish, NOT on_chain_error."""
        route = _mock_send(config_dir)

        handler = RineCallbackHandler(
            NOTIFY_HANDLE, config_dir=config_dir, api_url=API_URL, on=("agent_finish",)
        )
        # chain_error is NOT selected → a chain error must NOT notify.
        handler.on_chain_error(RuntimeError("boom"), run_id=uuid4())
        assert not route.called

        # agent_finish IS selected → it sends.
        handler.on_agent_finish(_agent_finish("all done"), run_id=uuid4())
        assert route.call_count == 1
