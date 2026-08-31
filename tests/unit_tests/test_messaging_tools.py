"""Behavioral tests for the messaging tools.

Each tool is exercised through its public LangChain interface (``tool.invoke({...})``
for sync, ``await tool.ainvoke({...})`` for async); assertions are on the
human-readable return string and observable HTTP. Only the rine HTTP boundary is
mocked (respx) — the SDK crypto/config/HTTP is real.

The ciphertext-never-leaks invariant is asserted by
encrypting a marker plaintext with a real HPKE round-trip and checking the raw
ciphertext string never appears in any tool return — on BOTH the sync ``.invoke`` and
async ``.ainvoke`` paths. Each ``_run`` test has an ``_arun`` parity twin
driving ``.ainvoke({...})`` against the SAME respx mock and asserting the SAME string.
``asyncio_mode = "auto"`` (pyproject) means async tests need no per-test marker.
"""

from __future__ import annotations

import json
import logging

import httpx
import respx
from rine import TRANSCRIPT_REFERENCE_REQUIRED, not_a_conversation_uuid
from rine._constants import EncryptionVersion, MessageType

from langchain_rine import (
    RineInboxTool,
    RineReadTool,
    RineReplyTool,
    RineSendAndWaitTool,
    RineSendTool,
    RineThreadTool,
)
from langchain_rine.tools._schemas import RineThreadInput
from langchain_rine.tools.groups_list import RineGroupRosterInput

from ..conftest import (
    AGENT_ID,
    API_URL,
    CONVERSATION_ID,
    GROUP_HANDLE,
    GROUP_ID,
    MESSAGE_ID,
    OTHER_AGENT_ID,
    OTHER_HANDLE,
    REPLY_ID,
    establish_mls_group_state,
    group_payload,
    hpke_peer_encrypt,
    hpke_self_encrypt,
    message_payload,
    mock_agents,
    mock_group_row,
    mock_recipient_keys,
    mock_sender_keys_unverifiable,
    mock_token,
    mock_webfinger,
    sent_message_payload,
)


def _send_tool(config_dir):
    return RineSendTool(config_dir=config_dir, api_url=API_URL)


# =====================================================================
# RineSendTool
# =====================================================================


class TestRineSendTool:
    @respx.mock
    def test_send_to_agent_when_recipient_exists_returns_confirmation(
        self, config_dir, other_keys
    ):
        mock_token()
        mock_agents()
        mock_webfinger()
        mock_recipient_keys(config_dir)
        route = respx.post(f"{API_URL}/messages").mock(
            return_value=httpx.Response(200, json=sent_message_payload())
        )

        out = _send_tool(config_dir).invoke({"to": OTHER_HANDLE, "body": "hello there"})

        assert route.called
        assert f"Sent message {MESSAGE_ID}" in out
        assert OTHER_HANDLE in out
        assert CONVERSATION_ID in out

    @respx.mock
    async def test_send_arun_matches_run(self, config_dir, other_keys):
        """``.ainvoke`` returns the same confirmation as ``.invoke`` (same mock).

        Also asserts the ciphertext-absence invariant holds on the async path.
        """
        mock_token()
        mock_agents()
        mock_webfinger()
        mock_recipient_keys(config_dir)
        route = respx.post(f"{API_URL}/messages").mock(
            return_value=httpx.Response(200, json=sent_message_payload())
        )

        out = await _send_tool(config_dir).ainvoke({"to": OTHER_HANDLE, "body": "hello there"})

        assert route.called
        assert f"Sent message {MESSAGE_ID}" in out
        assert OTHER_HANDLE in out
        assert CONVERSATION_ID in out
        assert "hello there" not in out  # ciphertext-absence holds on the async path too

    @respx.mock
    def test_send_auto_hpke_encrypts_body_on_the_wire(self, config_dir, other_keys):
        """The posted body carries HPKE ciphertext, not the plaintext 'hello there'."""
        mock_token()
        mock_agents()
        mock_webfinger()
        mock_recipient_keys(config_dir)
        route = respx.post(f"{API_URL}/messages").mock(
            return_value=httpx.Response(200, json=sent_message_payload())
        )

        _send_tool(config_dir).invoke({"to": OTHER_HANDLE, "body": "hello there"})

        body = json.loads(route.calls[0].request.content)
        assert body["encryption_version"] == EncryptionVersion.HPKE_V1
        assert "encrypted_payload" in body
        assert "hello there" not in route.calls[0].request.content.decode()

    @respx.mock
    def test_send_forwards_idempotency_key_and_message_type(self, config_dir, other_keys):
        """idempotency_key → 'Idempotency-Key' header; message_type → body 'type'.

        A refactor that dropped the dedup key would cause duplicate sends on retry;
        one that corrupted the type would mis-route. Both are silently untested
        otherwise — this pins them on the POST /messages request.
        """
        mock_token()
        mock_agents()
        mock_webfinger()
        mock_recipient_keys(config_dir)
        route = respx.post(f"{API_URL}/messages").mock(
            return_value=httpx.Response(200, json=sent_message_payload())
        )

        _send_tool(config_dir).invoke(
            {
                "to": OTHER_HANDLE,
                "body": "hi",
                "idempotency_key": "k1",
                "message_type": "rine.v1.custom",
            }
        )

        request = route.calls[0].request
        assert request.headers.get("Idempotency-Key") == "k1"
        assert json.loads(request.content)["type"] == "rine.v1.custom"

    @respx.mock
    def test_send_to_bare_name_that_answers_to_no_group_names_both_spellings(self, config_dir):
        """A bare target is read as a group reference, and refused as one.

        It cannot be an agent identifier — the agent namespace declines anything
        without an ``@`` before any request goes out — so the SDK asks this org's
        groups instead, and the refusal names both spellings rather than sending
        a group question to the agent directory.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [
                        group_payload(
                            name="logistics",
                            handle="#logistics@acme.rine.network",
                            group_id="cccccccc-dddd-eeee-ffff-000000000000",
                        )
                    ]
                },
            )
        )

        out = _send_tool(config_dir).invoke({"to": "nonexistent", "body": "hi"})

        assert "Not found: 'nonexistent' names neither an agent nor a group you can reach." in out
        # 🔴 The lead stopped naming the GROUP namespace, and the
        # remedy must not follow it: this refusal still ends on the group half of
        # the directory. If the predicate in ``_client._not_found_message`` ever
        # narrows back to ``startswith(_GROUP_NOT_FOUND_LEAD)``, this line is the
        # only thing between here and every bare-name miss being pointed at
        # ``rine_discover``, which cannot resolve a group handle.
        assert out.rstrip().endswith("Try rine_discover_groups to search the public directory.")
        # Both spellings, as worked values. The template this used to show
        # (``#group@org``) is gone because a model reading a refusal that names
        # the shape of an answer fills the shape in rather than supplying a real
        # one — so the assertion is that the sentence carries a group handle
        # somebody could actually send to, and says what the bare name was read
        # as. That half used to say a bare name was neither, which stopped being
        # true the day bare names resolved.
        assert "#logistics@acme.rine.network" in out
        assert "A bare name is read as a group this org is seated in" in out
        assert "rine_discover_groups" in out

    @respx.mock
    def test_send_to_missing_agent_returns_not_found_string(self, config_dir):
        """A 404 from the recipient-key fetch maps to the not-found string."""
        mock_token()
        mock_agents()
        mock_webfinger()
        respx.get(f"{API_URL}/agents/{OTHER_AGENT_ID}/keys").mock(
            return_value=httpx.Response(404, json={"detail": "agent not found"})
        )

        out = _send_tool(config_dir).invoke({"to": OTHER_HANDLE, "body": "hi"})

        assert "Not found" in out
        assert "agent not found" in out

    @respx.mock
    def test_send_to_mls_group_posts_a_real_mls_message(self, config_dir):
        """A send to a PQ MLS group genuinely encrypts and posts an ``mls-v1`` message.

        The connector operates MLS groups now: with local state established (a real
        founder admitted this agent), the send encrypts through the MLS core and hits
        ``POST /messages`` — no ceiling, no silent failure.
        """
        mock_token()
        mock_agents()
        establish_mls_group_state(config_dir)
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(
                200,
                json={"items": [group_payload(mls_group_id="mls-abc", mls_pending=False)]},
            )
        )
        send_route = respx.post(f"{API_URL}/messages").mock(
            return_value=httpx.Response(
                200, json={**sent_message_payload(), "group_id": GROUP_ID, "to_agent_id": None}
            )
        )

        out = _send_tool(config_dir).invoke({"to": GROUP_HANDLE, "body": "team update"})

        assert send_route.called
        # An MLS group send posts a to_handle (the group), not a 1:1 to_agent_id.
        body = json.loads(send_route.calls[0].request.content)
        assert body["to_handle"] == GROUP_HANDLE
        assert body["encryption_version"] == EncryptionVersion.MLS_V1
        assert "team update" not in send_route.calls[0].request.content.decode()  # ciphertext only
        assert f"Sent message {MESSAGE_ID}" in out
        assert GROUP_HANDLE in out

    @respx.mock
    async def test_send_to_hash_handle_routes_through_mls_group_path(self, config_dir):
        """A ``#``-prefixed target routes through the real MLS group path (queries /groups).

        ``rine_send`` is the group send tool too — a ``#`` handle drives the SDK's group
        resolution (GET /groups) and, with local MLS state, a genuine ``mls-v1`` send.
        Proven on the async path so the routing holds under ``.ainvoke`` as well.
        """
        mock_token()
        mock_agents()
        establish_mls_group_state(config_dir)
        groups_route = respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(
                200,
                json={"items": [group_payload(mls_group_id="mls-abc", mls_pending=False)]},
            )
        )
        send_route = respx.post(f"{API_URL}/messages").mock(
            return_value=httpx.Response(
                200, json={**sent_message_payload(), "group_id": GROUP_ID, "to_agent_id": None}
            )
        )

        out = await _send_tool(config_dir).ainvoke({"to": GROUP_HANDLE, "body": "team update"})

        assert groups_route.called  # group routing engaged, not the 1:1 /messages path
        assert send_route.called
        assert f"Sent message {MESSAGE_ID}" in out

    @respx.mock
    def test_send_never_raises_on_error(self, config_dir):
        """``_run`` must return a string even on error, never raise out of the tool."""
        mock_token()
        mock_agents()
        out = _send_tool(config_dir).invoke({"to": "nonexistent", "body": "hi"})
        assert isinstance(out, str)


# =====================================================================
# RineSendAndWaitTool
# =====================================================================


def _send_and_wait_reply_response(reply_ct):
    """The ``/messages/sync`` payload carrying an HPKE-encrypted reply to the caller."""
    return {
        "message": sent_message_payload(),
        "reply": message_payload(
            encryption_version=EncryptionVersion.HPKE_V1,
            encrypted_payload=reply_ct,
            msg_id=REPLY_ID,
            from_agent_id=AGENT_ID,
            msg_type=MessageType.TASK_RESPONSE,
        ),
        "conversation_id": CONVERSATION_ID,
        "status": "completed",
    }


class TestRineSendAndWaitTool:
    @respx.mock
    def test_send_and_wait_when_reply_arrives_returns_plaintext(self, config_dir, other_keys):
        mock_token()
        mock_agents()
        mock_webfinger()
        mock_recipient_keys(config_dir)
        # Reply is HPKE-encrypted to the caller agent so it decrypts to readable text.
        reply_ct = hpke_self_encrypt(config_dir, {"text": "ack: done"})
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        respx.post(f"{API_URL}/messages/sync").mock(
            return_value=httpx.Response(200, json=_send_and_wait_reply_response(reply_ct))
        )

        out = RineSendAndWaitTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"to": OTHER_HANDLE, "body": "do it", "wait_seconds": 5}
        )

        assert f"Reply from {OTHER_HANDLE}" in out
        assert "ack: done" in out
        # Exact verification note (M5): a fetchable sender key → 'signature verified'.
        assert "(signature verified)" in out
        # ciphertext never leaks into the return
        assert reply_ct not in out

    @respx.mock
    async def test_send_and_wait_arun_matches_run(self, config_dir, other_keys):
        """``.ainvoke`` returns the same plaintext reply as ``.invoke`` (same mock).

        Carries the async ciphertext-absence assertion on the reply path.
        """
        mock_token()
        mock_agents()
        mock_webfinger()
        mock_recipient_keys(config_dir)
        reply_ct = hpke_self_encrypt(config_dir, {"text": "ack: done"})
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        respx.post(f"{API_URL}/messages/sync").mock(
            return_value=httpx.Response(200, json=_send_and_wait_reply_response(reply_ct))
        )

        out = await RineSendAndWaitTool(config_dir=config_dir, api_url=API_URL).ainvoke(
            {"to": OTHER_HANDLE, "body": "do it", "wait_seconds": 5}
        )

        assert f"Reply from {OTHER_HANDLE}" in out
        assert "ack: done" in out
        assert "(signature verified)" in out
        assert reply_ct not in out  # ciphertext-absence on the async reply path

    @respx.mock
    def test_send_and_wait_unverifiable_reply_renders_signature_unverifiable(
        self, config_dir, other_keys
    ):
        """M5: an unverifiable reply renders '(signature unverifiable)', not 'verified'.

        The reply body still decrypts (recipient key local); the sender SIGNING key
        fetch 503s → verified=False, status='unverifiable'. Pins the exact rendered
        note so a hardcoded 'verified' suffix is caught.
        """
        mock_token()
        mock_agents()
        mock_webfinger()
        mock_recipient_keys(config_dir)
        reply_ct = hpke_self_encrypt(config_dir, {"text": "maybe done"})
        # Sender signing-key fetch fails → reply is unverifiable (but still decrypts).
        mock_sender_keys_unverifiable()  # GET /agents/{AGENT_ID}/keys → 503
        respx.post(f"{API_URL}/messages/sync").mock(
            return_value=httpx.Response(200, json=_send_and_wait_reply_response(reply_ct))
        )

        out = RineSendAndWaitTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"to": OTHER_HANDLE, "body": "do it", "wait_seconds": 5}
        )

        assert "maybe done" in out
        assert "(signature unverifiable)" in out
        assert "signature verified" not in out
        assert reply_ct not in out

    @respx.mock
    def test_send_and_wait_when_no_reply_returns_timeout_string(self, config_dir, other_keys):
        mock_token()
        mock_agents()
        mock_webfinger()
        mock_recipient_keys(config_dir)
        respx.post(f"{API_URL}/messages/sync").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": sent_message_payload(),
                    "reply": None,
                    "conversation_id": CONVERSATION_ID,
                    "status": "timeout",
                },
            )
        )

        out = RineSendAndWaitTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"to": OTHER_HANDLE, "body": "do it", "wait_seconds": 5}
        )

        assert f"Sent {MESSAGE_ID}" in out
        assert "no reply within 5s" in out

    def test_send_and_wait_to_group_returns_one_to_one_only_string(self, config_dir):
        """Group handle trips the SDK's typed UnsupportedTargetError (caught pre-request).

        The expected sentence is DERIVED from the tool's own registered id, not
        retyped: the refusal names the verb the model must call instead, so a
        rename here — or a slip back to a bare ``send_and_wait`` — would send the
        model to a tool-not-found with this suite still green.
        """
        tool = RineSendAndWaitTool(config_dir=config_dir, api_url=API_URL)
        send = RineSendTool(config_dir=config_dir, api_url=API_URL)

        out = tool.invoke({"to": GROUP_HANDLE, "body": "nope"})

        # 🔴 BOTH verbs derived. The second one is the instruction — the tool
        # the model is sent to next — so a rename of ``rine_send`` with this
        # sentence untouched spends the model's next turn on a tool-not-found,
        # and the cross-package pin cannot see it (all five constants agree).
        assert out == f"{tool.name} is 1:1 only; use {send.name} for groups."

    async def test_send_and_wait_to_group_arun_returns_one_to_one_only_string(self, config_dir):
        """The 1:1-only guard holds identically on the async path."""
        tool = RineSendAndWaitTool(config_dir=config_dir, api_url=API_URL)
        send = RineSendTool(config_dir=config_dir, api_url=API_URL)

        out = await tool.ainvoke({"to": GROUP_HANDLE, "body": "nope"})

        assert out == f"{tool.name} is 1:1 only; use {send.name} for groups."

    def test_send_and_wait_to_group_never_raises(self, config_dir):
        out = RineSendAndWaitTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"to": "#team@acme.rine.network", "body": "nope"}
        )
        assert isinstance(out, str)


def _self_keys_json(config_dir):
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


# =====================================================================
# RineInboxTool
# =====================================================================


def _inbox_response(ciphertext, encryption_version=EncryptionVersion.HPKE_V1):
    """A single-item ``/agents/{id}/messages`` page carrying one message."""
    return {
        "items": [
            message_payload(
                encryption_version=encryption_version,
                encrypted_payload=ciphertext,
            )
        ],
        "total": 1,
    }


class TestRineInboxTool:
    @respx.mock
    def test_inbox_with_new_messages_returns_decrypted_list(self, config_dir):
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        ciphertext = hpke_self_encrypt(config_dir, {"text": "inbox hello"})
        inbox_route = respx.get(f"{API_URL}/agents/{AGENT_ID}/messages").mock(
            return_value=httpx.Response(200, json=_inbox_response(ciphertext))
        )
        respx.post(f"{API_URL}/agents/{AGENT_ID}/messages/mark-delivered").mock(
            return_value=httpx.Response(200, json={"marked": 1})
        )

        out = RineInboxTool(config_dir=config_dir, api_url=API_URL).invoke({})

        assert "inbox hello" in out
        assert "1 new message" in out
        # status=new must be forwarded
        assert b"status=new" in inbox_route.calls[0].request.url.query
        # ciphertext invariant
        assert ciphertext not in out

    @respx.mock
    async def test_inbox_arun_matches_run(self, config_dir):
        """``.ainvoke`` returns the same decrypted list as ``.invoke`` (same mock).

        Carries the async ciphertext-absence assertion on the inbox path.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        ciphertext = hpke_self_encrypt(config_dir, {"text": "inbox hello"})
        inbox_route = respx.get(f"{API_URL}/agents/{AGENT_ID}/messages").mock(
            return_value=httpx.Response(200, json=_inbox_response(ciphertext))
        )
        respx.post(f"{API_URL}/agents/{AGENT_ID}/messages/mark-delivered").mock(
            return_value=httpx.Response(200, json={"marked": 1})
        )

        out = await RineInboxTool(config_dir=config_dir, api_url=API_URL).ainvoke({})

        assert "inbox hello" in out
        assert "1 new message" in out
        assert b"status=new" in inbox_route.calls[0].request.url.query
        assert ciphertext not in out  # ciphertext-absence on the async inbox path

    @respx.mock
    def test_inbox_marks_decryptable_messages_delivered(self, config_dir):
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        ciphertext = hpke_self_encrypt(config_dir, {"text": "ack me"})
        respx.get(f"{API_URL}/agents/{AGENT_ID}/messages").mock(
            return_value=httpx.Response(200, json=_inbox_response(ciphertext))
        )
        ack = respx.post(f"{API_URL}/agents/{AGENT_ID}/messages/mark-delivered").mock(
            return_value=httpx.Response(200, json={"marked": 1})
        )

        RineInboxTool(config_dir=config_dir, api_url=API_URL).invoke({})

        assert ack.called
        body = json.loads(ack.calls[0].request.content)
        assert body == {"message_ids": [MESSAGE_ID]}

    @respx.mock
    def test_inbox_ack_failure_warns_and_notes_without_losing_read(self, config_dir, caplog):
        """A transient ack (mark_delivered) failure must NOT lose the read.

        The message is still returned, a WARNING is logged, and a note is appended so
        the LLM knows the mail may reappear (best-effort promise).
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        ciphertext = hpke_self_encrypt(config_dir, {"text": "still here"})
        respx.get(f"{API_URL}/agents/{AGENT_ID}/messages").mock(
            return_value=httpx.Response(200, json=_inbox_response(ciphertext))
        )
        # The ack endpoint 500s → mark_delivered raises; the tool must swallow + note.
        respx.post(f"{API_URL}/agents/{AGENT_ID}/messages/mark-delivered").mock(
            return_value=httpx.Response(500, json={"detail": "ack down"})
        )

        with caplog.at_level(logging.WARNING, logger="langchain_rine.messaging"):
            out = RineInboxTool(config_dir=config_dir, api_url=API_URL).invoke({})

        # The read is NOT lost despite the failed ack.
        assert "still here" in out
        # The LLM is told these may reappear (the honesty note).
        assert "could not mark delivered" in out
        # And the failure is observable at WARNING, not silently swallowed.
        assert any("mark_delivered failed" in r.message for r in caplog.records), (
            "expected a WARNING when the ack fails"
        )

    @respx.mock
    def test_inbox_with_no_messages_returns_empty_string(self, config_dir):
        mock_token()
        mock_agents()
        ack = respx.post(f"{API_URL}/agents/{AGENT_ID}/messages/mark-delivered").mock(
            return_value=httpx.Response(200, json={"marked": 0})
        )
        respx.get(f"{API_URL}/agents/{AGENT_ID}/messages").mock(
            return_value=httpx.Response(200, json={"items": [], "total": 0})
        )

        out = RineInboxTool(config_dir=config_dir, api_url=API_URL).invoke({})

        assert out == "No new messages."
        # nothing to ack → mark-delivered not called
        assert not ack.called

    def test_inbox_client_construction_error_maps_to_string(self, config_dir, monkeypatch):
        """A client-construction-time raise is mapped via _format_error, never escapes.

        The client is built INSIDE the try block, so a constructor raise (e.g. a
        ConfigError reading credentials on a non-root host) returns a formatted string
        through ``.invoke`` rather than crashing the agent with a raw traceback.
        """
        from rine.errors import ConfigError

        def _boom(**_kwargs):
            raise ConfigError("could not read credentials")

        monkeypatch.setattr("langchain_rine._client.build_client", _boom)

        out = RineInboxTool(config_dir=config_dir, api_url=API_URL).invoke({})

        assert out == "could not read credentials"  # _format_error(ConfigError) → str(err)

    async def test_inbox_arun_client_construction_error_maps_to_string(
        self, config_dir, monkeypatch
    ):
        """On the async path: an async-client constructor raise is mapped, not raised."""
        from rine.errors import ConfigError

        def _boom(**_kwargs):
            raise ConfigError("could not read credentials")

        monkeypatch.setattr("langchain_rine._client.abuild_client", _boom)

        out = await RineInboxTool(config_dir=config_dir, api_url=API_URL).ainvoke({})

        assert out == "could not read credentials"

    @respx.mock
    def test_inbox_does_not_ack_undecryptable_message(self, config_dir):
        """An MLS message surfaces its decrypt_error and is NOT marked delivered."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/messages").mock(
            return_value=httpx.Response(
                200, json=_inbox_response("Q0lQSEVSVEVYVA==", EncryptionVersion.MLS_V1)
            )
        )
        ack = respx.post(f"{API_URL}/agents/{AGENT_ID}/messages/mark-delivered").mock(
            return_value=httpx.Response(200, json={"marked": 0})
        )

        out = RineInboxTool(config_dir=config_dir, api_url=API_URL).invoke({})

        assert "MLS" in out
        assert "Q0lQSEVSVEVYVA==" not in out  # ciphertext never leaked
        assert not ack.called  # undecryptable → not acked (would re-read next time)


# =====================================================================
# RineReadTool
# =====================================================================


class TestRineReadTool:
    @respx.mock
    def test_read_hpke_message_returns_plaintext_and_verification(self, config_dir):
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        ciphertext = hpke_self_encrypt(config_dir, {"text": "read me"})
        respx.get(f"{API_URL}/messages/{MESSAGE_ID}").mock(
            return_value=httpx.Response(
                200,
                json=message_payload(
                    encryption_version=EncryptionVersion.HPKE_V1,
                    encrypted_payload=ciphertext,
                ),
            )
        )

        out = RineReadTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"message_id": MESSAGE_ID}
        )

        assert "read me" in out
        # Exact rendered signal — not just the 'verification:' label (M5): a valid
        # round-trip with a fetchable sender key renders 'signature verified'.
        assert "verification: signature verified" in out
        assert ciphertext not in out  # ciphertext invariant

    @respx.mock
    async def test_read_arun_matches_run(self, config_dir):
        """``.ainvoke`` returns the same plaintext + verification as ``.invoke``.

        Carries the async ciphertext-absence assertion on the read path.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        ciphertext = hpke_self_encrypt(config_dir, {"text": "read me"})
        respx.get(f"{API_URL}/messages/{MESSAGE_ID}").mock(
            return_value=httpx.Response(
                200,
                json=message_payload(
                    encryption_version=EncryptionVersion.HPKE_V1,
                    encrypted_payload=ciphertext,
                ),
            )
        )

        out = await RineReadTool(config_dir=config_dir, api_url=API_URL).ainvoke(
            {"message_id": MESSAGE_ID}
        )

        assert "read me" in out
        assert "verification: signature verified" in out
        assert ciphertext not in out  # ciphertext-absence on the async read path

    @respx.mock
    def test_read_unverifiable_message_renders_signature_unverifiable(self, config_dir):
        """M5: an unfetchable sender key renders 'signature unverifiable', not 'verified'.

        The message body still decrypts (recipient key is local); only the sender's
        SIGNING key fetch fails (503), so the SDK yields verified=False,
        verification_status='unverifiable'. This is the security-honesty signal the
        LLM consumes — pinning the exact string catches a hardcoded 'verified' note.
        """
        mock_token()
        mock_agents()
        mock_sender_keys_unverifiable()  # GET /agents/{AGENT_ID}/keys → 503
        ciphertext = hpke_self_encrypt(config_dir, {"text": "trust but verify"})
        respx.get(f"{API_URL}/messages/{MESSAGE_ID}").mock(
            return_value=httpx.Response(
                200,
                json=message_payload(
                    encryption_version=EncryptionVersion.HPKE_V1,
                    encrypted_payload=ciphertext,
                ),
            )
        )

        out = RineReadTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"message_id": MESSAGE_ID}
        )

        assert "trust but verify" in out
        assert "verification: signature unverifiable" in out
        assert "signature verified" not in out
        assert ciphertext not in out

    @respx.mock
    def test_read_mls_message_naming_no_group_surfaces_structured_error(self, config_dir):
        """An ``mls-v1`` message that names no group surfaces the SDK's structured error.

        The SDK raises (a malformed message, not a missing key) rather than returning a
        soft decrypt_error, so the connector surfaces that exact guidance — never the
        raw ciphertext.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/messages/{MESSAGE_ID}").mock(
            return_value=httpx.Response(
                200,
                json=message_payload(
                    encryption_version=EncryptionVersion.MLS_V1,
                    encrypted_payload="Q0lQSEVSVEVYVA==",
                ),
            )
        )

        out = RineReadTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"message_id": MESSAGE_ID}
        )

        assert "mls-v1 message names no group" in out
        assert "malformed message, not a missing key" in out
        assert "Q0lQSEVSVEVYVA==" not in out

    @respx.mock
    def test_read_missing_message_returns_not_found_string(self, config_dir):
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/messages/{MESSAGE_ID}").mock(
            return_value=httpx.Response(404, json={"detail": "message not found"})
        )

        out = RineReadTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"message_id": MESSAGE_ID}
        )

        assert "Not found" in out
        assert "message not found" in out


# =====================================================================
# RineThreadTool
# =====================================================================


def _received_row(received_ct):
    """One ``direction="received"`` peer turn the caller can decrypt."""
    row = message_payload(
        encryption_version=EncryptionVersion.HPKE_V1,
        encrypted_payload=received_ct,
        from_agent_id=OTHER_AGENT_ID,
        to_agent_id=AGENT_ID,
    )
    row["direction"] = "received"
    row["sender_handle"] = OTHER_HANDLE
    return row


def _sent_row(self_ct):
    """One ``direction="sent"`` self turn the caller opens via ``self_encrypted_payload``."""
    row = message_payload(
        encryption_version=EncryptionVersion.HPKE_V1,
        encrypted_payload="ZHVtbXk=",  # peer-sealed copy the sender cannot open
        from_agent_id=AGENT_ID,
        to_agent_id=OTHER_AGENT_ID,
    )
    row["direction"] = "sent"
    row["self_encrypted_payload"] = self_ct
    return row


def _unavailable_sent_row():
    """A ``direction="sent"`` row with no self-blob → renders ``[unavailable]``."""
    row = message_payload(
        encryption_version=EncryptionVersion.HPKE_V1,
        encrypted_payload="ZHVtbXk=",
        from_agent_id=AGENT_ID,
        to_agent_id=OTHER_AGENT_ID,
    )
    row["direction"] = "sent"
    return row


def _thread_page(received_ct):
    """A both-sided ``GET /conversations/{id}/messages`` page (one received turn)."""
    return {"messages": [_received_row(received_ct)], "truncated": False}


class TestRineThreadTool:
    @respx.mock
    def test_thread_renders_role_tagged_transcript(self, config_dir, other_keys):
        """rine_thread returns the decrypted, role-tagged transcript."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        mock_recipient_keys(config_dir)
        # A received turn the caller can actually decrypt (sealed to AGENT_ID).
        received_ct = hpke_peer_encrypt(config_dir, {"text": "peer turn one"})
        respx.get(f"{API_URL}/conversations/{CONVERSATION_ID}/messages").mock(
            return_value=httpx.Response(200, json=_thread_page(received_ct))
        )

        out = RineThreadTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"conversation_id": CONVERSATION_ID}
        )

        assert f"[received] {OTHER_HANDLE}: peer turn one" in out
        assert received_ct not in out  # ciphertext never surfaces

    @respx.mock
    async def test_thread_arun_matches_run(self, config_dir, other_keys):
        """``.ainvoke`` returns the same transcript as ``.invoke`` (same mock)."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        mock_recipient_keys(config_dir)
        received_ct = hpke_peer_encrypt(config_dir, {"text": "peer turn one"})
        respx.get(f"{API_URL}/conversations/{CONVERSATION_ID}/messages").mock(
            return_value=httpx.Response(200, json=_thread_page(received_ct))
        )

        out = await RineThreadTool(config_dir=config_dir, api_url=API_URL).ainvoke(
            {"conversation_id": CONVERSATION_ID}
        )

        assert f"[received] {OTHER_HANDLE}: peer turn one" in out
        assert received_ct not in out

    @respx.mock
    def test_thread_empty_conversation_returns_empty_string(self, config_dir):
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/conversations/{CONVERSATION_ID}/messages").mock(
            return_value=httpx.Response(200, json={"messages": [], "truncated": False})
        )

        out = RineThreadTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"conversation_id": CONVERSATION_ID}
        )

        assert out == "No messages in this conversation."

    @respx.mock
    def test_thread_renders_sent_received_and_unavailable_rows(self, config_dir, other_keys):
        """Both-sided transcript renders all row kinds.

        ``[sent] you``, ``[received]``, and an ``[unavailable]`` passthrough all
        render via ``_render_thread_line``.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        mock_recipient_keys(config_dir)
        sent_ct = hpke_self_encrypt(config_dir, {"text": "my turn one"})
        received_ct = hpke_peer_encrypt(config_dir, {"text": "peer turn one"})
        page = {
            "messages": [
                _sent_row(sent_ct),
                _received_row(received_ct),
                _unavailable_sent_row(),
            ],
            "truncated": False,
        }
        respx.get(f"{API_URL}/conversations/{CONVERSATION_ID}/messages").mock(
            return_value=httpx.Response(200, json=page)
        )

        out = RineThreadTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"conversation_id": CONVERSATION_ID}
        )

        assert "[sent] you: my turn one" in out
        assert f"[received] {OTHER_HANDLE}: peer turn one" in out
        assert "[sent] you: [unavailable]" in out
        assert sent_ct not in out  # ciphertext never surfaces
        assert received_ct not in out


class TestRineThreadToolByGroup:
    """The transcript tool takes the name of the thing it reads.

    Every other group tool on this surface takes a group reference; this one
    used to demand a conversation UUID, an id of a different entity whose only
    source was a column in another tool's output.
    """

    @respx.mock
    def test_thread_by_group_handle_reads_the_groups_own_route(self, config_dir, other_keys):
        mock_token()
        mock_agents()
        mock_recipient_keys(config_dir)
        received_ct = hpke_peer_encrypt(config_dir, {"text": "the delivery is booked"})
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json={"items": [group_payload()]})
        )
        route = respx.get(f"{API_URL}/groups/{GROUP_ID}/messages").mock(
            return_value=httpx.Response(
                200, json={"messages": [_received_row(received_ct)], "truncated": False}
            )
        )

        out = RineThreadTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": GROUP_HANDLE}
        )

        assert route.called
        assert f"[received] {OTHER_HANDLE}: the delivery is booked" in out

    @respx.mock
    async def test_thread_by_group_handle_arun_matches_run(self, config_dir, other_keys):
        """``.ainvoke`` reads the same group by the same reference (same mock)."""
        mock_token()
        mock_agents()
        mock_recipient_keys(config_dir)
        received_ct = hpke_peer_encrypt(config_dir, {"text": "the delivery is booked"})
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json={"items": [group_payload()]})
        )
        route = respx.get(f"{API_URL}/groups/{GROUP_ID}/messages").mock(
            return_value=httpx.Response(
                200, json={"messages": [_received_row(received_ct)], "truncated": False}
            )
        )

        out = await RineThreadTool(config_dir=config_dir, api_url=API_URL).ainvoke(
            {"group": GROUP_HANDLE}
        )

        assert route.called
        assert f"[received] {OTHER_HANDLE}: the delivery is booked" in out

    @respx.mock
    def test_thread_naming_both_references_refuses_and_reads_nothing(self, config_dir):
        mock_token()
        mock_agents()
        conversation = respx.get(f"{API_URL}/conversations/{CONVERSATION_ID}/messages").mock(
            return_value=httpx.Response(200, json={"messages": [], "truncated": False})
        )
        group = respx.get(f"{API_URL}/groups/{GROUP_ID}/messages").mock(
            return_value=httpx.Response(200, json={"messages": [], "truncated": False})
        )

        out = RineThreadTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"conversation_id": CONVERSATION_ID, "group": GROUP_ID}
        )

        assert out == TRANSCRIPT_REFERENCE_REQUIRED
        assert not conversation.called and not group.called

    @respx.mock
    def test_thread_naming_neither_reference_refuses_and_reads_nothing(self, config_dir):
        mock_token()
        mock_agents()

        out = RineThreadTool(config_dir=config_dir, api_url=API_URL).invoke({})

        assert out == TRANSCRIPT_REFERENCE_REQUIRED
        assert not any(call.request.url.path.endswith("/messages") for call in respx.calls)

    @respx.mock
    def test_thread_with_a_group_name_in_conversation_id_names_group_and_sends_nothing(
        self, config_dir
    ):
        """The likeliest mistake, answered before it becomes a URL.

        Passing the group's NAME to the conversation parameter used to reach the
        server and come back a framework 422 carrying a pydantic error object.
        """
        mock_token()
        mock_agents()

        out = RineThreadTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"conversation_id": "quarterly planning group"}
        )

        assert out == not_a_conversation_uuid("quarterly planning group")
        # The refusal no longer carries the ``group=#logistics@…`` worked example
        # this line used to match — the sentence describes the three spellings
        # the parameter takes instead. What must survive is that the refusal
        # points the caller at the parameter that does read a group.
        assert "name the group instead" in out
        assert not any(call.request.url.path.endswith("/messages") for call in respx.calls)

    @respx.mock
    async def test_thread_arun_refuses_a_group_name_the_same_way(self, config_dir):
        """The async body owns no second copy of the rule."""
        mock_token()
        mock_agents()

        out = await RineThreadTool(config_dir=config_dir, api_url=API_URL).ainvoke(
            {"conversation_id": "quarterly planning group"}
        )

        assert out == not_a_conversation_uuid("quarterly planning group")
        assert not any(call.request.url.path.endswith("/messages") for call in respx.calls)

    @respx.mock
    def test_thread_by_group_uuid_needs_no_directory_lookup(self, config_dir, other_keys):
        """A UUID is already the id the route takes — resolving it would be a wasted call."""
        mock_token()
        mock_agents()
        mock_recipient_keys(config_dir)
        mock_group_row()
        directory = respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json={"items": [group_payload()]})
        )
        received_ct = hpke_peer_encrypt(config_dir, {"text": "ship it"})
        respx.get(f"{API_URL}/groups/{GROUP_ID}/messages").mock(
            return_value=httpx.Response(
                200, json={"messages": [_received_row(received_ct)], "truncated": False}
            )
        )

        out = RineThreadTool(config_dir=config_dir, api_url=API_URL).invoke({"group": GROUP_ID})

        assert not directory.called
        assert f"[received] {OTHER_HANDLE}: ship it" in out

    @respx.mock
    def test_thread_still_reads_a_dashless_conversation_uuid(self, config_dir):
        """The local refusal may only refuse what the wire would refuse.

        ``uuid.UUID`` accepts the dashless spelling, so a caller who has always
        passed one keeps working — the local check adds a refusal, not a
        stricter format.
        """
        mock_token()
        mock_agents()
        dashless = CONVERSATION_ID.replace("-", "")
        route = respx.get(f"{API_URL}/conversations/{dashless}/messages").mock(
            return_value=httpx.Response(200, json={"messages": [], "truncated": False})
        )

        out = RineThreadTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"conversation_id": dashless}
        )

        assert route.called
        assert out == "No messages in this conversation."


class TestRineThreadToolContract:
    """The shape of the tool, asserted where a model would read it."""

    def test_group_is_described_the_way_every_other_group_tool_describes_it(self):
        """One vocabulary per surface: the transcript tool's ``group`` is the roster tool's."""
        thread = RineThreadInput.model_fields["group"]
        roster = RineGroupRosterInput.model_fields["group"]

        assert thread.description == roster.description

    def test_neither_reference_is_required_on_its_own(self):
        """Requiring either would make the other unreachable."""
        fields = RineThreadInput.model_fields

        assert not fields["conversation_id"].is_required()
        assert not fields["group"].is_required()

    def test_the_tool_description_states_the_exactly_one_rule(self):
        """A model that never calls the tool wrong never sees the refusal."""
        assert TRANSCRIPT_REFERENCE_REQUIRED in RineThreadTool.model_fields["description"].default


# =====================================================================
# RineReplyTool
# =====================================================================


def _original_message_for_reply():
    """The ``GET /messages/{id}`` original ``reply()`` fetches to find the sender."""
    return message_payload(
        encryption_version=EncryptionVersion.HPKE_V1,
        encrypted_payload="ZHVtbXk=",
        from_agent_id=OTHER_AGENT_ID,
        to_agent_id=AGENT_ID,
    )


class TestRineReplyTool:
    @respx.mock
    def test_reply_to_message_returns_reply_id(self, config_dir, other_keys):
        mock_token()
        mock_agents()
        # reply() GETs the original to find the sender, then fetches their key.
        respx.get(f"{API_URL}/messages/{MESSAGE_ID}").mock(
            return_value=httpx.Response(200, json=_original_message_for_reply())
        )
        mock_recipient_keys(config_dir)
        route = respx.post(f"{API_URL}/messages/{MESSAGE_ID}/reply").mock(
            return_value=httpx.Response(200, json={**sent_message_payload(), "id": REPLY_ID})
        )

        out = RineReplyTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"message_id": MESSAGE_ID, "body": "on it"}
        )

        assert route.called
        assert f"Replied to {MESSAGE_ID} -> {REPLY_ID}." == out

    @respx.mock
    async def test_reply_arun_matches_run(self, config_dir, other_keys):
        """``.ainvoke`` returns the same reply-id confirmation as ``.invoke``."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/messages/{MESSAGE_ID}").mock(
            return_value=httpx.Response(200, json=_original_message_for_reply())
        )
        mock_recipient_keys(config_dir)
        route = respx.post(f"{API_URL}/messages/{MESSAGE_ID}/reply").mock(
            return_value=httpx.Response(200, json={**sent_message_payload(), "id": REPLY_ID})
        )

        out = await RineReplyTool(config_dir=config_dir, api_url=API_URL).ainvoke(
            {"message_id": MESSAGE_ID, "body": "on it"}
        )

        assert route.called
        assert f"Replied to {MESSAGE_ID} -> {REPLY_ID}." == out

    @respx.mock
    def test_reply_never_returns_ciphertext(self, config_dir, other_keys):
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/messages/{MESSAGE_ID}").mock(
            return_value=httpx.Response(200, json=_original_message_for_reply())
        )
        mock_recipient_keys(config_dir)
        respx.post(f"{API_URL}/messages/{MESSAGE_ID}/reply").mock(
            return_value=httpx.Response(
                200,
                json={**sent_message_payload(), "id": REPLY_ID, "encrypted_payload": "U0VDUkVU"},
            )
        )

        out = RineReplyTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"message_id": MESSAGE_ID, "body": "on it"}
        )

        assert "U0VDUkVU" not in out
        assert "encrypted_payload" not in out


# =====================================================================
# RineInboxTool — the status slice
# =====================================================================


class TestRineInboxStatus:
    @respx.mock
    def test_inbox_status_all_omits_the_status_query_param(self, config_dir):
        """'all' is a TOOL-layer word, not a server one.

        The route's ``status`` is a three-value Literal, so "every state" is
        expressed by omitting the parameter. Passing the word through would 422
        at the network — this pins the translation, not the string.
        """
        mock_token()
        mock_agents()
        route = respx.get(f"{API_URL}/agents/{AGENT_ID}/messages").mock(
            return_value=httpx.Response(200, json={"items": [], "total": 0})
        )

        RineInboxTool(config_dir=config_dir, api_url=API_URL).invoke({"status": "all"})

        assert b"status" not in route.calls[0].request.url.query

    @respx.mock
    async def test_inbox_arun_status_all_omits_the_status_query_param(self, config_dir):
        """The async twin makes the same translation — the two halves cannot drift."""
        mock_token()
        mock_agents()
        route = respx.get(f"{API_URL}/agents/{AGENT_ID}/messages").mock(
            return_value=httpx.Response(200, json={"items": [], "total": 0})
        )

        await RineInboxTool(config_dir=config_dir, api_url=API_URL).ainvoke({"status": "all"})

        assert b"status" not in route.calls[0].request.url.query

    @respx.mock
    def test_inbox_status_read_forwards_read_and_marks_nothing_delivered(self, config_dir):
        """A look-back slice acks nothing: re-marking a read message walks its state back."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        ciphertext = hpke_self_encrypt(config_dir, {"text": "old news"})
        route = respx.get(f"{API_URL}/agents/{AGENT_ID}/messages").mock(
            return_value=httpx.Response(200, json=_inbox_response(ciphertext))
        )
        ack = respx.post(f"{API_URL}/agents/{AGENT_ID}/messages/mark-delivered").mock(
            return_value=httpx.Response(200, json={"marked": 1})
        )

        out = RineInboxTool(config_dir=config_dir, api_url=API_URL).invoke({"status": "read"})

        assert b"status=read" in route.calls[0].request.url.query
        assert not ack.called
        assert "old news" in out

    @respx.mock
    async def test_inbox_arun_status_read_marks_nothing_delivered(self, config_dir):
        """The async twin gates the auto-ack the same way."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        ciphertext = hpke_self_encrypt(config_dir, {"text": "old news"})
        respx.get(f"{API_URL}/agents/{AGENT_ID}/messages").mock(
            return_value=httpx.Response(200, json=_inbox_response(ciphertext))
        )
        ack = respx.post(f"{API_URL}/agents/{AGENT_ID}/messages/mark-delivered").mock(
            return_value=httpx.Response(200, json={"marked": 1})
        )

        await RineInboxTool(config_dir=config_dir, api_url=API_URL).ainvoke({"status": "read"})

        assert not ack.called

    @respx.mock
    def test_inbox_status_all_marks_nothing_delivered(self, config_dir):
        """Auto-ack is the NEW slice's alone — 'all' spans states that are already acked."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        ciphertext = hpke_self_encrypt(config_dir, {"text": "everything"})
        respx.get(f"{API_URL}/agents/{AGENT_ID}/messages").mock(
            return_value=httpx.Response(200, json=_inbox_response(ciphertext))
        )
        ack = respx.post(f"{API_URL}/agents/{AGENT_ID}/messages/mark-delivered").mock(
            return_value=httpx.Response(200, json={"marked": 1})
        )

        RineInboxTool(config_dir=config_dir, api_url=API_URL).invoke({"status": "all"})

        assert not ack.called

    @respx.mock
    def test_inbox_status_names_the_slice_it_returned(self, config_dir):
        """A 'read' page must not be announced as new mail."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
            return_value=httpx.Response(200, json=_self_keys_json(config_dir))
        )
        ciphertext = hpke_self_encrypt(config_dir, {"text": "old news"})
        respx.get(f"{API_URL}/agents/{AGENT_ID}/messages").mock(
            return_value=httpx.Response(200, json=_inbox_response(ciphertext))
        )

        out = RineInboxTool(config_dir=config_dir, api_url=API_URL).invoke({"status": "read"})

        assert out.startswith("1 read message(s):")
        assert "new message" not in out


# =====================================================================
# The send's success line names BOTH spellings
# =====================================================================


class TestSendLineNamesBothSpellings:
    """What ``rine_send`` reports back after a send that went through.

    It echoes both: the caller's own spelling AND what the server resolved
    it to, so a model can correlate the line against its own turn and still see
    which group or agent actually received the message. The resolved half is read
    off the server's answer (``MessageRead.group_handle`` / ``recipient_handle``)
    rather than resolved a second time here.

    Both halves of every tool are driven: a migration that moves ``_run`` and
    leaves ``_arun`` behind reads green under a suite that drives only one.
    """

    @respx.mock
    def test_send_to_a_uuid_when_the_server_names_the_handle_reports_both(
        self, config_dir, other_keys
    ):
        mock_token()
        mock_agents()
        mock_webfinger()
        mock_recipient_keys(config_dir)
        respx.post(f"{API_URL}/messages").mock(
            return_value=httpx.Response(
                200, json={**sent_message_payload(), "recipient_handle": OTHER_HANDLE}
            )
        )

        out = _send_tool(config_dir).invoke({"to": OTHER_AGENT_ID, "body": "hello there"})

        assert f"Sent message {MESSAGE_ID} to {OTHER_AGENT_ID} → {OTHER_HANDLE}" in out
        assert CONVERSATION_ID in out

    @respx.mock
    async def test_send_to_a_uuid_reports_both_spellings_on_the_async_half_too(
        self, config_dir, other_keys
    ):
        mock_token()
        mock_agents()
        mock_webfinger()
        mock_recipient_keys(config_dir)
        respx.post(f"{API_URL}/messages").mock(
            return_value=httpx.Response(
                200, json={**sent_message_payload(), "recipient_handle": OTHER_HANDLE}
            )
        )

        out = await _send_tool(config_dir).ainvoke({"to": OTHER_AGENT_ID, "body": "hello there"})

        assert f"Sent message {MESSAGE_ID} to {OTHER_AGENT_ID} → {OTHER_HANDLE}" in out

    @respx.mock
    def test_send_when_the_spelling_is_the_resolved_handle_draws_no_arrow(
        self, config_dir, other_keys
    ):
        """Nothing to reconcile, so the line does not say the same thing twice."""
        mock_token()
        mock_agents()
        mock_webfinger()
        mock_recipient_keys(config_dir)
        respx.post(f"{API_URL}/messages").mock(
            return_value=httpx.Response(
                200, json={**sent_message_payload(), "recipient_handle": OTHER_HANDLE}
            )
        )

        out = _send_tool(config_dir).invoke({"to": OTHER_HANDLE, "body": "hello there"})

        assert f"Sent message {MESSAGE_ID} to {OTHER_HANDLE} (conversation" in out
        assert "→" not in out

    @respx.mock
    def test_send_when_the_server_echoes_an_empty_handle_never_prints_a_bare_arrow(
        self, config_dir, other_keys
    ):
        """The falsy arm of the same guard, which ``is None`` did not cover.

        ``group_handle`` is ``str | None`` on the wire, so an empty string is a
        value the type permits and no client can rule out. Under a ``is None``
        test it is neither ``None`` nor equal to the spelling, so the arrow is
        drawn at nothing. All three Python connectors guard this the same way.
        """
        mock_token()
        mock_agents()
        mock_webfinger()
        mock_recipient_keys(config_dir)
        respx.post(f"{API_URL}/messages").mock(
            return_value=httpx.Response(
                200, json={**sent_message_payload(), "recipient_handle": ""}
            )
        )

        out = _send_tool(config_dir).invoke({"to": OTHER_HANDLE, "body": "hello there"})

        assert f"Sent message {MESSAGE_ID} to {OTHER_HANDLE} (conversation" in out
        assert "→" not in out

    @respx.mock
    def test_send_when_the_server_names_no_handle_omits_the_arrow(self, config_dir, other_keys):
        """A 201 carrying no handle falls back to the no-arrow form."""
        mock_token()
        mock_agents()
        mock_webfinger()
        mock_recipient_keys(config_dir)
        respx.post(f"{API_URL}/messages").mock(
            return_value=httpx.Response(200, json=sent_message_payload())
        )

        out = _send_tool(config_dir).invoke({"to": OTHER_AGENT_ID, "body": "hello there"})

        assert f"Sent message {MESSAGE_ID} to {OTHER_AGENT_ID} (conversation" in out
        assert "None" not in out
        assert "→" not in out
