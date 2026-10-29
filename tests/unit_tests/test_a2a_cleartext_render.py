"""A cleartext A2A message says it was not end-to-end encrypted wherever a model reads it."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from rine import A2A_CLEARTEXT_NOTICE, DecryptedMessage
from rine.a2a_cleartext import read_a2a_cleartext

from langchain_rine._format import render_inbox, render_single_message

#: One text part "plain" as the server serves it: unpadded base64url of the parts' JSON.
WIRE_PAYLOAD = "W3sidGV4dCI6ICJwbGFpbiJ9XQ"


def _cleartext_row() -> DecryptedMessage:
    """An A2A row as the SDK hands it over after reading it."""
    return read_a2a_cleartext(
        DecryptedMessage(
            id=uuid4(),
            conversation_id=uuid4(),
            type="rine.v1.a2a_message",
            encryption_version="none",
            encrypted_payload=WIRE_PAYLOAD,
            created_at=datetime.now(UTC),
            sender_handle="external@acme.rine.network",
        )
    )


def _encrypted_row() -> DecryptedMessage:
    return DecryptedMessage(
        id=uuid4(),
        conversation_id=uuid4(),
        type="rine.v1.task_request",
        encrypted_payload="OPAQUE-CIPHERTEXT-NEVER-SURFACED",
        created_at=datetime.now(UTC),
        sender_handle="alice@acme.rine.network",
        plaintext='{"text":"hello"}',
        verified=True,
        verification_status="verified",
    )


def test_single_message_when_it_is_a2a_cleartext_then_the_body_leads_with_the_notice():
    """A single read leads the body with the notice and never calls it verified."""
    out = render_single_message(_cleartext_row())

    assert f"body: [{A2A_CLEARTEXT_NOTICE}] " in out
    assert "signature verified" not in out


def test_inbox_when_it_holds_a2a_cleartext_then_only_that_row_carries_the_notice():
    """An inbox page marks the cleartext row only; the encrypted row is untouched."""
    out = render_inbox([_cleartext_row(), _encrypted_row()], status="new")

    assert out.count(A2A_CLEARTEXT_NOTICE) == 1
    assert "body=hello " in out


def test_single_message_when_it_is_encrypted_then_it_renders_as_before():
    """An encrypted message renders exactly as it did before the notice existed."""
    out = render_single_message(_encrypted_row())

    assert out.endswith("body: hello\nverification: signature verified")


def test_resume_payload_when_the_reply_is_a2a_cleartext_then_the_text_leads_with_the_notice():
    """The resume payload a graph hands its model leads with the notice."""
    from langchain_rine._resume import build_human_response

    args = build_human_response(_cleartext_row())["args"]

    assert args["text"].startswith(f"[unverified sender] [{A2A_CLEARTEXT_NOTICE}] ")
