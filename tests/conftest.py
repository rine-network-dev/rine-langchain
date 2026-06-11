"""Shared fixtures + helpers for the langchain-rine behavioral suite.

All tests mock ONLY the rine HTTP boundary with respx — the rine SDK itself is
real (never mocked). The patterns mirror ``rine-sdk/tests/`` (token + agents +
keys mocks; a real HPKE self-encrypt round-trip so the ciphertext-never-leaks
invariant and plaintext assertions exercise the real crypto path).

Tools are driven through their public LangChain interface: a tool is instantiated
with ``config_dir``/``api_url`` overrides (the documented per-tool kwargs) and
called via ``.invoke(...)`` / ``.ainvoke(...)``. We assert on the returned string
and observable HTTP, never on private internals.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from rine import SyncRineClient
from rine._config import save_credentials
from rine._constants import MessageType
from rine._crypto.keys import (
    encryption_public_key_to_jwk,
    generate_agent_keys,
    load_agent_keys,
    save_agent_keys,
    signing_public_key_to_jwk,
)
from rine._crypto.message import decrypt_message_sync, encrypt_message

API_URL = "https://rine.network"

# Fixed UUIDs reused across the suite (match the SDK test fixtures' style).
AGENT_ID = "00000000-1111-2222-3333-444444444444"
OTHER_AGENT_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
GROUP_ID = "bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
MESSAGE_ID = "cccccccc-dddd-eeee-ffff-000000000001"
CONVERSATION_ID = "cccccccc-dddd-eeee-ffff-000000000002"
REPLY_ID = "dddddddd-dddd-dddd-dddd-dddddddddddd"

OTHER_HANDLE = "other@acme.rine.network"
GROUP_HANDLE = "#ops@acme.rine.network"

TOKEN_RESPONSE = {"access_token": "test-token", "expires_in": 900}

AGENT_PAYLOAD = {
    "id": AGENT_ID,
    "name": "test-agent",
    "handle": "test-agent@myorg.rine.network",
    "human_oversight": True,
    "incoming_policy": "accept_all",
    "outgoing_policy": "send_all",
    "created_at": "2024-01-01T00:00:00Z",
}


@pytest.fixture
def config_dir(tmp_path) -> str:
    """A config dir with valid credentials + the caller agent's keys on disk.

    This is what a tool's ``config_dir`` override points at, so the real SDK
    resolves auth/keys exactly as it would in production — no auth is mocked
    away, only the HTTP boundary is.
    """
    cdir = str(tmp_path)
    save_credentials(cdir, {"default": {"client_id": "cid", "client_secret": "csecret"}})
    save_agent_keys(cdir, AGENT_ID, generate_agent_keys())
    return cdir


@pytest.fixture
def other_keys(config_dir):
    """Generate + persist the recipient agent's keys; return them.

    Persisted under the same config dir so a self-mocked ``/keys`` fetch returns
    a real public key and a real HPKE round-trip is possible in tests that need
    plaintext back.
    """
    keys = generate_agent_keys()
    save_agent_keys(config_dir, OTHER_AGENT_ID, keys)
    return keys


# --- HTTP boundary mocks (used inside @respx.mock test bodies) ---


def mock_token() -> None:
    """Stub the OAuth2 token endpoint."""
    import respx

    respx.post(f"{API_URL}/oauth/token").mock(
        return_value=httpx.Response(200, json=TOKEN_RESPONSE)
    )


def mock_agents() -> None:
    """Stub ``GET /agents`` (single-agent org → auto agent resolution)."""
    import respx

    respx.get(f"{API_URL}/agents").mock(
        return_value=httpx.Response(200, json={"items": [AGENT_PAYLOAD]})
    )


def mock_webfinger(handle_uuid: str = OTHER_AGENT_ID) -> None:
    """Stub WebFinger so a handle resolves to ``handle_uuid``."""
    import respx

    respx.get(f"{API_URL}/.well-known/webfinger").mock(
        return_value=httpx.Response(200, json={"aliases": [f"{API_URL}/agents/{handle_uuid}"]})
    )


def mock_recipient_keys(config_dir: str, agent_id: str = OTHER_AGENT_ID) -> None:
    """Stub ``GET /agents/{id}/keys`` with the persisted recipient public key."""
    import respx

    keys = load_agent_keys(config_dir, agent_id)
    respx.get(f"{API_URL}/agents/{agent_id}/keys").mock(
        return_value=httpx.Response(
            200,
            json={
                "agent_id": agent_id,
                "encryption_public_key": encryption_public_key_to_jwk(keys.encryption.public_key),
                "signing_public_key": signing_public_key_to_jwk(keys.signing.public_key),
            },
        )
    )


def mock_sender_keys_unverifiable(agent_id: str = AGENT_ID) -> None:
    """Stub ``GET /agents/{id}/keys`` with a server error → sender unverifiable.

    The message still decrypts (the recipient private key is on disk), but the
    SDK cannot fetch the sender's *signing* key to verify the envelope, so it
    returns ``verified=False, verification_status="unverifiable"`` (the only
    reachable ``verified=False`` plaintext case — an ``"invalid"`` signature
    raises ``CryptoError`` instead). Drives the signature-honesty render path.
    """
    import respx

    respx.get(f"{API_URL}/agents/{agent_id}/keys").mock(
        return_value=httpx.Response(503, json={"detail": "key service unavailable"})
    )


def hpke_self_encrypt(config_dir: str, payload: dict[str, Any]) -> str:
    """HPKE-encrypt ``payload`` from AGENT_ID to AGENT_ID (self-readable).

    Returns the base64url ciphertext — used to build a message the caller agent
    can actually decrypt, so plaintext + ciphertext-invariant assertions run the
    real crypto path.
    """
    own_pk = load_agent_keys(config_dir, AGENT_ID).encryption.public_key
    return encrypt_message(config_dir, AGENT_ID, own_pk, payload).encrypted_payload


def decrypt_self_wire(config_dir: str, encrypted_payload_b64: str) -> str:
    """Decrypt a wire ciphertext that was HPKE-sealed to AGENT_ID.

    The inverse of :func:`hpke_self_encrypt`. Used to read back the *content* a
    tool/listener actually put on the wire (e.g. the handler's agent-name + verb
    notification text) without trusting the tool's own formatter — proves the real
    encrypted body, not just that *a* POST happened. Sender verification is skipped
    by returning a 503 on the sender-keys fetch (caller mocks that), so this returns
    the raw decrypted plaintext string regardless of verification status.
    """
    import respx

    keys = load_agent_keys(config_dir, AGENT_ID)
    client = SyncRineClient(config_dir=config_dir, api_url=API_URL)
    try:
        with respx.mock:
            respx.get(f"{API_URL}/agents/{AGENT_ID}/keys").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "agent_id": AGENT_ID,
                        "encryption_public_key": encryption_public_key_to_jwk(
                            keys.encryption.public_key
                        ),
                        "signing_public_key": signing_public_key_to_jwk(keys.signing.public_key),
                    },
                )
            )
            return decrypt_message_sync(
                config_dir, AGENT_ID, encrypted_payload_b64, client._http
            ).plaintext
    finally:
        client.close()


def message_payload(
    *,
    encryption_version: str,
    encrypted_payload: str,
    msg_id: str = MESSAGE_ID,
    from_agent_id: str = AGENT_ID,
    to_agent_id: str = AGENT_ID,
    msg_type: str = MessageType.TASK_REQUEST,
    group_id: str | None = None,
) -> dict[str, Any]:
    """Build a server message payload dict (what ``/messages/{id}`` returns)."""
    payload: dict[str, Any] = {
        "id": msg_id,
        "conversation_id": CONVERSATION_ID,
        "from_agent_id": from_agent_id,
        "to_agent_id": to_agent_id,
        "type": msg_type,
        "encrypted_payload": encrypted_payload,
        "encryption_version": encryption_version,
        "sender_signing_kid": f"rine:{from_agent_id}",
        "created_at": "2024-01-01T00:00:00Z",
    }
    if group_id is not None:
        payload["group_id"] = group_id
    return payload


def sent_message_payload(
    *, to_agent_id: str = OTHER_AGENT_ID, encrypted_payload: str = "ZHVtbXk="
) -> dict[str, Any]:
    """A ``MessageRead`` payload for a just-sent message (ciphertext opaque)."""
    return {
        "id": MESSAGE_ID,
        "conversation_id": CONVERSATION_ID,
        "from_agent_id": AGENT_ID,
        "to_agent_id": to_agent_id,
        "type": MessageType.TASK_REQUEST,
        "encrypted_payload": encrypted_payload,
        "encryption_version": "hpke-v1",
        "sender_signing_kid": f"rine:{AGENT_ID}",
        "created_at": "2024-01-01T00:00:00Z",
    }


def group_payload(
    *,
    name: str = "ops",
    handle: str = GROUP_HANDLE,
    group_id: str = GROUP_ID,
    member_count: int = 2,
    mls_enabled: bool = False,
    mls_group_id: str | None = None,
    mls_pending: bool = False,
    description: str | None = None,
) -> dict[str, Any]:
    """A ``GroupRead`` payload (server ``/groups`` item shape)."""
    return {
        "id": group_id,
        "name": name,
        "handle": handle,
        "description": description,
        "enrollment_policy": "closed",
        "visibility": "private",
        "member_count": member_count,
        "created_at": "2024-01-01T00:00:00Z",
        "mls_enabled": mls_enabled,
        "mls_group_id": mls_group_id,
        "mls_pending": mls_pending,
    }
