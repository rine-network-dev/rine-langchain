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
from rine._crypto.dm_envelope import RecipientKeys
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
# A GROUP's running thread, which is a different object from the 1:1
# CONVERSATION_ID above: the group one is what GroupRead.conversation_id
# carries and what rine_thread is handed to read a group's backlog.
GROUP_CONVERSATION_ID = "cccccccc-dddd-eeee-ffff-000000000003"
REPLY_ID = "dddddddd-dddd-dddd-dddd-dddddddddddd"
WELCOME_ID = "77777777-8888-9999-aaaa-bbbbbbbbbbbb"

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

# A second agent in the same org — the shape an app that runs more than one
# persona actually has, and the one under which the SDK refuses to guess which
# agent a tool call acts as unless something names it.
SECOND_AGENT_ID = "11111111-2222-3333-4444-555555555555"
SECOND_AGENT_PAYLOAD = {
    **AGENT_PAYLOAD,
    "id": SECOND_AGENT_ID,
    "name": "second-agent",
    "handle": "second-agent@myorg.rine.network",
}


@pytest.fixture(autouse=True)
def _no_ambient_acting_agent(monkeypatch):
    """Keep the developer's own ``RINE_AGENT`` out of every test in this package.

    The SDK client constructor reads it (rung 3 of the ladder), so a machine
    that exports one would silently give every tool call an actor the test never
    asked for. A test that wants the env rung sets it itself, after this ran.
    """
    monkeypatch.delenv("RINE_AGENT", raising=False)


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


def mock_group_row(group_id: str = GROUP_ID, **payload: Any) -> None:
    """Stub ``GET /groups/{id}`` — the one indexed read a group reference costs.

    A reference is resolved before it becomes a URL, and a UUID is
    checked against its own row rather than against a listing. Every verb that
    takes a group therefore pays this read once, whatever spelling it was given.
    """
    import respx

    respx.get(f"{API_URL}/groups/{group_id}").mock(
        return_value=httpx.Response(200, json=group_payload(group_id=group_id, **payload))
    )


def mock_two_agents() -> None:
    """Stub ``GET /agents`` with a two-agent org (auto-resolution is refused)."""
    import respx

    respx.get(f"{API_URL}/agents").mock(
        return_value=httpx.Response(200, json={"items": [AGENT_PAYLOAD, SECOND_AGENT_PAYLOAD]})
    )


def mock_group_mls_init(group_id: str = GROUP_ID) -> Any:
    """Stub what a group create fires after ``POST /groups``: MLS founding + re-read.

    Founding opens the init window (``mls/init``) and publishes the founder's init
    commit (``mls/commit``). Create then **re-reads** the group, because the server
    latches ``mls_group_id`` after the create response was minted — without that
    ``GET`` a brand-new MLS group renders as sender-key. The re-read is guarded in
    the SDK, so leaving it unmocked would not fail a test; it would silently return
    the unlatched create response.

    Returns the ``mls/init`` route so a caller can assert the founder opened the
    init window.
    """
    import respx

    init = respx.post(f"{API_URL}/groups/{group_id}/mls/init").mock(
        return_value=httpx.Response(
            200, json={"members": [{"agent_id": AGENT_ID, "has_key_package": True}]}
        )
    )
    respx.post(f"{API_URL}/groups/{group_id}/mls/commit").mock(
        return_value=httpx.Response(200, json={"epoch": 1})
    )
    respx.get(f"{API_URL}/groups/{group_id}").mock(
        return_value=httpx.Response(
            200, json=group_payload(group_id=group_id, mls_enabled=True, mls_group_id="mls-xyz")
        )
    )
    return init


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


def establish_mls_group_state(config_dir: str) -> None:
    """Give AGENT_ID genuine local MLS state for ``GROUP_ID``, so a group send posts.

    A real second agent founds the post-quantum (X-Wing) MLS group and admits
    AGENT_ID via a real Welcome; AGENT_ID installs it and can then encrypt/decrypt
    in the group. Only the HTTP boundary is mocked — the MLS crypto is the real
    Rust core, exactly as in ``rine-sdk``'s own MLS suite. Call inside an active
    ``@respx.mock`` after ``mock_token()``; it also stubs the group's Welcome and
    handshake endpoints (empty past install), leaving the caller to mock only the
    group resolve (``GET /groups``) and the ``POST /messages`` send.
    """
    import base64
    import json

    import respx
    from rine._mls import (
        mls_group_id_bytes,
        open_mls_client,
        process_mls_welcomes,
        publish_mls_key_packages,
    )

    # Mint + capture AGENT_ID's KeyPackage (public half) the founder will admit.
    kp_route = respx.post(f"{API_URL}/agents/{AGENT_ID}/mls/key-packages").mock(
        return_value=httpx.Response(201, json={"stored": 10})
    )
    client = SyncRineClient(config_dir=config_dir, api_url=API_URL)
    try:
        publish_mls_key_packages(client._http, config_dir, AGENT_ID)
        key_package = base64.b64decode(
            json.loads(kp_route.calls[0].request.content)["packages"][0], validate=True
        )

        # A real second agent founds the group admitting AGENT_ID's KeyPackage.
        peer_dir = f"{config_dir}/_peer"
        save_agent_keys(peer_dir, OTHER_AGENT_ID, generate_agent_keys())
        peer = open_mls_client(peer_dir, OTHER_AGENT_ID)
        group = peer.create_group_with_members(mls_group_id_bytes(GROUP_ID), [key_package])
        group.save()
        welcome = group.last_commit().welcomes[0]

        # Deliver the Welcome once (then nothing) so AGENT_ID installs its own state.
        respx.get(f"{API_URL}/agents/{AGENT_ID}/mls/welcomes").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "welcomes": [
                            {
                                "id": WELCOME_ID,
                                "group_id": GROUP_ID,
                                "blob": base64.b64encode(welcome).decode("ascii"),
                                "created_at": "2024-01-01T00:00:00Z",
                            }
                        ]
                    },
                ),
                *[httpx.Response(200, json={"welcomes": []}) for _ in range(9)],
            ]
        )
        # The pre-send sync finds no newer epoch to catch up to.
        respx.get(f"{API_URL}/groups/{GROUP_ID}/mls/handshakes").mock(
            return_value=httpx.Response(200, json={"handshakes": []})
        )
        process_mls_welcomes(client._http, config_dir, AGENT_ID)
    finally:
        client.close()


def hpke_self_encrypt(config_dir: str, payload: dict[str, Any]) -> str:
    """HPKE-encrypt ``payload`` from AGENT_ID to AGENT_ID (self-readable).

    Returns the base64url ciphertext — used to build a message the caller agent
    can actually decrypt, so plaintext + ciphertext-invariant assertions run the
    real crypto path.
    """
    # Classical on purpose: the recipient's ML-KEM half is left out so this helper
    # keeps producing `hpke-v1`, which is what the fixtures around it declare.
    own = load_agent_keys(config_dir, AGENT_ID)
    recipient = RecipientKeys(encryption=own.encryption.public_key)
    return encrypt_message(config_dir, AGENT_ID, recipient, payload).encrypted_payload


def hpke_peer_encrypt(config_dir: str, payload: dict[str, Any]) -> str:
    """HPKE-encrypt ``payload`` from OTHER_AGENT_ID to AGENT_ID, signed by the peer.

    A row the server reports as coming from another agent is signed by that agent's
    own key, so a fixture standing in for a received turn is sealed with the peer's
    keys — sealing it with the caller's own key would attribute a signature to an
    agent that did not make it. Needs the ``other_keys`` fixture (peer keys on disk)
    and ``mock_recipient_keys`` (the peer's published signing key).
    """
    # Classical on purpose, matching ``hpke_self_encrypt``: no ML-KEM half, so the
    # ciphertext stays `hpke-v1` as the fixtures around it declare.
    own = load_agent_keys(config_dir, AGENT_ID)
    recipient = RecipientKeys(encryption=own.encryption.public_key)
    return encrypt_message(config_dir, OTHER_AGENT_ID, recipient, payload).encrypted_payload


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
                config_dir, AGENT_ID, encrypted_payload_b64, client._http, AGENT_ID
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
    enrollment_policy: str = "closed",
    mls_enabled: bool = False,
    mls_group_id: str | None = None,
    mls_pending: bool = False,
    description: str | None = None,
    conversation_id: str | None = None,
    member_agent_ids: list[str] | None = None,
) -> dict[str, Any]:
    """A ``GroupRead`` payload (server ``/groups`` item shape).

    ``conversation_id`` defaults to ``None`` and the key is always emitted,
    which is the server's own shape: it is nullable because a group's
    conversation is created lazily on its first post, and a group nobody has
    posted in therefore has none.

    ``member_agent_ids`` defaults to **empty**, and the key is likewise always
    emitted. Empty is the server's honest answer for a group the org is in but
    no agent of it is seated in, and defaulting it that way keeps every test
    that does not care about seats from quietly asserting one.

    ``enrollment_policy`` is an argument because the default pair here —
    ``closed`` with ``mls_enabled=False`` — is a row no shipped client creates:
    every create sends ``mls_enabled=True`` and the server column defaults true.
    A group that really runs sender keys is ``open``, and without this the
    sender-key arm of every verb was only ever driven on the unreachable row.
    """
    return {
        "id": group_id,
        "name": name,
        "handle": handle,
        "description": description,
        "enrollment_policy": enrollment_policy,
        "visibility": "private",
        "member_count": member_count,
        "created_at": "2024-01-01T00:00:00Z",
        "mls_enabled": mls_enabled,
        "mls_group_id": mls_group_id,
        "mls_pending": mls_pending,
        "conversation_id": conversation_id,
        "member_agent_ids": member_agent_ids or [],
    }
