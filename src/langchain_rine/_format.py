"""Human-readable renderers for SDK output models.

Every tool returns a *string* the LLM reads — never a raw model and **never**
ciphertext. These helpers turn ``DecryptedMessage`` / ``GroupRead`` / ``AgentProfile``
/ ``AgentSummary`` into plain text, surfacing ``plaintext``, ``verified`` /
``verification_status``, and ``decrypt_error`` honestly. The encrypted
``encrypted_payload`` field is deliberately never read here (SPEC §2.4 / REQ-07,
REQ-10): it must not enter the LLM context.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rine import AgentProfile, AgentSummary, DecryptedMessage, GroupRead


def _verified_note(msg: DecryptedMessage) -> str:
    """Return a short signature-verification suffix for a decrypted message."""
    if msg.verified:
        return "signature verified"
    return f"signature {msg.verification_status}"


def render_message_body(msg: DecryptedMessage) -> str:
    """Render the readable content of one message: plaintext or decrypt_error.

    Never returns ``encrypted_payload``. If the message could not be decrypted
    (e.g. MLS or an unsupported version), the SDK's ``decrypt_error`` is surfaced
    verbatim so the failure is loud, not silent. Otherwise the decrypted plaintext
    is unwrapped: rine payloads are ``{"text": "..."}`` JSON (the convention every
    tool sends and the MCP surface ``JSON.parse``s), so the human ``text`` field is
    surfaced; a JSON object without ``text`` falls back to compact JSON and a
    non-JSON plaintext to the raw string (SPEC §6.2).
    """
    if msg.decrypt_error:
        return f"[unreadable] {msg.decrypt_error}"
    if msg.plaintext is None:
        return "[no plaintext]"
    return _unwrap_text(msg.plaintext)


def _unwrap_text(plaintext: str) -> str:
    """Surface the ``text`` field of a JSON payload; fall back to compact JSON / raw."""
    try:
        payload = json.loads(plaintext)
    except (ValueError, TypeError):
        return plaintext
    if isinstance(payload, dict):
        text = payload.get("text")
        if isinstance(text, str):
            return text
        return json.dumps(payload, separators=(",", ":"))
    return plaintext


def render_message(msg: DecryptedMessage) -> str:
    """Render one message as a multi-field readable line (no ciphertext)."""
    sender = msg.sender_handle or (str(msg.from_agent_id) if msg.from_agent_id else "unknown")
    parts = [
        f"id={msg.id}",
        f"from={sender}",
        f"type={msg.type}",
    ]
    if msg.group_handle or msg.group_id:
        parts.append(f"group={msg.group_handle or msg.group_id}")
    parts.append(f"body={render_message_body(msg)}")
    if not msg.decrypt_error:
        parts.append(f"({_verified_note(msg)})")
    return " | ".join(parts)


def render_single_message(msg: DecryptedMessage) -> str:
    """Render one fetched message as readable multi-line text (no ciphertext)."""
    sender = msg.sender_handle or (str(msg.from_agent_id) if msg.from_agent_id else "unknown")
    lines = [f"from: {sender}", f"type: {msg.type}"]
    if msg.group_handle or msg.group_id:
        lines.append(f"group: {msg.group_handle or msg.group_id}")
    lines.append(f"body: {render_message_body(msg)}")
    if not msg.decrypt_error:
        lines.append(f"verification: {_verified_note(msg)}")
    return "\n".join(lines)


def render_inbox(messages: list[DecryptedMessage]) -> str:
    """Render an inbox page as a numbered list, or an empty-state string."""
    if not messages:
        return "No new messages."
    lines = [f"{i}. {render_message(m)}" for i, m in enumerate(messages, start=1)]
    return f"{len(messages)} new message(s):\n" + "\n".join(lines)


def render_agent_summary(a: AgentSummary) -> str:
    """Render a directory search hit as a single readable line."""
    verified = "verified" if a.verified else "unverified"
    tier = f", trust_tier={a.trust_tier}" if a.trust_tier is not None else ""
    cat = f", category={a.category}" if a.category else ""
    desc = f" — {a.description}" if a.description else ""
    return f"{a.handle} ({a.name}, {verified}{tier}{cat}){desc}"


def render_discover(agents: list[AgentSummary]) -> str:
    """Render a discover() page as a numbered list, or an empty-state string."""
    if not agents:
        return "No agents matched."
    lines = [f"{i}. {render_agent_summary(a)}" for i, a in enumerate(agents, start=1)]
    return f"{len(agents)} agent(s) found:\n" + "\n".join(lines)


def render_profile(p: AgentProfile) -> str:
    """Render a full agent profile as readable multi-line text."""
    return (
        f"handle: {p.handle}\n"
        f"name: {p.name}\n"
        f"description: {p.description or '(none)'}\n"
        f"category: {p.category or '(none)'}\n"
        f"verified: {p.verified}\n"
        f"trust_tier: {p.trust_tier if p.trust_tier is not None else '(none)'}\n"
        f"human_oversight: {p.human_oversight}"
    )


def group_is_mls(g: GroupRead) -> bool:
    """Return True if a group is MLS-encrypted (Python cannot read/post it).

    Any of the MLS latches (``mls_enabled``, a non-null ``mls_group_id``, or
    ``mls_pending`` during the init window) means the sender-key path is frozen.
    """
    return bool(g.mls_enabled or g.mls_group_id is not None or g.mls_pending)


def render_group(g: GroupRead) -> str:
    """Render a group's details + the sender-key-vs-MLS self-diagnosis line."""
    if group_is_mls(g):
        diagnosis = (
            "[WARN] MLS group — this Python integration cannot read or post here "
            "(use the rine CLI/MCP/TS SDK, or recreate as sender-key)."
        )
    else:
        diagnosis = "[OK] sender-key group — fully readable/postable from here."
    return (
        f"handle: {g.handle}\n"
        f"name: {g.name}\n"
        f"id: {g.id}\n"
        f"members: {g.member_count}\n"
        f"enrollment: {g.enrollment_policy}\n"
        f"visibility: {g.visibility}\n"
        f"mls_enabled: {g.mls_enabled}\n"
        f"mls_group_id: {g.mls_group_id or '(none)'}\n"
        f"{diagnosis}"
    )
