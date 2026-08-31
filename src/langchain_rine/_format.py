"""Human-readable renderers for SDK output models.

Every tool returns a *string* the LLM reads — never a raw model and **never**
ciphertext. These helpers turn ``DecryptedMessage`` / ``GroupRead`` / ``AgentProfile``
/ ``AgentSummary`` into plain text, surfacing ``plaintext``, ``verified`` /
``verification_status``, and ``decrypt_error`` honestly. The encrypted
``encrypted_payload`` field is deliberately never read here: it must not enter the
LLM context.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

# One definition per language: the SDK owns the predicate, so a group renders the
# same way on every surface. Every consumer imports it from the SDK — this module
# is a renderer, not a pass-through.
from rine.format import (
    UnfoundedGroupRepair,
    group_is_mls,
    group_mls_init_in_flight,
    group_mls_never_founded,
    unfounded_group_note,
)

# Same rule, one question further on: which of this org's own agents are seated
# in a group is the SDK's sentence, not a fourth copy of it here.
from rine.group_words import own_agents_clause

# And the same rule for "explain the missing-sender-key refusal once per render,
# then mark": the two-state machine is the SDK's, held per render and passed in.
# A copy of it here would be the fifth, and a module-level flag would leak the
# state across renders, groups and agents.
from rine.sender_key_words import SenderKeyRefusal

if TYPE_CHECKING:
    from rine import (
        AgentProfile,
        AgentSummary,
        DecryptedMessage,
        GroupRead,
        ThreadEntry,
        WhoAmI,
    )

#: What a group with no conversation renders as, wherever this surface prints
#: one. A group's conversation is created lazily on its first post, so a group
#: nobody has spoken in has no id yet — a normal state of a healthy group, and
#: never an error, a permission refusal or a missing feature. One definition
#: because ``render_group`` and ``render_group_list`` both print it and a model
#: reading the two must not be told two different things.
NO_CONVERSATION = "(none — nothing has been said in this group yet)"

#: How THIS surface spells the verb that founds a group's MLS state.
#:
#: langchain ships ``rine_group_reclaim`` and it really does found an unlatched
#: group, so every sentence this package prints about a group whose ratchet tree
#: was never founded names that tool. One constant rather than the literal at
#: each call site: the spelling is the load-bearing token, and a verb only half
#: this surface's outputs name is worse than one none of them do. Typed as the
#: SDK's union so a misspelling is a type error here rather than a refusal at
#: render time.
UNFOUNDED_REPAIR: UnfoundedGroupRepair = "tool-group-reclaim"


def _render_thread_line(e: ThreadEntry) -> str:
    """One thread turn as a role-tagged line: ``[sent] you: …`` / ``[received] h: …``."""
    who = "you" if e.direction == "sent" else (e.sender_handle or "unknown")
    return f"[{e.direction}] {who}: {e.text}"


def render_thread(entries: list[ThreadEntry]) -> str:
    """Render a both-sided transcript (oldest→newest) for ``rine_thread``.

    Each turn is a role-tagged line; ``[unavailable]`` text passes through. Mirrors
    the TS connectors' ``renderThread`` so transcripts read identically.
    """
    if not entries:
        return "No messages in this conversation."
    return "\n".join(_render_thread_line(e) for e in entries)


def _verified_note(msg: DecryptedMessage) -> str:
    """Return a short signature-verification suffix for a decrypted message."""
    if msg.verified:
        return "signature verified"
    return f"signature {msg.verification_status}"


def render_message_body(msg: DecryptedMessage, refusal: SenderKeyRefusal) -> str:
    """Render the readable content of one message: plaintext or decrypt_error.

    Never returns ``encrypted_payload``. If the message could not be decrypted
    (e.g. MLS or an unsupported version), the SDK's ``decrypt_error`` is surfaced
    verbatim so the failure is loud, not silent. Otherwise the decrypted plaintext
    is unwrapped: rine payloads are ``{"text": "..."}`` JSON (the convention every
    tool sends and the MCP surface ``JSON.parse``s), so the human ``text`` field is
    surfaced; a JSON object without ``text`` falls back to compact JSON and a
    non-JSON plaintext to the raw string.

    One cause is rendered in two tiers: a row nobody here holds a sender key for
    is explained in full on the FIRST such row of a render and marked with the
    SDK's bounded preview on every later one. ``refusal`` is that render's state
    and is **required** — a default would let a new loop pay the whole ~600-char
    explanation per row and still read green. Every other cause is untouched.
    """
    if msg.decrypt_error:
        return f"[unreadable] {refusal.text_for(msg) or msg.decrypt_error}"
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


def render_message(msg: DecryptedMessage, refusal: SenderKeyRefusal) -> str:
    """Render one message as a multi-field readable line (no ciphertext).

    ``refusal`` is the enclosing render's — see :func:`render_message_body`.
    """
    sender = msg.sender_handle or (str(msg.from_agent_id) if msg.from_agent_id else "unknown")
    parts = [
        f"id={msg.id}",
        f"from={sender}",
        f"type={msg.type}",
    ]
    if msg.group_handle or msg.group_id:
        parts.append(f"group={msg.group_handle or msg.group_id}")
    parts.append(f"body={render_message_body(msg, refusal)}")
    if not msg.decrypt_error:
        parts.append(f"({_verified_note(msg)})")
    return " | ".join(parts)


def render_single_message(msg: DecryptedMessage) -> str:
    """Render one fetched message as readable multi-line text (no ciphertext).

    One message was asked for, so a sender-key refusal is explained in full: a
    fresh :class:`~rine.SenderKeyRefusal` has explained nothing yet.
    """
    sender = msg.sender_handle or (str(msg.from_agent_id) if msg.from_agent_id else "unknown")
    lines = [f"from: {sender}", f"type: {msg.type}"]
    if msg.group_handle or msg.group_id:
        lines.append(f"group: {msg.group_handle or msg.group_id}")
    lines.append(f"body: {render_message_body(msg, SenderKeyRefusal())}")
    if not msg.decrypt_error:
        lines.append(f"verification: {_verified_note(msg)}")
    return "\n".join(lines)


def render_inbox(messages: list[DecryptedMessage], *, status: str = "new") -> str:
    """Render an inbox page as a numbered list, or an empty-state string.

    ``status`` is the slice that was asked for, and it names the messages: a
    ``read`` page reported as "3 new message(s)" tells the model the opposite of
    what it fetched. ``all`` has no adjective because the page spans every state.

    The page is ONE render and holds a single :class:`~rine.SenderKeyRefusal`, so
    eight rows with no sender key carry the explanation once, then a marker.
    """
    noun = "" if status == "all" else f"{status} "
    if not messages:
        return f"No {noun}messages."
    refusal = SenderKeyRefusal()
    lines = [f"{i}. {render_message(m, refusal)}" for i, m in enumerate(messages, start=1)]
    return f"{len(messages)} {noun}message(s):\n" + "\n".join(lines)


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


def render_whoami(me: WhoAmI) -> str:
    """Render this agent's own rine identity as readable multi-line text.

    Revoked agents are dropped: a revoked handle can neither send nor be sent
    to, so listing it would offer the model an address that cannot work.
    """
    handles = [a.handle for a in me.agents if a.revoked_at is None]
    return (
        f"org: {me.org.name}\n"
        f"org_slug: {me.org.slug or '(none)'}\n"
        f"trust_tier: {me.trust_tier}\n"
        f"handles: {', '.join(handles) if handles else '(none)'}"
    )


def render_group(g: GroupRead) -> str:
    """Render a group's details + its encryption-type line.

    Four states, not two. The second is the MLS init window: the group has no
    ``mls_group_id`` to render yet, while the server already accepts nothing but
    MLS there, so sends from here are MLS sends. Rendering that window as either
    of the other states states the opposite of what is happening.

    The third is a group created to run MLS whose ratchet tree was never
    founded. It reads to every other question exactly like the fourth — an
    ordinary sender-key group — so it used to fall into it, and the agent was
    never told the group had not got the encryption it was created for. It stays
    ``[OK]``: every claim that marker makes is true here, the group carries
    traffic, and only the regime is other than intended. The sentence is the
    SDK's, in this surface's spelling of the verb that founds a group.
    """
    if group_is_mls(g):
        diagnosis = (
            "[OK] post-quantum MLS group (X-Wing) — end-to-end encrypted, "
            "readable and postable from here."
        )
    elif group_mls_init_in_flight(g):
        diagnosis = (
            "[OK] MLS group, initialising — end-to-end encrypted. Sends from here already use MLS."
        )
    elif group_mls_never_founded(g):
        diagnosis = (
            "[OK] sender-key group — end-to-end encrypted, readable and postable from here. "
            f"{unfounded_group_note(UNFOUNDED_REPAIR)}"
        )
    else:
        diagnosis = (
            "[OK] sender-key group — end-to-end encrypted, readable and postable from here."
        )
    return (
        f"handle: {g.handle}\n"
        f"name: {g.name}\n"
        f"id: {g.id}\n"
        f"conversation_id: {g.conversation_id or NO_CONVERSATION}\n"
        f"your_agents: {own_agents_clause(g)}\n"
        f"members: {g.member_count}\n"
        f"enrollment: {g.enrollment_policy}\n"
        f"visibility: {g.visibility}\n"
        f"mls_enabled: {g.mls_enabled}\n"
        f"mls_group_id: {g.mls_group_id or '(none)'}\n"
        f"{diagnosis}"
    )
