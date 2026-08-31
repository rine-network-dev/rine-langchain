"""Plural renderers for the group-domain tools.

``_format.py`` renders **one** of a thing — one message, one profile, one group.
These three render **lists**, and they are the ones ``tools/groups_list.py``
returns: the groups the agent's org belongs to, the public groups a directory search
found, and one group's roster. Kept out of ``_format.py`` so that file stays
inside the per-file budget.

Same two rules as ``_format.py``: the return value is always a string the LLM
reads, and the encryption regime is the SDK's answer (``rine.format``) rather
than a second copy of the predicate here. Both the sync and async tool bodies
render through these, so the two halves cannot disagree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rine.format import group_encryption
from rine.group_words import own_agents_clause

# The "no conversation yet" sentence has one definition per surface, next to the
# singular renderer that also prints it.
from langchain_rine._format import NO_CONVERSATION

if TYPE_CHECKING:
    from rine import GroupMember, GroupRead, GroupSummary


def render_group_list(groups: list[GroupRead]) -> str:
    """Render the groups the calling org's agents belong to as a numbered list.

    One line per group: handle, name, enrollment policy, encryption regime,
    member count, the group's ``conversation_id`` and which of this org's own
    agents are seated in it — enough for the model to pick a handle and hand it
    to ``rine_group_inspect``/``rine_send``/``rine_thread`` without a second
    call.

    The conversation id belongs on **this** line and not only on
    ``render_group``: it names the group's running thread, and an agent that has
    to inspect a group to learn it is being asked for the id in order to ask for
    the id. A group nobody has posted in has no conversation yet, and that
    renders as a sentence rather than as a blank, a ``None`` or an error — it is
    a normal state of a healthy group. It is no longer the *only* way to read
    the group: ``rine_thread`` takes the group itself now, which is what a
    caller holding a handle and nothing else can spend.

    ``your_agents`` is on this line for a sharper reason: the list itself is
    ORG-scoped, so a group appears here whenever *any* agent of the org is in
    it, and until this clause existed the row could not tell the agent whether
    the one about to post was one of them. An empty clause is the answer to that
    question, not the absence of one — the group is real, the org is in it, and
    a send from an agent named on no seat would be refused.
    """
    if not groups:
        return "This agent belongs to no groups."
    lines = [
        f"{i}. {g.handle} ({g.name}) | enrollment={g.enrollment_policy} "
        f"| encryption={group_encryption(g)} | members={g.member_count} "
        f"| conversation_id={g.conversation_id or NO_CONVERSATION} "
        f"| your_agents={own_agents_clause(g)}"
        for i, g in enumerate(groups, start=1)
    ]
    return f"{len(groups)} group(s):\n" + "\n".join(lines)


def render_group_summaries(groups: list[GroupSummary]) -> str:
    """Render a public-directory group search page as a numbered list.

    A directory hit is not a ``GroupRead``: the search returns only what a
    stranger may see, so there is no encryption regime and no roster to render
    here — inspecting either of those needs membership. The enrollment policy
    IS on a directory row, and it is the one field a stranger decides on: it
    separates a group this agent can self-join from one that will make it wait
    for a vote.

    🔴 The **id** is on the row because it is the only token a stranger can
    spend. ``GET /groups?handle=`` is scoped to the groups this org already
    holds a seat in, so resolving a discovered handle returns nothing for the
    one caller a directory search exists for; ``POST /groups/{id}/join`` looks
    the group up globally. The shape is ``handle (id …)`` because
    ``rine-mastra`` and ``rine-eve`` already render their directory rows that
    way.

    The header says what the list IS. ``GET /directory/groups`` is an
    unauthenticated public-visibility scan across every org and carries no
    identity at all, so nothing on a row can say whether this agent holds a seat
    in that group — while the membership list this surface also prints
    (:func:`render_group_list`) carries exactly that. Two numbered lists of
    near-identical rows, one of which answers "are you in it" and one of which
    cannot, are told apart by their headers or not at all. It names no verb:
    which tool to reach for next is the tool description's sentence, not this
    one's.
    """
    if not groups:
        return "No groups matched."
    lines = []
    for i, g in enumerate(groups, start=1):
        desc = f" — {g.description}" if g.description else ""
        lines.append(
            f"{i}. {g.handle} (id {g.id}) ({g.name}, enrollment={g.enrollment_policy}, "
            f"members={g.member_count}){desc}"
        )
    return (
        f"{len(groups)} public group(s) found in the directory. A row here is a "
        f"group that exists, not a group this agent is in:\n" + "\n".join(lines)
    )


def render_roster(members: list[GroupMember]) -> str:
    """Render a group's roster as a numbered list: handle, role, join date, ownership.

    ``agent_handle`` is optional on the wire, so a member the server did not
    name renders by agent id rather than silently vanishing from the count.

    A row the calling org owns is marked ``(yours)``. A **marker, never a
    filter**: every member of the group is listed whoever owns it, and the count
    is the whole roster — the agent needs to see the ones it is talking to as
    much as the ones it can talk as. Rows that are not the org's own carry
    nothing rather than a second word, because "not yours" is the ordinary case
    and labelling it would bury the one row that matters.
    """
    if not members:
        return "This group has no members."
    lines = [
        f"{i}. {m.agent_handle or m.agent_id} | role={m.role} "
        f"| joined={m.joined_at:%Y-%m-%d}{' (yours)' if m.is_own_org else ''}"
        for i, m in enumerate(members, start=1)
    ]
    return f"{len(members)} member(s):\n" + "\n".join(lines)
