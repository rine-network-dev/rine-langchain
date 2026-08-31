"""What a plural admission did, rendered for a model.

A roster on ``rine_group_create`` and a batch on ``rine_group_invite`` are the
same operation at two entry points, so they answer in the same vocabulary: one
outcome per requested agent, in the order they were named. A batch never fails
whole — naming 32 agents and having one of them revoked costs that one agent
its seat, not the other 31 theirs — which is only useful if the agent is told
which one.

The report says whether a **row** was minted for each agent, which is a different
question from whether a ratchet-tree **leaf** was. An agent can be invited and
still hold no leaf: it may join, and it will read nothing until one exists.

The row is not always an invitation. On a majority- or unanimity-enrollment group
an invite files a join request the group's electorate decides, so that entry comes
back ``nominated`` and seats nobody until the vote carries. Both kinds hold a seat
against the group's ceiling; only one of them is spendable by the agent it names.

Sync and async tool bodies share this module, so the two can never render the
same answer differently — which is the whole reason it exists in a package
whose every tool is written twice.

Each outcome is worded by the SDK's :func:`rine.describe_admission_entry`
rather than here. This surface used to print the wire token — ``skipped
(not_applicable)`` — which on an open group labelled as skipped every agent the
very same call had just DM'd an invitation notice. The token is the machine's
vocabulary; an agent reading it cannot tell a revoked identity from a group that
needs no invitation at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rine import (
    admission_headline,
    describe_admission_entry,
    describe_mls_admission,
    render_request_row,
)
from rine.format import describe_group_encryption

from langchain_rine._format import UNFOUNDED_REPAIR

if TYPE_CHECKING:
    from rine._mls.admission import MlsAdmissionResult
    from rine.group_admission import GroupAdmissionReport, GroupCreateWithRoster
    from rine.types import InviteResult, JoinRequestRead, VoteResponse

#: The refusal when a caller names both invite targets, or neither.
INVITE_TARGET_REQUIRED = (
    "Name exactly one of agent_to_invite (one invitation) or agents_to_invite (a batch)."
)

#: What an approving vote that crossed the threshold actually did. Appended only
#: when the SDK reports ``mls_seated``, never on the server's ``approved`` alone.
#: The seat is best-effort — a sender-key group has no tree to mint a leaf in,
#: and an add commit that fails is logged and swallowed so the vote can still be
#: reported — so a sentence conditioned on the status alone claims a Welcome for
#: every vote that minted none.
VOTE_SEATED = (
    "The applicant is now a member; their ratchet-tree leaf and Welcome were minted "
    "as part of this vote."
)

#: The same admission, when no leaf and no Welcome were minted. The membership
#: stands either way, so this reports what is missing and who can supply it
#: rather than reading as a failure.
VOTE_NOT_SEATED = (
    "The applicant is now a member. No ratchet-tree leaf or Welcome was minted by "
    "this vote, so on a group that runs MLS they read nothing until a member runs "
    "resume-admission on it."
)

#: A carried vote that seated nobody, and the fourth status a vote can answer.
#: A member may nominate an agent, and the electorate approving that nomination
#: is not the same event as the agent agreeing to be in the group — so a carried
#: vote on an agent that has not consented resolves the request to a spendable
#: invitation rather than to a seat. Reporting it as "approved" would claim a
#: membership nobody has; letting it fall through to the bare status line would
#: report a decided request as an unrecognised word.
VOTE_CARRIED_AWAITING_CONSENT = (
    "The electorate approved this request. The agent it names was nominated by a member "
    "rather than having applied itself, so it now holds a spendable invitation instead of "
    "a seat — it becomes a member by accepting that invitation, and the vote seats the "
    "member, which is what grants the group's keys."
)


def render_admission(report: GroupAdmissionReport) -> str:
    """Per-agent invitation outcomes, one line each, in the order they were named."""
    lines = [f"  {describe_admission_entry(entry)}" for entry in report.entries]
    # 🔴 The head is the SDK's, derived from who was REACHED. Composing it from
    # ``invited``/``skipped`` printed "0 invited, N skipped" on every open group
    # — the server answers ``not_applicable`` for each valid agent there — over
    # per-agent lines saying those same agents may join whenever they like.
    return "\n".join([admission_headline(report), *lines])


def render_created(group: GroupCreateWithRoster) -> str:
    """The whole ``rine_group_create`` answer: what was made, and what a roster did.

    The encryption sentence is the SDK's :func:`describe_group_encryption`,
    which covers the case a local three-state label cannot: a group whose MLS
    init **failed** did not get the MLS it was created for, and reads exactly
    like a sender-key group to anything that only asks "is it MLS?". It does not
    hold no keys — that sentence shipped, was false, and is gone.
    """
    head = (
        f"Created group {group.handle} (id {group.id}, enrollment "
        f"{group.enrollment_policy}).\n"
        f"Encryption: {describe_group_encryption(group, UNFOUNDED_REPAIR)}"
    )
    mls = render_mls(group.mls_admission)
    if group.roster is None:
        return f"{head}{mls}"
    return (
        f"{head}\nRoster invited (nobody is a member until they join; an invitation "
        f"holds a seat until it is accepted or expires):\n"
        f"{render_admission(group.roster)}{mls}"
    )


def render_mls(result: MlsAdmissionResult | None) -> str:
    """The MLS half, or nothing at all when there was none to report.

    A roster **invites**; the founding Welcome is what decides whether an
    invitee can read a word of the group. An agent whose KeyPackage pool was
    empty holds a perfectly good invitation and no leaf, and the voucher counts
    above cannot say so — which is how a forty-agent roster with three empty
    pools came to answer "40 invited, 0 skipped" and nothing else.
    """
    return "" if result is None else f"\n{describe_mls_admission(result)}"


def render_requests(group: str, outstanding: str, rows: list[JoinRequestRead]) -> str:
    """What this group still owes an answer on, oldest first.

    The filter is named in both the head and the empty answer, because "nothing
    outstanding" under ``pending`` and under ``live`` are different facts: a group
    with no vote queue can still be holding a dozen unaccepted invitations, and
    each of those holds a ratchet-tree seat against the group's ceiling.

    ``pending`` holds agents that applied themselves **and** agents a member
    nominated; both are decided by the same vote, so they are one queue
    and not two.

    Each row is the SDK's :func:`rine.render_request_row`, so it says the same
    thing here as on every other surface — including whether this agent's own
    vote would count, which is the one fact that decides whether calling
    ``rine_group_vote`` is worth a round trip.
    """
    if not rows:
        return f"Nothing outstanding in `{group}` under filter `{outstanding}`."
    lines = [render_request_row(row) for row in sorted(rows, key=lambda r: r.created_at)]
    return "\n".join([f"Outstanding in `{group}` ({outstanding}): {len(rows)}", *lines])


def render_vote(group: str, request_id: str, choice: str, result: VoteResponse) -> str:
    """The vote as cast, and what crossing the threshold did.

    ``mls_seated`` is derived by the SDK, not sent by the server, and it is the
    only thing here that knows whether a leaf and a Welcome exist.

    ``approved`` stays the only status that claims a seat. A carried vote on a
    nomination nobody consented to answers ``invited`` and seats nobody, so
    widening this branch to "anything that is not denied" would report a
    membership that does not exist — and, on the surfaces that act on it, admit
    an agent that was never asked.
    """
    head = (
        f"Vote `{choice}` recorded on request `{request_id}` in `{group}`. "
        f"Request status: `{result.status}`."
    )
    if result.status == "invited":
        return f"{head} {VOTE_CARRIED_AWAITING_CONSENT}"
    if result.status != "approved":
        return head
    return f"{head} {VOTE_SEATED if result.mls_seated else VOTE_NOT_SEATED}"


def render_invite_one(group: str, agent_to_invite: str, result: InviteResult) -> str:
    """One invitation, in the words of the row the server actually wrote.

    A closed group mints a voucher and this reads exactly as it always has. A
    majority- or unanimity-enrollment group files a join request instead, and the
    status alone (``pending``) is what tells them apart — printed under the word
    "Invited" it would read as an invitation in an unexplained state, which does
    not hold: nobody is seated until the vote carries.
    """
    if result.status == "pending":
        return (
            f"Nominated {agent_to_invite} for {group}: a join request the group's "
            f"electorate decides. This invite counts as one approval, and nobody is "
            f"seated until the vote carries."
        )
    return f"Invited {agent_to_invite} to {group} (status {result.status})."


__all__ = [
    "INVITE_TARGET_REQUIRED",
    "VOTE_CARRIED_AWAITING_CONSENT",
    "VOTE_NOT_SEATED",
    "VOTE_SEATED",
    "render_admission",
    "render_created",
    "render_invite_one",
    "render_mls",
    "render_requests",
    "render_vote",
]
