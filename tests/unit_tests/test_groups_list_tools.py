"""Behavioral tests for the group-enumeration tools.

``rine_groups`` and ``rine_group_roster`` are authenticated reads over
``client.groups``; ``rine_discover_groups`` hits the UNAUTHENTICATED public group
directory — asserted the same way ``test_discovery_tools.py`` asserts it for
agents, by not mocking ``/oauth/token`` and confirming the call still succeeds.

Tools are driven through their public LangChain interface: ``.invoke({...})``
(sync) and ``.ainvoke({...})`` (async). respx intercepts httpx regardless of
transport, so the same mock backs both paths — which is how the ``_arun`` twin is
held to the same output as ``_run``.
"""

from __future__ import annotations

from typing import Any

import httpx
import respx

from langchain_rine import RineDiscoverGroupsTool, RineGroupRosterTool, RineGroupsTool

from ..conftest import (
    AGENT_ID,
    API_URL,
    GROUP_CONVERSATION_ID,
    GROUP_HANDLE,
    GROUP_ID,
    OTHER_AGENT_ID,
    SECOND_AGENT_ID,
    group_payload,
    mock_agents,
    mock_token,
    mock_two_agents,
)

OTHER_GROUP_ID = "11111111-2222-3333-4444-555555555555"
THIRD_GROUP_ID = "22222222-3333-4444-5555-666666666666"
OTHER_GROUP_HANDLE = "#lobby@acme.rine.network"


def directory_group_payload(
    *,
    group_id: str = GROUP_ID,
    name: str = "ops",
    handle: str = GROUP_HANDLE,
    description: str | None = "Operations coordination.",
    member_count: int = 4,
    enrollment_policy: str = "open",
) -> dict[str, Any]:
    """A ``/directory/groups`` search hit (the server's DirectoryGroupSummary)."""
    return {
        "id": group_id,
        "name": name,
        "handle": handle,
        "description": description,
        "enrollment_policy": enrollment_policy,
        "member_count": member_count,
        "created_at": "2024-01-01T00:00:00Z",
    }


def member_payload(
    *,
    agent_id: str = AGENT_ID,
    role: str = "member",
    agent_handle: str | None = None,
    joined_at: str = "2024-01-01T00:00:00Z",
    is_own_org: bool = True,
) -> dict[str, Any]:
    """A ``GroupMember`` payload (``GET /groups/{id}/members`` item shape).

    ``is_own_org`` defaults to ``True`` because that is what most of these rows
    are: the roster of a group this agent's org is in. A test that wants another
    org's seat says so, which is the only way that row is ever interesting.
    """
    return {
        "id": "eeeeeeee-ffff-0000-1111-222222222222",
        "group_id": GROUP_ID,
        "agent_id": agent_id,
        "role": role,
        "joined_at": joined_at,
        "agent_handle": agent_handle,
        "is_own_org": is_own_org,
    }


def _two_group_page() -> dict[str, Any]:
    """A ``GET /groups`` page with one MLS group and one sender-key group."""
    return {
        "items": [
            group_payload(name="ops", mls_enabled=True, mls_group_id="mls-1"),
            group_payload(
                name="lobby",
                handle=OTHER_GROUP_HANDLE,
                group_id=OTHER_GROUP_ID,
                member_count=7,
            ),
        ]
    }


class TestRineGroupsTool:
    @respx.mock
    def test_groups_with_two_memberships_returns_numbered_list(self, config_dir):
        """Both groups render, each with its handle, policy, encryption and member count."""
        mock_token()
        mock_agents()
        route = respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json=_two_group_page())
        )

        out = RineGroupsTool(config_dir=config_dir, api_url=API_URL).invoke({})

        assert route.called
        assert out.startswith("2 group(s):")
        assert GROUP_HANDLE in out
        assert OTHER_GROUP_HANDLE in out
        assert "enrollment=closed" in out
        assert "members=7" in out

    @respx.mock
    def test_groups_names_the_encryption_regime_per_group(self, config_dir):
        """Three groups, three regimes, and no two of them collapse into one.

        The regime comes from the SDK's ``group_encryption``, so the labels
        never merge — which is how a sender-key group used to be announced as
        MLS on other surfaces. The third row is the one that had no label of its
        own: closed, created to run MLS, and its ratchet tree never founded.

        🔴 Asserted with the trailing delimiter, not as a bare substring.
        ``"encryption=mls" in out`` is satisfied by ``encryption=mls-pending``
        and ``encryption=mls-never-founded`` alike, so it would read green on
        exactly the confusion this row exists to catch.
        """
        mock_token()
        mock_agents()
        page = _two_group_page()
        page["items"].append(
            group_payload(
                name="unfounded",
                handle="#unfounded@acme.rine.network",
                group_id=THIRD_GROUP_ID,
                mls_enabled=True,
            )
        )
        respx.get(f"{API_URL}/groups").mock(return_value=httpx.Response(200, json=page))

        out = RineGroupsTool(config_dir=config_dir, api_url=API_URL).invoke({})

        assert "encryption=mls |" in out
        assert "encryption=sender-key |" in out
        assert "encryption=mls-never-founded |" in out

    @respx.mock
    def test_groups_carries_each_groups_conversation_id_on_the_listed_row(self, config_dir):
        """The listed row carries the argument ``rine_thread`` needs.

        This is the catch-up affordance: an agent lists its groups, takes the
        conversation id off that same answer and reads the backlog — without
        having posted in the group and without having been handed a raw inbox
        row. The id has to be on the **plural** row; putting it only on
        ``rine_group_inspect`` would make the agent ask for the id in order to
        ask for the id.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(
                200,
                json={"items": [group_payload(name="ops", conversation_id=GROUP_CONVERSATION_ID)]},
            )
        )

        out = RineGroupsTool(config_dir=config_dir, api_url=API_URL).invoke({})

        assert f"conversation_id={GROUP_CONVERSATION_ID}" in out

    @respx.mock
    def test_groups_when_nothing_was_said_in_a_group_says_so_instead_of_none(self, config_dir):
        """A group with no conversation reads as a fact, not as a bug or a refusal.

        The conversation is created on the group's first post, so a group nobody
        has spoken in legitimately has none. Rendering that as ``None`` — or
        dropping the clause — tells the model the tool is broken; the sentence
        tells it there is nothing to read yet.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(
                200, json={"items": [group_payload(name="ops", conversation_id=None)]}
            )
        )

        out = RineGroupsTool(config_dir=config_dir, api_url=API_URL).invoke({})

        assert "conversation_id=(none — nothing has been said in this group yet)" in out
        assert "None" not in out

    @respx.mock
    async def test_groups_arun_matches_run(self, config_dir):
        """``.ainvoke`` returns the same list as ``.invoke`` against the same mock."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json=_two_group_page())
        )

        out = await RineGroupsTool(config_dir=config_dir, api_url=API_URL).ainvoke({})

        assert out.startswith("2 group(s):")
        assert GROUP_HANDLE in out
        assert OTHER_GROUP_HANDLE in out

    @respx.mock
    def test_groups_names_which_of_this_orgs_agents_are_seated_in_each_group(self, config_dir):
        """The org-scoped row answers the agent-scoped question it used to raise.

        ``GET /groups`` lists a group whenever **any** agent of the org is in it,
        so before this clause the row could not tell the agent whether the one
        about to post was among them. It answers in **handles** — the spelling
        ``rine_whoami`` already gave the caller, and one the acting-agent header
        takes as readily as a UUID, so the row names a seat without having to be
        joined against a second tool's output first.

        Both seats are this org's own. ``member_agent_ids`` is own-org by
        construction, so a fixture naming a peer-org agent would be pinning the
        rendering of a state the server cannot produce.
        """
        mock_token()
        mock_two_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [
                        group_payload(name="ops", member_agent_ids=[AGENT_ID, SECOND_AGENT_ID])
                    ]
                },
            )
        )

        out = RineGroupsTool(config_dir=config_dir, api_url=API_URL).invoke({})

        assert "your_agents=test-agent@myorg.rine.network, second-agent@myorg.rine.network" in out
        assert AGENT_ID not in out

    @respx.mock
    def test_groups_when_no_agent_of_this_org_is_seated_says_so_instead_of_nothing(
        self, config_dir
    ):
        """An empty seat list is the answer, not the absence of one.

        The org is in the group — that is why the row is here at all — and no
        agent of it holds a seat, so a send would be refused. Dropping the clause
        would read as a rendering fault; printing an empty value would read as a
        group with no members.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(
                200, json={"items": [group_payload(name="ops", member_agent_ids=[])]}
            )
        )

        out = RineGroupsTool(config_dir=config_dir, api_url=API_URL).invoke({})

        assert "your_agents=(none — no agent of your org is seated in this group)" in out

    @respx.mock
    def test_groups_with_no_memberships_returns_the_empty_state(self, config_dir):
        """No groups is a plain sentence, never an empty string the model must guess at."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(return_value=httpx.Response(200, json={"items": []}))

        out = RineGroupsTool(config_dir=config_dir, api_url=API_URL).invoke({})

        assert out == "This agent belongs to no groups."

    @respx.mock
    def test_groups_api_error_returns_a_formatted_string_not_a_raise(self, config_dir):
        """A server error surfaces as a tool message so the agent continues."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(500, json={"detail": "boom"})
        )

        out = RineGroupsTool(config_dir=config_dir, api_url=API_URL).invoke({})

        assert "500" in out


class TestRineDiscoverGroupsTool:
    @respx.mock
    def test_discover_groups_with_query_sends_q_and_renders_hits(self, config_dir):
        """The q filter reaches the wire and each hit renders handle + member count."""
        route = respx.get(f"{API_URL}/directory/groups").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [directory_group_payload()],
                    "total_estimate": 1,
                    "has_more": False,
                },
            )
        )

        out = RineDiscoverGroupsTool(config_dir=config_dir, api_url=API_URL).invoke({"q": "ops"})

        assert route.called
        assert route.calls[0].request.url.params["q"] == "ops"
        # The header says what the list IS. The directory scan carries no
        # identity, so no row can answer "am I in it" — and this surface's
        # membership list prints near-identical rows that do.
        assert out.startswith(
            "1 public group(s) found in the directory. A row here is a group "
            "that exists, not a group this agent is in:"
        )
        assert GROUP_HANDLE in out
        assert "members=4" in out
        assert "Operations coordination." in out

    @respx.mock
    def test_discover_groups_prints_the_id_rine_group_join_can_spend(self, config_dir):
        """A directory row carries the only reference a stranger can act on.

        ``GET /groups?handle=`` is scoped to the groups this org already sits in,
        so resolving a discovered handle finds nothing for the one caller
        discovery exists for; ``POST /groups/{id}/join`` is global. The row shape
        matches what ``rine-mastra`` and ``rine-eve`` already print.
        """
        respx.get(f"{API_URL}/directory/groups").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [directory_group_payload()],
                    "total_estimate": 1,
                    "has_more": False,
                },
            )
        )

        out = RineDiscoverGroupsTool(config_dir=config_dir, api_url=API_URL).invoke({"q": "ops"})

        assert f"{GROUP_HANDLE} (id {GROUP_ID})" in out

    @respx.mock
    async def test_discover_groups_arun_matches_run(self, config_dir):
        """``.ainvoke`` returns the same page as ``.invoke`` against the same mock."""
        respx.get(f"{API_URL}/directory/groups").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [directory_group_payload()],
                    "total_estimate": 1,
                    "has_more": False,
                },
            )
        )

        out = await RineDiscoverGroupsTool(config_dir=config_dir, api_url=API_URL).ainvoke(
            {"q": "ops"}
        )

        assert out.startswith(
            "1 public group(s) found in the directory. A row here is a group "
            "that exists, not a group this agent is in:"
        )
        assert GROUP_HANDLE in out

    @respx.mock
    def test_discover_groups_renders_the_enrollment_policy_the_directory_sent(self, config_dir):
        """A directory row carries the policy, and the agent is told which one.

        It is the only field on a discovery hit that decides what happens next:
        an ``open`` group is self-joinable with ``rine_group_join``, a
        ``majority`` one makes the caller wait for the electorate. Rendering the
        row without it leaves the model guessing, which is what the Python SDK's
        ``GroupSummary`` used to force by dropping the field on parse.
        """
        respx.get(f"{API_URL}/directory/groups").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [
                        directory_group_payload(),
                        directory_group_payload(
                            group_id=OTHER_GROUP_ID,
                            handle=OTHER_GROUP_HANDLE,
                            enrollment_policy="unanimity",
                        ),
                    ],
                    "total_estimate": 2,
                    "has_more": False,
                },
            )
        )

        out = RineDiscoverGroupsTool(config_dir=config_dir, api_url=API_URL).invoke({})

        assert "enrollment=open" in out
        assert "enrollment=unanimity" in out

    @respx.mock
    def test_discover_groups_is_unauthenticated(self, config_dir):
        """No OAuth token is fetched and no Authorization header is sent.

        Same posture as ``rine_discover``: the public group directory is readable
        without credentials, and asserting it here is what stops a later change
        from quietly making group discovery require an agent.
        """
        route = respx.get(f"{API_URL}/directory/groups").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [directory_group_payload()],
                    "total_estimate": 1,
                    "has_more": False,
                },
            )
        )

        RineDiscoverGroupsTool(config_dir=config_dir, api_url=API_URL).invoke({})

        assert "authorization" not in {k.lower() for k in route.calls[0].request.headers}

    @respx.mock
    def test_discover_groups_with_no_matches_returns_the_empty_state(self, config_dir):
        """An empty page is a plain sentence, not an empty string."""
        respx.get(f"{API_URL}/directory/groups").mock(
            return_value=httpx.Response(
                200, json={"items": [], "total_estimate": 0, "has_more": False}
            )
        )

        out = RineDiscoverGroupsTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"q": "nothing"}
        )

        assert out == "No groups matched."


class TestRineGroupRosterTool:
    @respx.mock
    def test_roster_renders_handle_role_and_join_date(self, config_dir):
        """Each member line carries the three fields the description promises."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json={"items": [group_payload()]})
        )
        route = respx.get(f"{API_URL}/groups/{GROUP_ID}/members").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [
                        member_payload(role="admin", agent_handle="boss@acme"),
                        member_payload(
                            agent_id=OTHER_AGENT_ID,
                            agent_handle="other@acme",
                            joined_at="2024-03-04T00:00:00Z",
                        ),
                    ]
                },
            )
        )

        out = RineGroupRosterTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": GROUP_HANDLE}
        )

        assert route.called
        assert out.startswith("2 member(s):")
        assert "boss@acme | role=admin | joined=2024-01-01" in out
        assert "other@acme | role=member | joined=2024-03-04" in out

    @respx.mock
    def test_roster_marks_this_orgs_own_seats_and_still_lists_every_other_one(self, config_dir):
        """``is_own_org`` is a marker, never a filter.

        A roster is authorised for the org but is not narrowed to it: the agent
        needs to see the ones it is talking **to** as much as the ones it can
        talk **as**. So the stranger's row is present and unmarked, the own row
        carries ``(yours)``, and the count is the whole roster.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json={"items": [group_payload()]})
        )
        respx.get(f"{API_URL}/groups/{GROUP_ID}/members").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [
                        member_payload(agent_handle="mine@acme"),
                        member_payload(
                            agent_id=OTHER_AGENT_ID,
                            agent_handle="stranger@other",
                            is_own_org=False,
                        ),
                    ]
                },
            )
        )

        out = RineGroupRosterTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": GROUP_HANDLE}
        )

        assert out.startswith("2 member(s):")
        assert "mine@acme | role=member | joined=2024-01-01 (yours)" in out
        assert "stranger@other | role=member | joined=2024-01-01" in out
        assert "stranger@other | role=member | joined=2024-01-01 (yours)" not in out

    @respx.mock
    async def test_roster_arun_matches_run(self, config_dir):
        """``.ainvoke`` returns the same roster as ``.invoke`` against the same mock."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json={"items": [group_payload()]})
        )
        respx.get(f"{API_URL}/groups/{GROUP_ID}/members").mock(
            return_value=httpx.Response(
                200, json={"items": [member_payload(role="admin", agent_handle="boss@acme")]}
            )
        )

        out = await RineGroupRosterTool(config_dir=config_dir, api_url=API_URL).ainvoke(
            {"group": GROUP_HANDLE}
        )

        assert "boss@acme | role=admin | joined=2024-01-01" in out

    @respx.mock
    def test_roster_member_without_a_handle_renders_by_agent_id(self, config_dir):
        """``agent_handle`` is optional on the wire; the member must still be counted."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json={"items": [group_payload()]})
        )
        respx.get(f"{API_URL}/groups/{GROUP_ID}/members").mock(
            return_value=httpx.Response(200, json={"items": [member_payload()]})
        )

        out = RineGroupRosterTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": GROUP_HANDLE}
        )

        assert out.startswith("1 member(s):")
        assert AGENT_ID in out

    @respx.mock
    def test_roster_of_an_unknown_group_returns_a_formatted_not_found(self, config_dir):
        """A group this agent cannot resolve surfaces as a tool message, never a raise."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(return_value=httpx.Response(200, json={"items": []}))

        out = RineGroupRosterTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": "#ghost@acme"}
        )

        assert "Not found" in out or "ghost" in out
