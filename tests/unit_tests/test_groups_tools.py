"""Behavioral tests for the group tools.

Covers create (post-quantum MLS by default, a sender-key group on request,
single-call with description), invite, remove, and inspect's encryption-type line. The
``#``-prefix group send/read routing is asserted via RineSendTool against a
sender-key group (verifies group send rides the messaging tool, not a separate
group-send tool) and via RineInboxTool returning a group message.

Tools are driven through their public LangChain interface: ``.invoke({...})``
(sync) and ``.ainvoke({...})`` (async). respx intercepts httpx regardless
of sync/async transport, so the same mock backs both paths.
"""

from __future__ import annotations

import json
import re

import httpx
import pytest
import respx
from rine import GroupTarget, SyncRineClient
from rine._constants import EncryptionVersion

from langchain_rine import (
    RineGroupCreateTool,
    RineGroupInspectTool,
    RineGroupInvitesTool,
    RineGroupInviteTool,
    RineGroupJoinTool,
    RineGroupRemoveTool,
    RineInboxTool,
    RineSendTool,
)
from langchain_rine.tools.groups import RineGroupCreateInput

from ..conftest import (
    AGENT_ID,
    API_URL,
    GROUP_CONVERSATION_ID,
    GROUP_HANDLE,
    GROUP_ID,
    MESSAGE_ID,
    OTHER_AGENT_ID,
    group_payload,
    message_payload,
    mock_agents,
    mock_group_mls_init,
    mock_mls_group_services,
    mock_token,
)

#: A second invitee, so a batch is genuinely plural. Local rather than in
#: conftest: only the admission tests need a third party.
THIRD_AGENT_ID = "cccccccc-dddd-eeee-ffff-111111111111"

# =====================================================================
# RineGroupCreateTool
# =====================================================================


class TestRineGroupCreateTool:
    def test_create_input_schema_offers_the_encryption_choice_without_defaulting_it(self):
        """The agent can ask for a sender-key group; omitting it keeps MLS.

        The toggle used to be hidden here, so an agent that wanted a broadcast
        group could only get an MLS one — while the TypeScript surfaces shipped
        the flag. Optional, and with no default of its own: an unnamed value
        sends no ``mls_enabled`` at all, and the group the server makes for it is
        the MLS group it always made, off the default declared on ``GroupCreate``.
        """
        fields = set(RineGroupCreateInput.model_fields)
        assert fields == {
            "name",
            "enrollment",
            "visibility",
            "description",
            "vote_duration_hours",
            "members",
            "enable_mls",
        }
        field = RineGroupCreateInput.model_fields["enable_mls"]
        assert not field.is_required()
        assert field.default is None
        assert "sender-key" in (field.description or "")

    def test_create_input_schema_requires_visibility_with_no_default(self):
        """Every client-side visibility default is gone.

        The model is told by the schema, at the point of choice, rather than by a
        refusal after it has already guessed — and the description says what each
        value does, including that ``public`` turns the member-joined signal off.
        """
        field = RineGroupCreateInput.model_fields["visibility"]
        assert field.is_required()
        assert "dir.rine.network" in (field.description or "")
        assert "member-joined signal off" in (field.description or "")

    @respx.mock
    def test_create_group_without_a_visibility_is_refused_in_rines_words(self, config_dir):
        """The refusal names both choices, so nobody has to read the docs to pick.

        It is the SDK's sentence, not a copy of it: a second copy with no gate
        keeping the two in step is how the surfaces disagreed in the first place.
        Driven through ``_run`` — the method LangChain calls once it has parsed
        the arguments — because what is under test is this tool's own behaviour
        when no visibility arrives, not LangChain's argument parser.
        """
        mock_token()
        mock_agents()
        create = respx.post(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json=group_payload(name="ops"))
        )

        out = RineGroupCreateTool(config_dir=config_dir, api_url=API_URL)._run(name="ops")

        assert not create.called
        assert "dir.rine.network" in out
        assert 'visibility="public"' in out
        assert 'visibility="private"' in out

    @respx.mock
    def test_create_group_with_a_roster_invites_and_seats_nobody(self, config_dir):
        """A roster mints invitations, and the group still has one member."""
        mock_token()
        mock_agents()
        route = respx.post(f"{API_URL}/groups").mock(
            return_value=httpx.Response(
                200,
                json={
                    "group": group_payload(name="ops"),
                    "roster": {
                        "invited": 1,
                        "skipped": 1,
                        "entries": [
                            {"agent_id": OTHER_AGENT_ID, "status": "invited", "reason": None},
                            {
                                "agent_id": THIRD_AGENT_ID,
                                "status": "skipped",
                                "reason": "already_a_member",
                            },
                        ],
                    },
                },
            )
        )

        out = RineGroupCreateTool(config_dir=config_dir, api_url=API_URL).invoke(
            {
                "name": "ops",
                "visibility": "private",
                "members": [OTHER_AGENT_ID, THIRD_AGENT_ID],
            }
        )

        body = json.loads(route.calls[0].request.content)
        assert body["members"] == [OTHER_AGENT_ID, THIRD_AGENT_ID]
        assert "1 of 2 reached; 1 new invitation to accept." in out
        assert f"{THIRD_AGENT_ID}: already a member of this group" in out
        assert "nobody is a member until they join" in out

    @respx.mock
    def test_open_group_roster_is_not_reported_as_a_failure(self, config_dir):
        """An open group's roster reads as reached, not as skipped.

        Every entry comes back ``skipped/not_applicable`` because there is no
        voucher to mint — and the very same call DMs each of those agents an
        invitation notice. Printing the wire token told the agent its whole
        roster had failed. The word "skipped" is deliberately absent from the
        per-agent lines: it is the server's field name, not the fact.
        """
        mock_token()
        mock_agents()
        respx.post(f"{API_URL}/groups").mock(
            return_value=httpx.Response(
                200,
                json={
                    "group": group_payload(name="ops"),
                    "roster": {
                        "invited": 0,
                        "skipped": 1,
                        "entries": [
                            {
                                "agent_id": OTHER_AGENT_ID,
                                "status": "skipped",
                                "reason": "not_applicable",
                            }
                        ],
                    },
                },
            )
        )

        out = RineGroupCreateTool(config_dir=config_dir, api_url=API_URL).invoke(
            {
                "name": "ops",
                "visibility": "private",
                "enrollment": "open",
                "members": [OTHER_AGENT_ID],
            }
        )

        assert f"{OTHER_AGENT_ID}: this group takes open enrollment" in out
        assert "may join whenever it likes" in out
        assert "not_applicable" not in out
        assert f"{OTHER_AGENT_ID}: skipped" not in out

    @respx.mock
    def test_create_group_returns_mls_confirmation(self, config_dir):
        """A closed group is founded as post-quantum MLS — the confirmation says so.

        The latch lands server-side only after the create response is minted, so
        this passes only while create re-reads the group: without the re-read the
        brand-new MLS group would be announced as sender-key.
        """
        mock_token()
        mock_agents()
        respx.post(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json=group_payload(name="ops", mls_enabled=True))
        )
        mock_group_mls_init()

        out = RineGroupCreateTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"name": "ops", "visibility": "private"}
        )

        assert "Created group" in out
        assert GROUP_HANDLE in out
        assert "MLS over the post-quantum X-Wing ciphersuite" in out

    @respx.mock
    def test_create_when_the_founding_landed_and_the_re_read_failed_still_reports_mls(
        self, config_dir
    ):
        """A founded group is announced as MLS even when the confirming re-read is refused.

        🔴 The founding LANDS here and only the re-read that would carry the
        latch back is refused, so what the SDK hands this tool is the PRE-init
        create response: no latch, no pending, ``mls_enabled`` true — field for
        field the shape every predicate below the admission arm reads as a group
        whose ratchet tree was never founded. It is not one, and calling it one
        sends an agent to found what is already founded.

        ``mls_admission`` is what the founding returned and it exists nowhere
        else: ``rine._resources.group_ops.create_group`` carries it across the
        failed re-read for exactly this reason, and ``rine.format`` asks for it
        before it asks any predicate. Its TypeScript twin ``createdGroupMls``
        checks ``mlsAdmission`` first for the same reason.

        The never-founded rendering is NOT reachable on this path: a create that
        skips the founding is a create that sent ``mls_enabled=false``, and the
        row comes back sender-key. Which verb THIS surface names for a genuinely
        unfounded group is asserted where that state is real —
        ``test_inspect_group_created_for_mls_and_never_founded_names_this_agents_verb``.
        """
        mock_token()
        mock_agents()
        mock_mls_group_services(GROUP_ID, (AGENT_ID,))
        respx.post(f"{API_URL}/groups/{GROUP_ID}/mls/init").mock(
            return_value=httpx.Response(200, json={"members": []})
        )
        respx.post(f"{API_URL}/groups/{GROUP_ID}/mls/commit").mock(
            return_value=httpx.Response(200, json={"epoch": 1})
        )
        respx.get(f"{API_URL}/groups/{GROUP_ID}").mock(return_value=httpx.Response(503, json={}))
        respx.post(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json=group_payload(name="ops", mls_enabled=True))
        )

        out = RineGroupCreateTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"name": "ops", "visibility": "private"}
        )

        assert "Encryption: MLS over the post-quantum X-Wing ciphersuite." in out
        # The carried admission, printed. It is the whole of the evidence that
        # the founding landed, so a create that dropped it would go straight
        # back to telling this agent the group has no tree.
        assert "MLS: 0 seated across 1 commit(s), now at epoch 1." in out
        assert "its ratchet tree was never founded" not in out
        assert "rine_group_reclaim" not in out

    @respx.mock
    def test_create_group_defaults_to_mls_without_asking_for_it(self, config_dir):
        """New groups default to post-quantum MLS, and the body asks for nothing.

        MLS is the server's own default for a closed group, so what proves the
        group is an MLS one is the founding this create runs — not a flag
        echoing the default back at the server.
        """
        mock_token()
        mock_agents()
        route = respx.post(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json=group_payload(name="ops", mls_enabled=True))
        )
        init = mock_group_mls_init()

        RineGroupCreateTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"name": "ops", "visibility": "private"}
        )

        assert "mls_enabled" not in json.loads(route.calls[0].request.content)
        assert init.called

    @respx.mock
    def test_create_group_with_enable_mls_false_asks_the_server_and_founds_nothing(
        self, config_dir
    ):
        """A sender-key group is asked for on the wire, and no tree is founded.

        Both halves matter. Skipping the founding alone would leave a closed row
        the server still records as an MLS group with no latch, which every
        surface reports as a group whose ratchet tree was never founded, for the
        life of the group.
        """
        mock_token()
        mock_agents()
        route = respx.post(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json=group_payload(name="ops", mls_enabled=False))
        )
        init = mock_group_mls_init()

        out = RineGroupCreateTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"name": "ops", "visibility": "private", "enable_mls": False}
        )

        assert json.loads(route.calls[0].request.content)["mls_enabled"] is False
        assert not init.called
        assert "Created group" in out

    @respx.mock
    def test_create_group_with_description_is_a_single_call(self, config_dir):
        """description is passed straight to create — no follow-up groups.update PATCH."""
        mock_token()
        mock_agents()
        create = respx.post(f"{API_URL}/groups").mock(
            return_value=httpx.Response(
                200, json=group_payload(name="ops", description="war room")
            )
        )
        update = respx.patch(f"{API_URL}/groups/{GROUP_ID}").mock(
            return_value=httpx.Response(200, json=group_payload(name="ops"))
        )

        out = RineGroupCreateTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"name": "ops", "visibility": "private", "description": "war room"}
        )

        body = json.loads(create.calls[0].request.content)
        assert body["description"] == "war room"
        assert not update.called  # single-call, no PATCH workaround
        assert "Created group" in out

    @respx.mock
    def test_create_group_when_a_vote_window_is_named_then_the_hours_ride_the_create(
        self, config_dir
    ):
        """The window reaches the wire, rather than only the schema.

        Declaring the field is what a schema test proves. Drop the line that
        copies it into the create body and the tool still accepts
        ``vote_duration_hours=6``, still reports success, and founds a group on
        the server's 72-hour default — a window wrong by 12x, with every suite
        green. It has to ride the create itself: a group that exists for even a
        moment under the wrong deadline can expire a request under it.
        """
        mock_token()
        mock_agents()
        create = respx.post(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json=group_payload(name="ops"))
        )

        RineGroupCreateTool(config_dir=config_dir, api_url=API_URL).invoke(
            {
                "name": "ops",
                "visibility": "private",
                "enrollment": "majority",
                "vote_duration_hours": 6,
            }
        )

        body = json.loads(create.calls[0].request.content)
        assert body["vote_duration_hours"] == 6

    @respx.mock
    def test_create_group_name_conflict_returns_clear_string(self, config_dir):
        mock_token()
        mock_agents()
        respx.post(f"{API_URL}/groups").mock(
            return_value=httpx.Response(409, json={"detail": "group name taken"})
        )

        out = RineGroupCreateTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"name": "ops", "visibility": "private"}
        )

        assert "group name taken" in out

    @respx.mock
    async def test_create_group_arun_matches_run(self, config_dir):
        """``.ainvoke`` returns the same confirmation as ``.invoke``.

        Same mocks as the sync case, so the async create path is held to the same
        answer — including its own re-read of the freshly latched group.
        """
        mock_token()
        mock_agents()
        respx.post(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json=group_payload(name="ops", mls_enabled=True))
        )
        mock_group_mls_init()

        out = await RineGroupCreateTool(config_dir=config_dir, api_url=API_URL).ainvoke(
            {"name": "ops", "visibility": "private"}
        )

        assert "Created group" in out
        assert GROUP_HANDLE in out
        assert "MLS over the post-quantum X-Wing ciphersuite" in out

    @respx.mock
    async def test_create_group_arun_with_enable_mls_false_asks_the_server(self, config_dir):
        """The async twin carries ``enable_mls`` to the wire too.

        ``_run`` and ``_arun`` are two functions with two hand-written argument
        lists, and the async one has silently dropped a create keyword before —
        the reason the SDK asserts every create keyword on both halves. Asserted
        on the wire rather than on the answer, because a dropped keyword renders
        an identical confirmation.
        """
        mock_token()
        mock_agents()
        route = respx.post(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json=group_payload(name="ops", mls_enabled=False))
        )
        init = mock_group_mls_init()

        out = await RineGroupCreateTool(config_dir=config_dir, api_url=API_URL).ainvoke(
            {"name": "ops", "visibility": "private", "enable_mls": False}
        )

        assert json.loads(route.calls[0].request.content)["mls_enabled"] is False
        assert not init.called
        assert "Created group" in out


# =====================================================================
# RineGroupInviteTool / RineGroupRemoveTool
# =====================================================================


class TestRineGroupInviteTool:
    @respx.mock
    def test_invite_agent_returns_status_string(self, config_dir):
        """The control arm: a closed group still mints a voucher and says so."""
        mock_token()
        mock_agents()
        # invite resolves the group by handle, then the invitee UUID is a UUID (no webfinger).
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json={"items": [group_payload()]})
        )
        respx.post(f"{API_URL}/groups/{GROUP_ID}/invite").mock(
            return_value=httpx.Response(200, json={"status": "invited", "request_id": None})
        )

        out = RineGroupInviteTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": GROUP_HANDLE, "agent_to_invite": OTHER_AGENT_ID}
        )

        assert f"Invited {OTHER_AGENT_ID}" in out
        assert "status invited" in out
        assert "electorate" not in out

    @respx.mock
    def test_invite_into_a_voting_group_reports_a_nomination_not_an_invitation(self, config_dir):
        """On majority/unanimity the row is a join request, not a voucher.

        The server answers ``pending`` and nothing else changes on the wire, so a
        surface that prints the status under the word "Invited" would report an
        invitation in an unexplained state — which does not hold. Nobody is
        seated until the electorate carries the vote.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json={"items": [group_payload()]})
        )
        respx.post(f"{API_URL}/groups/{GROUP_ID}/invite").mock(
            return_value=httpx.Response(200, json={"status": "pending", "request_id": None})
        )

        out = RineGroupInviteTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": GROUP_HANDLE, "agent_to_invite": OTHER_AGENT_ID}
        )

        assert out.startswith(f"Nominated {OTHER_AGENT_ID}")
        assert "the group's electorate decides" in out
        assert "nobody is seated until the vote carries" in out
        assert "Invited" not in out

    @respx.mock
    async def test_invite_into_a_voting_group_reads_the_same_on_the_async_half(self, config_dir):
        """The async body is the sync body; both render through ``_admission``."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json={"items": [group_payload()]})
        )
        respx.post(f"{API_URL}/groups/{GROUP_ID}/invite").mock(
            return_value=httpx.Response(200, json={"status": "pending", "request_id": None})
        )

        out = await RineGroupInviteTool(config_dir=config_dir, api_url=API_URL).ainvoke(
            {"group": GROUP_HANDLE, "agent_to_invite": OTHER_AGENT_ID}
        )

        assert out.startswith(f"Nominated {OTHER_AGENT_ID}")
        assert "Invited" not in out

    @respx.mock
    def test_invite_a_batch_posts_agent_ids_and_reports_one_outcome_each(self, config_dir):
        """A batch drops what it cannot admit and says which one.

        The singular form is untouched — same field, same body, same sentence —
        so the plural form is a second entry point on one route, not a rewrite.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json={"items": [group_payload()]})
        )
        route = respx.post(f"{API_URL}/groups/{GROUP_ID}/invite").mock(
            return_value=httpx.Response(
                200,
                json={
                    "invited": 1,
                    "skipped": 1,
                    "entries": [
                        {"agent_id": OTHER_AGENT_ID, "status": "invited", "reason": None},
                        {
                            "agent_id": THIRD_AGENT_ID,
                            "status": "skipped",
                            "reason": "agent_revoked",
                        },
                    ],
                },
            )
        )

        out = RineGroupInviteTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": GROUP_HANDLE, "agents_to_invite": [OTHER_AGENT_ID, THIRD_AGENT_ID]}
        )

        body = json.loads(route.calls[0].request.content)
        assert body["agent_ids"] == [OTHER_AGENT_ID, THIRD_AGENT_ID]
        assert "agent_id" not in body
        assert "1 of 2 reached; 1 new invitation to accept." in out
        assert f"{THIRD_AGENT_ID}: that agent's identity is revoked" in out
        assert "agent_revoked" not in out

    @respx.mock
    async def test_invite_a_batch_reads_the_same_on_the_async_path(self, config_dir):
        """The async body renders through the same module, so it cannot diverge."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json={"items": [group_payload()]})
        )
        respx.post(f"{API_URL}/groups/{GROUP_ID}/invite").mock(
            return_value=httpx.Response(
                200,
                json={
                    "invited": 1,
                    "skipped": 0,
                    "entries": [{"agent_id": OTHER_AGENT_ID, "status": "invited", "reason": None}],
                },
            )
        )

        out = await RineGroupInviteTool(config_dir=config_dir, api_url=API_URL).ainvoke(
            {"group": GROUP_HANDLE, "agents_to_invite": [OTHER_AGENT_ID]}
        )

        assert "1 of 1 reached; 1 new invitation to accept." in out
        assert f"{OTHER_AGENT_ID}: invited" in out

    @respx.mock
    def test_invite_a_batch_into_a_voting_group_never_calls_a_nomination_an_invitation(
        self, config_dir
    ):
        """A nomination seats nobody and may be denied, so it is not one.

        The headline is the SDK's and is derived from the entries, so a surface
        that renders it inherits the distinction; the failure mode this pins is
        the entries coming back ``nominated`` while the head still counts them as
        invitations to accept.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json={"items": [group_payload()]})
        )
        respx.post(f"{API_URL}/groups/{GROUP_ID}/invite").mock(
            return_value=httpx.Response(
                200,
                json={
                    "invited": 2,
                    "skipped": 0,
                    "entries": [
                        {"agent_id": OTHER_AGENT_ID, "status": "nominated", "reason": None},
                        {"agent_id": THIRD_AGENT_ID, "status": "nominated", "reason": None},
                    ],
                },
            )
        )

        out = RineGroupInviteTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": GROUP_HANDLE, "agents_to_invite": [OTHER_AGENT_ID, THIRD_AGENT_ID]}
        )

        assert "2 of 2 reached; 2 new nominations awaiting a vote." in out
        assert f"{OTHER_AGENT_ID}: nominated, awaiting a vote" in out
        assert "invitation" not in out

    @respx.mock
    def test_invite_naming_both_targets_or_neither_never_reaches_the_wire(self, config_dir):
        """One invitation or a batch — naming both is ambiguous and is refused here."""
        mock_token()
        mock_agents()
        route = respx.post(f"{API_URL}/groups/{GROUP_ID}/invite").mock(
            return_value=httpx.Response(200, json={"status": "pending", "request_id": None})
        )
        tool = RineGroupInviteTool(config_dir=config_dir, api_url=API_URL)

        both = tool._run(
            group=GROUP_HANDLE,
            agent_to_invite=OTHER_AGENT_ID,
            agents_to_invite=[THIRD_AGENT_ID],
        )
        neither = tool._run(group=GROUP_HANDLE)

        assert not route.called
        for out in (both, neither):
            assert "Name exactly one of agent_to_invite" in out


class TestRineGroupRemoveTool:
    @respx.mock
    def test_remove_member_prints_the_sdk_sentence_for_this_removal(self, config_dir):
        """The tool delegates the sentence, it does not compose one.

        A server that accepts a bare DELETE holds no MLS state for the group, so
        this is an open-group eviction and the true sentence says what actually
        bounds the departed member's reach — each remaining member rotating on
        their next send. The one string this tool used to print claimed a
        removal that an open group has never had.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json={"items": [group_payload()]})
        )
        route = respx.delete(f"{API_URL}/groups/{GROUP_ID}/members/{OTHER_AGENT_ID}").mock(
            return_value=httpx.Response(204)
        )

        out = RineGroupRemoveTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": GROUP_HANDLE, "agent_id": OTHER_AGENT_ID}
        )

        assert route.called
        assert f"Removed {OTHER_AGENT_ID}" in out
        assert "no cryptographic eviction" in out
        assert "rotating their sender key on their next send" in out
        assert "keys rotated" not in out

    @respx.mock
    async def test_remove_member_reads_the_same_on_the_async_path(self, config_dir):
        """The async half is written twice in this package; it must not diverge."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json={"items": [group_payload()]})
        )
        respx.delete(f"{API_URL}/groups/{GROUP_ID}/members/{OTHER_AGENT_ID}").mock(
            return_value=httpx.Response(204)
        )

        out = await RineGroupRemoveTool(config_dir=config_dir, api_url=API_URL).ainvoke(
            {"group": GROUP_HANDLE, "agent_id": OTHER_AGENT_ID}
        )

        assert f"Removed {OTHER_AGENT_ID}" in out
        assert "no cryptographic eviction" in out

    def test_eviction_is_reported_as_a_membership_fact_not_a_ratchet_tree_one(
        self, config_dir, monkeypatch
    ):
        """An evicted agent gets rine's refusal, with a remedy.

        The MLS core answers every later operation on an evicted handle with
        *"this member was removed from the group at epoch 47"* — true about a
        ratchet tree, and unusable: it names no group and offers no next action.
        The SDK translates it; this asserts the tool passes that through rather
        than flattening it into a traceback or a second wording of its own.
        """
        from rine import GROUP_EVICTED_CODE, GroupEvictedError
        from rine._resources.groups import SyncGroups

        def _evicted(*_args, **_kwargs):
            raise GroupEvictedError(GROUP_ID)

        monkeypatch.setattr(SyncGroups, "remove_member", _evicted)
        out = RineGroupRemoveTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": GROUP_HANDLE, "agent_id": OTHER_AGENT_ID}
        )

        assert GROUP_ID in out
        assert "You were removed from group" in out
        assert "invite you again" in out
        assert "epoch 47" not in out
        assert GROUP_EVICTED_CODE not in out  # the code is to branch on, not to read


# =====================================================================
# RineGroupJoinTool / RineGroupInvitesTool
# =====================================================================

# Server response shapes copied field-for-field from rine-sdk's test fixtures
# (GroupMemberRead / JoinRequestRead — see rine-sdk/tests/test_groups.py).
_MEMBER_ID = "eeeeeeee-ffff-0000-1111-222222222222"
_REQUEST_ID = "ffffffff-0000-1111-2222-333333333333"

_MEMBER_PAYLOAD = {
    "id": _MEMBER_ID,
    "group_id": GROUP_ID,
    "agent_id": AGENT_ID,
    "role": "member",
    "joined_at": "2024-01-01T00:00:00Z",
    "agent_handle": "test-agent@myorg.rine.network",
}

_PENDING_REQUEST_PAYLOAD = {
    "id": _REQUEST_ID,
    "group_id": GROUP_ID,
    "agent_id": AGENT_ID,
    "invited_by": None,
    "message": "please let me in",
    "status": "pending",
    "expires_at": "2024-01-08T00:00:00Z",
    "created_at": "2024-01-01T00:00:00Z",
    "resolved_at": None,
    "your_vote": None,
}

_INVITE_PAYLOAD = {
    "id": _REQUEST_ID,
    "group_id": GROUP_ID,
    "agent_id": AGENT_ID,
    "invited_by": OTHER_AGENT_ID,
    "message": "join us",
    "status": "invited",
    "created_at": "2024-01-01T00:00:00Z",
    "group_handle": GROUP_HANDLE,
    "group_name": "ops",
}


def _directory_row(*, handle: str = GROUP_HANDLE, group_id: str = GROUP_ID) -> dict[str, object]:
    """A ``/directory/groups`` hit — the one rung that sees a group this org is not in.

    Local rather than shared with ``test_groups_list_tools``: what those tests
    need from a row is how it RENDERS, and what these need is only its handle
    and id, which is what the join ladder reads off it.
    """
    return {
        "id": group_id,
        "name": handle.lstrip("#").split("@")[0],
        "handle": handle,
        "description": "Operations coordination.",
        "enrollment_policy": "open",
        "member_count": 4,
        "created_at": "2024-01-01T00:00:00Z",
    }


class TestRineGroupJoinTool:
    @respx.mock
    def test_join_open_group_returns_joined_string(self, config_dir):
        mock_token()
        mock_agents()
        route = respx.post(f"{API_URL}/groups/{GROUP_ID}/join").mock(
            return_value=httpx.Response(200, json=_MEMBER_PAYLOAD)
        )

        out = RineGroupJoinTool(config_dir=config_dir, api_url=API_URL).invoke({"group": GROUP_ID})

        assert route.called
        assert f"Joined {GROUP_ID}" in out
        assert AGENT_ID in out

    @respx.mock
    def test_join_gated_group_returns_pending_string(self, config_dir):
        mock_token()
        mock_agents()
        respx.post(f"{API_URL}/groups/{GROUP_ID}/join").mock(
            return_value=httpx.Response(202, json=_PENDING_REQUEST_PAYLOAD)
        )

        out = RineGroupJoinTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": GROUP_ID, "message": "please let me in"}
        )

        assert f"Requested to join {GROUP_ID}" in out
        assert _REQUEST_ID in out
        assert "awaiting a member vote" in out

    @respx.mock
    def test_join_on_a_nomination_reports_that_it_recorded_consent(self, config_dir):
        """Joining a nomination is a consent step, not an application.

        Both answers are ``pending`` and ``invited_by`` is what tells them apart.
        Reported as an application, an operator cannot tell whether its agent
        asked for this or somebody proposed it — and the sentence that matters,
        that a member nominated it and the vote still decides, goes unsaid.
        """
        mock_token()
        mock_agents()
        respx.post(f"{API_URL}/groups/{GROUP_ID}/join").mock(
            return_value=httpx.Response(
                202, json={**_PENDING_REQUEST_PAYLOAD, "invited_by": OTHER_AGENT_ID}
            )
        )

        out = RineGroupJoinTool(config_dir=config_dir, api_url=API_URL).invoke({"group": GROUP_ID})

        assert "Consent recorded for the nomination" in out
        assert "electorate still has to carry the vote" in out
        assert "Requested to join" not in out

    @respx.mock
    def test_join_unknown_group_returns_clear_string(self, config_dir):
        mock_token()
        mock_agents()
        respx.post(f"{API_URL}/groups/{GROUP_ID}/join").mock(
            return_value=httpx.Response(404, json={"detail": "group not found"})
        )

        out = RineGroupJoinTool(config_dir=config_dir, api_url=API_URL).invoke({"group": GROUP_ID})

        assert "group not found" in out

    @respx.mock
    def test_join_a_handle_the_directory_prints_resolves_without_a_uuid(self, config_dir):
        """The reference ``rine_discover_groups`` prints, spent as it was printed.

        MEASURED on a sibling surface: a model called join with a handle it had
        just been shown and was refused CLIENT-SIDE, because resolution ran over
        the groups this org is already seated in — the one set a join cannot be
        aimed at. The directory is the rung that sees a group this org has no
        relationship with, and it is the last one tried.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(return_value=httpx.Response(200, json={"items": []}))
        respx.get(f"{API_URL}/groups/invites").mock(
            return_value=httpx.Response(200, json={"items": []})
        )
        respx.get(f"{API_URL}/groups/org-joinable").mock(
            return_value=httpx.Response(200, json={"id": None})
        )
        respx.get(f"{API_URL}/directory/groups").mock(
            return_value=httpx.Response(
                200,
                json={"items": [_directory_row()], "total_estimate": 1, "has_more": False},
            )
        )
        route = respx.post(f"{API_URL}/groups/{GROUP_ID}/join").mock(
            return_value=httpx.Response(200, json=_MEMBER_PAYLOAD)
        )

        out = RineGroupJoinTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": GROUP_HANDLE}
        )

        assert route.called
        assert "Joined" in out

    @respx.mock
    def test_join_a_directory_near_miss_is_rejected_rather_than_joined(self, config_dir):
        """The directory filter is a substring match, so its rows are candidates only.

        ``?q=ops`` returns every group whose name or description CONTAINS the
        string — ``#ops-archive@acme`` among them. Exact normalized-handle
        equality over those rows is the whole reason a fuzzy search is safe in
        front of a verb that cannot be taken back; without it this call would
        seat the agent in a group nobody named.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(return_value=httpx.Response(200, json={"items": []}))
        respx.get(f"{API_URL}/groups/invites").mock(
            return_value=httpx.Response(200, json={"items": []})
        )
        respx.get(f"{API_URL}/groups/org-joinable").mock(
            return_value=httpx.Response(200, json={"id": None})
        )
        respx.get(f"{API_URL}/directory/groups").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [
                        _directory_row(
                            handle="#ops-archive@acme.rine.network",
                            group_id=OTHER_AGENT_ID,
                        )
                    ],
                    "total_estimate": 1,
                    "has_more": False,
                },
            )
        )
        join = respx.post(f"{API_URL}/groups/{OTHER_AGENT_ID}/join")

        out = RineGroupJoinTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": GROUP_HANDLE}
        )

        assert not join.called
        assert "Not found" in out
        assert "a public group in the directory" in out

    @respx.mock
    def test_join_a_bare_name_an_invitation_names_spends_that_invitation(self, config_dir):
        """Every other group verb here takes a bare name, and so does this one — once.

        A bare name reaches exactly one set: the invitations already addressed
        to this agent. That set is bounded to groups that have asked for it, so
        a match in it cannot be a group the caller has no business in, which is
        what makes a reference carrying no org safe in front of a verb that
        cannot be taken back.
        """
        mock_token()
        mock_agents()
        seats = respx.get(f"{API_URL}/groups")
        directory = respx.get(f"{API_URL}/directory/groups")
        respx.get(f"{API_URL}/groups/invites").mock(
            return_value=httpx.Response(200, json={"items": [_INVITE_PAYLOAD]})
        )
        respx.post(f"{API_URL}/groups/{GROUP_ID}/join").mock(
            return_value=httpx.Response(200, json=_MEMBER_PAYLOAD)
        )

        out = RineGroupJoinTool(config_dir=config_dir, api_url=API_URL).invoke({"group": "ops"})

        assert "Joined" in out
        # The two rungs a bare name must not reach: this org's seats say nothing
        # about a group it is not in, and the directory is a network-wide
        # substring match with no org to tell two `ops` groups apart.
        assert not seats.called
        assert not directory.called

    @respx.mock
    def test_join_a_bare_name_no_invitation_names_is_refused_without_widening(self, config_dir):
        """A bare name nothing invited this agent to stops here, and says why.

        Deliberate and settled: the only rung that could see a
        group outside this agent's relationships is a network-wide substring
        match, and a wrong match on an irreversible verb is unrecoverable. The
        refusal says what the reference WAS read as, because every other group
        verb here taught the caller to write it.
        """
        mock_token()
        mock_agents()
        seats = respx.get(f"{API_URL}/groups")
        directory = respx.get(f"{API_URL}/directory/groups")
        respx.get(f"{API_URL}/groups/invites").mock(
            return_value=httpx.Response(200, json={"items": []})
        )

        out = RineGroupJoinTool(config_dir=config_dir, api_url=API_URL).invoke({"group": "ops"})

        assert "Not found" in out
        assert "A bare name is read as one of this agent's pending invitations" in out
        assert "Join takes a group's whole handle" in out
        assert not seats.called
        assert not directory.called

    @respx.mock
    async def test_join_arun_matches_run(self, config_dir):
        mock_token()
        mock_agents()
        respx.post(f"{API_URL}/groups/{GROUP_ID}/join").mock(
            return_value=httpx.Response(200, json=_MEMBER_PAYLOAD)
        )

        out = await RineGroupJoinTool(config_dir=config_dir, api_url=API_URL).ainvoke(
            {"group": GROUP_ID}
        )

        assert f"Joined {GROUP_ID}" in out


class TestRineGroupInvitesTool:
    @respx.mock
    def test_invites_lists_pending_invite(self, config_dir):
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups/invites").mock(
            return_value=httpx.Response(200, json={"items": [_INVITE_PAYLOAD]})
        )

        out = RineGroupInvitesTool(config_dir=config_dir, api_url=API_URL).invoke({})

        assert GROUP_HANDLE in out
        assert _REQUEST_ID in out
        assert "invited" in out
        assert "an invitation to spend" in out

    @respx.mock
    def test_invites_when_a_member_nominated_this_agent_then_the_row_names_the_consent_step(
        self, config_dir
    ):
        """This route is the only place a nominee sees its own nomination.

        A nomination is ``pending`` and the agent it names is not yet a member,
        so no other route reports it. Its own join call is the consent step, so a
        row that renders the bare token under the head "Pending invites" reads as
        an ordinary invitation and leaves consent unreachable.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups/invites").mock(
            return_value=httpx.Response(
                200, json={"items": [{**_INVITE_PAYLOAD, "status": "pending"}]}
            )
        )

        out = RineGroupInvitesTool(config_dir=config_dir, api_url=API_URL).invoke({})

        assert "status pending" in out
        assert "a nomination to consent to" in out
        assert "the electorate decides it" in out

    @respx.mock
    def test_invites_empty_returns_no_pending_string(self, config_dir):
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups/invites").mock(
            return_value=httpx.Response(200, json={"items": []})
        )

        out = RineGroupInvitesTool(config_dir=config_dir, api_url=API_URL).invoke({})

        assert out == "No pending group invites."

    @respx.mock
    async def test_invites_arun_matches_run(self, config_dir):
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups/invites").mock(
            return_value=httpx.Response(200, json={"items": [_INVITE_PAYLOAD]})
        )

        out = await RineGroupInvitesTool(config_dir=config_dir, api_url=API_URL).ainvoke({})

        assert GROUP_HANDLE in out


# =====================================================================
# RineGroupInspectTool — sender-key vs MLS self-diagnosis
# =====================================================================


class TestRineGroupInspectTool:
    @respx.mock
    def test_inspect_sender_key_group_reports_readable(self, config_dir):
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(
                200, json={"items": [group_payload(mls_enabled=False, mls_group_id=None)]}
            )
        )

        out = RineGroupInspectTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": GROUP_HANDLE}
        )

        assert "[OK] sender-key group" in out
        assert "readable and postable" in out
        assert GROUP_HANDLE in out

    @respx.mock
    def test_inspect_mls_group_reports_post_quantum_readable(self, config_dir):
        """An MLS group reports post-quantum encryption — readable and postable from here."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(
                200, json={"items": [group_payload(mls_enabled=True, mls_group_id="mls-xyz")]}
            )
        )

        out = RineGroupInspectTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": GROUP_HANDLE}
        )

        assert "[OK] post-quantum MLS group" in out
        assert "readable and postable" in out

    @respx.mock
    def test_inspect_group_when_mls_init_in_flight_reports_its_own_state(self, config_dir):
        """The init window gets its own render — neither of the other two.

        ``mls_pending`` is an init in flight, not state: nothing is decryptable
        from an MLS epoch until ``mls_group_id`` latches, so the render question
        (``rine.format.group_is_mls``) stays false. But the server already
        refuses a sender-key send here and this surface's sends take the MLS
        branch, so calling the window "sender-key" says the opposite of what is
        happening. It is a normal transient state, and it is reported as one.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [group_payload(mls_enabled=True, mls_group_id=None, mls_pending=True)]
                },
            )
        )

        out = RineGroupInspectTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": GROUP_HANDLE}
        )

        assert "[OK] MLS group, initialising" in out
        assert "Sends from here already use MLS." in out
        assert "sender-key" not in out

    @respx.mock
    def test_inspect_open_enrollment_group_reports_sender_key(self, config_dir):
        """The case the init window must never swallow.

        An open-enrollment group carries ``mls_enabled`` as intent while running
        on sender keys by design, and no init is or will be in flight for it.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(
                200,
                json={"items": [{**group_payload(mls_enabled=True), "enrollment_policy": "open"}]},
            )
        )

        out = RineGroupInspectTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": GROUP_HANDLE}
        )

        assert "[OK] sender-key group" in out
        assert "MLS group" not in out

    @respx.mock
    def test_inspect_group_created_for_mls_and_never_founded_names_this_agents_verb(
        self, config_dir
    ):
        """The fourth state: created to run MLS, ratchet tree never founded.

        Closed enrollment, ``mls_enabled`` true, no latch and no init in flight
        — field for field what an open group looks like, which is why this used
        to render as an ordinary sender-key group and the agent was never told
        the group had not got the encryption it was created for. The group is
        not frozen: the server takes the sender-key branch for it and a post
        into it is accepted, so the marker stays ``[OK]``.

        The remedy this agent can actually reach is ``rine_group_reclaim``;
        ``resume_admission`` is not a tool it has. This is the ONE state that
        carries a remedy, so it is also the only place the "names its own verb,
        never nobody's" rule can be asserted — the create path cannot reach it.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [
                        group_payload(mls_enabled=True, mls_group_id=None, mls_pending=False)
                    ]
                },
            )
        )

        out = RineGroupInspectTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": GROUP_HANDLE}
        )

        assert "[OK] sender-key group" in out
        assert "its ratchet tree was never founded" in out
        assert "Run rine_group_reclaim on it" in out
        assert "No verb on this surface founds it" not in out
        assert not re.search(r"broken|frozen|failed|cannot carry|holds no keys", out, re.I)

    @respx.mock
    def test_inspect_never_founded_group_extends_the_ordinary_sender_key_line(self, config_dir):
        """The fourth arm APPENDS to the third's sentence — it does not replace it.

        Two renders of the same tool, one ordinary sender-key group and one
        never-founded, asserted against each other: the never-founded diagnosis
        must begin with the ordinary one, so the two can never drift into
        describing the same traffic in two voices. Nothing in the source binds
        them, so this is the binding.
        """
        mock_token()
        mock_agents()
        tool = RineGroupInspectTool(config_dir=config_dir, api_url=API_URL)
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(
                200,
                json={"items": [{**group_payload(mls_enabled=True), "enrollment_policy": "open"}]},
            )
        )
        ordinary = tool.invoke({"group": GROUP_HANDLE}).splitlines()[-1]

        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(
                200, json={"items": [group_payload(mls_enabled=True, mls_group_id=None)]}
            )
        )
        never_founded = tool.invoke({"group": GROUP_HANDLE}).splitlines()[-1]

        assert never_founded.startswith(f"{ordinary} ")
        assert never_founded != ordinary

    @respx.mock
    def test_inspect_reports_a_distinct_line_for_each_of_the_four_states(self, config_dir):
        """The four renders are distinguishable, not three renders and an alias.

        A test that only proves the new branch fires cannot see the
        open-enrollment case being swallowed into it — and the never-founded row
        differs from the open one in ``enrollment_policy`` alone, so this is the
        only assertion here that would notice the two collapsing together.
        """
        mock_token()
        mock_agents()
        payloads = [
            group_payload(mls_enabled=True, mls_group_id="mls-xyz"),
            group_payload(mls_enabled=True, mls_group_id=None, mls_pending=True),
            group_payload(mls_enabled=True, mls_group_id=None, mls_pending=False),
            {**group_payload(mls_enabled=True), "enrollment_policy": "open"},
        ]
        renders = set()
        for payload in payloads:
            respx.get(f"{API_URL}/groups").mock(
                return_value=httpx.Response(200, json={"items": [payload]})
            )
            renders.add(
                RineGroupInspectTool(config_dir=config_dir, api_url=API_URL).invoke(
                    {"group": GROUP_HANDLE}
                )
            )

        assert len(renders) == 4

    @respx.mock
    def test_inspect_accepts_bare_group_name(self, config_dir):
        """M3: a bare group name ('ops') resolves on the handle's local part.

        Parity with invite/remove, which already accept a bare name. The cross-tool
        contradiction (inspect rejecting what siblings accept) would otherwise derail
        an LLM mid-task.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(
                200, json={"items": [group_payload(name="ops", handle=GROUP_HANDLE)]}
            )
        )

        out = RineGroupInspectTool(config_dir=config_dir, api_url=API_URL).invoke({"group": "ops"})

        # The bare name matched #ops@... — we get the group's diagnosis, not Not found.
        assert "Not found" not in out
        assert GROUP_HANDLE in out
        assert "sender-key group" in out

    @respx.mock
    def test_inspect_ambiguous_bare_name_refuses_and_names_both_groups(self, config_dir):
        """🔴 An ambiguous bare name is refused, and every candidate is named.

        Two groups answering to ``ops`` used to be resolved by taking whichever
        row ``GET /groups`` returned first — and that listing carries no
        ``ORDER BY``, so two identical calls could inspect two different groups
        and neither answer said so. The resolver refuses and names every
        candidate, so the agent picks with the spellings in front of it. A
        refusal where there used to be an answer is the trade.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [
                        group_payload(name="ops", handle=GROUP_HANDLE),
                        group_payload(
                            name="ops",
                            handle="#ops@beta.rine.network",
                            group_id="dddddddd-eeee-ffff-0000-111111111111",
                        ),
                    ]
                },
            )
        )

        out = RineGroupInspectTool(config_dir=config_dir, api_url=API_URL).invoke({"group": "ops"})

        assert GROUP_HANDLE in out
        assert "#ops@beta.rine.network" in out
        # A refusal, not a rendered group: no diagnosis line is printed.
        assert "sender-key group" not in out

    @respx.mock
    async def test_inspect_ambiguous_bare_name_refuses_on_the_async_half_too(self, config_dir):
        """The refusal is the resolver's, so both arms must spend it identically.

        The two bodies are hand-kept twins here — the async one is the sync one
        with ``await`` — so an ambiguity handled on one arm and answered on the
        other is a defect no assertion on the sync half can see.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [
                        group_payload(name="ops", handle=GROUP_HANDLE),
                        group_payload(
                            name="ops",
                            handle="#ops@beta.rine.network",
                            group_id="dddddddd-eeee-ffff-0000-111111111111",
                        ),
                    ]
                },
            )
        )

        out = await RineGroupInspectTool(config_dir=config_dir, api_url=API_URL).ainvoke(
            {"group": "ops"}
        )

        assert GROUP_HANDLE in out
        assert "#ops@beta.rine.network" in out
        assert "sender-key group" not in out

    @respx.mock
    def test_inspect_reports_the_groups_conversation_id(self, config_dir):
        """The detail block names the argument ``rine_thread`` needs.

        ``rine_groups`` carries the same pointer on its rows, so an agent never
        has to inspect a group just to learn how to read it — but the group's
        own detail block is where a model looks when it has already picked one.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(
                200,
                json={"items": [group_payload(conversation_id=GROUP_CONVERSATION_ID)]},
            )
        )

        out = RineGroupInspectTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": GROUP_HANDLE}
        )

        assert f"conversation_id: {GROUP_CONVERSATION_ID}" in out

    @respx.mock
    def test_inspect_when_nothing_was_said_in_the_group_says_so_instead_of_none(self, config_dir):
        """A group with no conversation is a healthy group, and reads as one.

        The conversation is created on the group's first post. Printing ``None``
        here would read as a broken tool; the sentence says there is nothing to
        read yet, which is a thing the model can act on.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json={"items": [group_payload(conversation_id=None)]})
        )

        out = RineGroupInspectTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": GROUP_HANDLE}
        )

        assert "conversation_id: (none — nothing has been said in this group yet)" in out
        assert "conversation_id: None" not in out

    @respx.mock
    def test_inspect_names_this_orgs_seated_agents_and_keeps_the_whole_member_count(
        self, config_dir
    ):
        """The detail block answers both questions, and does not confuse them.

        ``your_agents`` is which of this org's own agents may act here;
        ``members`` is still how big the group is. The first is additive — it
        replaces nothing — so a block that reports one seat and five members is
        reporting two facts, not contradicting itself.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(
                200,
                json={"items": [group_payload(member_count=5, member_agent_ids=[AGENT_ID])]},
            )
        )

        out = RineGroupInspectTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": GROUP_HANDLE}
        )

        assert "your_agents: test-agent@myorg.rine.network" in out
        assert AGENT_ID not in out
        assert "members: 5" in out

    @respx.mock
    def test_inspect_when_no_agent_of_this_org_is_seated_says_so_instead_of_none(self, config_dir):
        """A group the org is in and holds no seat in reads as exactly that.

        It is a state the agent has to be able to act on — ask a member for an
        invitation — and it is unreachable from ``members``, which counts every
        org's seats together.
        """
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json={"items": [group_payload(member_agent_ids=[])]})
        )

        out = RineGroupInspectTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": GROUP_HANDLE}
        )

        assert "your_agents: (none — no agent of your org is seated in this group)" in out
        assert "your_agents: None" not in out
        assert "your_agents: []" not in out

    @respx.mock
    def test_inspect_unknown_group_returns_not_found_string(self, config_dir):
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(return_value=httpx.Response(200, json={"items": []}))

        out = RineGroupInspectTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": "#missing@acme.rine.network"}
        )

        assert "Not found" in out


# =====================================================================
# Group-aware send / read via #-prefix routing (no separate group tools)
# =====================================================================


class TestGroupRoutingThroughMessagingTools:
    @respx.mock
    def test_send_to_hash_handle_routes_through_group_path(self, config_dir):
        """A '#'-prefixed `to` triggers the SDK sender-key group path, not 1:1 DM."""
        mock_token()
        mock_agents()
        # The group path resolves the group by handle (with members) — not WebFinger.
        # `member_agent_ids` is this org's own seated agents, and it must agree
        # with the roster mocked below: the server derives both from the same seat rows
        # (`own_seated_agent_ids` intersects the roster with `Agent.org_id`), so a group
        # whose members include AGENT_ID cannot also report an empty own-org seat list.
        # Leaving it at the fixture's empty default states a server shape that cannot
        # occur, and the SDK's pre-fan-out seat guard reads it and refuses the send.
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(
                200, json={"items": [group_payload(member_agent_ids=[AGENT_ID])]}
            )
        )
        respx.get(f"{API_URL}/groups/{GROUP_ID}/members").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "group_id": GROUP_ID,
                            "agent_id": AGENT_ID,
                            "joined_at": "2024-01-01T00:00:00Z",
                        }
                    ]
                },
            )
        )
        send_route = respx.post(f"{API_URL}/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    **message_payload(
                        encryption_version=EncryptionVersion.SENDER_KEY_V1,
                        encrypted_payload="ZHVtbXk=",
                    ),
                    "group_id": GROUP_ID,
                    "to_agent_id": None,
                },
            )
        )

        out = RineSendTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"to": GROUP_HANDLE, "body": "team update"}
        )

        # group body carries to_handle (the group), not a to_agent_id DM.
        body = json.loads(send_route.calls[0].request.content)
        assert body.get("to_handle") == GROUP_HANDLE
        assert body["encryption_version"] == EncryptionVersion.SENDER_KEY_V1
        assert f"Sent message {MESSAGE_ID}" in out

    @respx.mock
    def test_inbox_renders_group_message(self, config_dir):
        """A sender-key group message read via the inbox shows its group, no ciphertext."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/agents/{AGENT_ID}/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [
                        {
                            **message_payload(
                                encryption_version=EncryptionVersion.SENDER_KEY_V1,
                                encrypted_payload="R1JPVVBDVA==",
                                group_id=GROUP_ID,
                            ),
                            "group_handle": GROUP_HANDLE,
                        }
                    ],
                    "total": 1,
                },
            )
        )
        respx.post(f"{API_URL}/agents/{AGENT_ID}/messages/mark-delivered").mock(
            return_value=httpx.Response(200, json={"marked": 0})
        )

        out = RineInboxTool(config_dir=config_dir, api_url=API_URL).invoke({})

        # We have no sender-key state for this group → SDK sets a decrypt_error
        # (recovery is best-effort); the key point: the message surfaces with its
        # group, and the raw ciphertext never leaks.
        assert "R1JPVVBDVA==" not in out
        assert GROUP_HANDLE in out or str(GROUP_ID) in out


# =====================================================================
# The SDK's ladder, the resolve's own row, and both halves
# =====================================================================


class TestGroupInspectResolvesThroughTheSdk:
    """``rine_group_inspect`` resolves through the SDK's ladder.

    The package's own UUID/handle/bare-name matcher over ``groups.list()`` is
    gone; ``resolve_group`` is the one ladder, and the row it already read is
    what gets rendered. Both ``_run`` and ``_arun`` are driven at every claim —
    a sync/async pair where only one half moved reads green under a suite that
    drives only one.
    """

    @respx.mock
    def test_inspect_by_uuid_reads_the_group_exactly_once(self, config_dir):
        """The resolve's row is rendered, so no second ``GET /groups/{id}`` is paid."""
        mock_token()
        mock_agents()
        route = respx.get(f"{API_URL}/groups/{GROUP_ID}").mock(
            return_value=httpx.Response(200, json=group_payload(member_agent_ids=[AGENT_ID]))
        )

        out = RineGroupInspectTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": GROUP_ID}
        )

        assert route.call_count == 1
        assert GROUP_HANDLE in out
        assert "your_agents: test-agent@myorg.rine.network" in out

    @respx.mock
    async def test_inspect_by_uuid_reads_the_group_exactly_once_on_the_async_half(
        self, config_dir
    ):
        mock_token()
        mock_agents()
        route = respx.get(f"{API_URL}/groups/{GROUP_ID}").mock(
            return_value=httpx.Response(200, json=group_payload(member_agent_ids=[AGENT_ID]))
        )

        out = await RineGroupInspectTool(config_dir=config_dir, api_url=API_URL).ainvoke(
            {"group": GROUP_ID}
        )

        assert route.call_count == 1
        assert GROUP_HANDLE in out
        assert "your_agents: test-agent@myorg.rine.network" in out

    @respx.mock
    async def test_inspect_bare_name_resolves_on_the_async_half_too(self, config_dir):
        """M3 parity survives the migration on BOTH halves, not just the sync one."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(
                200, json={"items": [group_payload(name="ops", handle=GROUP_HANDLE)]}
            )
        )

        out = await RineGroupInspectTool(config_dir=config_dir, api_url=API_URL).ainvoke(
            {"group": "ops"}
        )

        assert "Not found" not in out
        assert GROUP_HANDLE in out

    @respx.mock
    def test_inspect_when_the_resolve_carries_no_row_refuses_instead_of_rereading(
        self, config_dir, monkeypatch
    ):
        """The seated ladder always carries the row; a ``None`` is refused as a contract break.

        ``GroupTarget.row`` is nullable for the JOIN ladder alone. If a seated
        resolve ever answered without one, the tool must say so rather than
        quietly paying a second ``GET /groups/{id}`` — so no group read is mocked
        here and the refusal must arrive anyway.
        """
        mock_token()
        mock_agents()
        monkeypatch.setattr(
            SyncRineClient,
            "resolve_group",
            lambda self, reference, agent=None: GroupTarget(
                id=GROUP_ID, handle=None, spelling=reference, row=None
            ),
        )

        out = RineGroupInspectTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": GROUP_HANDLE}
        )

        assert out.startswith(f"Not found: Group not found: {GROUP_HANDLE}")


class TestGroupSendLineNamesBothSpellings:
    """The group arm: the bare name the model typed, and the group it hit."""

    @respx.mock
    def test_send_to_a_bare_group_name_names_the_resolved_handle_too(self, config_dir):
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(
                200, json={"items": [group_payload(member_agent_ids=[AGENT_ID])]}
            )
        )
        respx.get(f"{API_URL}/groups/{GROUP_ID}/members").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "group_id": GROUP_ID,
                            "agent_id": AGENT_ID,
                            "joined_at": "2024-01-01T00:00:00Z",
                        }
                    ]
                },
            )
        )
        respx.post(f"{API_URL}/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    **message_payload(
                        encryption_version=EncryptionVersion.SENDER_KEY_V1,
                        encrypted_payload="ZHVtbXk=",
                    ),
                    "group_id": GROUP_ID,
                    "group_handle": GROUP_HANDLE,
                    "to_agent_id": None,
                },
            )
        )

        out = RineSendTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"to": "ops", "body": "team update"}
        )

        assert f"Sent message {MESSAGE_ID} to ops → {GROUP_HANDLE}" in out


@pytest.mark.parametrize("tool", [RineGroupCreateTool, RineGroupJoinTool])
def test_group_entry_tools_tell_the_agent_every_member_org_can_read_the_group(tool):
    """The tools that put an agent into a group say who can read it."""
    description = tool.model_fields["description"].default
    assert "Every member organisation can read this group" in description
