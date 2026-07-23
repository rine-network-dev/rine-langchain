"""Behavioral tests for the group tools.

Covers create (post-quantum MLS by default, single-call with description, no
mls_enabled field), invite, remove, and inspect's encryption-type line. The
``#``-prefix group send/read routing is asserted via RineSendTool against a
sender-key group (verifies group send rides the messaging tool, not a separate
group-send tool) and via RineCheckInboxTool returning a group message.

Tools are driven through their public LangChain interface: ``.invoke({...})``
(sync) and ``.ainvoke({...})`` (async). respx intercepts httpx regardless
of sync/async transport, so the same mock backs both paths.
"""

from __future__ import annotations

import json

import httpx
import respx
from rine._constants import EncryptionVersion

from langchain_rine import (
    RineCheckInboxTool,
    RineGroupCreateTool,
    RineGroupInspectTool,
    RineGroupInvitesTool,
    RineGroupInviteTool,
    RineGroupJoinTool,
    RineGroupRemoveTool,
    RineSendTool,
)
from langchain_rine.tools.groups import RineGroupCreateInput

from ..conftest import (
    AGENT_ID,
    API_URL,
    GROUP_HANDLE,
    GROUP_ID,
    MESSAGE_ID,
    OTHER_AGENT_ID,
    group_payload,
    message_payload,
    mock_agents,
    mock_token,
)

# =====================================================================
# RineGroupCreateTool
# =====================================================================


class TestRineGroupCreateTool:
    def test_create_input_schema_has_no_mls_enabled_field(self):
        """The toggle is hidden by design (the SDK picks the encryption)."""
        fields = set(RineGroupCreateInput.model_fields)
        assert "mls_enabled" not in fields
        assert fields == {"name", "enrollment", "visibility", "description"}

    @respx.mock
    def test_create_group_returns_mls_confirmation(self, config_dir):
        mock_token()
        mock_agents()
        respx.post(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json=group_payload(name="ops", mls_enabled=True))
        )
        respx.post(f"{API_URL}/groups/{GROUP_ID}/mls/init").mock(
            return_value=httpx.Response(200, json={"members": [{"agent_id": AGENT_ID}]})
        )

        out = RineGroupCreateTool(config_dir=config_dir, api_url=API_URL).invoke({"name": "ops"})

        assert "Created group" in out
        assert GROUP_HANDLE in out
        assert "post-quantum MLS E2EE" in out

    @respx.mock
    def test_create_group_enables_mls_on_the_wire(self, config_dir):
        """New groups default to post-quantum MLS: the create body carries mls_enabled=True."""
        mock_token()
        mock_agents()
        route = respx.post(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json=group_payload(name="ops", mls_enabled=True))
        )
        respx.post(f"{API_URL}/groups/{GROUP_ID}/mls/init").mock(
            return_value=httpx.Response(200, json={"members": [{"agent_id": AGENT_ID}]})
        )

        RineGroupCreateTool(config_dir=config_dir, api_url=API_URL).invoke({"name": "ops"})

        body = json.loads(route.calls[0].request.content)
        assert body["mls_enabled"] is True

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
            {"name": "ops", "description": "war room"}
        )

        body = json.loads(create.calls[0].request.content)
        assert body["description"] == "war room"
        assert not update.called  # single-call, no PATCH workaround
        assert "Created group" in out

    @respx.mock
    def test_create_group_name_conflict_returns_clear_string(self, config_dir):
        mock_token()
        mock_agents()
        respx.post(f"{API_URL}/groups").mock(
            return_value=httpx.Response(409, json={"detail": "group name taken"})
        )

        out = RineGroupCreateTool(config_dir=config_dir, api_url=API_URL).invoke({"name": "ops"})

        assert "group name taken" in out

    @respx.mock
    async def test_create_group_arun_matches_run(self, config_dir):
        """``.ainvoke`` returns the same sender-key confirmation as ``.invoke``."""
        mock_token()
        mock_agents()
        respx.post(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json=group_payload(name="ops"))
        )

        out = await RineGroupCreateTool(config_dir=config_dir, api_url=API_URL).ainvoke(
            {"name": "ops"}
        )

        assert "Created group" in out
        assert GROUP_HANDLE in out
        assert "sender-key E2EE" in out


# =====================================================================
# RineGroupInviteTool / RineGroupRemoveTool
# =====================================================================


class TestRineGroupInviteTool:
    @respx.mock
    def test_invite_agent_returns_status_string(self, config_dir):
        mock_token()
        mock_agents()
        # invite resolves the group by handle, then the invitee UUID is a UUID (no webfinger).
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json={"items": [group_payload()]})
        )
        respx.post(f"{API_URL}/groups/{GROUP_ID}/invite").mock(
            return_value=httpx.Response(200, json={"status": "pending", "request_id": None})
        )

        out = RineGroupInviteTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": GROUP_HANDLE, "agent_to_invite": OTHER_AGENT_ID}
        )

        assert f"Invited {OTHER_AGENT_ID}" in out
        assert "status pending" in out


class TestRineGroupRemoveTool:
    @respx.mock
    def test_remove_member_returns_rotation_string(self, config_dir):
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
        assert "keys rotated" in out


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
    "pending_mls_join": False,
    "pending_mls_remove": False,
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
    def test_join_unknown_group_returns_clear_string(self, config_dir):
        mock_token()
        mock_agents()
        respx.post(f"{API_URL}/groups/{GROUP_ID}/join").mock(
            return_value=httpx.Response(404, json={"detail": "group not found"})
        )

        out = RineGroupJoinTool(config_dir=config_dir, api_url=API_URL).invoke({"group": GROUP_ID})

        assert "group not found" in out

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
    def test_inspect_mls_pending_group_reports_post_quantum(self, config_dir):
        """The brief mls_pending init window also latches as MLS."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [
                        group_payload(mls_enabled=False, mls_group_id=None, mls_pending=True)
                    ]
                },
            )
        )

        out = RineGroupInspectTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"group": GROUP_HANDLE}
        )

        assert "[OK] post-quantum MLS group" in out

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
        respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json={"items": [group_payload()]})
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
    def test_check_inbox_renders_group_message(self, config_dir):
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

        out = RineCheckInboxTool(config_dir=config_dir, api_url=API_URL).invoke({})

        # We have no sender-key state for this group → SDK sets a decrypt_error
        # (recovery is best-effort); the key point: the message surfaces with its
        # group, and the raw ciphertext never leaks.
        assert "R1JPVVBDVA==" not in out
        assert GROUP_HANDLE in out or str(GROUP_ID) in out
