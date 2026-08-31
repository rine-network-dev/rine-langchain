"""``langchain-tests`` conformance for the four group tools.

create / invite / remove / inspect. One ``ToolsUnitTests`` + one ``ToolsIntegrationTests``
subclass each; the integration suite invokes against respx-mocked ``/groups`` routes. invite,
remove and inspect first resolve the group via ``GET /groups``. Every tool returns ``str``.
"""

from __future__ import annotations

import httpx
import respx
from langchain_tests.integration_tests import ToolsIntegrationTests
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_rine import (
    RineGroupCreateTool,
    RineGroupInspectTool,
    RineGroupInviteTool,
    RineGroupRemoveTool,
)
from tests.conftest import (
    API_URL,
    GROUP_HANDLE,
    GROUP_ID,
    OTHER_AGENT_ID,
    group_payload,
    mock_agents,
    mock_token,
)
from tests.integration_tests._conformance import RineToolConformance


def _mock_groups_list() -> None:
    respx.get(f"{API_URL}/groups").mock(
        return_value=httpx.Response(200, json={"items": [group_payload()]})
    )


class _GroupCreateSpec(RineToolConformance):
    tool_cls = RineGroupCreateTool
    # `visibility` is required with no default: a create that omits it
    # is refused before anything reaches the wire, so the conformance invocation
    # has to state one like every other caller now does.
    invoke_args = {"name": "ops", "visibility": "private"}

    def install_mocks(self) -> None:
        mock_token()
        mock_agents()
        respx.post(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json=group_payload(name="ops"))
        )


class TestRineGroupCreateUnit(_GroupCreateSpec, ToolsUnitTests):
    pass


class TestRineGroupCreateIntegration(_GroupCreateSpec, ToolsIntegrationTests):
    pass


class _GroupInviteSpec(RineToolConformance):
    tool_cls = RineGroupInviteTool
    invoke_args = {"group": GROUP_HANDLE, "agent_to_invite": OTHER_AGENT_ID}

    def install_mocks(self) -> None:
        mock_token()
        mock_agents()
        _mock_groups_list()
        respx.post(f"{API_URL}/groups/{GROUP_ID}/invite").mock(
            return_value=httpx.Response(200, json={"status": "pending", "request_id": None})
        )


class TestRineGroupInviteUnit(_GroupInviteSpec, ToolsUnitTests):
    pass


class TestRineGroupInviteIntegration(_GroupInviteSpec, ToolsIntegrationTests):
    pass


class _GroupRemoveSpec(RineToolConformance):
    tool_cls = RineGroupRemoveTool
    invoke_args = {"group": GROUP_HANDLE, "agent_id": OTHER_AGENT_ID}

    def install_mocks(self) -> None:
        mock_token()
        mock_agents()
        _mock_groups_list()
        respx.delete(f"{API_URL}/groups/{GROUP_ID}/members/{OTHER_AGENT_ID}").mock(
            return_value=httpx.Response(204)
        )


class TestRineGroupRemoveUnit(_GroupRemoveSpec, ToolsUnitTests):
    pass


class TestRineGroupRemoveIntegration(_GroupRemoveSpec, ToolsIntegrationTests):
    pass


class _GroupInspectSpec(RineToolConformance):
    tool_cls = RineGroupInspectTool
    invoke_args = {"group": GROUP_HANDLE}

    def install_mocks(self) -> None:
        mock_token()
        mock_agents()
        _mock_groups_list()


class TestRineGroupInspectUnit(_GroupInspectSpec, ToolsUnitTests):
    pass


class TestRineGroupInspectIntegration(_GroupInspectSpec, ToolsIntegrationTests):
    pass
