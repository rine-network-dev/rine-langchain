"""``langchain-tests`` conformance for the discovery tools.

``rine_discover`` + ``rine_inspect`` hit the unauthenticated directory endpoints (no token
fetch). One ``ToolsUnitTests`` + one ``ToolsIntegrationTests`` subclass each; the
integration suite invokes against respx-mocked ``/directory`` routes. Both return ``str``.
"""

from __future__ import annotations

import httpx
import respx
from langchain_tests.integration_tests import ToolsIntegrationTests
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_rine import RineDiscoverTool, RineInspectTool
from tests.conftest import API_URL, OTHER_AGENT_ID
from tests.integration_tests._conformance import RineToolConformance


class _DiscoverSpec(RineToolConformance):
    tool_cls = RineDiscoverTool
    invoke_args = {"q": "x"}

    def install_mocks(self) -> None:
        respx.get(f"{API_URL}/directory/agents").mock(
            return_value=httpx.Response(200, json={"items": [], "total": 0})
        )


class TestRineDiscoverUnit(_DiscoverSpec, ToolsUnitTests):
    pass


class TestRineDiscoverIntegration(_DiscoverSpec, ToolsIntegrationTests):
    pass


class _InspectSpec(RineToolConformance):
    tool_cls = RineInspectTool
    invoke_args = {"handle_or_id": OTHER_AGENT_ID}

    def install_mocks(self) -> None:
        respx.get(f"{API_URL}/directory/agents/{OTHER_AGENT_ID}").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": OTHER_AGENT_ID,
                    "name": "Translator",
                    "handle": "translator@acme.rine.network",
                    "verified": True,
                    "human_oversight": True,
                },
            )
        )


class TestRineInspectUnit(_InspectSpec, ToolsUnitTests):
    pass


class TestRineInspectIntegration(_InspectSpec, ToolsIntegrationTests):
    pass
