"""Behavioral tests for the discovery + self-identity tools.

``rine_discover`` and ``rine_inspect`` hit the UNAUTHENTICATED directory
endpoints — no OAuth token is fetched. Asserted by NOT mocking ``/oauth/token``
and confirming the calls still succeed. ``rine_whoami`` is the exception in this
module: it reads the caller's OWN org and agents, so it is authenticated and the
token mock is required.

Tools are driven through their public LangChain interface: ``.invoke({...})``
(sync) and ``.ainvoke({...})`` (async). respx intercepts httpx regardless
of sync/async transport, so the same mock backs both paths.
"""

from __future__ import annotations

import httpx
import respx

from langchain_rine import RineDiscoverTool, RineInspectTool, RineWhoamiTool

from ..conftest import AGENT_PAYLOAD, API_URL, OTHER_AGENT_ID, mock_agents, mock_token


class TestRineDiscoverTool:
    @respx.mock
    def test_discover_with_matches_returns_numbered_list(self, config_dir):
        route = respx.get(f"{API_URL}/directory/agents").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "id": OTHER_AGENT_ID,
                            "name": "Translator",
                            "handle": "translator@acme.rine.network",
                            "category": "language",
                            "verified": True,
                            "trust_tier": 2,
                            "description": "Translates documents.",
                        }
                    ],
                    "total": 1,
                },
            )
        )

        out = RineDiscoverTool(config_dir=config_dir, api_url=API_URL).invoke({"q": "translate"})

        assert route.called
        assert "1 agent(s) found" in out
        assert "translator@acme.rine.network" in out
        assert "Translator" in out
        # unauthenticated: the discover GET carries no Authorization header
        assert "authorization" not in {k.lower() for k in route.calls[0].request.headers}

    @respx.mock
    def test_discover_unverified_agent_renders_unverified_not_verified(self, config_dir):
        """An unverified directory hit renders 'unverified', never 'verified'.

        With no trust_tier/category/description either, a constant 'verified' render
        would mislead the LLM into trusting an unvetted agent; this pins the negative
        branch and the omitted-optional-fields render.
        """
        route = respx.get(f"{API_URL}/directory/agents").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "id": OTHER_AGENT_ID,
                            "name": "Sketchy",
                            "handle": "sketchy@acme.rine.network",
                            "verified": False,
                            "trust_tier": None,
                            # no category, no description
                        }
                    ],
                    "total": 1,
                },
            )
        )

        out = RineDiscoverTool(config_dir=config_dir, api_url=API_URL).invoke({"q": "x"})

        assert route.called
        # Renders the honest 'unverified' label, never the verified one. The render
        # is '... (Sketchy, unverified)' — a verified hit would read '..., verified)'.
        assert "unverified)" in out
        assert ", verified)" not in out
        # Absent optional fields are simply omitted (no trust_tier=/category= noise).
        assert "trust_tier=" not in out
        assert "category=" not in out

    @respx.mock
    def test_discover_with_no_matches_returns_empty_string(self, config_dir):
        respx.get(f"{API_URL}/directory/agents").mock(
            return_value=httpx.Response(200, json={"items": [], "total": 0})
        )

        out = RineDiscoverTool(config_dir=config_dir, api_url=API_URL).invoke({"q": "nothing"})

        assert out == "No agents matched."

    @respx.mock
    def test_discover_forwards_filters_as_query_params(self, config_dir):
        route = respx.get(f"{API_URL}/directory/agents").mock(
            return_value=httpx.Response(200, json={"items": [], "total": 0})
        )

        RineDiscoverTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"category": "finance", "verified": True, "limit": 5}
        )

        query = route.calls[0].request.url.query
        assert b"category=finance" in query
        assert b"limit=5" in query

    @respx.mock
    async def test_discover_arun_matches_run(self, config_dir):
        """``.ainvoke`` returns the same numbered list as ``.invoke`` (same mock)."""
        route = respx.get(f"{API_URL}/directory/agents").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "id": OTHER_AGENT_ID,
                            "name": "Translator",
                            "handle": "translator@acme.rine.network",
                            "category": "language",
                            "verified": True,
                            "trust_tier": 2,
                            "description": "Translates documents.",
                        }
                    ],
                    "total": 1,
                },
            )
        )

        out = await RineDiscoverTool(config_dir=config_dir, api_url=API_URL).ainvoke(
            {"q": "translate"}
        )

        assert route.called
        assert "1 agent(s) found" in out
        assert "translator@acme.rine.network" in out
        # unauthenticated on the async path too
        assert "authorization" not in {k.lower() for k in route.calls[0].request.headers}


class TestRineInspectTool:
    @respx.mock
    def test_inspect_existing_agent_returns_profile(self, config_dir):
        respx.get(f"{API_URL}/directory/agents/{OTHER_AGENT_ID}").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": OTHER_AGENT_ID,
                    "name": "Translator",
                    "handle": "translator@acme.rine.network",
                    "description": "Translates documents.",
                    "category": "language",
                    "verified": True,
                    "human_oversight": True,
                    "trust_tier": 2,
                },
            )
        )

        out = RineInspectTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"handle_or_id": OTHER_AGENT_ID}
        )

        assert "handle: translator@acme.rine.network" in out
        assert "name: Translator" in out
        assert "category: language" in out
        assert "verified: True" in out

    @respx.mock
    def test_inspect_unverified_agent_renders_false_and_none_placeholders(self, config_dir):
        """An unverified profile renders 'verified: False' + '(none)' placeholders.

        With no category/description/trust_tier, these are the negative branches a
        constant 'verified' or a hardcoded field would silently break.
        """
        respx.get(f"{API_URL}/directory/agents/{OTHER_AGENT_ID}").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": OTHER_AGENT_ID,
                    "name": "Sketchy",
                    "handle": "sketchy@acme.rine.network",
                    "verified": False,
                    "human_oversight": True,
                    # no description, no category, no trust_tier
                },
            )
        )

        out = RineInspectTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"handle_or_id": OTHER_AGENT_ID}
        )

        assert "verified: False" in out
        assert "verified: True" not in out
        assert "description: (none)" in out
        assert "category: (none)" in out
        assert "trust_tier: (none)" in out

    @respx.mock
    def test_inspect_missing_agent_returns_not_found_string(self, config_dir):
        respx.get(f"{API_URL}/directory/agents/{OTHER_AGENT_ID}").mock(
            return_value=httpx.Response(404, json={"detail": "no such agent"})
        )

        out = RineInspectTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"handle_or_id": OTHER_AGENT_ID}
        )

        assert "Not found" in out
        assert "no such agent" in out

    @respx.mock
    def test_inspect_is_unauthenticated(self, config_dir):
        """No OAuth token mock present — inspect must still succeed (public endpoint)."""
        route = respx.get(f"{API_URL}/directory/agents/{OTHER_AGENT_ID}").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": OTHER_AGENT_ID,
                    "name": "A",
                    "handle": "a@acme.rine.network",
                    "verified": False,
                    "human_oversight": True,
                },
            )
        )

        RineInspectTool(config_dir=config_dir, api_url=API_URL).invoke(
            {"handle_or_id": OTHER_AGENT_ID}
        )

        assert "authorization" not in {k.lower() for k in route.calls[0].request.headers}

    @respx.mock
    async def test_inspect_arun_matches_run(self, config_dir):
        """``.ainvoke`` returns the same profile render as ``.invoke`` (same mock)."""
        respx.get(f"{API_URL}/directory/agents/{OTHER_AGENT_ID}").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": OTHER_AGENT_ID,
                    "name": "Translator",
                    "handle": "translator@acme.rine.network",
                    "description": "Translates documents.",
                    "category": "language",
                    "verified": True,
                    "human_oversight": True,
                    "trust_tier": 2,
                },
            )
        )

        out = await RineInspectTool(config_dir=config_dir, api_url=API_URL).ainvoke(
            {"handle_or_id": OTHER_AGENT_ID}
        )

        assert "handle: translator@acme.rine.network" in out
        assert "name: Translator" in out
        assert "verified: True" in out


ORG_PAYLOAD = {
    "id": "99999999-8888-7777-6666-555555555555",
    "name": "My Org",
    "slug": "myorg",
    "trust_tier": 2,
    "agent_count": 1,
    "created_at": "2024-01-01T00:00:00Z",
}


class TestRineWhoamiTool:
    @respx.mock
    def test_whoami_renders_org_slug_trust_tier_and_live_handles(self, config_dir):
        """The four fields the description promises all reach the string."""
        mock_token()
        mock_agents()
        route = respx.get(f"{API_URL}/org").mock(
            return_value=httpx.Response(200, json=ORG_PAYLOAD)
        )

        out = RineWhoamiTool(config_dir=config_dir, api_url=API_URL).invoke({})

        assert route.called
        assert "org: My Org" in out
        assert "org_slug: myorg" in out
        assert "trust_tier: 2" in out
        assert "test-agent@myorg.rine.network" in out

    @respx.mock
    async def test_whoami_arun_matches_run(self, config_dir):
        """``.ainvoke`` returns the same identity as ``.invoke`` against the same mock."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/org").mock(return_value=httpx.Response(200, json=ORG_PAYLOAD))

        out = await RineWhoamiTool(config_dir=config_dir, api_url=API_URL).ainvoke({})

        assert "org: My Org" in out
        assert "test-agent@myorg.rine.network" in out

    @respx.mock
    def test_whoami_omits_a_revoked_handle(self, config_dir):
        """A revoked handle can neither send nor be sent to, so it is not offered.

        Listing it would hand the model an address every send to which fails.
        """
        mock_token()
        respx.get(f"{API_URL}/agents").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [
                        AGENT_PAYLOAD,
                        {
                            **AGENT_PAYLOAD,
                            "id": OTHER_AGENT_ID,
                            "handle": "retired@myorg.rine.network",
                            "revoked_at": "2024-06-01T00:00:00Z",
                        },
                    ]
                },
            )
        )
        respx.get(f"{API_URL}/org").mock(return_value=httpx.Response(200, json=ORG_PAYLOAD))

        out = RineWhoamiTool(config_dir=config_dir, api_url=API_URL).invoke({})

        assert "test-agent@myorg.rine.network" in out
        assert "retired@myorg.rine.network" not in out

    @respx.mock
    def test_whoami_api_error_returns_a_formatted_string_not_a_raise(self, config_dir):
        """A failed identity read surfaces as a tool message so the agent continues."""
        mock_token()
        mock_agents()
        respx.get(f"{API_URL}/org").mock(return_value=httpx.Response(500, json={"detail": "boom"}))

        out = RineWhoamiTool(config_dir=config_dir, api_url=API_URL).invoke({})

        assert "500" in out
