"""Config/auth + packaging behavior.

Config resolution is asserted through observable HTTP at the SDK boundary (which
client_id reaches the token endpoint, which config dir the SDK reads) rather than
by poking private attributes — a good test survives an internal rewrite that
keeps the same auth behavior. The import-safety test runs in a subprocess with
socket *connections* disabled to prove a side-effect-free import.
"""

from __future__ import annotations

import subprocess
import sys

import httpx
import pytest
import respx
from rine import ACTING_AGENT_SPELLING, AGENT_HEADER
from rine._config import save_credentials

import langchain_rine
from langchain_rine import RineGroupsTool, RineSendTool
from langchain_rine._client import _format_error

from ..conftest import (
    AGENT_ID,
    API_URL,
    SECOND_AGENT_ID,
    mock_agents,
    mock_token,
    mock_two_agents,
)

# --- one shared error-mapping fixture set ---


class TestErrorMapping:
    def test_not_found_error_maps_to_discover_hint(self):
        from rine.errors import NotFoundError

        out = _format_error(NotFoundError("no agent xyz"))
        assert "Not found" in out
        assert "rine_discover" in out

    def test_unknown_acting_agent_404_names_no_search_verb(self):
        """Both spellings of an unknown actor answer with the roster and nothing else.

        The public directory is unauthenticated and org-agnostic, so no search
        verb can answer "which of my own agents may I act as" — the list already
        inside the refusal is the answer. The refusals come from the SDK's own
        builders, so a change to either lead sentence fails here rather than
        silently restoring the hint.
        """
        from rine._actor import unknown_actor_refusal, unknown_handle_refusal
        from rine.errors import NotFoundError

        agents = [{"name": "ops", "handle": "ops@acme.rine.network", "id": AGENT_ID}]
        for detail in (
            unknown_actor_refusal(agents, "logistics-bot"),
            unknown_handle_refusal(agents, "logistics-bot@acme.rine.network"),
        ):
            out = _format_error(NotFoundError(detail))
            assert "ops@acme.rine.network" in out
            assert "rine_discover" not in out

    def test_server_group_404_names_the_membership_list_before_the_public_directory(self):
        """A group 404 that carries no roster gets ``rine_groups`` first.

        The public directory answers only ``visibility == "public"``, so naming it
        alone sends a member of a PRIVATE group to a search that structurally
        cannot resolve their group. ``rine_groups`` lists every group the caller's
        org holds a seat in, public or private, which is what a mistyped handle on
        a send almost always is. The ORDER is deliberate, so it is asserted.

        The detail here is the server's own bare literal — the one 404 that still
        arrives with nothing but the lead.
        """
        from rine.errors import NotFoundError

        out = _format_error(NotFoundError("Group not found: ops"))
        assert out == (
            "Not found: Group not found: ops. "
            "Try rine_groups to find the right group handle, "
            "or rine_discover_groups to search the public directory."
        )
        assert out.index("rine_groups") < out.index("rine_discover_groups")

    def test_group_404_carrying_the_roster_names_only_the_public_directory(self):
        """``rine_groups`` is not offered as a remedy for a message that already ran it.

        The SDK's refusal lists the org's groups inline, so repeating the verb
        that produces that list tells the caller to fetch what it is reading.
        Only the public directory can still find something new.

        Asserted whole rather than by substring: the last roster row is the one
        place this function can put punctuation a caller would copy back into a
        handle, so what that row ends in is the assertion.
        """
        from rine._group_ref import match_group_by_local_name
        from rine.errors import NotFoundError

        groups = [{"id": AGENT_ID, "handle": "#planning@acme.rine.network", "name": "Planning"}]
        with pytest.raises(NotFoundError) as excinfo:
            match_group_by_local_name(groups, "ops")

        out = _format_error(excinfo.value)
        assert out == (
            "Not found: Group not found: ops. Name one of these groups:\n"
            "  - #planning@acme.rine.network (Planning)\n"
            "Try rine_discover_groups to search the public directory."
        )
        assert "rine_groups" not in out

    def test_group_404_in_an_org_with_no_groups_names_only_the_public_directory(self):
        """The empty arm carries no roster and still must not name ``rine_groups``.

        An org that holds a seat nowhere gets an empty list back from that verb,
        so the sentence would send the caller to a dead end. This arm is one
        line, so the remedy joins it with a space rather than a newline.
        """
        from rine._group_ref import match_group_by_local_name
        from rine.errors import NotFoundError

        with pytest.raises(NotFoundError) as excinfo:
            match_group_by_local_name([], "ops")

        out = _format_error(excinfo.value)
        assert out == (
            "Not found: Group not found: ops. This org is in no groups yet. "
            "Try rine_discover_groups to search the public directory."
        )

    def test_authentication_error_maps_to_onboard_hint(self):
        from rine.errors import AuthenticationError

        out = _format_error(AuthenticationError("bad creds"))
        assert "auth failed" in out.lower()
        assert "langchain_rine.onboard" in out

    def test_rate_limit_error_surfaces_retry_after(self):
        from rine.errors import RateLimitError

        out = _format_error(RateLimitError("slow down", retry_after=12))
        assert "retry after 12" in out

    def test_missing_mls_group_state_tells_the_agent_how_to_get_it(self):
        # Python does MLS now, so this is a recoverable state gap, not a ceiling:
        # the SDK's own guidance has to reach the model-facing string intact.
        from rine.errors import NoMlsGroupStateError

        out = _format_error(NoMlsGroupStateError("group-1"))
        assert "group-1" in out
        assert "Welcome" in out

    def test_generic_api_error_includes_status_and_detail(self):
        from rine.errors import RineApiError

        out = _format_error(RineApiError(418, "teapot"))
        assert "418" in out
        assert "teapot" in out

    def test_config_error_stringifies(self):
        """The ConfigError branch surfaces the SDK's message verbatim (str(err))."""
        from rine.errors import ConfigError

        out = _format_error(ConfigError("multiple agents; pass agent="))
        assert out == "multiple agents; pass agent="

    def test_not_a_member_refusal_keeps_the_servers_sentence_and_adds_this_stacks(self):
        """Both halves reach the model, because each knows something the other cannot.

        Only the server knows **which** sibling holds the seat. Only this stack
        knows that its caller has no way to set ``X-Rine-Agent`` — the SDK writes
        that header from ``agent=`` or ``RINE_AGENT``. Forwarding the refusal
        unchanged would send an agent after a header no tool here exposes.
        """
        from rine.errors import AuthorizationError

        out = _format_error(
            AuthorizationError(
                "agent bob is not a member of this group; your agent alice is — "
                "retry with X-Rine-Agent: alice@acme.rine.network"
            )
        )

        assert "your agent alice is" in out
        # The remedy half is asserted as the SDK's own exported sentence rather
        # than a retyped copy of it. That sentence has changed spelling before
        # (``agent=<name>`` became ``agent=``) and a literal here would have had
        # to be re-typed to match — which is the edit that silently turns this
        # gate into a test of whatever the SDK happens to say.
        assert ACTING_AGENT_SPELLING in out
        assert "RINE_AGENT" in out
        assert out.startswith("403:")

    def test_a_403_that_is_not_about_membership_is_forwarded_unchanged(self):
        """The acting-agent sentence is appended to one refusal, not to every 403."""
        from rine.errors import AuthorizationError

        out = _format_error(AuthorizationError("Admin access required"))

        assert out == "403: Admin access required"

    def test_non_rine_exception_stringifies(self):
        out = _format_error(ValueError("plain boom"))
        assert out == "plain boom"

    def test_no_error_maps_to_a_raw_traceback(self):
        """The mapper never returns a multi-line traceback."""
        from rine.errors import NotFoundError

        out = _format_error(NotFoundError("x"))
        assert "Traceback" not in out


# --- config resolution precedence ---


class TestConfigResolution:
    @respx.mock
    def test_env_credentials_beat_config_dir_file(self, tmp_path, monkeypatch):
        """RINE_CLIENT_ID/SECRET env wins over a different client_id in the file."""
        cdir = str(tmp_path)
        save_credentials(cdir, {"default": {"client_id": "file-cid", "client_secret": "file-sec"}})
        monkeypatch.setenv("RINE_CLIENT_ID", "env-cid")
        monkeypatch.setenv("RINE_CLIENT_SECRET", "env-sec")

        token = respx.post(f"{API_URL}/oauth/token").mock(
            return_value=httpx.Response(200, json={"access_token": "t", "expires_in": 900})
        )
        mock_agents()
        respx.get(f"{API_URL}/directory/agents/{'a' * 8}").mock(
            return_value=httpx.Response(404, json={"detail": "x"})
        )

        # Any authenticated call triggers the token fetch; inspect a tool that auths.
        from langchain_rine import RineInboxTool

        respx.get(f"{API_URL}/agents/00000000-1111-2222-3333-444444444444/messages").mock(
            return_value=httpx.Response(200, json={"items": [], "total": 0})
        )
        respx.post(
            f"{API_URL}/agents/00000000-1111-2222-3333-444444444444/messages/mark-delivered"
        ).mock(return_value=httpx.Response(200, json={"marked": 0}))
        RineInboxTool(config_dir=cdir, api_url=API_URL).invoke({})

        assert token.called
        body = dict(httpx.QueryParams(token.calls[0].request.content.decode()))
        assert body["client_id"] == "env-cid"  # env creds used, not file creds

    @respx.mock
    def test_constructor_config_dir_overrides_env_config_dir(self, tmp_path, monkeypatch):
        """The tool's config_dir kwarg takes precedence over RINE_CONFIG_DIR env."""
        good = tmp_path / "good"
        good.mkdir()
        save_credentials(
            str(good), {"default": {"client_id": "good-cid", "client_secret": "good-sec"}}
        )
        # env points at an EMPTY dir; constructor points at the good one.
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setenv("RINE_CONFIG_DIR", str(empty))

        token = respx.post(f"{API_URL}/oauth/token").mock(
            return_value=httpx.Response(200, json={"access_token": "t", "expires_in": 900})
        )
        mock_agents()
        respx.get(f"{API_URL}/directory/agents").mock(
            return_value=httpx.Response(200, json={"items": [], "total": 0})
        )

        # discover is unauthenticated; use an authenticated path to force creds read.
        from langchain_rine import RineInboxTool

        respx.get(f"{API_URL}/agents/00000000-1111-2222-3333-444444444444/messages").mock(
            return_value=httpx.Response(200, json={"items": [], "total": 0})
        )
        respx.post(
            f"{API_URL}/agents/00000000-1111-2222-3333-444444444444/messages/mark-delivered"
        ).mock(return_value=httpx.Response(200, json={"marked": 0}))
        RineInboxTool(config_dir=str(good), api_url=API_URL).invoke({})

        assert token.called
        body = dict(httpx.QueryParams(token.calls[0].request.content.decode()))
        assert body["client_id"] == "good-cid"  # constructor dir won over env dir


# --- which agent a tool acts as ---


class TestTheActingAgent:
    """The connector passes the acting agent through, it never picks one.

    ``build_client``/``abuild_client`` omit a ``None`` agent, so a tool that
    names nobody reaches the SDK's own ladder — which now has a ``RINE_AGENT``
    rung under the tool's ``agent`` field. That is a behaviour change for a
    multi-agent org with no line of connector code behind it, so it is pinned
    here at the wire, on both the sync and the async path (they are separate
    builders, and a rung added to one and not the other is the drift this
    catches).
    """

    @respx.mock
    def test_a_two_agent_org_acts_as_the_agent_the_environment_names(
        self, config_dir, monkeypatch
    ):
        monkeypatch.setenv("RINE_AGENT", "second-agent")
        mock_token()
        mock_two_agents()
        route = respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json={"items": []})
        )

        out = RineGroupsTool(config_dir=config_dir, api_url=API_URL).invoke({})

        assert route.called, out
        assert route.calls[0].request.headers[AGENT_HEADER] == SECOND_AGENT_ID

    @respx.mock
    async def test_the_async_path_reads_the_environment_too(self, config_dir, monkeypatch):
        monkeypatch.setenv("RINE_AGENT", "second-agent")
        mock_token()
        mock_two_agents()
        route = respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json={"items": []})
        )

        out = await RineGroupsTool(config_dir=config_dir, api_url=API_URL).ainvoke({})

        assert route.called, out
        assert route.calls[0].request.headers[AGENT_HEADER] == SECOND_AGENT_ID

    @respx.mock
    def test_the_tools_own_agent_field_beats_the_environment(self, config_dir, monkeypatch):
        # Cross-rung precedence is silent: the tool's field is rung 2, the env
        # var rung 3, and nothing is said about the one that lost. Named by UUID
        # on purpose — every tool also forwards this field as a per-call
        # ``agent=``, and that path puts the value on the wire verbatim.
        monkeypatch.setenv("RINE_AGENT", "second-agent")
        mock_token()
        mock_two_agents()
        route = respx.get(f"{API_URL}/groups").mock(
            return_value=httpx.Response(200, json={"items": []})
        )

        out = RineGroupsTool(config_dir=config_dir, api_url=API_URL, agent=AGENT_ID).invoke({})

        assert route.called, out
        assert route.calls[0].request.headers[AGENT_HEADER] == AGENT_ID


# --- import safety + public surface ---

_IMPORT_GUARD = """
import socket
def _blocked(*a, **k):
    raise AssertionError("import-time network connection attempt!")
# Block *connections* (not the socket class — the LangChain tree imports ssl,
# which subclasses socket.socket; replacing the class would break import).
socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.create_connection = _blocked
import langchain_rine  # must import cleanly with no connection attempted
print("OK", langchain_rine.__version__)
"""


class TestImportSafety:
    def test_import_performs_no_network_call(self):
        """Importing langchain_rine builds no client and hits no socket."""
        result = subprocess.run(
            [sys.executable, "-c", _IMPORT_GUARD],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith("OK")

    def test_init_exports_all_tools_toolkit_and_callback(self):
        """Every tool class + the toolkit + the callback are public exports."""
        expected = {
            "RineSendTool",
            "RineSendAndWaitTool",
            "RineInboxTool",
            "RineReadTool",
            "RineReplyTool",
            "RineDiscoverTool",
            "RineInspectTool",
            "RineGroupCreateTool",
            "RineGroupInviteTool",
            "RineGroupRemoveTool",
            "RineGroupInspectTool",
            "RineToolkit",
            "RineCallbackHandler",
        }
        assert expected.issubset(set(langchain_rine.__all__))
        for name in expected:
            assert hasattr(langchain_rine, name)

    def test_crewai_listener_name_is_not_exported(self):
        """A sibling-package ``RineNotificationListener`` must NOT leak into this package."""
        assert "RineNotificationListener" not in langchain_rine.__all__
        assert not hasattr(langchain_rine, "RineNotificationListener")

    def test_no_rine_token_concept_anywhere(self):
        """The Python SDK has no RINE_TOKEN; the package must not invent one.

        String-absence proof: ``RINE_TOKEN`` appears in none of the package's source
        files (it is not a config knob — auth is client_id/client_secret only).
        """
        import pathlib

        pkg_root = pathlib.Path(langchain_rine.__file__).parent
        for path in pkg_root.rglob("*.py"):
            assert "RINE_TOKEN" not in path.read_text(), f"RINE_TOKEN leaked into {path}"

    def test_tool_construction_builds_no_client(self):
        """Constructing a tool must not build a SyncRineClient (lazy on first _run)."""
        tool = RineSendTool()
        # The private client attr stays None until first use — observable proxy for
        # "no client constructed at __init__".
        assert tool._client is None
        assert tool._aclient is None
