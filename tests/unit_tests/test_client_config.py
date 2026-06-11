"""Config/auth + packaging behavior (SPEC §7, REQ-01/02/05/06/18).

Config resolution is asserted through observable HTTP at the SDK boundary (which
client_id reaches the token endpoint, which config dir the SDK reads) rather than
by poking private attributes — a good test survives an internal rewrite that
keeps the same auth behavior. The import-safety test runs in a subprocess with
socket *connections* disabled to prove a side-effect-free import (REQ-01 / I-1).
"""

from __future__ import annotations

import subprocess
import sys

import httpx
import respx
from rine._config import save_credentials

import langchain_rine
from langchain_rine import RineSendTool
from langchain_rine._client import _format_error

from ..conftest import API_URL, mock_agents

# --- one shared error-mapping fixture set (REQ-18) ---


class TestErrorMapping:
    def test_not_found_error_maps_to_discover_hint(self):
        from rine.errors import NotFoundError

        out = _format_error(NotFoundError("no agent xyz"))
        assert "Not found" in out
        assert "rine_discover" in out

    def test_authentication_error_maps_to_onboard_hint(self):
        from rine.errors import AuthenticationError

        out = _format_error(AuthenticationError("bad creds"))
        assert "auth failed" in out.lower()
        assert "langchain_rine.onboard" in out

    def test_rate_limit_error_surfaces_retry_after(self):
        from rine.errors import RateLimitError

        out = _format_error(RateLimitError("slow down", retry_after=12))
        assert "retry after 12" in out

    def test_mls_unsupported_error_maps_to_ceiling(self):
        from rine.errors import MlsUnsupportedError

        out = _format_error(MlsUnsupportedError())
        assert "MLS" in out

    def test_generic_api_error_includes_status_and_detail(self):
        from rine.errors import RineApiError

        out = _format_error(RineApiError(418, "teapot"))
        assert "418" in out
        assert "teapot" in out

    def test_config_error_stringifies(self):
        """AC-2: the ConfigError branch surfaces the SDK's message verbatim (str(err))."""
        from rine.errors import ConfigError

        out = _format_error(ConfigError("multiple agents; pass agent="))
        assert out == "multiple agents; pass agent="

    def test_non_rine_exception_stringifies(self):
        out = _format_error(ValueError("plain boom"))
        assert out == "plain boom"

    def test_no_error_maps_to_a_raw_traceback(self):
        """The mapper never returns a multi-line traceback (REQ-18)."""
        from rine.errors import NotFoundError

        out = _format_error(NotFoundError("x"))
        assert "Traceback" not in out


# --- config resolution precedence (REQ-06) ---


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
        from langchain_rine import RineCheckInboxTool

        respx.get(f"{API_URL}/agents/00000000-1111-2222-3333-444444444444/messages").mock(
            return_value=httpx.Response(200, json={"items": [], "total": 0})
        )
        respx.post(
            f"{API_URL}/agents/00000000-1111-2222-3333-444444444444/messages/mark-delivered"
        ).mock(return_value=httpx.Response(200, json={"marked": 0}))
        RineCheckInboxTool(config_dir=cdir, api_url=API_URL).invoke({})

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
        from langchain_rine import RineCheckInboxTool

        respx.get(f"{API_URL}/agents/00000000-1111-2222-3333-444444444444/messages").mock(
            return_value=httpx.Response(200, json={"items": [], "total": 0})
        )
        respx.post(
            f"{API_URL}/agents/00000000-1111-2222-3333-444444444444/messages/mark-delivered"
        ).mock(return_value=httpx.Response(200, json={"marked": 0}))
        RineCheckInboxTool(config_dir=str(good), api_url=API_URL).invoke({})

        assert token.called
        body = dict(httpx.QueryParams(token.calls[0].request.content.decode()))
        assert body["client_id"] == "good-cid"  # constructor dir won over env dir


# --- import safety + public surface (REQ-01/02 / I-1) ---

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
        """REQ-01/I-1: importing langchain_rine builds no client and hits no socket."""
        result = subprocess.run(
            [sys.executable, "-c", _IMPORT_GUARD],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith("OK")

    def test_init_exports_all_tools_toolkit_and_callback(self):
        """REQ-02: every tool class + the toolkit + the callback are public exports."""
        expected = {
            "RineSendTool",
            "RineSendAndWaitTool",
            "RineCheckInboxTool",
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
        """SPEC §10/§13: the Python SDK has no RINE_TOKEN; the package must not invent one.

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
        # "no client constructed at __init__" (I-1).
        assert tool._client is None
        assert tool._aclient is None
