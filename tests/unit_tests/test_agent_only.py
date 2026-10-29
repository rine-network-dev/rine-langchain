"""An agent runtime acts as its agent and tells the model nothing about the owner.

SPEC 4.1: an agent runtime never loads ``root.key``. A host that onboarded holds one in
its owner dir, and the runtime must still sign in as the agent.
"""

from __future__ import annotations

import base64
import json

import httpx
import respx
from rine import PHRASE_INSTRUCTION
from rine._principal import agent_principal
from rine.testing import write_agent_identity, write_owner
from rine.types import RegistrationResult

from langchain_rine import onboard as onboard_module
from langchain_rine._client import _format_error, build_client

from ..conftest import AGENT_ID, API_URL, mock_agents


def _asserted_issuer(token_route) -> str:
    """The ``iss`` claim of the key assertion the first sign-in carried."""
    body = dict(httpx.QueryParams(token_route.calls[0].request.content.decode()))
    segment = body["client_assertion"].split(".")[1]
    return json.loads(base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4)))["iss"]


@respx.mock
def test_client_when_host_owner_dir_holds_a_root_then_still_signs_in_as_the_agent(
    tmp_path, monkeypatch
) -> None:
    """The assertion's issuer is the agent, whatever RINE_OWNER_DIR holds."""
    config_dir = str(tmp_path / "runtime")
    owner_dir = str(tmp_path / "owner")
    write_agent_identity(config_dir, AGENT_ID)
    write_owner(owner_dir)
    monkeypatch.setenv("RINE_OWNER_DIR", owner_dir)
    token = respx.post(f"{API_URL}/oauth/token").mock(
        return_value=httpx.Response(200, json={"access_token": "t", "expires_in": 900})
    )
    mock_agents()

    with build_client(config_dir=config_dir, api_url=API_URL, agent=None) as client:
        client.list_agents()

    assert _asserted_issuer(token) == str(agent_principal(AGENT_ID))


def test_auth_failure_when_shown_to_the_model_then_names_no_owner_command() -> None:
    """The model cannot run an owner command, so the auth failure does not name one."""
    from rine.errors import AuthenticationError

    shown = _format_error(AuthenticationError("no identity"))

    assert "langchain_rine.onboard" in shown
    assert "org recover" not in shown


class _FakeClient:
    """Stands in for the client that mints the agent; the key files are not under test."""

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def create_agent(self, name: str):
        from types import SimpleNamespace

        return SimpleNamespace(
            handle=f"{name}@acme.rine.network", id=AGENT_ID, verification_words="alpha bravo"
        )


def test_run_onboard_when_org_registered_then_prints_phrase_location_and_instruction(
    tmp_path, monkeypatch, capsys
) -> None:
    """The operator is told where the phrase is and what to do with it."""
    phrase_path = str(tmp_path / "owner" / "recovery-phrase.txt")
    monkeypatch.setattr(
        onboard_module,
        "onboard",
        lambda **_: RegistrationResult(org_id="org", org_words="one two", phrase_path=phrase_path),
    )
    monkeypatch.setattr(onboard_module, "SyncRineClient", lambda **_: _FakeClient())

    handle = onboard_module.run_onboard(
        email="a@acme.test",
        org_slug="acme",
        org_name="Acme",
        agent_name="scout",
        api_url=API_URL,
        config_dir=str(tmp_path / "runtime"),
    )

    out = capsys.readouterr().out
    assert handle == "scout@acme.rine.network"
    assert phrase_path in out
    assert PHRASE_INSTRUCTION in out
    assert "scout@acme.rine.network" in out
