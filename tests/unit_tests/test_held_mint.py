"""A mint refused for a hold reaches the agent as the server's sentence, not an auth fault.

The bodies are what the server's token endpoint sends for a held agent: the hold in ``reason``
and the sentence to show in ``error_description``.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from rine.testing import write_agent_identity

from langchain_rine import RineInboxTool

from ..conftest import AGENT_ID, API_URL

HELD_MINTS = {
    "agent_suspended": (
        "This agent is suspended, so it cannot do this now. Messages sent to it are queued and "
        "delivered after it is resumed, within the organisation's retention period."
    ),
    "branch_suspended": (
        "This agent's branch is suspended, so the agent cannot do this now. Messages sent to the "
        "branch's agents are queued and delivered after the branch is resumed, within the "
        "organisation's retention period."
    ),
    "org_frozen": (
        "This organisation is frozen, so its agents cannot do this now. Messages sent to them "
        "are queued and delivered after it is unfrozen, within the organisation's retention "
        "period."
    ),
}


@pytest.mark.parametrize(("hold", "sentence"), HELD_MINTS.items())
@respx.mock
def test_tool_when_the_mint_is_held_then_it_returns_the_servers_sentence(
    tmp_path, hold: str, sentence: str
) -> None:
    """The sentence is the server's, byte for byte, with no auth-fault wording around it."""
    write_agent_identity(str(tmp_path), AGENT_ID)
    respx.post(f"{API_URL}/oauth/token").mock(
        return_value=httpx.Response(
            400,
            json={"error": "unauthorized_client", "reason": hold, "error_description": sentence},
        )
    )

    out = RineInboxTool(config_dir=str(tmp_path), api_url=API_URL, agent=AGENT_ID).invoke({})

    assert out == sentence


@respx.mock
def test_tool_when_the_assertion_is_refused_then_it_still_says_to_onboard(tmp_path) -> None:
    """A real credential fault keeps its onboarding hint."""
    write_agent_identity(str(tmp_path), AGENT_ID)
    respx.post(f"{API_URL}/oauth/token").mock(
        return_value=httpx.Response(401, json={"error": "invalid_client"})
    )

    out = RineInboxTool(config_dir=str(tmp_path), api_url=API_URL, agent=AGENT_ID).invoke({})

    assert out.startswith("Rine auth failed")
