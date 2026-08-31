"""Behavioral tests for RineToolkit.

A toolkit curates a subset of the 25 tools and shares ONE client across them.
These tests assert the curation (``include`` → which tool classes) and the
shared-client invariant (a single ``build_client`` call per toolkit instance,
with the config overrides propagated to every returned tool). Import-safety is
preserved: constructing the toolkit builds no client — it appears only inside
``get_tools()``.
"""

from __future__ import annotations

from unittest.mock import patch

from langchain_rine import (
    RineDiscoverGroupsTool,
    RineDiscoverTool,
    RineFulfillTool,
    RineInboxTool,
    RineInspectTool,
    RinePayTool,
    RineReadTool,
    RineReplyTool,
    RineSendAndWaitTool,
    RineSendTool,
    RineThreadTool,
    RineToolkit,
    RineWhoamiTool,
)

from ..conftest import API_URL

_MESSAGING_CLASSES = {
    RineSendTool,
    RineSendAndWaitTool,
    RineInboxTool,
    RineReadTool,
    RineReplyTool,
    RineThreadTool,
}
_DISCOVERY_CLASSES = {
    RineDiscoverTool,
    RineInspectTool,
    RineDiscoverGroupsTool,
    RineWhoamiTool,
}
_PAYMENTS_CLASSES = {RinePayTool, RineFulfillTool}


class TestToolkitCuration:
    def test_default_returns_all_twenty_five_tools(self):
        """The default toolkit exposes all 25 tools (incl. the 2 payment tools)."""
        tools = RineToolkit().get_tools()
        assert len(tools) == 25

    def test_include_messaging_returns_exactly_the_messaging_tools(self):
        """include='messaging' returns exactly the messaging tools (no fund tools)."""
        tools = RineToolkit(include="messaging").get_tools()
        assert {type(t) for t in tools} == _MESSAGING_CLASSES

    def test_include_payments_returns_exactly_the_two_payment_tools(self):
        """include='payments' returns exactly rine_pay + rine_fulfill (opt-in domain).

        Fund-moving tools live in their own domain so a messaging-only agent never gets a
        wallet-spending tool by default.
        """
        tools = RineToolkit(include="payments").get_tools()
        assert {type(t) for t in tools} == _PAYMENTS_CLASSES

    def test_include_discovery_returns_exactly_the_four_discovery_tools(self):
        """include='discovery' returns the agent pair plus group discovery and whoami."""
        tools = RineToolkit(include="discovery").get_tools()
        assert {type(t) for t in tools} == _DISCOVERY_CLASSES
        assert len(tools) == 4

    def test_include_list_returns_the_union_of_named_domains(self):
        """A list value returns the union of the named domains."""
        tools = RineToolkit(include=["messaging", "discovery"]).get_tools()
        assert {type(t) for t in tools} == _MESSAGING_CLASSES | _DISCOVERY_CLASSES
        # union, no duplicates, no group/payment tools leaked in
        assert len(tools) == len(_MESSAGING_CLASSES) + len(_DISCOVERY_CLASSES)


class TestToolkitSharedClient:
    def test_one_toolkit_builds_the_client_exactly_once(self):
        """A single toolkit instance ⇒ exactly one build_client call.

        Spy on the ``build_client`` symbol the toolkit module actually calls and
        assert it fires once even though 25 tools are produced — they share the
        single client, never build their own.
        """
        sentinel = object()
        with patch("langchain_rine.toolkit.build_client", return_value=sentinel) as spy:
            tools = RineToolkit(config_dir="/cfg", api_url=API_URL).get_tools()

        assert spy.call_count == 1
        # the one shared client is the very object injected into every tool
        assert all(t._client is sentinel for t in tools)

    def test_overrides_propagate_to_every_returned_tool(self):
        """config_dir/api_url/agent propagate to every tool from the toolkit."""
        sentinel = object()
        with patch("langchain_rine.toolkit.build_client", return_value=sentinel):
            tools = RineToolkit(config_dir="/cfg", api_url=API_URL, agent="bot@org").get_tools()

        assert tools  # non-empty so the `all` below is not vacuous
        for tool in tools:
            assert tool.config_dir == "/cfg"
            assert tool.api_url == API_URL
            assert tool.agent == "bot@org"

    def test_constructing_the_toolkit_builds_no_client(self):
        """Building the toolkit at/after import does nothing; client is lazy.

        The shared client appears only inside ``get_tools()`` — never at
        construction — so importing/instantiating the toolkit stays
        side-effect-free.
        """
        toolkit = RineToolkit(config_dir="/cfg", api_url=API_URL)
        assert toolkit._client is None  # nothing built yet

        sentinel = object()
        with patch("langchain_rine.toolkit.build_client", return_value=sentinel):
            toolkit.get_tools()
        # only now (after get_tools) has the single shared client been built
        assert toolkit._client is sentinel
