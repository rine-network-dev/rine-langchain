"""Behavioral tests for RineToolkit (SPEC §7, AC-10/11).

A toolkit curates a subset of the 11 tools and shares ONE client across them.
These tests assert the curation (``include`` → which tool classes) and the
shared-client invariant (a single ``build_client`` call per toolkit instance,
with the config overrides propagated to every returned tool). Import-safety is
preserved: constructing the toolkit builds no client — it appears only inside
``get_tools()``.
"""

from __future__ import annotations

from unittest.mock import patch

from langchain_rine import (
    RineCheckInboxTool,
    RineDiscoverTool,
    RineInspectTool,
    RineReadTool,
    RineReplyTool,
    RineSendAndWaitTool,
    RineSendTool,
    RineToolkit,
)

from ..conftest import API_URL

_MESSAGING_CLASSES = {
    RineSendTool,
    RineSendAndWaitTool,
    RineCheckInboxTool,
    RineReadTool,
    RineReplyTool,
}
_DISCOVERY_CLASSES = {RineDiscoverTool, RineInspectTool}


class TestToolkitCuration:
    def test_default_returns_all_eleven_tools(self):
        """AC-10: the default toolkit exposes all 11 tools."""
        tools = RineToolkit().get_tools()
        assert len(tools) == 11

    def test_include_messaging_returns_exactly_the_five_messaging_tools(self):
        """AC-10: include='messaging' returns exactly the 5 messaging tools."""
        tools = RineToolkit(include="messaging").get_tools()
        assert {type(t) for t in tools} == _MESSAGING_CLASSES

    def test_include_list_returns_the_union_of_named_domains(self):
        """AC-10: a list value returns the union of the named domains."""
        tools = RineToolkit(include=["messaging", "discovery"]).get_tools()
        assert {type(t) for t in tools} == _MESSAGING_CLASSES | _DISCOVERY_CLASSES
        # union, no duplicates, no group tools leaked in
        assert len(tools) == len(_MESSAGING_CLASSES) + len(_DISCOVERY_CLASSES)


class TestToolkitSharedClient:
    def test_one_toolkit_builds_the_client_exactly_once(self):
        """AC-11: a single toolkit instance ⇒ exactly one build_client call.

        Spy on the ``build_client`` symbol the toolkit module actually calls and
        assert it fires once even though 11 tools are produced — they share the
        single client, never build their own.
        """
        sentinel = object()
        with patch("langchain_rine.toolkit.build_client", return_value=sentinel) as spy:
            tools = RineToolkit(config_dir="/cfg", api_url=API_URL).get_tools()

        assert spy.call_count == 1
        # the one shared client is the very object injected into every tool
        assert all(t._client is sentinel for t in tools)

    def test_overrides_propagate_to_every_returned_tool(self):
        """AC-11: config_dir/api_url/agent propagate to every tool from the toolkit."""
        sentinel = object()
        with patch("langchain_rine.toolkit.build_client", return_value=sentinel):
            tools = RineToolkit(config_dir="/cfg", api_url=API_URL, agent="bot@org").get_tools()

        assert tools  # non-empty so the `all` below is not vacuous
        for tool in tools:
            assert tool.config_dir == "/cfg"
            assert tool.api_url == API_URL
            assert tool.agent == "bot@org"

    def test_constructing_the_toolkit_builds_no_client(self):
        """I-1: building the toolkit at/after import does nothing; client is lazy.

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
