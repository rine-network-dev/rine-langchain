"""Shared rig for the ``langchain-tests`` conformance subclasses (SPEC §13, PHASE0 §7).

Each rine tool ships two conformance subclasses — one ``ToolsUnitTests`` and one
``ToolsIntegrationTests`` (``langchain_tests.unit_tests`` / ``.integration_tests``).
The unit suite only constructs the tool and inspects its ``name``/``args_schema``; the
integration suite actually **invokes** it (``.invoke`` + ``.ainvoke``) over the real
``rine`` SDK, so the SDK's HTTP boundary must be live for the call to succeed.

This module supplies the wiring once (no per-tool duplication):

- :class:`RineToolConformance` — a mixin both suites inherit. It overrides only the
  ``langchain-tests`` *properties* (never a ``test_*`` method, so
  ``test_no_overrides_DO_NOT_OVERRIDE`` stays green) and adds one autouse fixture that:
  1. builds a real config dir + persists the recipient keys (via the conftest
     ``config_dir``/``other_keys`` fixtures) so construction needs no env;
  2. enters ``respx.mock`` and installs the per-tool HTTP mocks, so the invoke the
     integration suite performs hits the mocked boundary, not the network.

A concrete tool spec sets three class attributes — ``tool_cls`` (the tool class),
``invoke_args`` (a minimal valid ``args_schema`` dict), and ``needs_other_keys`` — and
implements :meth:`install_mocks` (the respx routes for that one tool). The Unit/Integration
subclasses are then just ``class …Unit(Spec, ToolsUnitTests)`` /
``class …Integration(Spec, ToolsIntegrationTests)``.

v0.1 ships the string-return default (``response_format="content"``, D5): every tool
returns a ``str`` and never sets a ``content_and_artifact`` artifact, so the integration
suite's artifact assertion never fires.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.conftest import API_URL

if TYPE_CHECKING:
    from langchain_rine._client import BaseRineTool


class RineToolConformance:
    """Mixin wiring the conftest rig + per-tool respx mocks into both conformance suites.

    Subclasses set ``tool_cls`` / ``invoke_args`` / ``needs_other_keys`` and implement
    :meth:`install_mocks`. Only properties + fixtures are defined here — no ``test_*``
    method is touched, so ``test_no_overrides_DO_NOT_OVERRIDE`` passes.
    """

    tool_cls: type[BaseRineTool]
    invoke_args: dict[str, Any]
    needs_other_keys: bool = False

    @property
    def tool_constructor(self) -> type[BaseRineTool]:
        """The tool class under conformance test."""
        return self.tool_cls

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        """Construct against the real test config dir + mocked API — no env needed."""
        return {"config_dir": self._cfg, "api_url": API_URL}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        """A minimal valid ``args_schema`` dict for the tool under test."""
        return self.invoke_args

    def install_mocks(self) -> None:
        """Register the respx routes this tool's invoke needs (override per tool)."""
        raise NotImplementedError

    @pytest.fixture(autouse=True)
    def _rine_rig(self, config_dir: str, request: pytest.FixtureRequest) -> Any:
        """Wire the real config dir + per-tool HTTP mocks around each conformance test.

        Runs before the ``langchain-tests`` ``tool`` fixture (autouse, no deps), so
        ``tool_constructor_params`` can read ``self._cfg``. The integration suite invokes
        the tool inside this ``respx.mock`` window; the unit suite simply ignores the
        unused routes.
        """
        import respx

        self._cfg = config_dir
        if self.needs_other_keys:
            request.getfixturevalue("other_keys")
        with respx.mock:
            self.install_mocks()
            yield
