"""Smoke guard for the (non-CI) live-LLM E2E driver.

The live E2E (``e2e/e2e_live_agent.py``) spends real Together.ai tokens and touches the
production rine network, so it never runs in CI. But it MUST not silently rot: this test
``py_compile``s the driver + its responder thread so a syntax error or a broken import
*shape* is caught by the unit suite without invoking the live model. (It does not import the
modules — importing the driver would require ``langchain``/``langchain-openai`` at module
top-level, and the point is a cheap, dependency-light compile check.)
"""

from __future__ import annotations

import py_compile
from pathlib import Path

import pytest

_E2E_DIR = Path(__file__).resolve().parents[2] / "e2e"
_E2E_MODULES = ["e2e_live_agent.py", "_responder.py", "__init__.py"]


@pytest.mark.parametrize("module", _E2E_MODULES)
def test_e2e_module_compiles(module: str) -> None:
    """Each shipped e2e module byte-compiles (no syntax/indentation rot)."""
    path = _E2E_DIR / module
    assert path.exists(), f"missing e2e module: {path}"
    py_compile.compile(str(path), doraise=True)
