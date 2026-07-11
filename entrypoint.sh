#!/usr/bin/env bash
# Idempotent rig bootstrap for the contained langchain-rine test rig.
#
# Ensures the named-volume venv at $VENV exists and holds the bind-mounted local
# rine-sdk (editable) + langchain-rine[dev] (editable, pulling langchain-core / langgraph /
# langchain-openai / langchain-tests), then exec's whatever command compose/`run` passed.
# The venv lives on a named volume, so the (slow) first install is cached for every later
# run and never touches the host.
set -euo pipefail

# Private, non-exported sentinel path. Named `_RIG_INSTALLED` (not `MARKER`) so a test
# that passes `-e MARKER=…` into the rig is not clobbered: reassigning an already-exported
# env var here would leak the rig value into the exec'd child process env.
VENV="${VENV:-/opt/venv}"
_RIG_INSTALLED="$VENV/.rig-installed"

if [ ! -f "$_RIG_INSTALLED" ]; then
  echo ">>> [rig] bootstrapping venv at $VENV (first run — installs the LangChain tree + deps; slow)"
  # Resolve uv to its absolute path BEFORE clearing the venv. `uv venv --clear` wipes
  # $VENV/bin (which is first on PATH); bash's command-hash would otherwise cache a now
  # -deleted $VENV/bin/uv and fail the next call ("No such file or directory") on a
  # re-bootstrap (marker removed but volume still populated).
  UV="$(command -v uv)"
  "$UV" venv --clear "$VENV"
  # Force langchain-rine's `rine` dependency to resolve to the bind-mounted LOCAL SDK
  # (unreleased Phase-0 fixes), overriding the published floor pinned in pyproject.toml.
  # pyproject keeps the honest published pin (`rine>=0.2.2`); dev/tests run on the local
  # source (./rine-sdk, currently 0.2.2). The override is the rig-only escape hatch.
  echo "rine @ file:///work/rine-sdk" > /tmp/rig-overrides.txt
  "$UV" pip install --python "$VENV/bin/python" \
    --override /tmp/rig-overrides.txt \
    -e /work/rine-sdk \
    -e "/work/rine-langchain[dev]"
  touch "$_RIG_INSTALLED"
  echo ">>> [rig] bootstrap complete"
fi

exec "$@"
