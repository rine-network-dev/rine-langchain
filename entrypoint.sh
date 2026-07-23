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
  echo ">>> [rig] bootstrapping venv at $VENV (first run — installs the LangChain tree + deps; slow)" >&2
  # Resolve uv to its absolute path BEFORE clearing the venv. `uv venv --clear` wipes
  # $VENV/bin (which is first on PATH); bash's command-hash would otherwise cache a now
  # -deleted $VENV/bin/uv and fail the next call ("No such file or directory") on a
  # re-bootstrap (marker removed but volume still populated).
  UV="$(command -v uv)"
  "$UV" venv --clear "$VENV"
  # `rine` MUST be the bind-mounted local SDK installed EDITABLE, so the rig always runs the
  # CURRENT SDK source and an unreleased SDK change can actually fail these tests. The local
  # path satisfies langchain-rine's `rine>=…` floor, so nothing is fetched from PyPI.
  # Two ways to silently lose editability — neither may be reintroduced, because the venv
  # lives on a named volume and a non-editable install freezes an SDK snapshot for the life
  # of that volume (this is what green-lit the payee suite against a pre-quote-ref SDK):
  #   1. `--override rine @ file:///work/rine-sdk` — an override outranks `-e` and copies.
  #   2. `-e "/work/rine-sdk[payments]"` — uv drops editability when a path carries extras.
  #   3. installing both projects in ONE `uv pip install` — langchain-rine[dev] requires
  #      `rine>=…`, and that dependency edge outranks the sibling `-e`, so rine is copied.
  # Hence two steps: pin the SDK editable first, then langchain-rine[dev] finds `rine`
  # already satisfied and leaves it alone. eth-account (EIP-3009 signing for rine_pay)
  # comes from langchain-rine's `dev` extra, which lists it directly, not via
  # `rine[payments]` — so the SDK path never needs an extra.
  # rine-mls (rine's compiled MLS + ML-KEM core) comes from the prebuilt wheelhouse
  # mounted at /work/wheels, never from source: the SDK's `[tool.uv.sources]` points it
  # at a crate, and building that crate needs a Rust toolchain no rig image carries.
  # `--no-index` is what proves the wheel came from there and not from PyPI, where
  # rine-mls is not published yet — so the SDK install below ignores that one source
  # and takes this wheel as satisfying its `rine-mls` dependency.
  "$UV" pip install --python "$VENV/bin/python" \
    --no-index --find-links /work/wheels rine-mls
  "$UV" pip install --python "$VENV/bin/python" --no-sources-package rine-mls \
    -e /work/rine-sdk
  "$UV" pip install --python "$VENV/bin/python" -e "/work/rine-langchain[dev]"
  touch "$_RIG_INSTALLED"
  echo ">>> [rig] bootstrap complete" >&2
fi

exec "$@"
