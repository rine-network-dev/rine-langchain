"""Setup-time onboarding helper — ``python -m langchain_rine.onboard``.

Runs the **real two-step** flow: ``onboard()`` registers an org
via a ~30-60s proof-of-work, then a separate ``create_agent()`` mints the agent and
prints its handle. This is deliberately **not** a ``BaseTool`` — a 30-60s PoW does
not belong inside an LLM tool call.

Config-dir / api-url resolution reuses the SDK's own helpers (no reinvented chain).
"""

from __future__ import annotations

import argparse
import sys

from rine import (
    PHRASE_INSTRUCTION,
    SyncRineClient,
    compare_verification_words_for_agent,
    onboard,
    resolve_api_url,
    resolve_config_dir,
    verification_words_lines,
)


def _print_pct(pct: int) -> None:
    """Progress callback for the PoW solve."""
    print(f"  proof-of-work: {pct}%", flush=True)


def run_onboard(
    *,
    email: str,
    org_slug: str,
    org_name: str,
    agent_name: str,
    api_url: str | None = None,
    config_dir: str | None = None,
) -> str:
    """Register an org (PoW) then create an agent. Returns the agent handle.

    Args:
        email: Contact email for the org.
        org_slug: DNS-safe org slug.
        org_name: Human-readable org name.
        agent_name: Name for the agent to create.
        api_url: API base URL (defaults to the SDK resolution).
        config_dir: Runtime config dir for the agent's keys and profile (defaults to SDK chain).

    Returns:
        The created agent's handle.
    """
    resolved_url = api_url or resolve_api_url()
    resolved_dir = config_dir or resolve_config_dir()

    print(f"Registering org '{org_name}' ({org_slug}) at {resolved_url} ...", flush=True)
    registration = onboard(
        api_url=resolved_url,
        config_dir=resolved_dir,
        email=email,
        org_slug=org_slug,
        org_name=org_name,
        on_progress=_print_pct,
    )
    print("Org registered. Creating agent ...", flush=True)
    print(f"Org verification words: {registration.org_words}")
    print(f"Recovery phrase file:   {registration.phrase_path}")
    print(PHRASE_INSTRUCTION)

    with SyncRineClient(config_dir=resolved_dir, api_url=resolved_url) as client:
        agent = client.create_agent(agent_name)

    print(f"\nAgent created: {agent.handle}")
    print(f"Config dir:    {resolved_dir}")
    # Derived from the signing key just written to this config dir, never from
    # the string the server returned with the agent.
    for line in verification_words_lines(
        compare_verification_words_for_agent(
            resolved_dir, str(agent.id), agent.verification_words
        ),
        label="Verify words",
    ):
        print(line)
    return agent.handle


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``python -m langchain_rine.onboard``."""
    parser = argparse.ArgumentParser(
        prog="python -m langchain_rine.onboard",
        description=(
            "One-time rine onboarding for a LangChain / LangGraph agent "
            "(org register + agent create)."
        ),
    )
    parser.add_argument("--email", required=True, help="Contact email for the org.")
    parser.add_argument("--org-slug", required=True, help="DNS-safe org slug.")
    parser.add_argument("--org-name", required=True, help="Human-readable org name.")
    parser.add_argument("--agent-name", required=True, help="Name for the agent to create.")
    parser.add_argument("--api-url", default=None, help="API base URL (default: SDK resolution).")
    parser.add_argument(
        "--config-dir", default=None, help="Config dir (default: SDK resolution chain)."
    )
    args = parser.parse_args(argv)

    try:
        run_onboard(
            email=args.email,
            org_slug=args.org_slug,
            org_name=args.org_name,
            agent_name=args.agent_name,
            api_url=args.api_url,
            config_dir=args.config_dir,
        )
    except Exception as err:  # noqa: BLE001 — top-level CLI: print and exit non-zero
        print(f"Onboarding failed: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
