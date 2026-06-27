#!/usr/bin/env python3
"""Mint a per-pass GitHub App installation token for shell ``gh`` consumers.

Shell daemons (codex-automation publisher, merge-arbiter wrapper) call ``gh``
directly with the operator's personal auth, draining the operator's API
budget. The Python paths already route through the App installation token via
:mod:`aragora.swarm.github_app_auth`; this script extends that to shell:

    tok="$(python3 scripts/gh_app_env.py --print-token 2>/dev/null || true)"
    [ -n "$tok" ] && export GH_TOKEN="$tok"

or, eval-able form::

    eval "$(python3 scripts/gh_app_env.py)"

Contract (load-bearing for the daemons):

* **Silent-safe** — when App config is absent, the mint fails, or aragora is
  not importable, exit 0 and print *nothing* so callers degrade gracefully to
  the operator's existing gh auth instead of crashing the pass.
* **No leakage** — the token is written only to stdout in the designated
  output mode; it is never logged, echoed to stderr, or included in
  diagnostics.
* **Per-pass freshness** — installation tokens expire after one hour, so
  callers must invoke this at the top of each pass, never once at daemon
  start. Each invocation is a fresh process, so the token is always fresh.

Callers that export the result should also export
``ARAGORA_GITHUB_AUTH_SOURCE=github_app_installation`` so
:func:`aragora.swarm.github_app_auth.github_cli_env` and
:func:`gh_subprocess_run` can recognize (and drop, for write ops) the
App-sourced token.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_minter() -> Callable[[], str | None]:
    """Import the App token minter; raises if aragora is unavailable."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from aragora.swarm.github_app_auth import get_github_app_installation_token

    return get_github_app_installation_token


def main(argv: list[str] | None = None, minter: Callable[[], str | None] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Mint a GitHub App installation token for shell gh calls. "
            "Prints nothing (exit 0) when App config is absent."
        ),
    )
    parser.add_argument(
        "--print-token",
        action="store_true",
        help="Print the bare token to stdout (for command substitution).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress diagnostics on stderr (never affects token output).",
    )
    args = parser.parse_args(argv)

    token: str | None = None
    try:
        mint = minter if minter is not None else _resolve_minter()
        token = mint()
    except Exception:  # noqa: BLE001 - silent-safe contract: degrade, never crash callers
        token = None

    token = (token or "").strip()
    if not token:
        if not args.quiet:
            # Diagnostic only — must never include token material.
            print(
                "gh_app_env: GitHub App config absent or mint failed; "
                "no token emitted (callers keep existing gh auth)",
                file=sys.stderr,
            )
        return 0

    if args.print_token:
        print(token)
    else:
        print(f"GH_TOKEN={token}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
