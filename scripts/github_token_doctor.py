#!/usr/bin/env python3
"""Report GitHub API quota by identity without printing token material."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_USERS = ("an0mium", "scarmani")


@dataclass(frozen=True)
class TokenCapacity:
    source: str
    available: bool
    core_remaining: int | None = None
    core_limit: int | None = None
    graphql_remaining: int | None = None
    graphql_limit: int | None = None
    graphql_reset: int | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _completed_error(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stderr or result.stdout or "").strip()


def _rate_limit_for_token(token: str, *, timeout: float = 20.0) -> TokenCapacity:
    env = dict(os.environ)
    env["GH_TOKEN"] = token
    env["GITHUB_TOKEN"] = token
    result = subprocess.run(  # noqa: S603 - controlled gh invocation
        ["gh", "api", "rate_limit"],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        return TokenCapacity(source="", available=False, error=_completed_error(result))
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        return TokenCapacity(
            source="", available=False, error=f"rate_limit JSON decode failed: {exc}"
        )
    resources = payload.get("resources")
    if not isinstance(resources, dict):
        return TokenCapacity(source="", available=False, error="rate_limit missing resources")
    core = resources.get("core") if isinstance(resources.get("core"), dict) else {}
    graphql = resources.get("graphql") if isinstance(resources.get("graphql"), dict) else {}
    return TokenCapacity(
        source="",
        available=True,
        core_remaining=_int_or_none(core.get("remaining")),
        core_limit=_int_or_none(core.get("limit")),
        graphql_remaining=_int_or_none(graphql.get("remaining")),
        graphql_limit=_int_or_none(graphql.get("limit")),
        graphql_reset=_int_or_none(graphql.get("reset")),
    )


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _with_source(capacity: TokenCapacity, source: str) -> TokenCapacity:
    return TokenCapacity(
        source=source,
        available=capacity.available,
        core_remaining=capacity.core_remaining,
        core_limit=capacity.core_limit,
        graphql_remaining=capacity.graphql_remaining,
        graphql_limit=capacity.graphql_limit,
        graphql_reset=capacity.graphql_reset,
        error=capacity.error,
    )


def _gh_user_token(user: str, *, timeout: float = 10.0) -> str | None:
    result = subprocess.run(  # noqa: S603 - controlled gh invocation
        ["gh", "auth", "token", "--user", user],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        return None
    token = result.stdout.strip()
    return token or None


def probe_gh_user(user: str) -> TokenCapacity:
    token = _gh_user_token(user)
    if not token:
        return TokenCapacity(
            source=f"gh-user:{user}", available=False, error="gh token unavailable"
        )
    try:
        return _with_source(_rate_limit_for_token(token), f"gh-user:{user}")
    finally:
        token = ""


def probe_app_token() -> TokenCapacity:
    try:
        from aragora.swarm.github_app_auth import get_github_app_installation_token

        token = get_github_app_installation_token()
    except Exception as exc:  # noqa: BLE001 - diagnostic only, no token leakage
        return TokenCapacity(source="github-app", available=False, error=f"mint failed: {exc}")
    if not token:
        return TokenCapacity(source="github-app", available=False, error="app token unavailable")
    try:
        return _with_source(_rate_limit_for_token(token), "github-app")
    finally:
        token = ""


def probe_sources(users: Sequence[str], *, include_app: bool = True) -> list[TokenCapacity]:
    capacities = [probe_gh_user(user) for user in users]
    if include_app:
        capacities.append(probe_app_token())
    return capacities


def _format_capacity(capacity: TokenCapacity) -> str:
    if not capacity.available:
        return f"{capacity.source}: unavailable ({capacity.error})"
    return (
        f"{capacity.source}: core={capacity.core_remaining}/{capacity.core_limit} "
        f"graphql={capacity.graphql_remaining}/{capacity.graphql_limit} "
        f"graphql_reset={capacity.graphql_reset}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--user",
        action="append",
        dest="users",
        default=[],
        help="gh account to probe via `gh auth token --user`; repeatable.",
    )
    parser.add_argument("--no-app", action="store_true", help="Skip GitHub App token probe.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    users = tuple(args.users) if args.users else DEFAULT_USERS
    capacities = probe_sources(users, include_app=not args.no_app)
    payload = {
        "ok": any(item.available for item in capacities),
        "sources": [c.to_dict() for c in capacities],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for capacity in capacities:
            print(_format_capacity(capacity))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
