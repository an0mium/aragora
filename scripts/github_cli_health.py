#!/usr/bin/env python3
"""Bounded GitHub CLI health probe for automation publisher flows."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_TIMEOUT_SECONDS = 20
CONNECTIVITY_ERROR_TOKENS = (
    "error connecting to api.github.com",
    "could not resolve host: api.github.com",
    "could not resolve host: github.com",
    "failed to connect to github.com",
    "failed to connect to api.github.com",
    "network is unreachable",
    "connection timed out",
    "connection refused",
    "temporary failure in name resolution",
    "no route to host",
)


@dataclass(frozen=True)
class GitHubCLIHealth:
    ready: bool
    auth_ok: bool
    api_ok: bool
    mode: str
    error: str
    repo: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def is_github_connectivity_error(message: str) -> bool:
    lowered = str(message or "").strip().lower()
    return any(token in lowered for token in CONNECTIVITY_ERROR_TOKENS)


def _run(args: list[str], *, cwd: Path, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        message = stderr or f"command timed out after {timeout_seconds}s: {' '.join(args)}"
        return subprocess.CompletedProcess(args=args, returncode=124, stdout=stdout, stderr=message)


def check_github_cli_health(
    repo_root: Path,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> GitHubCLIHealth:
    repo_root = repo_root.resolve()
    repo_label = str(repo_root)
    if shutil.which("gh") is None:
        return GitHubCLIHealth(
            ready=False,
            auth_ok=False,
            api_ok=False,
            mode="gh_missing",
            error="gh CLI not found on PATH",
            repo=repo_label,
        )

    auth_proc = _run(["gh", "auth", "status"], cwd=repo_root, timeout_seconds=timeout_seconds)
    if auth_proc.returncode != 0:
        error = auth_proc.stderr.strip() or auth_proc.stdout.strip() or "gh auth status failed"
        mode = "connectivity_failed" if is_github_connectivity_error(error) else "auth_failed"
        return GitHubCLIHealth(
            ready=False,
            auth_ok=False,
            api_ok=False,
            mode=mode,
            error=error,
            repo=repo_label,
        )

    api_proc = _run(["gh", "api", "rate_limit"], cwd=repo_root, timeout_seconds=timeout_seconds)
    if api_proc.returncode != 0:
        error = api_proc.stderr.strip() or api_proc.stdout.strip() or "gh api rate_limit failed"
        mode = "connectivity_failed" if is_github_connectivity_error(error) else "api_failed"
        return GitHubCLIHealth(
            ready=False,
            auth_ok=True,
            api_ok=False,
            mode=mode,
            error=error,
            repo=repo_label,
        )

    return GitHubCLIHealth(
        ready=True,
        auth_ok=True,
        api_ok=True,
        mode="ready",
        error="",
        repo=repo_label,
    )


def ensure_github_cli_ready(
    repo_root: Path,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> GitHubCLIHealth:
    health = check_github_cli_health(repo_root, timeout_seconds=timeout_seconds)
    if not health.ready:
        raise RuntimeError(health.error or health.mode)
    return health


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check whether gh can reach GitHub from here.")
    parser.add_argument("--repo", default=".", help="Path inside the target repository")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Timeout for each gh probe (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable output")
    parser.add_argument("--quiet", action="store_true", help="Suppress normal output")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    health = check_github_cli_health(
        Path(args.repo),
        timeout_seconds=int(args.timeout_seconds),
    )

    if args.json:
        print(json.dumps(health.to_dict(), indent=2))
    elif not args.quiet:
        if health.ready:
            print(f"gh ready for GitHub automation in {health.repo}")
        else:
            print(f"gh unavailable in {health.repo}: [{health.mode}] {health.error}".strip())
    return 0 if health.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
