from __future__ import annotations

import subprocess
from pathlib import Path

PR_9320_HEAD_REF = "refs/pull/9320/head"
PR_9320_LOCAL_REF = "refs/cdg-historical-backfill/9320/head"
PR_9320_HEAD_SHA = "aba6b14c94eca3a9c825b1a303ea67684d5f8daa"
HISTORICAL_HEAD_FETCH_ENV = "ARAGORA_CDG_FETCH_HISTORICAL_HEAD"


def ensure_pr_9320_head(
    repo_root: Path,
    *,
    remote: str = "origin",
    expected_sha: str = PR_9320_HEAD_SHA,
) -> str:
    """Fetch and authenticate the historical PR head missing from ordinary clones."""
    probe = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--verify", f"{expected_sha}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "fetch",
                "--no-tags",
                "--force",
                remote,
                f"{PR_9320_HEAD_REF}:{PR_9320_LOCAL_REF}",
            ],
            check=True,
        )
        ref = PR_9320_LOCAL_REF
    else:
        ref = PR_9320_HEAD_SHA
    resolved = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "--verify", f"{ref}^{{commit}}"],
        text=True,
    ).strip()
    if resolved != expected_sha:
        raise AssertionError(f"{PR_9320_HEAD_REF} resolved to {resolved}, expected {expected_sha}")
    return resolved
