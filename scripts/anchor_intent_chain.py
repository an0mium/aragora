#!/usr/bin/env python3
"""Anchor the intent-chain head hash outside the laptop (TET phase T2).

Spec: ``docs/specs/TAMPER_EVIDENT_TRAIL.md`` Component 2, step 2 ("External
anchor"). The local chain (``aragora/trail/intent_chain.py``) is only
internally consistent; this script periodically commits its head hash to
infrastructure the writing machine cannot rewrite, so any later rewrite of an
anchored prefix is detectable by reconciliation (phase T3).

Primary anchor: a GitHub **commit status** on ``origin/main`` HEAD under the
dedicated context ``aragora/trail-anchor`` with description
``trail-anchor seq=<N> head=<hash12>``. Commit statuses are server-side
timestamped and themselves witnessed by the Enterprise audit stream (the
witness then witnesses the anchor). Token resolution prefers the GitHub App
installation path (``aragora.swarm.github_app_auth.github_cli_env``) and
falls back to ambient ``gh`` auth.

Optional anchor: ``--rekor`` additionally submits the head hash to the
Sigstore Rekor public transparency log when a ``rekor-cli`` or ``cosign``
binary is present — and degrades SILENTLY to commit-status-only when neither
is installed (noted in output; no hard dependency, per spec).

Safety model (mirrors the run's other bounded scripts):

- Dry-run by default; ``--apply`` gates every network mutation.
- ``--max-anchors`` caps mutations per invocation (default 2: one status +
  one optional rekor entry).
- Fail-closed: a chain that fails ``verify_chain`` is NEVER anchored (an
  anchor must only ever attest a valid chain head); exit 1 on any failure.
- Empty chain is a clean no-op (exit 0): there is nothing to attest.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:  # direct script invocation
    sys.path.insert(0, str(_REPO_ROOT))

from aragora.trail.intent_chain import (  # noqa: E402
    chain_head_hash,
    default_chain_path,
    read_records,
    verify_chain,
)

DEFAULT_REPO = "synaptent/aragora"
ANCHOR_CONTEXT = "aragora/trail-anchor"
GH_TIMEOUT_SECONDS = 60
REKOR_TIMEOUT_SECONDS = 120

EXIT_OK = 0
EXIT_FAILURE = 1


def _gh_env() -> dict[str, str] | None:
    """Prefer the App-installation token env; ``None`` means ambient auth."""
    try:
        from aragora.swarm.github_app_auth import github_cli_env

        return github_cli_env()
    except Exception:  # noqa: BLE001 - any auth-path failure falls back to ambient gh
        return None


def default_run_gh(args: list[str]) -> tuple[int, str]:
    """Run ``gh`` with App-token env when available; return (rc, stdout)."""
    try:
        proc = subprocess.run(
            ["gh", *args],
            capture_output=True,
            text=True,
            timeout=GH_TIMEOUT_SECONDS,
            env=_gh_env(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, f"gh invocation failed: {exc}"
    out = proc.stdout if proc.returncode == 0 else (proc.stderr or proc.stdout)
    return proc.returncode, out.strip()


def resolve_main_head(repo: str, run_gh: Callable[[list[str]], tuple[int, str]]) -> str:
    """SHA of the repo's main branch head, resolved server-side via the API."""
    rc, out = run_gh(["api", f"repos/{repo}/commits/main", "--jq", ".sha"])
    sha = out.strip().splitlines()[-1].strip() if out.strip() else ""
    if rc != 0 or len(sha) != 40:
        raise RuntimeError(f"could not resolve {repo} main head: {out[:200]}")
    return sha


def build_status_args(repo: str, sha: str, seq: int, head_hash: str) -> list[str]:
    """``gh api`` argv that posts the anchor commit status (server-timestamped)."""
    return [
        "api",
        "--method",
        "POST",
        f"repos/{repo}/statuses/{sha}",
        "-f",
        "state=success",
        "-f",
        f"context={ANCHOR_CONTEXT}",
        "-f",
        f"description=trail-anchor seq={seq} head={head_hash[:12]}",
    ]


def find_rekor_backend() -> list[str] | None:
    """Argv prefix for an available Rekor uploader, or ``None``."""
    if shutil.which("rekor-cli"):
        return ["rekor-cli", "upload", "--artifact"]
    if shutil.which("cosign"):
        return ["cosign", "upload-blob"]  # placeholder; cosign needs a registry ref
    return None


def submit_rekor(head_hash: str, *, apply: bool, log: Callable[[str], None]) -> bool:
    """Best-effort Rekor anchor; ``False`` only on an *attempted* failure.

    Absence of any Rekor-capable binary is a SILENT degrade (spec): the run
    still succeeds on the commit-status anchor alone, with a note in output.
    """
    backend = find_rekor_backend()
    if backend is None or backend[0] == "cosign":
        # cosign's blob upload targets an OCI registry, not Rekor directly —
        # treat it as unavailable rather than half-anchor somewhere else.
        log(json.dumps({"rekor": "unavailable", "note": "commit-status anchor only"}))
        return True
    if not apply:
        log(json.dumps({"rekor": "dry-run", "cmd": [*backend, "<head-hash-artifact>"]}))
        return True
    with tempfile.NamedTemporaryFile("w", suffix=".trail-head", delete=False) as fh:
        fh.write(head_hash + "\n")
        artifact = fh.name
    try:
        proc = subprocess.run(
            [*backend, artifact],
            capture_output=True,
            text=True,
            timeout=REKOR_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log(json.dumps({"rekor": "failed", "error": str(exc)[:200]}))
        return False
    finally:
        Path(artifact).unlink(missing_ok=True)
    if proc.returncode != 0:
        log(json.dumps({"rekor": "failed", "error": (proc.stderr or proc.stdout)[:200]}))
        return False
    log(json.dumps({"rekor": "anchored", "head": head_hash[:12]}))
    return True


def run_anchor(
    *,
    chain_path: Path,
    repo: str,
    apply: bool,
    rekor: bool,
    max_anchors: int,
    run_gh: Callable[[list[str]], tuple[int, str]],
    log: Callable[[str], None] = print,
) -> int:
    """One bounded anchor pass. Returns a process exit code."""
    records = read_records(chain_path)
    if not records:
        log(json.dumps({"result": "no-op", "reason": "empty chain", "chain": str(chain_path)}))
        return EXIT_OK

    ok, broken_seq = verify_chain(chain_path)
    if not ok:
        log(
            json.dumps(
                {
                    "result": "fail-closed",
                    "reason": f"chain verification failed at seq={broken_seq}",
                    "chain": str(chain_path),
                }
            )
        )
        return EXIT_FAILURE

    head_hash = chain_head_hash(chain_path)
    seq = int(records[-1]["seq"])
    if head_hash is None:  # unreachable after verify, kept fail-closed
        return EXIT_FAILURE

    try:
        sha = resolve_main_head(repo, run_gh)
    except RuntimeError as exc:
        log(json.dumps({"result": "fail-closed", "reason": str(exc)}))
        return EXIT_FAILURE

    status_args = build_status_args(repo, sha, seq, head_hash)
    plan = {
        "mode": "apply" if apply else "dry-run",
        "chain": str(chain_path),
        "seq": seq,
        "head": head_hash,
        "anchor_target": {"repo": repo, "sha": sha, "context": ANCHOR_CONTEXT},
        "gh_args": status_args,
    }
    log(json.dumps(plan))

    # Mutation budget: a single sequential pass, decremented before each
    # external write. This is a per-invocation cap (the singleton-cadence
    # caller owns cross-invocation bounding); there is no shared state to
    # race — one process, one pass, no loops.
    anchors_remaining = max_anchors

    if apply:
        if anchors_remaining <= 0:
            log(json.dumps({"result": "fail-closed", "reason": "max-anchors exhausted"}))
            return EXIT_FAILURE
        anchors_remaining -= 1
        rc, out = run_gh(status_args)
        if rc != 0:
            log(json.dumps({"result": "fail-closed", "reason": f"status post failed: {out[:200]}"}))
            return EXIT_FAILURE
        log(json.dumps({"result": "anchored", "seq": seq, "head": head_hash[:12]}))

    if rekor:
        if apply and anchors_remaining <= 0:
            log(json.dumps({"rekor": "skipped", "reason": "max-anchors exhausted"}))
        else:
            if apply:
                anchors_remaining -= 1
            if not submit_rekor(head_hash, apply=apply, log=log):
                return EXIT_FAILURE

    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--chain",
        default=str(default_chain_path()),
        help="Path to the intent-chain JSONL working copy",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repo owner/name")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Post the anchor (default: dry-run plan only — nothing leaves the machine)",
    )
    parser.add_argument(
        "--rekor",
        action="store_true",
        help="Additionally anchor to Sigstore Rekor when a rekor-cli binary exists "
        "(silently degrades to commit-status-only otherwise)",
    )
    parser.add_argument(
        "--max-anchors",
        type=int,
        default=2,
        help="Maximum external mutations per invocation (default: 2)",
    )
    args = parser.parse_args(argv)
    try:
        return run_anchor(
            chain_path=Path(args.chain),
            repo=args.repo,
            apply=args.apply,
            rekor=args.rekor,
            max_anchors=max(0, args.max_anchors),
            run_gh=default_run_gh,
        )
    except Exception as exc:  # noqa: BLE001 - top-level fail-closed boundary
        print(json.dumps({"result": "fail-closed", "reason": str(exc)[:300]}))
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
