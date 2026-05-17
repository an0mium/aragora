#!/usr/bin/env python3
"""Read-only four-bucket PR triage classifier.

Implements ``docs/governance/OPERATOR_DELEGATION_POLICY.md`` (Stage 1
of the rollout per ``docs/roadmap/OPERATOR_DELEGATION_ROLLOUT.md``).

For every open PR on the current repo, classifies it into exactly one
of four buckets:

  A — recommend auto-merge (NOT DRAFT + CLEAN/review-only merge state +
      green CI + mergeable + tests + no held/protected touch + ≤1500 LOC +
      trusted author)
  B — recommend auto-close (superseded by newer / >60d stale +
      inactive / CI red >7d)
  C — needs operator y/n (touches held / protected / large diff /
      CI red / CI pending / non-trusted author / unresolved review /
      no tests with code changes / merge-state not clean or authorized /
      flag/label/external-dependency tripwire / draft)
  D — strategic check-in (PR works but plausibly conflicts with
      canonical direction; not auto-classifiable from gh metadata
      alone — reserved for future enhancement)

The classifier is read-only by design: it NEVER mutates GitHub state.
The downstream Stage 2 (``scripts/auto_merge_bucket_a.py``) is what
acts on Bucket A; this script only emits the recommendation table.
Bucket A is still exact-head gated here: otherwise-eligible candidates
are demoted to Bucket C unless ``aragora review-queue merge-packet
--pr N --json`` reports ``admin_squash_allowed=true``, ``not_ready=[]``,
``unresolved_dissent=false`` at the current head SHA. Tier 3 and Tier 4
PRs remain Bucket C unless that merge packet proves the required human
risk settlement / preapproval has already been recorded.

Bucket B supersede is similarly exact-head gated: a candidate
superseder is only accepted if it would itself qualify for Bucket A
(``_would_qualify_for_bucket_a``). This implements the policy's
"newer PR is in Bucket A or already merged" requirement — a draft /
held / CI-red / large / non-trusted / merge-packet-blocked candidate
cannot supersede an older PR even if file overlap is high.

Pure stdlib (argparse, dataclasses, datetime, json, shutil,
subprocess, sys, pathlib, typing). No ``aragora.*`` imports. No
third-party dependencies.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Policy constants — keep in sync with docs/governance/OPERATOR_DELEGATION_POLICY.md
# ---------------------------------------------------------------------------

# Held PRs (from the policy's canonical hold list). Whoever updates the
# policy doc's hold list MUST update both this set and the equivalent in
# ``scripts/apply_operator_decisions.py``.
HELD_PR_NUMBERS: frozenset[int] = frozenset({4990, 7173, 7215, 7240, 7243, 7245, 7249, 7252})

# Protected file paths — from the policy's irreducible tripwire list.
PROTECTED_PATHS: frozenset[str] = frozenset(
    {
        "CLAUDE.md",
        "aragora/__init__.py",
        ".env",
        ".envrc",
        "scripts/nomic_loop.py",
        "docs/AGENT_OPERATING_CONTRACT.md",
        "automation.toml",
    }
)

# Trusted authors — Bucket A is gated on author membership here. Adding
# a new entry is itself an operator-only tripwire (the policy doc names
# this explicitly).
TRUSTED_AUTHORS: frozenset[str] = frozenset({"an0mium"})

# Labels that cannot be added by a Bucket-A PR without an operator look.
OPERATOR_ONLY_LABELS: frozenset[str] = frozenset({"boss-ready", "autonomous"})

# Dependency / integration manifest edits are a conservative proxy for the
# policy's "new external dependency / network call / secret read" tripwire.
EXTERNAL_DEPENDENCY_PATHS: frozenset[str] = frozenset(
    {
        "package.json",
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "bun.lockb",
        "pyproject.toml",
        "poetry.lock",
        "uv.lock",
        "requirements.txt",
        "requirements-dev.txt",
        "Pipfile",
        "Pipfile.lock",
        "Cargo.toml",
        "Cargo.lock",
        "go.mod",
        "go.sum",
        "Gemfile",
        "Gemfile.lock",
    }
)

# Bucket A diff-size cap. PRs above this go to Bucket C regardless of
# other criteria — large diffs trip more invariants than the
# metadata-only classifier can verify.
LARGE_DIFF_LOC = 1500

# Auto-close thresholds (days). Both must hold for the stale-draft
# Bucket B path.
STALE_AGE_DAYS = 60
STALE_INACTIVITY_DAYS = 30

# CI-red threshold (days) for the auto-close path.
CI_RED_THRESHOLD_DAYS = 7

# File-overlap threshold for the supersede path. The policy doc names 0.8.
SUPERSEDE_OVERLAP_THRESHOLD = 0.80

BUCKET_A = "A"
BUCKET_B = "B"
BUCKET_C = "C"
BUCKET_D = "D"  # currently never auto-emitted; reserved (see module docstring)

_BUCKET_LABELS: dict[str, str] = {
    BUCKET_A: "BUCKET A — recommend AUTO-MERGE",
    BUCKET_B: "BUCKET B — recommend AUTO-CLOSE",
    BUCKET_C: "BUCKET C — needs operator y/n",
    BUCKET_D: "BUCKET D — strategic check-in",
}


# ---------------------------------------------------------------------------
# Result + classification
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ClassificationResult:
    """One PR's classification. Bucket + ≤120-char justification + the
    recommended human action (MERGE / CLOSE / DEFER / DECIDE / STAY HELD).
    """

    pr_number: int
    bucket: str
    reason: str
    title: str
    recommended_action: str


MergePacketProvider = Callable[[int], dict[str, Any] | None]


def _result(
    pr_number: int,
    title: str,
    bucket: str,
    reason: str,
    recommended_action: str,
) -> ClassificationResult:
    return ClassificationResult(
        pr_number=pr_number,
        bucket=bucket,
        reason=reason[:200],  # cap reason length defensively
        title=title[:80],
        recommended_action=recommended_action,
    )


def _is_protected(path: str) -> bool:
    return path in PROTECTED_PATHS


def _is_test_file(path: str) -> bool:
    if path.startswith("tests/"):
        return True
    if "/__tests__/" in path:
        return True
    if path.endswith((".test.tsx", ".test.ts", ".test.jsx", ".test.js")):
        return True
    if path.endswith("_test.py"):
        return True
    return False


def _is_code_file(path: str) -> bool:
    if _is_test_file(path):
        return False
    if path.endswith((".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs")):
        return True
    return False


def _parse_age_days(iso: str, now: datetime.datetime) -> int:
    """Days between ``iso`` and ``now``; 0 on parse failure or empty."""

    if not iso:
        return 0
    try:
        dt = datetime.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return 0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return max(0, (now - dt).days)


def _file_paths(pr: dict[str, Any]) -> list[str]:
    files = pr.get("files") or []
    return [str(f.get("path", "")) for f in files if isinstance(f, dict) and f.get("path")]


def _labels(pr: dict[str, Any]) -> set[str]:
    raw = pr.get("labels") or []
    out: set[str] = set()
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, dict):
            name = str(item.get("name") or "")
        else:
            name = str(item)
        if name:
            out.add(name)
    return out


def _truthy_policy_field(pr: dict[str, Any], *names: str) -> bool:
    for name in names:
        if bool(pr.get(name)):
            return True
    tripwires = pr.get("policyTripwires") or pr.get("policy_tripwires") or []
    if not isinstance(tripwires, list):
        return False
    normalized = {str(item).strip().lower().replace("-", "_") for item in tripwires}
    return any(name.strip().lower().replace("-", "_") in normalized for name in names)


def _flag_or_label_tripwire(pr: dict[str, Any]) -> str | None:
    if _truthy_policy_field(
        pr,
        "flagFlip",
        "flag_flip",
        "hasFlagFlip",
        "addsAutonomousLabel",
        "label_add",
        "boss_ready",
        "autonomous_label",
    ):
        return "flag flip / operator-only label tripwire"

    blocked = sorted(label for label in _labels(pr) if label in OPERATOR_ONLY_LABELS)
    if blocked:
        return f"operator-only label present ({blocked[0]})"
    return None


def _external_dependency_tripwire(pr: dict[str, Any], file_paths: list[str]) -> str | None:
    if _truthy_policy_field(
        pr,
        "externalDependency",
        "external_dependency",
        "networkCall",
        "network_call",
        "secretRead",
        "secret_read",
        "new_external_dependency",
    ):
        return "external dependency / network / secret-read tripwire"

    for path in file_paths:
        name = Path(path).name
        if path in EXTERNAL_DEPENDENCY_PATHS or name in EXTERNAL_DEPENDENCY_PATHS:
            return f"external dependency manifest touched ({path})"
        parts = set(Path(path).parts)
        if "secrets" in parts or "credentials" in parts:
            return f"secret/credential path touched ({path})"
    return None


def _unresolved_review_tripwire(pr: dict[str, Any]) -> str | None:
    count = pr.get("unresolvedReviewComments") or pr.get("unresolved_review_comments")
    if isinstance(count, int) and count > 0:
        return f"unresolved review comments ({count})"
    if bool(count):
        return "unresolved review comments"
    if _truthy_policy_field(pr, "unresolvedReview", "unresolved_review"):
        return "unresolved review comments"

    threads = pr.get("reviewThreads") or pr.get("review_threads") or []
    if isinstance(threads, list):
        unresolved = [t for t in threads if isinstance(t, dict) and t.get("isResolved") is False]
        if unresolved:
            return f"unresolved review comments ({len(unresolved)})"
    return None


def _merge_state_bucket_a_blocker(
    pr: dict[str, Any],
    *,
    merge_packet_provider: MergePacketProvider | None,
) -> str | None:
    """Return None if the merge-state gate allows Bucket A.

    #7283 permits either CLEAN or the common branch-protection state where
    GitHub reports BLOCKED only because review is required while Aragora's
    exact-head merge packet says admin squash is otherwise allowed.
    """

    mss = str(pr.get("mergeStateStatus") or "")
    if mss == "CLEAN":
        return None
    review = str(pr.get("reviewDecision") or "")
    if mss == "BLOCKED" and review == "REVIEW_REQUIRED":
        packet_blocker = _merge_packet_bucket_a_blocker(
            pr, merge_packet_provider=merge_packet_provider
        )
        if packet_blocker is None:
            return None
        return (
            "merge state status: BLOCKED but review-only/admin-squash "
            f"exception not authorized ({packet_blocker})"
        )
    return f"merge state status: {mss or '(unknown)'} (policy requires CLEAN or review-only admin-squash authorization)"


def _would_qualify_for_bucket_a(
    pr: dict[str, Any],
    *,
    now: datetime.datetime,
    merge_packet_provider: MergePacketProvider | None,
) -> bool:
    """Return True iff ``pr`` would land in Bucket A on its own merits.

    Mirrors the gates ``classify()`` applies, *minus* the supersede
    check itself (to avoid mutual recursion between supersede candidates
    that overlap heavily). The supersede precedence rule already
    requires the superseder to be NEWER (higher PR number), so this
    predicate is only consulted on the newer side — no infinite
    descent is possible.

    Used by ``_find_superseder`` to enforce the policy requirement
    that "the newer PR is in Bucket A or already merged" (we can't
    see merged PRs from the open list, so we treat Bucket-A-eligible
    open PRs as the safe proxy).
    """

    n = int(pr.get("number") or 0)
    if n <= 0:
        return False
    if n in HELD_PR_NUMBERS:
        return False
    file_paths = _file_paths(pr)
    if any(_is_protected(p) for p in file_paths):
        return False
    if _flag_or_label_tripwire(pr) is not None:
        return False
    if _external_dependency_tripwire(pr, file_paths) is not None:
        return False
    additions = int(pr.get("additions") or 0)
    deletions = int(pr.get("deletions") or 0)
    if additions + deletions > LARGE_DIFF_LOC:
        return False
    checks = pr.get("statusCheckRollup") or []
    if any(isinstance(c, dict) and c.get("conclusion") == "FAILURE" for c in checks):
        return False
    if any(isinstance(c, dict) and c.get("status") in ("IN_PROGRESS", "QUEUED") for c in checks):
        return False
    if any(
        isinstance(c, dict)
        and c.get("status") == "COMPLETED"
        and c.get("conclusion") not in ("SUCCESS", "SKIPPED", "NEUTRAL")
        for c in checks
    ):
        return False
    author_raw = pr.get("author") or {}
    author = author_raw.get("login", "") if isinstance(author_raw, dict) else str(author_raw)
    if author not in TRUSTED_AUTHORS:
        return False
    if bool(pr.get("isDraft")):
        return False
    if str(pr.get("mergeable") or "") != "MERGEABLE":
        return False
    if _merge_state_bucket_a_blocker(pr, merge_packet_provider=merge_packet_provider) is not None:
        return False
    has_tests = any(_is_test_file(p) for p in file_paths)
    has_code = any(_is_code_file(p) for p in file_paths)
    if has_code and not has_tests:
        return False
    if str(pr.get("reviewDecision") or "") == "CHANGES_REQUESTED":
        return False
    if _unresolved_review_tripwire(pr) is not None:
        return False
    if _merge_packet_bucket_a_blocker(pr, merge_packet_provider=merge_packet_provider) is not None:
        return False
    # The ``now`` parameter is accepted for API symmetry with classify(),
    # but the Bucket-A gates above do not depend on age (those are only
    # used by the Bucket B stale-draft path, which is not a Bucket A
    # criterion).
    del now
    return True


def _find_superseder(
    pr: dict[str, Any],
    all_open: list[dict[str, Any]],
    *,
    now: datetime.datetime,
    merge_packet_provider: MergePacketProvider | None,
) -> int | None:
    """Return the PR number of a newer open PR that supersedes ``pr``.

    Supersede requires ALL of:
      - newer (higher PR number)
      - file-overlap ≥ ``SUPERSEDE_OVERLAP_THRESHOLD``
      - the newer PR would itself qualify for Bucket A
        (per ``_would_qualify_for_bucket_a``)

    The Bucket-A-eligibility gate is the policy's "newer PR is in
    Bucket A or already merged" requirement: a draft / held / CI-red /
    CI-pending / large / non-trusted / merge-packet-blocked candidate
    cannot supersede an older PR. (We cannot see merged PRs from the
    open list, so we treat Bucket-A-eligible open PRs as the safe
    proxy for "or already merged.")
    """

    pr_n = int(pr.get("number", 0))
    if pr_n <= 0:
        return None
    pr_files = set(_file_paths(pr))
    if not pr_files:
        return None
    for other in all_open:
        other_n = int(other.get("number", 0) or 0)
        if other_n <= pr_n:
            continue
        other_files = set(_file_paths(other))
        if not other_files:
            continue
        overlap_count = len(pr_files & other_files)
        overlap_ratio = overlap_count / len(pr_files)
        if overlap_ratio < SUPERSEDE_OVERLAP_THRESHOLD:
            continue
        if not _would_qualify_for_bucket_a(
            other, now=now, merge_packet_provider=merge_packet_provider
        ):
            continue
        return other_n
    return None


def _first_merge_packet_entry(packet: dict[str, Any], pr_number: int) -> dict[str, Any] | None:
    entries = packet.get("entries")
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if int(entry.get("pr_number") or 0) == pr_number:
            return entry
    if len(entries) == 1 and isinstance(entries[0], dict):
        return entries[0]
    return None


def _load_merge_packet(pr_number: int) -> dict[str, Any] | None:
    cmd = [
        sys.executable,
        "-m",
        "aragora.cli.main",
        "review-queue",
        "merge-packet",
        "--pr",
        str(pr_number),
        "--json",
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or str(exc)
        return {"_error": stderr.strip()[:300] or "merge-packet command failed"}
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {"_error": f"merge-packet returned non-JSON: {exc}"}
    if not isinstance(data, dict):
        return {"_error": "merge-packet returned non-object JSON"}
    return data


def _embedded_merge_packet(pr: dict[str, Any]) -> dict[str, Any] | None:
    packet = pr.get("mergePacket")
    return packet if isinstance(packet, dict) else None


def _merge_packet_bucket_a_blocker(
    pr: dict[str, Any],
    *,
    merge_packet_provider: MergePacketProvider | None,
) -> str | None:
    """Return ``None`` only when the exact-head merge packet authorizes Bucket A."""

    pr_number = int(pr.get("number") or 0)
    head_sha = str(pr.get("headRefOid") or "")
    packet = _embedded_merge_packet(pr)
    if packet is None and merge_packet_provider is not None:
        packet = merge_packet_provider(pr_number)
    if packet is None:
        return "merge-packet not checked (required for Bucket A)"
    if packet.get("_error"):
        return f"merge-packet failed: {packet['_error']}"

    not_ready = packet.get("not_ready")
    if not_ready != []:
        return f"merge-packet not_ready={not_ready!r}"

    entry = _first_merge_packet_entry(packet, pr_number)
    if entry is None:
        return "merge-packet missing PR entry"

    packet_head = str(entry.get("head_sha") or "")
    if not head_sha:
        return "PR headRefOid missing; cannot verify exact head"
    if packet_head != head_sha:
        return "merge-packet head mismatch"

    if entry.get("admin_squash_allowed") is not True:
        return "merge-packet admin_squash_allowed is not true"
    if entry.get("unresolved_dissent") is not False:
        return "merge-packet unresolved_dissent is not false"
    try:
        tier = int(entry.get("tier") or 0)
    except (TypeError, ValueError):
        return "merge-packet tier is invalid"
    if tier >= 3 and entry.get("requires_human_risk_settlement") is not False:
        return "Tier 3/4 PR lacks recorded human settlement/preapproval"
    return None


def classify(
    pr: dict[str, Any],
    all_open: list[dict[str, Any]],
    *,
    now: datetime.datetime | None = None,
    merge_packet_provider: MergePacketProvider | None = None,
) -> ClassificationResult:
    """Run the four-bucket classification on one PR.

    Bucket precedence (most-restrictive wins): C (held) → C (protected) →
    C (large) → B (CI red 7d+) → C (CI red recent) → C (CI pending) →
    C (CI non-green) → B (stale draft) → B (superseded by a Bucket-A
    -eligible newer PR) → C (non-trusted) → C (draft) → C (not
    mergeable) → C (merge-state not CLEAN/authorized) → C (code
    without tests) → C (CHANGES_REQUESTED / unresolved review) →
    C (merge-packet blocker) → A.

    Bucket D is never auto-emitted by this classifier — it's reserved
    for explicit operator/agent escalation in a future stage.
    """

    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)

    n = int(pr.get("number") or 0)
    title = str(pr.get("title") or "")
    author_raw = pr.get("author") or {}
    author = author_raw.get("login", "") if isinstance(author_raw, dict) else str(author_raw)
    file_paths = _file_paths(pr)
    additions = int(pr.get("additions") or 0)
    deletions = int(pr.get("deletions") or 0)
    net_loc = additions + deletions
    mergeable = str(pr.get("mergeable") or "")
    is_draft = bool(pr.get("isDraft"))
    checks = pr.get("statusCheckRollup") or []
    ci_success = sum(1 for c in checks if isinstance(c, dict) and c.get("conclusion") == "SUCCESS")
    ci_failure = sum(1 for c in checks if isinstance(c, dict) and c.get("conclusion") == "FAILURE")
    ci_pending = sum(
        1 for c in checks if isinstance(c, dict) and c.get("status") in ("IN_PROGRESS", "QUEUED")
    )
    ci_non_green = [
        str(c.get("conclusion") or "(missing)")
        for c in checks
        if isinstance(c, dict)
        and c.get("status") == "COMPLETED"
        and c.get("conclusion") not in ("SUCCESS", "SKIPPED", "NEUTRAL")
    ]
    ci_total = len(checks)
    age_days = _parse_age_days(str(pr.get("createdAt") or ""), now)
    updated_days = _parse_age_days(str(pr.get("updatedAt") or ""), now)
    review = str(pr.get("reviewDecision") or "")

    # --- Bucket C: held PR ---
    if n in HELD_PR_NUMBERS:
        return _result(n, title, BUCKET_C, f"held (#{n} is on the policy hold list)", "STAY HELD")

    # --- Bucket C: protected file edits ---
    protected_hit = [p for p in file_paths if _is_protected(p)]
    if protected_hit:
        return _result(
            n,
            title,
            BUCKET_C,
            f"edits protected file ({protected_hit[0]})",
            "DECIDE",
        )

    # --- Bucket C: operator-only flag/label tripwire ---
    flag_tripwire = _flag_or_label_tripwire(pr)
    if flag_tripwire is not None:
        return _result(n, title, BUCKET_C, flag_tripwire, "DECIDE")

    # --- Bucket C: external dependency / network / secret-read tripwire ---
    external_tripwire = _external_dependency_tripwire(pr, file_paths)
    if external_tripwire is not None:
        return _result(n, title, BUCKET_C, external_tripwire, "DECIDE")

    # --- Bucket C: large diff ---
    if net_loc > LARGE_DIFF_LOC:
        return _result(
            n,
            title,
            BUCKET_C,
            f"large diff ({net_loc} LOC > {LARGE_DIFF_LOC})",
            "DECIDE",
        )

    # --- Bucket B: CI red ≥7 days (use updated_days as a proxy for "no
    # recent fix attempts" since the rollup doesn't carry per-check age) ---
    if ci_failure > 0 and updated_days >= CI_RED_THRESHOLD_DAYS:
        return _result(
            n,
            title,
            BUCKET_B,
            (
                f"CI red ≥{CI_RED_THRESHOLD_DAYS}d ({ci_failure} failures, "
                f"{updated_days}d since update)"
            ),
            "CLOSE",
        )

    # --- Bucket C: CI red but recent ---
    if ci_failure > 0:
        return _result(n, title, BUCKET_C, f"CI red ({ci_failure} failures)", "DECIDE")

    # --- Bucket C: CI pending ---
    if ci_pending > 0:
        return _result(
            n,
            title,
            BUCKET_C,
            f"CI pending ({ci_pending} in-flight, {ci_success}/{ci_total} green)",
            "DEFER",
        )

    # --- Bucket C: completed but non-green CI (for example CANCELLED/TIMED_OUT) ---
    if ci_non_green:
        return _result(
            n,
            title,
            BUCKET_C,
            f"CI non-green ({ci_non_green[0]})",
            "DECIDE",
        )

    # --- Bucket B: stale draft ---
    if is_draft and age_days >= STALE_AGE_DAYS and updated_days >= STALE_INACTIVITY_DAYS:
        return _result(
            n,
            title,
            BUCKET_B,
            (
                f"stale draft ({age_days}d old, {updated_days}d inactive — "
                f"thresholds {STALE_AGE_DAYS}/{STALE_INACTIVITY_DAYS})"
            ),
            "CLOSE",
        )

    # --- Bucket B: superseded by newer Bucket-A-eligible PR ---
    superseder = _find_superseder(
        pr,
        all_open,
        now=now,
        merge_packet_provider=merge_packet_provider,
    )
    if superseder is not None:
        return _result(
            n,
            title,
            BUCKET_B,
            (
                f"superseded by #{superseder} "
                f"(≥{int(SUPERSEDE_OVERLAP_THRESHOLD * 100)}% file overlap, "
                f"newer + Bucket-A-eligible)"
            ),
            "CLOSE",
        )

    # --- Bucket C: non-trusted author ---
    if author not in TRUSTED_AUTHORS:
        return _result(
            n,
            title,
            BUCKET_C,
            f"non-trusted author ({author or '(unknown)'})",
            "DECIDE",
        )

    # --- Bucket C: draft (policy explicitly excludes draft from A) ---
    # The policy says "PR is not draft" for Bucket A. Draft PRs that
    # are otherwise clean go to C with recommended action "READY?" —
    # the operator decides whether to mark-ready (which then re-runs
    # the classifier on the next pass).
    if is_draft:
        return _result(
            n,
            title,
            BUCKET_C,
            f"draft (policy requires non-draft for Bucket A; {ci_success}/{ci_total} CI green)",
            "READY?",
        )

    # --- Bucket C: not mergeable ---
    if mergeable != "MERGEABLE":
        return _result(
            n,
            title,
            BUCKET_C,
            f"not mergeable (mergeable={mergeable or '(unknown)'})",
            "DECIDE",
        )

    # --- Bucket C: merge-state status not CLEAN/authorized ---
    merge_state_blocker = _merge_state_bucket_a_blocker(
        pr, merge_packet_provider=merge_packet_provider
    )
    if merge_state_blocker is not None:
        return _result(
            n,
            title,
            BUCKET_C,
            merge_state_blocker,
            "DECIDE",
        )

    # --- Bucket C: code changes without tests ---
    has_tests = any(_is_test_file(p) for p in file_paths)
    has_code = any(_is_code_file(p) for p in file_paths)
    if has_code and not has_tests:
        return _result(
            n,
            title,
            BUCKET_C,
            f"code changes without test files ({len(file_paths)} files touched)",
            "DECIDE",
        )

    # --- Bucket C: changes requested in review ---
    if review == "CHANGES_REQUESTED":
        return _result(
            n,
            title,
            BUCKET_C,
            "review decision: CHANGES_REQUESTED",
            "DECIDE",
        )

    # --- Bucket C: unresolved review comments from another agent ---
    unresolved_review = _unresolved_review_tripwire(pr)
    if unresolved_review is not None:
        return _result(n, title, BUCKET_C, unresolved_review, "DECIDE")

    # --- Bucket C: merge-packet does not authorize exact-head Bucket A ---
    merge_packet_blocker = _merge_packet_bucket_a_blocker(
        pr,
        merge_packet_provider=merge_packet_provider,
    )
    if merge_packet_blocker is not None:
        return _result(
            n,
            title,
            BUCKET_C,
            merge_packet_blocker,
            "DECIDE",
        )

    # --- Bucket A: default if all gates pass, including the exact-head merge packet ---
    return _result(
        n,
        title,
        BUCKET_A,
        (
            f"ready + CLEAN + green CI ({ci_success}/{ci_total}), merge-packet authorized, "
            f"{net_loc} LOC, {len(file_paths)} files, tests present, author={author}"
        ),
        "MERGE",
    )


# ---------------------------------------------------------------------------
# I/O — gh shell-out + output formatting
# ---------------------------------------------------------------------------


_GH_JSON_FIELDS = (
    "number,title,isDraft,author,mergeable,mergeStateStatus,additions,"
    "deletions,changedFiles,createdAt,updatedAt,headRefName,"
    "headRefOid,statusCheckRollup,reviewDecision,labels,files"
)


def fetch_open_prs(*, limit: int = 100) -> list[dict[str, Any]]:
    """Shell out to ``gh pr list`` with the field set the classifier needs."""

    cmd = [
        "gh",
        "pr",
        "list",
        "--state",
        "open",
        "-L",
        str(limit),
        "--json",
        _GH_JSON_FIELDS,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise SystemExit(f"gh pr list failed: {stderr[:300]}") from exc
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"gh pr list returned non-JSON: {exc}") from exc


def _print_human(results: Sequence[ClassificationResult]) -> None:
    by_bucket: dict[str, list[ClassificationResult]] = {
        BUCKET_A: [],
        BUCKET_B: [],
        BUCKET_C: [],
        BUCKET_D: [],
    }
    for r in results:
        by_bucket.setdefault(r.bucket, []).append(r)

    for bucket in (BUCKET_A, BUCKET_B, BUCKET_C, BUCKET_D):
        entries = sorted(by_bucket[bucket], key=lambda r: r.pr_number)
        print(_BUCKET_LABELS[bucket])
        if not entries:
            print("  (none)")
        else:
            for r in entries:
                print(f"  #{r.pr_number} — {r.recommended_action} — {r.reason}")
        print()

    summary = "  ".join(
        f"{b}: {len(by_bucket[b])}" for b in (BUCKET_A, BUCKET_B, BUCKET_C, BUCKET_D)
    )
    print(f"summary: {summary}    total: {len(results)}")


def _print_json(results: Sequence[ClassificationResult]) -> None:
    print(
        json.dumps(
            {
                "policy_doc": ("docs/governance/OPERATOR_DELEGATION_POLICY.md"),
                "rollout_doc": ("docs/roadmap/OPERATOR_DELEGATION_ROLLOUT.md"),
                "results": [
                    dataclasses.asdict(r)
                    for r in sorted(results, key=lambda r: (r.bucket, r.pr_number))
                ],
                "summary": {
                    b: sum(1 for r in results if r.bucket == b)
                    for b in (BUCKET_A, BUCKET_B, BUCKET_C, BUCKET_D)
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triage_open_prs.py",
        description=(
            "Read-only four-bucket PR triage classifier per "
            "docs/governance/OPERATOR_DELEGATION_POLICY.md."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit results as JSON (default: human-readable table).",
    )
    parser.add_argument(
        "--bucket",
        choices=[BUCKET_A, BUCKET_B, BUCKET_C, BUCKET_D],
        help="Filter output to one bucket only.",
    )
    parser.add_argument(
        "--include-held",
        action="store_true",
        default=True,
        help=(
            "Always include held PRs (default: yes for visibility — held "
            "PRs always show as Bucket C with reason 'held')."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max PRs to fetch from gh (default: 100).",
    )
    parser.add_argument(
        "--from-json",
        type=Path,
        help=(
            "Read PR data from a JSON file instead of calling gh (used by tests and offline runs)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.from_json:
        if not args.from_json.exists():
            print(f"ERROR: file not found: {args.from_json}", file=sys.stderr)
            return 2
        try:
            prs = json.loads(args.from_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"ERROR: invalid JSON: {exc}", file=sys.stderr)
            return 2
    else:
        if shutil.which("gh") is None:
            print(
                "ERROR: gh CLI not found on PATH — install gh (https://cli.github.com) and retry.",
                file=sys.stderr,
            )
            return 2
        prs = fetch_open_prs(limit=args.limit)

    if not isinstance(prs, list):
        print("ERROR: PR data must be a JSON array", file=sys.stderr)
        return 2

    if args.limit:
        prs = prs[: args.limit]

    merge_packet_provider = None if args.from_json else _load_merge_packet
    results = [
        classify(pr, prs, merge_packet_provider=merge_packet_provider)
        for pr in prs
        if isinstance(pr, dict)
    ]

    if args.bucket:
        results = [r for r in results if r.bucket == args.bucket]

    if args.json:
        _print_json(results)
    else:
        _print_human(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
