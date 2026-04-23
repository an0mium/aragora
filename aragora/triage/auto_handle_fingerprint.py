"""Pure fingerprinting helpers for auto-handle calibration (#6372).

This module carries the domain-logic half of the original
:mod:`aragora.triage.auto_handle_calibration`: decision-class fingerprints
and the stable SHA-256-derived decision id. None of the helpers touch
disk, the network, or any SQLite schema — they are fully pure and can be
unit-tested in isolation.

Keeping these functions out of the SQLite store module lets reviewers
reason about persistence and fingerprinting independently, which is what
the 8/8 Mode 3 panel on PR #6468 asked for. The store module re-imports
the public symbols so existing callers at ``aragora.triage`` see no API
change.
"""

from __future__ import annotations

import hashlib

__all__ = [
    "AUTO_HANDLE_PATH_ADMIN_MERGE_ALLOWED",
    "AUTO_HANDLE_PATH_FIRE_AND_FORGET",
    "auto_handle_decision_id",
    "bucket_count",
    "classify_scope",
    "fingerprint_admin_merge_class",
    "fingerprint_low_risk_class",
]


# ---------------------------------------------------------------------------
# Auto-handle path identifiers
# ---------------------------------------------------------------------------


#: The ``fire_and_forget`` low-risk-merge auto-handle path in
#: :mod:`aragora.swarm.tranche_integrate`.
AUTO_HANDLE_PATH_FIRE_AND_FORGET = "fire_and_forget"

#: The ``admin_merge_allowed`` review-gate-bypass auto-handle path in
#: :mod:`aragora.ralph.supervisor`.
AUTO_HANDLE_PATH_ADMIN_MERGE_ALLOWED = "admin_merge_allowed"


# ---------------------------------------------------------------------------
# Fingerprinting helpers (pure, no I/O)
# ---------------------------------------------------------------------------


def bucket_count(value: int) -> str:
    """Return a coarse-grained label for ``value`` suitable for fingerprints.

    Buckets compress continuous integer inputs (file counts, lane counts,
    required-check counts) into a small vocabulary so calibration classes
    remain stable across small fluctuations.
    """
    if value <= 1:
        return "1"
    if value <= 3:
        return "2-3"
    if value <= 6:
        return "4-6"
    return "7+"


def classify_scope(paths: list[str]) -> str:
    """Summarise a changed-files list into its top-level directory roots.

    Returns a deterministic ``+``-joined token of up to three distinct
    roots (``aragora``, ``tests``, ``docs``, ...); ``+more`` is appended
    when the PR touches more than three roots. Empty path lists return
    ``"unknown"`` so the fingerprint slot is never the empty string.
    """
    roots: list[str] = []
    for raw in paths:
        text = str(raw or "").strip().strip("/")
        if not text:
            continue
        head = text.split("/", 1)[0]
        roots.append(head or "root")
    unique = sorted(dict.fromkeys(roots))
    if not unique:
        return "unknown"
    return "+".join(unique[:3]) + ("+more" if len(unique) > 3 else "")


def fingerprint_low_risk_class(
    *,
    changed_files: list[str],
    review_tier: int | None,
    lane_count: int,
) -> str:
    """Fingerprint a ``fire_and_forget`` low-risk merge decision.

    Combines review tier, lane-count bucket, file-count bucket, and the
    scope roots into a stable string used as the ``decision_class``
    column in the auto-handle calibration store.
    """
    return (
        f"tier={review_tier if review_tier is not None else 'unknown'}"
        f"|lanes={bucket_count(max(lane_count, 0))}"
        f"|files={bucket_count(len(changed_files))}"
        f"|scope={classify_scope(changed_files)}"
    )


def fingerprint_admin_merge_class(
    *,
    base_branch: str | None,
    required_checks_count: int,
    target_kind: str | None,
) -> str:
    """Fingerprint an ``admin_merge_allowed`` review-gate-bypass decision.

    Combines base branch, bucketed required-check count, and target kind
    (``fleet`` / ``user`` / ``unknown``). Defaults to ``"unknown"`` when
    any discriminator is missing to keep the slot populated.
    """
    return (
        f"base={str(base_branch or 'unknown').strip() or 'unknown'}"
        f"|checks={bucket_count(max(required_checks_count, 0))}"
        f"|target={str(target_kind or 'unknown').strip() or 'unknown'}"
    )


def auto_handle_decision_id(
    *,
    auto_handle_path: str,
    pr_url: str,
    decision_class: str,
) -> str:
    """Return a deterministic id for an auto-handle decision.

    The id is a prefix-disambiguated 24-char SHA-256 digest over the
    tuple ``(auto_handle_path, pr_url, decision_class)``. Stability lets
    callers de-dupe retries and the store enforce ``ON CONFLICT DO NOTHING``
    on ``decision_id``.
    """
    payload = "\x1f".join(
        (
            str(auto_handle_path or "").strip(),
            str(pr_url or "").strip(),
            str(decision_class or "").strip(),
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"{str(auto_handle_path or '').strip()}:{digest}"
