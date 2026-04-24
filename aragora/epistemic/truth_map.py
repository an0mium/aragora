"""Organizational Truth Map report (DIC-18 / #6028).

Read-only operator report aggregating ExecutableClaim verification results
(DIC-14) and optional CruxFinderResult summaries (DIC-15).
Default OFF — callers must explicitly invoke ``build_truth_map`` or
``build_truth_map_from_manifests``.  No queue mutation, no side effects.
"""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aragora.epistemic.claim_verifier import ClaimResult, ClaimStatus

if TYPE_CHECKING:
    from aragora.debate.crux_mode import CruxFinderResult


@dataclass
class ClaimRow:
    claim_id: str
    statement: str
    owner: str
    status: str
    evidence_age_hours: float | None
    verifier_kind: str
    verifier_command: str
    follow_up_link: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CruxSummaryRow:
    debate_id: str
    question: str
    convergence_barrier: float
    top_cruxes: list[dict[str, Any]]
    crux_count: int
    open_cruxes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OrgTruthMapReport:
    """Read-only aggregated truth map for an Aragora deployment."""

    generated_at: str
    claims: list[ClaimRow] = field(default_factory=list)
    crux_summaries: list[CruxSummaryRow] = field(default_factory=list)
    total_claims: int = 0
    passing_claims: int = 0
    failing_claims: int = 0
    stale_claims: int = 0
    unsupported_claims: int = 0
    error_claims: int = 0
    open_crux_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "claims": [c.to_dict() for c in self.claims],
            "crux_summaries": [cs.to_dict() for cs in self.crux_summaries],
            "summary": {
                "total_claims": self.total_claims,
                "passing": self.passing_claims,
                "failing": self.failing_claims,
                "stale": self.stale_claims,
                "unsupported": self.unsupported_claims,
                "error": self.error_claims,
                "open_crux_count": self.open_crux_count,
            },
        }


def build_truth_map(
    *,
    claim_results: list[ClaimResult],
    claim_metadata: dict[str, dict[str, Any]] | None = None,
    crux_results: list[CruxFinderResult] | None = None,
    top_k_cruxes: int = 3,
    open_crux_score_threshold: float = 0.3,
) -> OrgTruthMapReport:
    """Build an OrgTruthMapReport from pre-computed claim and crux inputs."""
    meta = claim_metadata or {}
    rows: list[ClaimRow] = []
    for cr in claim_results:
        m = meta.get(cr.claim_id, {})
        verif: dict[str, Any] = m.get("verification", {})
        rows.append(
            ClaimRow(
                claim_id=cr.claim_id,
                statement=m.get("statement", cr.detail.get("statement", "")),
                owner=m.get("owner", cr.detail.get("owner", "")),
                status=cr.status.value if isinstance(cr.status, ClaimStatus) else str(cr.status),
                evidence_age_hours=cr.detail.get("evidence_age_hours"),
                verifier_kind=verif.get("kind", cr.detail.get("verifier_kind", "")),
                verifier_command=verif.get("command", cr.detail.get("verifier_command", "")),
                follow_up_link=cr.detail.get("follow_up_link", ""),
            )
        )

    counts: dict[ClaimStatus, int] = {s: 0 for s in ClaimStatus}
    for row in rows:
        try:
            counts[ClaimStatus(row.status)] += 1
        except ValueError:
            pass

    crux_rows: list[CruxSummaryRow] = []
    open_crux_total = 0
    for cfr in crux_results or []:
        open_count = sum(
            1 for c in cfr.analysis.cruxes if c.crux_score >= open_crux_score_threshold
        )
        open_crux_total += open_count
        crux_rows.append(
            CruxSummaryRow(
                debate_id=cfr.debate_id,
                question=cfr.question,
                convergence_barrier=cfr.convergence_barrier(),
                top_cruxes=[c.to_dict() for c in cfr.top_cruxes()[:top_k_cruxes]],
                crux_count=len(cfr.analysis.cruxes),
                open_cruxes=open_count,
            )
        )

    return OrgTruthMapReport(
        generated_at=datetime.datetime.utcnow().isoformat() + "Z",
        claims=rows,
        crux_summaries=crux_rows,
        total_claims=len(rows),
        passing_claims=counts[ClaimStatus.PASS],
        failing_claims=counts[ClaimStatus.FAIL],
        stale_claims=counts[ClaimStatus.STALE],
        unsupported_claims=counts[ClaimStatus.UNSUPPORTED],
        error_claims=counts[ClaimStatus.ERROR],
        open_crux_count=open_crux_total,
    )


def build_truth_map_from_manifests(
    *,
    manifest_paths: list[Path],
    repo_root: Path | None = None,
    crux_results: list[CruxFinderResult] | None = None,
    top_k_cruxes: int = 3,
    open_crux_score_threshold: float = 0.3,
    dry_run: bool = True,
) -> OrgTruthMapReport:
    """Load DIC-13 YAML manifests, verify claims, and build a truth map."""
    import yaml  # project-level dep; local import keeps module testable without it

    from aragora.epistemic.claim_verifier import ClaimVerifier

    verifier = ClaimVerifier(repo_root=repo_root, dry_run=dry_run)
    all_results: list[ClaimResult] = []
    all_metadata: dict[str, dict[str, Any]] = {}
    for path in manifest_paths:
        with open(path) as fh:
            manifest = yaml.safe_load(fh)
        for claim in manifest.get("claims", []):
            all_metadata[claim.get("claim_id", "<unknown>")] = claim
        all_results.extend(verifier.verify_manifest(path))

    return build_truth_map(
        claim_results=all_results,
        claim_metadata=all_metadata,
        crux_results=crux_results,
        top_k_cruxes=top_k_cruxes,
        open_crux_score_threshold=open_crux_score_threshold,
    )
