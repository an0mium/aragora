"""PR intelligence brief — receipt + settlement linkage schema.

Type contracts for the receipt extension described in
docs/plans/2026-04-19-pr-intelligence-brief.md (#6307).

This module is **schema only**. No I/O, no hashing, no orchestration.
Behavior — brief-receipt writing, repair-receipt linking, export —
ships in successor PRs (the #6306 orchestrator, #6304 UI, #6305
policy). This module just defines the shapes those PRs agree on.

Composes, but does not redefine:
  - aragora.review.protocol.ReviewBrief — wrapped by ``BriefReceipt``
  - aragora.cli.commands.review_queue.SettlementReceipt — referenced by
    path in ``SettlementLinkage`` (keeps backwards compatibility with
    existing settlement receipts already on disk)

Acceptance-criteria linkage (from #6307 body):
  - "A settled PR can be traced back to the exact brief packet and
    head SHA"  →  BriefReceipt.brief.packet_sha + SettlementLinkage.head_sha
  - "Dissent survives in receipts instead of being collapsed into one
    summary line"  →  BriefReceipt.brief.dissent (tuple of DissentingView)
  - "Receipt payload is stable enough for operator UI and external
    export"  →  frozen dataclasses + tuple sequences + deterministic
    receipt_id preimage (documented on BriefReceipt)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from aragora.review.protocol import ReviewBrief


# --- Evidence and validation references -----------------------------------


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """A pointer to a piece of supporting evidence cited by the brief.

    Deliberately short: ``quote`` is 1-2 lines, not a full paragraph.
    Deep evidence lives where the ref points; the brief just names the
    location so the operator (or UI) can expand on demand.
    """

    kind: str  # "file" | "test" | "commit" | "artifact" | "issue" | "pr" | "external"
    path: str  # for file/test: repo-relative path; for commit/pr/issue: canonical ref; for external: URL
    sha: str = ""  # git SHA or artifact hash where applicable
    line_range: tuple[int, int] | None = None
    quote: str = ""  # short excerpt (≤2 lines recommended)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.line_range is not None:
            d["line_range"] = list(self.line_range)
        return d


@dataclass(frozen=True, slots=True)
class ValidationRef:
    """A pointer to an automated check whose result backs the brief.

    Different from ``EvidenceRef`` because validation refs have a
    pass/fail outcome and link to a live run (not a static artifact).
    """

    kind: str  # "ci_check" | "test_suite" | "receipt" | "benchmark" | "manual_review"
    name: str  # human-readable check name (e.g. "Version Alignment", "test-fast (server)")
    result: str  # "success" | "failure" | "skipped" | "cancelled" | "pending"
    url: str = ""  # link to the live run or artifact

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- Brief receipt --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BriefReceipt:
    """Persisted, SHA-bound form of a ReviewBrief.

    Wraps a ``ReviewBrief`` (which already carries agent_roster, dissent,
    per-finding cost, overall_confidence, disagreement_score, and a
    ``packet_sha`` preimage spec) and adds the evidence + validation
    references that the raw brief does not yet carry.

    Dissent survival: because ``ReviewBrief.dissent`` is already a
    ``tuple[DissentingView, ...]`` with a frozen shape, the dissent array
    is preserved verbatim in this receipt. There is no summarization step
    that would collapse multi-agent disagreement into one line.

    Receipt-ID preimage (deterministic, #6305/#6304 can rely on this):
      1. Call ``BriefReceipt.to_dict()``.
      2. Remove the ``"receipt_id"`` key from the result.
      3. Serialize the remainder as canonical JSON:
         ``json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)``.
      4. UTF-8 encode, take ``hashlib.sha256(bytes).hexdigest()``.
      Rule lives in the docstring (not in code) because the schema module
      is intentionally behavior-free; the orchestrator (#6306 successor)
      implements the hash and holds it under test.
    """

    brief: ReviewBrief
    evidence_refs: tuple[EvidenceRef, ...]
    validation_refs: tuple[ValidationRef, ...]
    receipt_id: str  # SHA-256 over to_dict() minus this key; see docstring
    created_at: str  # ISO-8601 with timezone
    advisory_only: bool = True
    settlement_note: str = (
        "This receipt records an advisory brief. It does not approve or block merge. "
        "Human settlement required."
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["brief"] = self.brief.to_dict()
        d["evidence_refs"] = [e.to_dict() for e in self.evidence_refs]
        d["validation_refs"] = [v.to_dict() for v in self.validation_refs]
        return d


# --- Settlement linkage ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class SettlementLinkage:
    """Binds one brief receipt to its settlement and any later repair attempts.

    Preserves the full audit trail required by #6307 acceptance:
      brief (advisory machine output)
        → settlement (human action, keyed by head_sha + packet_sha)
          → repair attempts (if settlement was request_changes)

    Kept as a separate dataclass so a settlement without a pre-existing
    brief (e.g. legacy PR settled via the pre-#6306 review-queue loop)
    does not force consumers to synthesize a fake brief receipt.

    Storage handoff: the existing SettlementReceipt on disk is referenced
    by path (``settlement_receipt_path``), not by re-embedding its
    fields. That keeps backwards compatibility with settlement receipts
    already on disk from the pre-#6307 review-queue loop. Consumers that
    need the full settlement payload read the file at that path.
    """

    brief_receipt_id: str  # BriefReceipt.receipt_id; empty if no prior brief
    settlement_receipt_path: str  # filesystem path to existing SettlementReceipt JSON
    head_sha: str  # duplicated from settlement for fast-lookup / cross-validation
    packet_sha: str  # duplicated from brief for fast-lookup / cross-validation
    pr_number: int
    repo: str  # owner/name
    action: str  # "approve" | "request_changes" | "defer"
    settled_at: str  # ISO-8601 with timezone
    repair_receipt_paths: tuple[str, ...] = ()  # filesystem paths to repair receipts, growing
    advisory_only: bool = False  # settlement is a human action; not advisory

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["repair_receipt_paths"] = list(self.repair_receipt_paths)
        return d
