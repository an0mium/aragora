"""
Human-oversight attestation records (#8230, ODR-6).

Extracts the **machine-readable form of what this repository already does
manually for Tier-4 settlements**: who accepted the risk (the oversight
identity, distinct from the execution identity), what they saw (exact head
SHA, digest of the counted evidence), when, and via which mechanism
(``aragora/human-settlement`` commit status, Tier-4 preapproval comment,
local settlement receipt).

Honesty rules (mirroring the ODR profile, ``docs/specs/OPEN_DECISION_RECEIPT.md``):

1. **Never fabricate.** Every field is copied from a real settlement artifact.
2. **Absence is recorded, not implied.** Missing pieces become explicit
   entries in :attr:`OversightAttestationRecord.absences` with a reason.
3. **Autonomous is first-class.** Decisions settled by the model-quorum gate
   with no human in the loop produce an explicit ``autonomous`` disposition
   record — precisely what EU AI Act Article 14 oversight tooling needs to
   mechanically filter "decisions no human ever looked at."

The settlement mechanisms mined here are the live ones defined (read-only
precedent) in ``aragora/cli/commands/review_queue.py``:

- the gate-trusted ``aragora/human-settlement`` commit status, whose
  *creator login* is the oversight identity;
- the Tier-4 preapproval comment ("Tier-4 Human Settlement Authorization"
  marker + exact head SHA + authorized-action token + human-risk acceptance);
- local settlement receipts under ``.aragora/review-queue/receipts/``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from aragora.gauntlet.odr_export import jcs_canonicalize

__all__ = [
    "HUMAN_SETTLEMENT_STATUS_CONTEXT",
    "TIER_FOUR_AUTHORIZED_MERGE_TOKENS",
    "TIER_FOUR_SETTLEMENT_MARKER",
    "OversightAttestationRecord",
    "attestation_from_local_settlement_receipt",
    "attestation_from_settlement_status",
    "attestation_from_tier_four_comment",
    "autonomous_attestation",
    "classify_settled_pr",
    "evidence_digest",
    "find_tier_four_preapproval_comment",
]

# Canonical gate definitions. The source of truth is
# aragora/cli/commands/review_queue.py (read-only here; the merge gate owns
# them). tests/compliance/test_oversight_attestation.py asserts these stay in
# sync so the attestation extractor can never drift from what the gate counts.
HUMAN_SETTLEMENT_STATUS_CONTEXT = "aragora/human-settlement"
TIER_FOUR_SETTLEMENT_MARKER = "Tier-4 Human Settlement Authorization"
TIER_FOUR_AUTHORIZED_MERGE_TOKENS = ("admin_squash_merge", "admin squash")

# Mechanism identifiers (stable strings; consumed by the oversight pack).
MECHANISM_SETTLEMENT_STATUS = "github_settlement_status"
MECHANISM_TIER_FOUR_COMMENT = "tier4_preapproval_comment"
MECHANISM_LOCAL_SETTLEMENT_RECEIPT = "local_settlement_receipt"
MECHANISM_MODEL_QUORUM = "model_quorum_autonomous"

# The literal the ODR profile uses when a source genuinely recorded no
# identity metadata ("never a guess" — OPEN_DECISION_RECEIPT.md section 4.4).
UNDISCLOSED = "undisclosed"


def evidence_digest(items: list[dict[str, Any]]) -> str:
    """SHA-256 hex digest over the JCS bytes of the evidence item list.

    ``items`` are normalized dicts (e.g. ``{"type": ..., "ref": ...,
    "digest": ...}``). The list is sorted by its canonical bytes first so the
    digest is independent of collection order. Deterministic across platforms
    because the basis is RFC 8785 canonicalization.
    """
    ordered = sorted(items, key=lambda item: jcs_canonicalize(item))
    return hashlib.sha256(jcs_canonicalize(ordered)).hexdigest()


@dataclass
class OversightAttestationRecord:
    """Machine-readable human-oversight attestation for one settled decision.

    ``disposition`` is ``"human_attested"`` or ``"autonomous"`` (first-class,
    mirroring ODR section 4.7). All identity/observation fields are extracted
    from real settlement artifacts; anything the artifact does not contain is
    listed in ``absences`` with a reason instead of being invented.
    """

    disposition: str
    mechanism: str
    subject: dict[str, Any] = field(default_factory=dict)
    attestor_id: str | None = None
    attestor_role: str | None = None
    attested_at: str | None = None
    observed: dict[str, Any] = field(default_factory=dict)
    references: list[dict[str, Any]] = field(default_factory=list)
    absences: list[dict[str, str]] = field(default_factory=list)

    def record_absence(self, field_name: str, reason: str) -> None:
        """Record that ``field_name`` is genuinely missing from the source."""
        self.absences.append({"field": field_name, "reason": reason})

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition,
            "mechanism": self.mechanism,
            "subject": dict(self.subject),
            "attestor_id": self.attestor_id,
            "attestor_role": self.attestor_role,
            "attested_at": self.attested_at,
            "observed": dict(self.observed),
            "references": [dict(ref) for ref in self.references],
            "absences": [dict(a) for a in self.absences],
        }

    def to_odr_attestation(self) -> dict[str, Any]:
        """Project onto the ODR v0.1 ``attestation`` block (schema-conformant).

        The ODR schema allows exactly ``disposition``, ``attestor``,
        ``attested_at`` and ``method`` at the top level (additionalProperties
        is false); ``attestor`` accepts additional properties, so the
        observation facts ride there. An unknown attestor identity uses the
        profile's literal ``"undisclosed"`` — never a guess — and the absence
        stays visible in the full record.
        """
        if self.disposition != "human_attested":
            block: dict[str, Any] = {"disposition": "autonomous"}
            if self.mechanism:
                block["method"] = self.mechanism
            return block

        attestor: dict[str, Any] = {
            "id": self.attestor_id or UNDISCLOSED,
        }
        if self.attestor_role:
            attestor["role"] = self.attestor_role
        head_sha = self.observed.get("head_sha")
        if head_sha:
            attestor["observed_head_sha"] = head_sha
        observed_digest = self.observed.get("evidence_digest")
        if observed_digest:
            attestor["observed_evidence_digest"] = observed_digest

        block = {
            "disposition": "human_attested",
            "attestor": attestor,
            "method": self.mechanism,
        }
        if self.attested_at:
            block["attested_at"] = self.attested_at
        return block


def _subject(repo: str | None, pr_number: int | None, head_sha: str | None) -> dict[str, Any]:
    subject: dict[str, Any] = {}
    if repo:
        subject["repo"] = repo
    if pr_number is not None:
        subject["pr_number"] = int(pr_number)
    if head_sha:
        subject["head_sha"] = head_sha
    return subject


def attestation_from_settlement_status(
    status: dict[str, Any],
    *,
    repo: str | None = None,
    pr_number: int | None = None,
    head_sha: str | None = None,
    evidence_items: list[dict[str, Any]] | None = None,
) -> OversightAttestationRecord:
    """Build an attestation from an ``aragora/human-settlement`` commit status.

    ``status`` is the GitHub commit-status object (``GET
    /repos/{repo}/commits/{sha}/statuses`` element): the **creator login** is
    the oversight identity, ``created_at`` is when, and ``target_url``
    (typically the Tier-4 preapproval comment) is the mechanism reference.
    """
    record = OversightAttestationRecord(
        disposition="human_attested",
        mechanism=MECHANISM_SETTLEMENT_STATUS,
        subject=_subject(repo, pr_number, head_sha),
    )

    creator = status.get("creator") or {}
    login = str(creator.get("login") or "").strip() if isinstance(creator, dict) else ""
    if login:
        record.attestor_id = login
        record.attestor_role = "settlement_status_creator"
    else:
        record.record_absence(
            "attestor_id",
            "settlement status carries no creator login; oversight identity "
            "cannot be derived from this artifact",
        )

    created_at = str(status.get("created_at") or "").strip()
    if created_at:
        record.attested_at = created_at
    else:
        record.record_absence("attested_at", "settlement status has no created_at timestamp")

    if head_sha:
        record.observed["head_sha"] = head_sha
    else:
        record.record_absence("observed.head_sha", "no head SHA supplied for the settled decision")

    if evidence_items:
        record.observed["evidence_digest"] = evidence_digest(evidence_items)
        record.observed["evidence_items"] = [dict(item) for item in evidence_items]
    else:
        record.record_absence(
            "observed.evidence_digest",
            "no counted evidence artifacts were supplied; digest of what the "
            "overseer saw cannot be computed",
        )

    context = str(status.get("context") or "").strip()
    if context:
        record.references.append({"type": "github_status_context", "ref": context})
    target_url = str(status.get("target_url") or "").strip()
    if target_url:
        record.references.append({"type": "github_status_target_url", "ref": target_url})
    description = str(status.get("description") or "").strip()
    if description:
        record.references.append({"type": "github_status_description", "ref": description})
    return record


def attestation_from_tier_four_comment(
    comment: dict[str, Any],
    *,
    repo: str | None = None,
    pr_number: int | None = None,
    head_sha: str | None = None,
    evidence_items: list[dict[str, Any]] | None = None,
) -> OversightAttestationRecord:
    """Build an attestation from a Tier-4 human preapproval comment.

    ``comment`` is a GitHub issue-comment object (``user.login``, ``body``,
    ``created_at``/``createdAt``, ``html_url``/``url``). The caller is
    responsible for having matched it via
    :func:`find_tier_four_preapproval_comment` (same predicate the merge gate
    applies).
    """
    record = OversightAttestationRecord(
        disposition="human_attested",
        mechanism=MECHANISM_TIER_FOUR_COMMENT,
        subject=_subject(repo, pr_number, head_sha),
    )

    user = comment.get("user") or comment.get("author") or {}
    login = str(user.get("login") or "").strip() if isinstance(user, dict) else ""
    if login:
        record.attestor_id = login
        record.attestor_role = "preapproval_comment_author"
    else:
        record.record_absence("attestor_id", "preapproval comment carries no author login")

    created_at = str(comment.get("created_at") or comment.get("createdAt") or "").strip()
    if created_at:
        record.attested_at = created_at
    else:
        record.record_absence("attested_at", "preapproval comment has no creation timestamp")

    if head_sha:
        record.observed["head_sha"] = head_sha
    else:
        record.record_absence("observed.head_sha", "no head SHA supplied for the settled decision")

    if evidence_items:
        record.observed["evidence_digest"] = evidence_digest(evidence_items)
        record.observed["evidence_items"] = [dict(item) for item in evidence_items]
    else:
        record.record_absence(
            "observed.evidence_digest",
            "no counted evidence artifacts were supplied; digest of what the "
            "overseer saw cannot be computed",
        )

    body = str(comment.get("body") or "")
    if body:
        record.observed["comment_body_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    url = str(comment.get("html_url") or comment.get("url") or "").strip()
    if url:
        record.references.append({"type": "github_comment_url", "ref": url})
    comment_id = comment.get("id")
    if comment_id is not None:
        record.references.append({"type": "github_comment_id", "ref": str(comment_id)})
    return record


def attestation_from_local_settlement_receipt(
    payload: dict[str, Any],
    *,
    receipt_path: str | None = None,
) -> OversightAttestationRecord:
    """Build an attestation from a local review-queue settlement receipt.

    ``payload`` is the JSON stored under ``.aragora/review-queue/receipts/``
    (a trusted operator-controlled store, exact-head bound): ``actor`` is the
    oversight identity, ``reviewed_at`` is when, ``head_sha`` is what was
    settled, and ``packet_sha`` (the merge-packet digest counted at settlement
    time) is the digest of what the overseer saw.
    """
    pr_number_raw = payload.get("pr_number")
    pr_number = int(pr_number_raw) if pr_number_raw is not None else None
    head_sha = str(payload.get("head_sha") or "").strip() or None

    record = OversightAttestationRecord(
        disposition="human_attested",
        mechanism=MECHANISM_LOCAL_SETTLEMENT_RECEIPT,
        subject=_subject(None, pr_number, head_sha),
    )
    pr_url = str(payload.get("pr_url") or "").strip()
    if pr_url:
        record.subject["pr_url"] = pr_url

    actor = str(payload.get("actor") or "").strip()
    if actor:
        record.attestor_id = actor
        record.attestor_role = "settlement_receipt_actor"
    else:
        record.record_absence("attestor_id", "settlement receipt has no actor field")

    reviewed_at = str(payload.get("reviewed_at") or "").strip()
    if reviewed_at:
        record.attested_at = reviewed_at
    else:
        record.record_absence("attested_at", "settlement receipt has no reviewed_at timestamp")

    if head_sha:
        record.observed["head_sha"] = head_sha
    else:
        record.record_absence("observed.head_sha", "settlement receipt has no head_sha")

    packet_sha = str(payload.get("packet_sha") or "").strip()
    if packet_sha:
        record.observed["evidence_digest"] = packet_sha
        record.observed["evidence_digest_source"] = "merge_packet_sha"
    else:
        record.record_absence(
            "observed.evidence_digest",
            "settlement receipt has no packet_sha; digest of the merge packet "
            "the overseer saw was not recorded",
        )

    for ref_field, ref_type in (
        ("action", "settlement_action"),
        ("github_event", "github_event"),
        ("reason", "settlement_reason"),
        ("session_id", "settlement_session_id"),
    ):
        value = str(payload.get(ref_field) or "").strip()
        if value:
            record.references.append({"type": ref_type, "ref": value})
    if receipt_path:
        record.references.append({"type": "settlement_receipt_path", "ref": str(receipt_path)})
    return record


def autonomous_attestation(
    *,
    repo: str | None = None,
    pr_number: int | None = None,
    head_sha: str | None = None,
    reason: str,
    references: list[dict[str, Any]] | None = None,
) -> OversightAttestationRecord:
    """Build the explicit ``autonomous`` disposition record.

    Used for decisions settled by the model-quorum gate with no human in the
    loop. The non-intervention is recorded as a first-class fact (the ODR
    profile's honesty rule), not implied by a missing field.
    """
    record = OversightAttestationRecord(
        disposition="autonomous",
        mechanism=MECHANISM_MODEL_QUORUM,
        subject=_subject(repo, pr_number, head_sha),
    )
    record.observed["non_intervention_reason"] = reason
    if head_sha:
        record.observed["head_sha"] = head_sha
    if references:
        record.references.extend(dict(ref) for ref in references)
    return record


def find_tier_four_preapproval_comment(
    comments: list[dict[str, Any]],
    *,
    head_sha: str,
) -> dict[str, Any] | None:
    """Return the first comment satisfying the Tier-4 preapproval predicate.

    Same conditions the merge gate applies
    (``review_queue._has_tier_four_human_preapproval_comment``): the exact
    marker, the exact head SHA, an authorized-action token, and an explicit
    human-risk settlement sentence.
    """
    head = str(head_sha or "").strip()
    if not head:
        return None
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        body = str(comment.get("body") or "")
        lowered = body.lower()
        if TIER_FOUR_SETTLEMENT_MARKER not in body:
            continue
        if head not in body:
            continue
        if not any(token in lowered for token in TIER_FOUR_AUTHORIZED_MERGE_TOKENS):
            continue
        if "human-risk settlement" not in lowered:
            continue
        return comment
    return None


def classify_settled_pr(
    *,
    repo: str | None = None,
    pr_number: int | None = None,
    head_sha: str | None = None,
    statuses: list[dict[str, Any]] | None = None,
    comments: list[dict[str, Any]] | None = None,
    evidence_items: list[dict[str, Any]] | None = None,
) -> OversightAttestationRecord:
    """Classify one settled PR into its oversight attestation.

    Precedence mirrors the gate's trust order: the gate-trusted
    ``aragora/human-settlement`` commit status first, then an exact-head
    Tier-4 preapproval comment, otherwise the explicit ``autonomous``
    disposition. Pure function over already-fetched data — callers own I/O.
    """
    for status in statuses or []:
        if not isinstance(status, dict):
            continue
        context = str(status.get("context") or "").strip()
        state = str(status.get("state") or "").strip().lower()
        if context == HUMAN_SETTLEMENT_STATUS_CONTEXT and state == "success":
            return attestation_from_settlement_status(
                status,
                repo=repo,
                pr_number=pr_number,
                head_sha=head_sha,
                evidence_items=evidence_items,
            )

    if head_sha and comments:
        comment = find_tier_four_preapproval_comment(comments, head_sha=head_sha)
        if comment is not None:
            return attestation_from_tier_four_comment(
                comment,
                repo=repo,
                pr_number=pr_number,
                head_sha=head_sha,
                evidence_items=evidence_items,
            )

    return autonomous_attestation(
        repo=repo,
        pr_number=pr_number,
        head_sha=head_sha,
        reason=(
            "settled via the model-quorum merge gate; no "
            f"'{HUMAN_SETTLEMENT_STATUS_CONTEXT}' status and no exact-head "
            "Tier-4 preapproval comment present"
        ),
    )
