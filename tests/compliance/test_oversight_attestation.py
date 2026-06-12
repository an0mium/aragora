"""Tests for human-oversight attestation extraction (#8230, ODR-6).

Fixtures mirror the *real* settlement artifacts this repo produces:

- the ``aragora/human-settlement`` commit status from PR #8169 (the scarmani
  inaugural Tier-4 settlement);
- the Tier-4 preapproval comment posted on that PR;
- a local review-queue settlement receipt (shape of
  ``.aragora/review-queue/receipts/pr-8154-...json``).
"""

from __future__ import annotations

import pytest

from aragora.compliance.oversight_attestation import (
    HUMAN_SETTLEMENT_STATUS_CONTEXT,
    TIER_FOUR_AUTHORIZED_MERGE_TOKENS,
    TIER_FOUR_SETTLEMENT_MARKER,
    UNDISCLOSED,
    attestation_from_local_settlement_receipt,
    attestation_from_settlement_status,
    attestation_from_tier_four_comment,
    autonomous_attestation,
    classify_settled_pr,
    evidence_digest,
    find_tier_four_preapproval_comment,
)

HEAD_8169 = "bdb4ee5b9c9f4d7c130ef75cb01cd0530a62f241"

# Shape: GET /repos/{repo}/commits/{sha}/statuses element (real PR #8169 data).
STATUS_8169 = {
    "context": "aragora/human-settlement",
    "state": "success",
    "creator": {"login": "an0mium"},
    "created_at": "2026-06-11T18:14:09Z",
    "updated_at": "2026-06-11T18:14:09Z",
    "description": "Tier 4 exact-head human-risk settlement recorded for PR #8169",
    "target_url": "https://github.com/synaptent/aragora/pull/8169#issuecomment-4683612872",
}

# Shape: GET /repos/{repo}/issues/comments/{id} (real PR #8169 preapproval).
COMMENT_8169 = {
    "id": 4683612872,
    "user": {"login": "an0mium"},
    "created_at": "2026-06-11T18:14:08Z",
    "html_url": "https://github.com/synaptent/aragora/pull/8169#issuecomment-4683612872",
    "body": (
        "Tier-4 Human Settlement Authorization\n\n"
        "PR: #8169\n"
        f"Exact head: {HEAD_8169}\n"
        "Authorized action: admin_squash_merge and branch_protection_reconcile, "
        "only if #8169 is non-draft and live exact-head checks/merge-packet "
        "remain otherwise green.\n\n"
        "Human-risk settlement: I accept the Tier 4 risk for this PR."
    ),
}

# Shape: .aragora/review-queue/receipts/pr-8154-...json (real fields).
LOCAL_RECEIPT = {
    "session_id": "recorded-8154-5af592dd619c-admin_squash_merge",
    "reviewed_at": "2026-06-11T04:34:42+00:00",
    "actor": "an0mium",
    "action": "admin_squash_merge",
    "reason": "external exact-head admin squash observed for PR #8154",
    "pr_number": 8154,
    "pr_url": "https://github.com/synaptent/aragora/pull/8154",
    "head_sha": "5af592dd619c02c023088a0625ff70f2a3c1d5c6",
    "packet_sha": "sha256:f887561b04a7ec0bb5ff841654c2434f7b41a871d0a7997df894aca1fdf3e7d3",
    "github_event": "ADMIN_SQUASH_MERGE",
}


class TestGateConstantsStayInSync:
    """The extractor must count exactly what the merge gate counts."""

    def test_constants_match_review_queue(self):
        review_queue = pytest.importorskip("aragora.cli.commands.review_queue")
        assert HUMAN_SETTLEMENT_STATUS_CONTEXT == review_queue.HUMAN_SETTLEMENT_CONTEXT
        assert TIER_FOUR_SETTLEMENT_MARKER == review_queue.TIER_FOUR_SETTLEMENT_MARKER
        assert TIER_FOUR_AUTHORIZED_MERGE_TOKENS == review_queue.TIER_FOUR_AUTHORIZED_MERGE_TOKENS


class TestSettlementStatusExtraction:
    def test_real_8169_shape(self):
        record = attestation_from_settlement_status(
            STATUS_8169,
            repo="synaptent/aragora",
            pr_number=8169,
            head_sha=HEAD_8169,
            evidence_items=[{"type": "merge_packet", "digest": "sha256:abc"}],
        )
        assert record.disposition == "human_attested"
        assert record.attestor_id == "an0mium"
        assert record.attestor_role == "settlement_status_creator"
        assert record.attested_at == "2026-06-11T18:14:09Z"
        assert record.mechanism == "github_settlement_status"
        assert record.subject == {
            "repo": "synaptent/aragora",
            "pr_number": 8169,
            "head_sha": HEAD_8169,
        }
        assert record.observed["head_sha"] == HEAD_8169
        assert record.observed["evidence_digest"]
        refs = {r["type"]: r["ref"] for r in record.references}
        assert refs["github_status_context"] == "aragora/human-settlement"
        assert "issuecomment-4683612872" in refs["github_status_target_url"]
        assert not record.absences

    def test_missing_creator_is_recorded_absent_not_fabricated(self):
        status = {k: v for k, v in STATUS_8169.items() if k != "creator"}
        record = attestation_from_settlement_status(status, head_sha=HEAD_8169)
        assert record.attestor_id is None
        assert any(a["field"] == "attestor_id" for a in record.absences)
        # ODR projection uses the profile's "undisclosed" literal, never a guess.
        odr = record.to_odr_attestation()
        assert odr["attestor"]["id"] == UNDISCLOSED

    def test_missing_evidence_items_recorded_absent(self):
        record = attestation_from_settlement_status(STATUS_8169, head_sha=HEAD_8169)
        assert "evidence_digest" not in record.observed
        assert any(a["field"] == "observed.evidence_digest" for a in record.absences)

    def test_missing_head_sha_recorded_absent(self):
        record = attestation_from_settlement_status(STATUS_8169)
        assert any(a["field"] == "observed.head_sha" for a in record.absences)


class TestTierFourCommentExtraction:
    def test_finds_real_8169_comment(self):
        found = find_tier_four_preapproval_comment([COMMENT_8169], head_sha=HEAD_8169)
        assert found is COMMENT_8169

    def test_rejects_wrong_head(self):
        assert find_tier_four_preapproval_comment([COMMENT_8169], head_sha="deadbeef" * 5) is None

    def test_rejects_missing_marker(self):
        comment = dict(COMMENT_8169)
        comment["body"] = comment["body"].replace(TIER_FOUR_SETTLEMENT_MARKER, "Approval")
        assert find_tier_four_preapproval_comment([comment], head_sha=HEAD_8169) is None

    def test_rejects_missing_risk_acceptance(self):
        comment = dict(COMMENT_8169)
        comment["body"] = comment["body"].replace("Human-risk settlement", "Looks good")
        assert find_tier_four_preapproval_comment([comment], head_sha=HEAD_8169) is None

    def test_extraction_fields(self):
        record = attestation_from_tier_four_comment(
            COMMENT_8169,
            repo="synaptent/aragora",
            pr_number=8169,
            head_sha=HEAD_8169,
        )
        assert record.disposition == "human_attested"
        assert record.attestor_id == "an0mium"
        assert record.attestor_role == "preapproval_comment_author"
        assert record.attested_at == "2026-06-11T18:14:08Z"
        assert record.mechanism == "tier4_preapproval_comment"
        assert record.observed["comment_body_sha256"]
        refs = {r["type"] for r in record.references}
        assert "github_comment_url" in refs
        assert "github_comment_id" in refs


class TestLocalReceiptExtraction:
    def test_real_receipt_shape(self):
        record = attestation_from_local_settlement_receipt(
            LOCAL_RECEIPT, receipt_path="/tmp/pr-8154.json"
        )
        assert record.disposition == "human_attested"
        assert record.attestor_id == "an0mium"
        assert record.attestor_role == "settlement_receipt_actor"
        assert record.attested_at == "2026-06-11T04:34:42+00:00"
        assert record.subject["pr_number"] == 8154
        assert record.observed["head_sha"] == LOCAL_RECEIPT["head_sha"]
        # packet_sha is the digest of what the overseer saw at settlement time.
        assert record.observed["evidence_digest"] == LOCAL_RECEIPT["packet_sha"]
        assert record.observed["evidence_digest_source"] == "merge_packet_sha"
        assert not record.absences

    def test_missing_pieces_recorded_absent(self):
        record = attestation_from_local_settlement_receipt({"pr_number": 99})
        missing = {a["field"] for a in record.absences}
        assert {
            "attestor_id",
            "attested_at",
            "observed.head_sha",
            "observed.evidence_digest",
        } <= missing


class TestAutonomousDisposition:
    def test_explicit_autonomous_record(self):
        record = autonomous_attestation(
            repo="synaptent/aragora",
            pr_number=8239,
            head_sha="a" * 40,
            reason="merged via model-quorum gate",
        )
        assert record.disposition == "autonomous"
        assert record.mechanism == "model_quorum_autonomous"
        assert record.observed["non_intervention_reason"] == "merged via model-quorum gate"

    def test_odr_projection_is_explicit_not_missing(self):
        odr = autonomous_attestation(reason="no human in loop").to_odr_attestation()
        assert odr["disposition"] == "autonomous"
        assert "attestor" not in odr


class TestClassifySettledPr:
    def test_status_wins(self):
        record = classify_settled_pr(
            repo="synaptent/aragora",
            pr_number=8169,
            head_sha=HEAD_8169,
            statuses=[STATUS_8169],
            comments=[COMMENT_8169],
        )
        assert record.disposition == "human_attested"
        assert record.mechanism == "github_settlement_status"

    def test_comment_fallback(self):
        record = classify_settled_pr(
            pr_number=8169,
            head_sha=HEAD_8169,
            statuses=[],
            comments=[COMMENT_8169],
        )
        assert record.mechanism == "tier4_preapproval_comment"

    def test_non_success_status_does_not_count(self):
        failed = dict(STATUS_8169, state="failure")
        record = classify_settled_pr(head_sha=HEAD_8169, statuses=[failed], comments=[])
        assert record.disposition == "autonomous"

    def test_autonomous_default_with_reason(self):
        record = classify_settled_pr(pr_number=8239, head_sha="b" * 40)
        assert record.disposition == "autonomous"
        assert "model-quorum" in record.observed["non_intervention_reason"]


class TestOdrProjection:
    def test_human_attested_projection_schema_valid(self):
        jsonschema = pytest.importorskip("jsonschema")
        from aragora.gauntlet.odr_export import load_odr_schema

        schema = load_odr_schema()
        attestation_schema = schema["properties"]["attestation"]
        record = attestation_from_settlement_status(STATUS_8169, pr_number=8169, head_sha=HEAD_8169)
        jsonschema.Draft202012Validator(attestation_schema).validate(record.to_odr_attestation())

    def test_autonomous_projection_schema_valid(self):
        jsonschema = pytest.importorskip("jsonschema")
        from aragora.gauntlet.odr_export import load_odr_schema

        attestation_schema = load_odr_schema()["properties"]["attestation"]
        record = autonomous_attestation(reason="no human in loop")
        jsonschema.Draft202012Validator(attestation_schema).validate(record.to_odr_attestation())

    def test_odr_export_accepts_record_via_duck_typing(self):
        from aragora.gauntlet.odr_export import _map_attestation

        record = attestation_from_settlement_status(STATUS_8169, pr_number=8169, head_sha=HEAD_8169)
        block = _map_attestation(record)
        assert block["disposition"] == "human_attested"
        assert block["attestor"]["id"] == "an0mium"
        assert block["attestor"]["observed_head_sha"] == HEAD_8169
        assert block["method"] == "github_settlement_status"

    def test_odr_export_full_receipt_with_attestation(self):
        jsonschema = pytest.importorskip("jsonschema")
        from aragora.gauntlet.odr_export import decision_receipt_to_odr, load_odr_schema
        from aragora.gauntlet.receipt_models import DecisionReceipt

        receipt = DecisionReceipt(
            receipt_id="r-1",
            gauntlet_id="g-1",
            timestamp="2026-06-11T18:14:09+00:00",
            input_summary="test",
            input_hash="a" * 64,
            risk_summary={},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=0,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.0,
        )
        record = attestation_from_settlement_status(STATUS_8169, pr_number=8169, head_sha=HEAD_8169)
        odr = decision_receipt_to_odr(receipt, attestation=record)
        jsonschema.Draft202012Validator(load_odr_schema()).validate(odr)
        assert odr["attestation"]["attestor"]["id"] == "an0mium"


class TestEvidenceDigest:
    def test_order_independent_and_deterministic(self):
        a = {"type": "comment", "digest": "x"}
        b = {"type": "receipt", "digest": "y"}
        assert evidence_digest([a, b]) == evidence_digest([b, a])
        assert len(evidence_digest([a])) == 64
