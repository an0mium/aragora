"""Tests for DIC-16 receipt epistemic provenance helpers.

Covers:
- ReceiptVerification backwards compat (old callers unaffected)
- ReceiptVerification new optional fields
- receipt_verification_from_claim_result: all five DIC-14 statuses
- receipt_verification_from_crux: happy path + edge cases
- KM adapter propagates the new provenance fields into item metadata
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from aragora.export.decision_receipt import ReceiptVerification
from aragora.export.receipt_epistemic import (
    receipt_verification_from_claim_result,
    receipt_verification_from_crux,
)


# ---------------------------------------------------------------------------
# ReceiptVerification backwards compat
# ---------------------------------------------------------------------------


class TestReceiptVerificationBackwardsCompat:
    def test_legacy_constructor_still_works(self) -> None:
        rv = ReceiptVerification(
            claim="The system is healthy.",
            verified=True,
            method="manual",
            proof_hash="abc123",
        )
        assert rv.claim == "The system is healthy."
        assert rv.verified is True
        assert rv.method == "manual"
        assert rv.proof_hash == "abc123"

    def test_new_optional_fields_default_to_none_or_empty(self) -> None:
        rv = ReceiptVerification(claim="x", verified=False, method="y")
        assert rv.claim_id is None
        assert rv.crux_id is None
        assert rv.evidence_ids == []
        assert rv.verification_status is None
        assert rv.source_receipt_id is None


# ---------------------------------------------------------------------------
# receipt_verification_from_claim_result
# ---------------------------------------------------------------------------


class TestReceiptVerificationFromClaimResult:
    def test_pass_status_sets_verified_true(self) -> None:
        rv = receipt_verification_from_claim_result(
            claim_id="b0.truth.success_rate",
            statement="Benchmark surface is fresh.",
            status="pass",
        )
        assert rv.verified is True
        assert rv.verification_status == "pass"
        assert rv.claim_id == "b0.truth.success_rate"
        assert rv.crux_id is None
        assert rv.method == "claim_verifier"

    @pytest.mark.parametrize("status", ["fail", "stale", "unsupported", "error"])
    def test_non_pass_status_sets_verified_false(self, status: str) -> None:
        rv = receipt_verification_from_claim_result(
            claim_id="some.claim",
            statement="stmt",
            status=status,  # type: ignore[arg-type]
        )
        assert rv.verified is False
        assert rv.verification_status == status

    def test_pass_computes_proof_hash(self) -> None:
        rv = receipt_verification_from_claim_result(
            claim_id="claim.x", statement="s", status="pass"
        )
        assert rv.proof_hash is not None
        assert len(rv.proof_hash) == 16

    def test_explicit_proof_hash_overrides_computed(self) -> None:
        rv = receipt_verification_from_claim_result(
            claim_id="claim.x",
            statement="s",
            status="pass",
            proof_hash="custom_hash",
        )
        assert rv.proof_hash == "custom_hash"

    def test_evidence_ids_preserved(self) -> None:
        rv = receipt_verification_from_claim_result(
            claim_id="c",
            statement="s",
            status="pass",
            evidence_ids=["docs/status/B0.md", "workflow:benchmark_truth"],
        )
        assert rv.evidence_ids == ["docs/status/B0.md", "workflow:benchmark_truth"]

    def test_empty_claim_id_raises(self) -> None:
        with pytest.raises(ValueError, match="claim_id"):
            receipt_verification_from_claim_result(claim_id="", statement="s", status="pass")

    def test_unknown_status_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown status"):
            receipt_verification_from_claim_result(
                claim_id="c",
                statement="s",
                status="unknown",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# receipt_verification_from_crux
# ---------------------------------------------------------------------------


class TestReceiptVerificationFromCrux:
    def test_basic_crux(self) -> None:
        rv = receipt_verification_from_crux(
            crux_id="crux.guard.expansion",
            question="Should Aragora widen B2 guard expansion now?",
            load_bearing_score=0.86,
        )
        assert rv.crux_id == "crux.guard.expansion"
        assert rv.claim_id is None
        assert rv.verified is False
        assert rv.method == "crux_set"
        assert rv.verification_status == "open"
        assert rv.proof_hash is None

    def test_empty_crux_id_raises(self) -> None:
        with pytest.raises(ValueError, match="crux_id"):
            receipt_verification_from_crux(crux_id="", question="q", load_bearing_score=0.5)

    def test_load_bearing_score_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="load_bearing_score"):
            receipt_verification_from_crux(crux_id="c", question="q", load_bearing_score=-0.1)

    def test_load_bearing_score_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="load_bearing_score"):
            receipt_verification_from_crux(crux_id="c", question="q", load_bearing_score=1.1)

    def test_boundary_scores_accepted(self) -> None:
        for score in [0.0, 1.0]:
            rv = receipt_verification_from_crux(crux_id="c", question="q", load_bearing_score=score)
            assert rv.crux_id == "c"


# ---------------------------------------------------------------------------
# KM adapter propagates DIC-16 provenance fields
# ---------------------------------------------------------------------------


class TestKMAdapterProvenanceFields:
    """Verify _verification_to_knowledge_item preserves the new DIC-16 fields."""

    def _make_receipt(self, receipt_id: str = "rcpt_test") -> Any:
        receipt = MagicMock()
        receipt.receipt_id = receipt_id
        receipt.gauntlet_id = "gauntlet_test"
        receipt.verdict = "approved"
        receipt.confidence = 0.9
        return receipt

    def _make_adapter(self) -> Any:
        from aragora.knowledge.mound.adapters.receipt_adapter import ReceiptAdapter

        adapter = ReceiptAdapter.__new__(ReceiptAdapter)
        adapter._mound = MagicMock()
        return adapter

    def test_claim_id_appears_in_metadata(self) -> None:
        adapter = self._make_adapter()
        rv = receipt_verification_from_claim_result(
            claim_id="b0.truth.success_rate",
            statement="Benchmark surface is fresh.",
            status="pass",
            evidence_ids=["docs/status/B0.md"],
            source_receipt_id="rcpt_test",
        )
        item = adapter._verification_to_knowledge_item(
            rv, self._make_receipt(), workspace_id="ws1", tags=["tag:a"]
        )
        assert item.metadata["claim_id"] == "b0.truth.success_rate"
        assert item.metadata["verification_status"] == "pass"
        assert item.metadata["evidence_ids"] == ["docs/status/B0.md"]
        assert item.metadata["source_receipt_id"] == "rcpt_test"
        assert item.metadata["crux_id"] is None

    def test_crux_id_appears_in_metadata(self) -> None:
        adapter = self._make_adapter()
        rv = receipt_verification_from_crux(
            crux_id="crux.guard.expansion",
            question="Should Aragora widen B2?",
            load_bearing_score=0.86,
            evidence_gap_ids=["gap.soak_policy"],
            source_receipt_id="rcpt_test",
        )
        item = adapter._verification_to_knowledge_item(
            rv, self._make_receipt(), workspace_id="ws1", tags=[]
        )
        assert item.metadata["crux_id"] == "crux.guard.expansion"
        assert item.metadata["verification_status"] == "open"
        assert item.metadata["evidence_ids"] == ["gap.soak_policy"]
        assert item.metadata["claim_id"] is None

    def test_legacy_receipt_verification_metadata_unchanged(self) -> None:
        adapter = self._make_adapter()
        rv = ReceiptVerification(
            claim="Legacy claim text.",
            verified=True,
            method="manual",
            proof_hash="legacy_hash",
        )
        item = adapter._verification_to_knowledge_item(
            rv, self._make_receipt(), workspace_id=None, tags=[]
        )
        assert item.metadata["claim_id"] is None
        assert item.metadata["crux_id"] is None
        assert item.metadata["evidence_ids"] == []
        assert item.metadata["verification_status"] is None
        # Legacy fields still present
        assert item.metadata["verification_method"] == "manual"
        assert item.metadata["proof_hash"] == "legacy_hash"
