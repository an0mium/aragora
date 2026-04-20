"""Tests for aragora.review.receipt — schema-only BriefReceipt + linkage contracts."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from aragora.review import (
    BriefReceipt,
    DissentingView,
    DissentPosition,
    EvidenceRef,
    Recommendation,
    ReviewBrief,
    ReviewRole,
    RoleFinding,
    SettlementLinkage,
    ValidationRef,
)

UTC = timezone.utc


# --- Helpers -------------------------------------------------------------


def _minimal_brief(**overrides) -> ReviewBrief:
    defaults = dict(
        pr_number=6307,
        repo="synaptent/aragora",
        head_sha="f1a640ee2",
        base_sha="eeef02721",
        packet_sha="abc123",
        recommendation=Recommendation.APPROVE_CANDIDATE,
        top_line="Bounded schema extension; no behavior.",
        role_findings=(),
        dissent=(),
        validation_summary="pre-commit clean.",
        overall_confidence=0.9,
        disagreement_score=0.05,
        total_cost_usd=0.12,
        total_wall_clock_ms=3500,
        agent_roster=("claude-opus-4-7", "gpt-5-4"),
        generated_at=datetime.now(UTC).isoformat(),
    )
    defaults.update(overrides)
    return ReviewBrief(**defaults)


def _minimal_receipt(**overrides) -> BriefReceipt:
    defaults = dict(
        brief=_minimal_brief(),
        evidence_refs=(),
        validation_refs=(),
        receipt_id="receipt-sha-xyz",
        created_at=datetime.now(UTC).isoformat(),
    )
    defaults.update(overrides)
    return BriefReceipt(**defaults)


# --- EvidenceRef --------------------------------------------------------


class TestEvidenceRef:
    def test_frozen(self) -> None:
        ref = EvidenceRef(kind="file", path="aragora/review/receipt.py")
        with pytest.raises((AttributeError, TypeError)):
            ref.path = "elsewhere"  # type: ignore[misc]

    def test_line_range_serializes_as_list(self) -> None:
        ref = EvidenceRef(
            kind="file",
            path="aragora/review/receipt.py",
            line_range=(42, 58),
            quote="def to_dict(self) -> dict[str, Any]:",
        )
        d = ref.to_dict()
        assert d["line_range"] == [42, 58]
        assert d["kind"] == "file"
        assert d["path"] == "aragora/review/receipt.py"

    def test_line_range_omitted_when_none(self) -> None:
        ref = EvidenceRef(kind="commit", path="main@f1a640ee2", sha="f1a640ee2")
        d = ref.to_dict()
        assert d["line_range"] is None

    def test_json_roundtrip(self) -> None:
        ref = EvidenceRef(
            kind="file",
            path="aragora/cli/commands/review_queue.py",
            line_range=(130, 151),
            quote="class SettlementReceipt: ...",
        )
        roundtrip = json.loads(json.dumps(ref.to_dict()))
        assert roundtrip["kind"] == "file"
        assert roundtrip["line_range"] == [130, 151]


# --- ValidationRef -------------------------------------------------------


class TestValidationRef:
    def test_frozen(self) -> None:
        ref = ValidationRef(kind="ci_check", name="lint", result="success")
        with pytest.raises((AttributeError, TypeError)):
            ref.result = "failure"  # type: ignore[misc]

    def test_to_dict_roundtrip(self) -> None:
        ref = ValidationRef(
            kind="ci_check",
            name="Version Alignment",
            result="success",
            url="https://github.com/synaptent/aragora/actions/runs/12345",
        )
        roundtrip = json.loads(json.dumps(ref.to_dict()))
        assert roundtrip["kind"] == "ci_check"
        assert roundtrip["name"] == "Version Alignment"
        assert roundtrip["result"] == "success"


# --- BriefReceipt --------------------------------------------------------


class TestBriefReceipt:
    def test_advisory_only_default_is_true(self) -> None:
        # Acceptance criterion: BriefReceipt wraps the machine brief, which
        # is advisory. Settlement is a separate human action elsewhere.
        receipt = _minimal_receipt()
        assert receipt.advisory_only is True

    def test_settlement_note_default_says_advisory(self) -> None:
        receipt = _minimal_receipt()
        assert "advisory" in receipt.settlement_note.lower()
        assert "human settlement" in receipt.settlement_note.lower()

    def test_to_dict_nests_brief_and_refs(self) -> None:
        receipt = _minimal_receipt(
            evidence_refs=(EvidenceRef(kind="file", path="aragora/review/protocol.py"),),
            validation_refs=(ValidationRef(kind="ci_check", name="lint", result="success"),),
        )
        d = receipt.to_dict()
        assert d["brief"]["pr_number"] == 6307
        assert d["brief"]["recommendation"] == "approve_candidate"
        assert d["evidence_refs"][0]["path"] == "aragora/review/protocol.py"
        assert d["validation_refs"][0]["name"] == "lint"

    def test_frozen(self) -> None:
        receipt = _minimal_receipt()
        with pytest.raises((AttributeError, TypeError)):
            receipt.advisory_only = False  # type: ignore[misc]

    def test_sequence_fields_are_immutable_tuples(self) -> None:
        # Same safety property as ReviewBrief: attribute reassignment is
        # blocked, but without tuple types `receipt.evidence_refs.append(...)`
        # would still be possible mid-flight and break receipt_id binding.
        receipt = _minimal_receipt(
            evidence_refs=(EvidenceRef(kind="file", path="x.py"),),
            validation_refs=(ValidationRef(kind="ci_check", name="lint", result="success"),),
        )
        assert isinstance(receipt.evidence_refs, tuple)
        assert isinstance(receipt.validation_refs, tuple)
        with pytest.raises(AttributeError):
            receipt.evidence_refs.append(EvidenceRef(kind="file", path="y.py"))  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            receipt.validation_refs.append(  # type: ignore[attr-defined]
                ValidationRef(kind="ci_check", name="x", result="success")
            )

    def test_dissent_survives_in_receipt(self) -> None:
        # Acceptance criterion (#6307 body): "Dissent survives in receipts
        # instead of being collapsed into one summary line."
        brief = _minimal_brief(
            dissent=(
                DissentingView(
                    agent="grok-3",
                    position=DissentPosition.REQUEST_CHANGES,
                    reason="Security concern in auth path.",
                    role=ReviewRole.SECURITY,
                ),
                DissentingView(
                    agent="gpt-5-4",
                    position=DissentPosition.DEFER,
                    reason="Needs more validation data.",
                ),
            ),
        )
        receipt = _minimal_receipt(brief=brief)
        d = receipt.to_dict()
        assert len(d["brief"]["dissent"]) == 2
        assert d["brief"]["dissent"][0]["position"] == "request_changes"
        assert d["brief"]["dissent"][1]["position"] == "defer"
        # Per-dissent reasons are preserved; no collapse into a summary line.
        assert d["brief"]["dissent"][0]["reason"] == "Security concern in auth path."
        assert d["brief"]["dissent"][1]["reason"] == "Needs more validation data."

    def test_brief_sha_binding_survives_receipt(self) -> None:
        # Acceptance criterion: "A settled PR can be traced back to the
        # exact brief packet and head SHA."
        brief = _minimal_brief(
            head_sha="f1a640ee2deadbeef",
            packet_sha="packet-sha-locked",
        )
        receipt = _minimal_receipt(brief=brief)
        d = receipt.to_dict()
        assert d["brief"]["head_sha"] == "f1a640ee2deadbeef"
        assert d["brief"]["packet_sha"] == "packet-sha-locked"

    def test_receipt_id_preimage_is_documented(self) -> None:
        # The preimage rule lives in the docstring (not in code) because
        # this module is intentionally behavior-free. The orchestrator
        # (#6306 successor) implements the hash and holds it under test.
        # This guard is here so #6304/#6305 can't drift silently.
        from aragora.review.receipt import BriefReceipt as BR

        doc = BR.__doc__ or ""
        assert "Receipt-ID preimage" in doc
        assert 'Remove the ``"receipt_id"`` key' in doc
        assert "canonical JSON" in doc
        assert "sha256" in doc.lower()

    def test_json_roundtrip(self) -> None:
        receipt = _minimal_receipt()
        roundtrip = json.loads(json.dumps(receipt.to_dict()))
        assert roundtrip["brief"]["pr_number"] == 6307
        assert roundtrip["advisory_only"] is True


# --- SettlementLinkage --------------------------------------------------


class TestSettlementLinkage:
    def _minimal_linkage(self, **overrides) -> SettlementLinkage:
        defaults = dict(
            brief_receipt_id="receipt-sha-xyz",
            settlement_receipt_path=".aragora/review-queue/settlements/pr-6307-f1a640ee2def-approve.json",
            head_sha="f1a640ee2deadbeef",
            packet_sha="packet-sha-locked",
            pr_number=6307,
            repo="synaptent/aragora",
            action="approve",
            settled_at=datetime.now(UTC).isoformat(),
        )
        defaults.update(overrides)
        return SettlementLinkage(**defaults)

    def test_advisory_only_defaults_to_false_because_settlement_is_human(self) -> None:
        # Settlement is a human action, not an advisory machine output.
        # Unlike BriefReceipt.advisory_only=True, SettlementLinkage tracks
        # a human settlement decision.
        linkage = self._minimal_linkage()
        assert linkage.advisory_only is False

    def test_frozen(self) -> None:
        linkage = self._minimal_linkage()
        with pytest.raises((AttributeError, TypeError)):
            linkage.action = "request_changes"  # type: ignore[misc]

    def test_repair_receipt_paths_is_immutable_tuple(self) -> None:
        linkage = self._minimal_linkage(
            repair_receipt_paths=(".aragora/repair/pr-6307-attempt-1.json",),
        )
        assert isinstance(linkage.repair_receipt_paths, tuple)
        with pytest.raises(AttributeError):
            linkage.repair_receipt_paths.append(  # type: ignore[attr-defined]
                ".aragora/repair/pr-6307-attempt-2.json"
            )

    def test_to_dict_serializes_repair_paths_as_list(self) -> None:
        linkage = self._minimal_linkage(
            repair_receipt_paths=(
                ".aragora/repair/pr-6307-attempt-1.json",
                ".aragora/repair/pr-6307-attempt-2.json",
            ),
        )
        d = linkage.to_dict()
        assert d["repair_receipt_paths"] == [
            ".aragora/repair/pr-6307-attempt-1.json",
            ".aragora/repair/pr-6307-attempt-2.json",
        ]

    def test_empty_brief_receipt_id_allowed_for_legacy_settlements(self) -> None:
        # Pre-#6307 settlements already on disk have no associated
        # BriefReceipt. Consumers must still be able to link them.
        linkage = self._minimal_linkage(brief_receipt_id="")
        assert linkage.brief_receipt_id == ""

    def test_trace_contract_brief_to_settlement_to_repair(self) -> None:
        # Acceptance: "Preserve linkage between machine brief, human
        # settlement, and later repair receipts."
        linkage = self._minimal_linkage(
            brief_receipt_id="brief-001",
            settlement_receipt_path=".aragora/review-queue/settlements/pr-6307-request_changes.json",
            action="request_changes",
            repair_receipt_paths=(
                ".aragora/repair/pr-6307-attempt-1.json",
                ".aragora/repair/pr-6307-attempt-2.json",
            ),
        )
        d = linkage.to_dict()
        # All three audit-trail elements are present and connected.
        assert d["brief_receipt_id"] == "brief-001"
        assert "settlements/" in d["settlement_receipt_path"]
        assert len(d["repair_receipt_paths"]) == 2

    def test_json_roundtrip(self) -> None:
        linkage = self._minimal_linkage()
        roundtrip = json.loads(json.dumps(linkage.to_dict()))
        assert roundtrip["pr_number"] == 6307
        assert roundtrip["action"] == "approve"
        assert roundtrip["advisory_only"] is False


# --- Cross-module contract coherence --------------------------------------


class TestContractCoherence:
    def test_brief_receipt_composes_with_protocol_brief(self) -> None:
        # BriefReceipt must accept the exact ReviewBrief shape defined in
        # aragora.review.protocol without any adapter. If this test fails,
        # #6307 has drifted from #6334.
        from aragora.review.protocol import ReviewBrief as ProtocolBrief

        assert ReviewBrief is ProtocolBrief  # same module, same class
        receipt = _minimal_receipt(brief=_minimal_brief())
        assert isinstance(receipt.brief, ProtocolBrief)
