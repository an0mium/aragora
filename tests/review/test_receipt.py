"""Tests for aragora.review.receipt — schema-only BriefReceipt + linkage contracts."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from aragora.review import (
    BriefReceipt,
    DissentingView,
    DissentPosition,
    EvidenceKind,
    EvidenceRef,
    Recommendation,
    ReviewBrief,
    ReviewRole,
    RoleFinding,
    SettlementAction,
    SettlementLinkage,
    ValidationKind,
    ValidationRef,
    ValidationResult,
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
        ref = EvidenceRef(kind=EvidenceKind.FILE, path="aragora/review/receipt.py")
        with pytest.raises((AttributeError, TypeError)):
            ref.path = "elsewhere"  # type: ignore[misc]

    def test_line_range_serializes_as_list(self) -> None:
        ref = EvidenceRef(
            kind=EvidenceKind.FILE,
            path="aragora/review/receipt.py",
            line_range=(42, 58),
            quote="def to_dict(self) -> dict[str, Any]:",
        )
        d = ref.to_dict()
        assert d["line_range"] == [42, 58]
        assert d["kind"] == "file"
        assert d["path"] == "aragora/review/receipt.py"

    def test_line_range_omitted_when_none(self) -> None:
        ref = EvidenceRef(kind=EvidenceKind.COMMIT, path="main@f1a640ee2", sha="f1a640ee2")
        d = ref.to_dict()
        assert d["line_range"] is None

    def test_kind_enum_values_match_canonical_strings(self) -> None:
        # These are the canonical serialized strings consumers branch on.
        # Drift breaks the orchestrator / UI / export contract silently.
        assert EvidenceKind.FILE.value == "file"
        assert EvidenceKind.TEST.value == "test"
        assert EvidenceKind.COMMIT.value == "commit"
        assert EvidenceKind.ARTIFACT.value == "artifact"
        assert EvidenceKind.ISSUE.value == "issue"
        assert EvidenceKind.PR.value == "pr"
        assert EvidenceKind.EXTERNAL.value == "external"

    def test_kind_serialized_as_string(self) -> None:
        ref = EvidenceRef(kind=EvidenceKind.FILE, path="x.py")
        assert ref.to_dict()["kind"] == "file"

    def test_json_roundtrip(self) -> None:
        ref = EvidenceRef(
            kind=EvidenceKind.FILE,
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
        ref = ValidationRef(
            kind=ValidationKind.CI_CHECK, name="lint", result=ValidationResult.SUCCESS
        )
        with pytest.raises((AttributeError, TypeError)):
            ref.result = ValidationResult.FAILURE  # type: ignore[misc]

    def test_kind_enum_values(self) -> None:
        assert ValidationKind.CI_CHECK.value == "ci_check"
        assert ValidationKind.TEST_SUITE.value == "test_suite"
        assert ValidationKind.RECEIPT.value == "receipt"
        assert ValidationKind.BENCHMARK.value == "benchmark"
        assert ValidationKind.MANUAL_REVIEW.value == "manual_review"

    def test_result_enum_values(self) -> None:
        # Mirror the GitHub Actions conclusion vocabulary; this is the
        # contract the orchestrator / UI / export share.
        assert ValidationResult.SUCCESS.value == "success"
        assert ValidationResult.FAILURE.value == "failure"
        assert ValidationResult.SKIPPED.value == "skipped"
        assert ValidationResult.CANCELLED.value == "cancelled"
        assert ValidationResult.PENDING.value == "pending"

    def test_enum_fields_serialize_as_strings(self) -> None:
        ref = ValidationRef(
            kind=ValidationKind.CI_CHECK,
            name="Version Alignment",
            result=ValidationResult.SUCCESS,
        )
        d = ref.to_dict()
        assert d["kind"] == "ci_check"
        assert d["result"] == "success"

    def test_to_dict_roundtrip(self) -> None:
        ref = ValidationRef(
            kind=ValidationKind.CI_CHECK,
            name="Version Alignment",
            result=ValidationResult.SUCCESS,
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
            evidence_refs=(EvidenceRef(kind=EvidenceKind.FILE, path="aragora/review/protocol.py"),),
            validation_refs=(
                ValidationRef(
                    kind=ValidationKind.CI_CHECK, name="lint", result=ValidationResult.SUCCESS
                ),
            ),
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
            evidence_refs=(EvidenceRef(kind=EvidenceKind.FILE, path="x.py"),),
            validation_refs=(
                ValidationRef(
                    kind=ValidationKind.CI_CHECK, name="lint", result=ValidationResult.SUCCESS
                ),
            ),
        )
        assert isinstance(receipt.evidence_refs, tuple)
        assert isinstance(receipt.validation_refs, tuple)
        with pytest.raises(AttributeError):
            receipt.evidence_refs.append(EvidenceRef(kind=EvidenceKind.FILE, path="y.py"))  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            receipt.validation_refs.append(  # type: ignore[attr-defined]
                ValidationRef(
                    kind=ValidationKind.CI_CHECK, name="x", result=ValidationResult.SUCCESS
                )
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
            settlement_receipt_id="settlement-sha-abc",
            settlement_receipt_path=".aragora/review-queue/settlements/pr-6307-f1a640ee2def-approve.json",
            head_sha="f1a640ee2deadbeef",
            packet_sha="packet-sha-locked",
            pr_number=6307,
            repo="synaptent/aragora",
            action=SettlementAction.APPROVE,
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

    def test_action_enum_values(self) -> None:
        # These three are the only valid settlement actions; they match
        # the review-queue `act` CLI so locking them in the contract
        # prevents UI/export from inventing new strings.
        assert SettlementAction.APPROVE.value == "approve"
        assert SettlementAction.REQUEST_CHANGES.value == "request_changes"
        assert SettlementAction.DEFER.value == "defer"

    def test_action_is_an_enum_not_string(self) -> None:
        # P1 regression guard: `action` must be typed, not raw str.
        linkage = self._minimal_linkage()
        assert isinstance(linkage.action, SettlementAction)

    def test_action_serialized_as_string(self) -> None:
        linkage = self._minimal_linkage(action=SettlementAction.REQUEST_CHANGES)
        assert linkage.to_dict()["action"] == "request_changes"

    def test_frozen(self) -> None:
        linkage = self._minimal_linkage()
        with pytest.raises((AttributeError, TypeError)):
            linkage.action = SettlementAction.REQUEST_CHANGES  # type: ignore[misc]

    def test_repair_receipt_paths_is_immutable_tuple(self) -> None:
        linkage = self._minimal_linkage(
            repair_receipt_paths=(".aragora/repair/pr-6307-attempt-1.json",),
        )
        assert isinstance(linkage.repair_receipt_paths, tuple)
        with pytest.raises(AttributeError):
            linkage.repair_receipt_paths.append(  # type: ignore[attr-defined]
                ".aragora/repair/pr-6307-attempt-2.json"
            )

    def test_repair_receipt_ids_is_immutable_tuple(self) -> None:
        linkage = self._minimal_linkage(
            repair_receipt_ids=("repair-sha-001",),
        )
        assert isinstance(linkage.repair_receipt_ids, tuple)
        with pytest.raises(AttributeError):
            linkage.repair_receipt_ids.append("repair-sha-002")  # type: ignore[attr-defined]

    def test_portable_ids_alongside_paths(self) -> None:
        # P2 regression guard: export-portable IDs must exist alongside
        # local-only paths. An exported payload dereferences by ID on
        # another machine; a local consumer may fast-path via the path.
        linkage = self._minimal_linkage(
            settlement_receipt_id="settlement-content-sha-01",
            settlement_receipt_path=".aragora/review-queue/settlements/pr-6307-approve.json",
            repair_receipt_ids=("repair-content-sha-01", "repair-content-sha-02"),
            repair_receipt_paths=(
                ".aragora/repair/pr-6307-attempt-1.json",
                ".aragora/repair/pr-6307-attempt-2.json",
            ),
        )
        assert linkage.settlement_receipt_id == "settlement-content-sha-01"
        assert len(linkage.repair_receipt_ids) == 2
        assert len(linkage.repair_receipt_paths) == 2
        # Both fields survive the JSON trip so external consumers can pick
        # the one appropriate to their address space.
        d = linkage.to_dict()
        assert d["settlement_receipt_id"] == "settlement-content-sha-01"
        assert d["repair_receipt_ids"] == ["repair-content-sha-01", "repair-content-sha-02"]

    def test_to_dict_serializes_both_id_and_path_lists(self) -> None:
        linkage = self._minimal_linkage(
            repair_receipt_ids=("repair-sha-01", "repair-sha-02"),
            repair_receipt_paths=(
                ".aragora/repair/pr-6307-attempt-1.json",
                ".aragora/repair/pr-6307-attempt-2.json",
            ),
        )
        d = linkage.to_dict()
        assert d["repair_receipt_ids"] == ["repair-sha-01", "repair-sha-02"]
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
            settlement_receipt_id="settlement-001",
            settlement_receipt_path=".aragora/review-queue/settlements/pr-6307-request_changes.json",
            action=SettlementAction.REQUEST_CHANGES,
            repair_receipt_ids=("repair-001", "repair-002"),
            repair_receipt_paths=(
                ".aragora/repair/pr-6307-attempt-1.json",
                ".aragora/repair/pr-6307-attempt-2.json",
            ),
        )
        d = linkage.to_dict()
        # All three audit-trail elements are present and connected,
        # both by portable ID and local path.
        assert d["brief_receipt_id"] == "brief-001"
        assert d["settlement_receipt_id"] == "settlement-001"
        assert "settlements/" in d["settlement_receipt_path"]
        assert d["repair_receipt_ids"] == ["repair-001", "repair-002"]
        assert len(d["repair_receipt_paths"]) == 2

    def test_json_roundtrip(self) -> None:
        linkage = self._minimal_linkage()
        roundtrip = json.loads(json.dumps(linkage.to_dict()))
        assert roundtrip["pr_number"] == 6307
        assert roundtrip["action"] == "approve"
        assert roundtrip["advisory_only"] is False
        assert roundtrip["settlement_receipt_id"] == "settlement-sha-abc"

    def test_settlement_receipt_id_preimage_is_documented(self) -> None:
        # Introducing portable IDs without a derivation contract would
        # mean different producers generate different IDs for the same
        # receipt. This test locks the preimage rule in the docstring
        # parallel to ReviewBrief.packet_sha and BriefReceipt.receipt_id.
        from aragora.review.receipt import SettlementLinkage as SL

        doc = SL.__doc__ or ""
        assert "Settlement-receipt-ID preimage" in doc
        assert 'Remove the ``"receipt_path"`` key' in doc
        assert "canonical JSON" in doc
        assert "sha256" in doc.lower()

    def test_repair_receipt_id_preimage_is_documented(self) -> None:
        # Repair receipts use the same rule; guard that the docstring
        # says so explicitly so the future repair lane can't pick a
        # different derivation strategy silently.
        from aragora.review.receipt import SettlementLinkage as SL

        doc = SL.__doc__ or ""
        assert "Repair-receipt-ID preimage" in doc
        assert "same rule" in doc.lower()


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
