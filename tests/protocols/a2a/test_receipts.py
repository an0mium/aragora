"""Tests for the A2A agent-readable decision receipt envelope (AGT-02 sub-deliverable)."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from aragora.protocols.a2a.receipts import (
    AGENT_RECEIPT_SCHEMA_VERSION,
    DEFAULT_FRESHNESS_SLA_SECONDS,
    DEFAULT_SIGNATURE_ALGORITHM,
    DEFAULT_SETTLEMENT_WINDOW_SECONDS,
    AgentReceipt,
    DissentEntry,
    ReputationDelta,
)

ISSUER_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
ATTACKER_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(bytes(range(32, 64)))
ISSUER_PUBLIC_KEY = ISSUER_PRIVATE_KEY.public_key()
ATTACKER_PUBLIC_KEY = ATTACKER_PRIVATE_KEY.public_key()
SIGNATURE_KEY_ID = "test-issuer-key-v1"


def _build_basic_receipt(**overrides) -> AgentReceipt:
    base = dict(
        issuer="aragora.ai",
        subject_kind="decision",
        subject={"decision": "ship", "rationale": "tests pass"},
        issued_at="2026-04-17T12:00:00Z",
        signing_key=ISSUER_PRIVATE_KEY,
        signature_key_id=SIGNATURE_KEY_ID,
    )
    base.update(overrides)
    return AgentReceipt.build(**base)


class TestDissentEntry:
    def test_validates_confidence(self) -> None:
        with pytest.raises(ValueError):
            DissentEntry(agent_id="a", statement="x", confidence=1.5)
        with pytest.raises(ValueError):
            DissentEntry(agent_id="a", statement="x", confidence=-0.1)

    def test_requires_non_empty_fields(self) -> None:
        with pytest.raises(ValueError):
            DissentEntry(agent_id=" ", statement="x")
        with pytest.raises(ValueError):
            DissentEntry(agent_id="a", statement="")

    def test_json_roundtrip(self) -> None:
        d = DissentEntry(agent_id="alice", statement="x is risky", confidence=0.4)
        assert DissentEntry.from_json(d.to_json()) == d


class TestReputationDelta:
    def test_json_roundtrip(self) -> None:
        delta = ReputationDelta(
            agent_id="alice",
            domain="prediction_market",
            delta=-0.12,
            reason="manifold resolution wrong",
        )
        assert ReputationDelta.from_json(delta.to_json()) == delta


class TestAgentReceiptBuild:
    def test_build_requires_non_empty_issuer(self) -> None:
        with pytest.raises(ValueError):
            AgentReceipt.build(
                issuer=" ",
                subject_kind="decision",
                subject={"x": 1},
                signing_key=ISSUER_PRIVATE_KEY,
            )

    def test_build_requires_non_empty_subject_kind(self) -> None:
        with pytest.raises(ValueError):
            AgentReceipt.build(
                issuer="aragora.ai",
                subject_kind="",
                subject={"x": 1},
                signing_key=ISSUER_PRIVATE_KEY,
            )

    def test_build_validates_freshness_sla(self) -> None:
        with pytest.raises(ValueError):
            AgentReceipt.build(
                issuer="aragora.ai",
                subject_kind="decision",
                subject={"x": 1},
                freshness_sla_seconds=0,
                signing_key=ISSUER_PRIVATE_KEY,
            )

    def test_build_validates_settlement_window(self) -> None:
        with pytest.raises(ValueError):
            AgentReceipt.build(
                issuer="aragora.ai",
                subject_kind="decision",
                subject={"x": 1},
                settlement_window_seconds=-1,
                signing_key=ISSUER_PRIVATE_KEY,
            )

    def test_build_requires_signing_key(self) -> None:
        with pytest.raises(ValueError, match="signing_key"):
            AgentReceipt.build(
                issuer="aragora.ai",
                subject_kind="decision",
                subject={"x": 1},
            )

    def test_build_requires_non_empty_signature_key_id(self) -> None:
        with pytest.raises(ValueError, match="signature_key_id"):
            AgentReceipt.build(
                issuer="aragora.ai",
                subject_kind="decision",
                subject={"x": 1},
                signing_key=ISSUER_PRIVATE_KEY,
                signature_key_id=" ",
            )

    def test_build_rejects_public_verification_key_as_signing_key(self) -> None:
        with pytest.raises(ValueError, match="private key"):
            AgentReceipt.build(
                issuer="aragora.ai",
                subject_kind="decision",
                subject={"x": 1},
                signing_key=ISSUER_PUBLIC_KEY,
            )

    def test_default_freshness_and_settlement_applied(self) -> None:
        receipt = _build_basic_receipt()
        assert receipt.freshness_sla_seconds == DEFAULT_FRESHNESS_SLA_SECONDS
        assert receipt.settlement_window_seconds == DEFAULT_SETTLEMENT_WINDOW_SECONDS

    def test_schema_version_recorded(self) -> None:
        receipt = _build_basic_receipt()
        assert receipt.schema_version == AGENT_RECEIPT_SCHEMA_VERSION
        assert receipt.signature_algorithm == DEFAULT_SIGNATURE_ALGORITHM
        assert receipt.signature_key_id == SIGNATURE_KEY_ID
        assert receipt.verification_key_sha256
        assert receipt.to_json()["verification_key_sha256"] == receipt.verification_key_sha256


class TestContentAddressing:
    def test_identical_inputs_yield_identical_receipt_id(self) -> None:
        a = _build_basic_receipt()
        b = _build_basic_receipt()
        assert a.receipt_id == b.receipt_id
        assert a.signature == b.signature

    def test_different_subject_yields_different_receipt_id(self) -> None:
        a = _build_basic_receipt(subject={"decision": "ship", "rationale": "tests pass"})
        b = _build_basic_receipt(subject={"decision": "hold", "rationale": "tests pass"})
        assert a.receipt_id != b.receipt_id
        assert a.signature != b.signature

    def test_different_dissent_changes_signature(self) -> None:
        a = _build_basic_receipt()
        b = _build_basic_receipt(
            dissent=(DissentEntry(agent_id="bob", statement="risky", confidence=0.6),),
        )
        assert a.signature != b.signature

    def test_different_freshness_sla_changes_signature(self) -> None:
        a = _build_basic_receipt(freshness_sla_seconds=3600)
        b = _build_basic_receipt(freshness_sla_seconds=7200)
        assert a.signature != b.signature

    def test_different_signing_key_preserves_content_id_but_changes_signature(self) -> None:
        a = _build_basic_receipt(signing_key=ISSUER_PRIVATE_KEY)
        b = _build_basic_receipt(signing_key=ATTACKER_PRIVATE_KEY)
        assert a.receipt_id == b.receipt_id
        assert a.signature != b.signature
        assert a.verification_key_sha256 != b.verification_key_sha256


class TestSignatureVerification:
    def test_verify_signature_succeeds_on_valid_receipt(self) -> None:
        receipt = _build_basic_receipt()
        assert receipt.verify_signature(verification_key=ISSUER_PUBLIC_KEY) is True

    def test_verify_signature_uses_key_resolver(self) -> None:
        receipt = _build_basic_receipt()

        def resolve_key(issuer: str, key_id: str):
            assert issuer == "aragora.ai"
            assert key_id == SIGNATURE_KEY_ID
            return ISSUER_PUBLIC_KEY

        assert receipt.verify_signature(key_resolver=resolve_key) is True

    def test_verify_signature_fails_without_verification_key(self) -> None:
        receipt = _build_basic_receipt()
        assert receipt.verify_signature() is False

    def test_verify_signature_fails_with_wrong_key(self) -> None:
        receipt = _build_basic_receipt()
        assert receipt.verify_signature(verification_key=ATTACKER_PUBLIC_KEY) is False

    def test_verify_signature_fails_when_subject_mutated(self) -> None:
        receipt = _build_basic_receipt()
        tampered = replace(
            receipt,
            subject={"decision": "TAMPERED", "rationale": "tests pass"},
        )
        assert tampered.verify_signature(verification_key=ISSUER_PUBLIC_KEY) is False

    def test_verify_signature_fails_when_dissent_mutated(self) -> None:
        receipt = _build_basic_receipt(
            dissent=(DissentEntry(agent_id="bob", statement="risky"),),
        )
        # Reuse signature but with no dissent
        tampered = replace(
            receipt,
            dissent=(),
        )
        assert tampered.verify_signature(verification_key=ISSUER_PUBLIC_KEY) is False

    def test_attacker_rebuilt_changed_subject_fails_under_issuer_key(self) -> None:
        original = _build_basic_receipt()
        forged = _build_basic_receipt(
            subject={"decision": "hold", "rationale": "attacker changed subject"},
            signing_key=ATTACKER_PRIVATE_KEY,
        )

        assert original.verify_signature(verification_key=ISSUER_PUBLIC_KEY) is True
        assert forged.issuer == original.issuer
        assert forged.signature_key_id == original.signature_key_id
        assert forged.verify_signature(verification_key=ISSUER_PUBLIC_KEY) is False
        assert forged.verify_signature(verification_key=ATTACKER_PUBLIC_KEY) is True

    def test_verify_signature_fails_when_signature_metadata_mutated(self) -> None:
        receipt = _build_basic_receipt()
        tampered = replace(receipt, signature_key_id="attacker-key")
        assert tampered.verify_signature(verification_key=ISSUER_PUBLIC_KEY) is False


class TestFreshnessAndSettlement:
    def test_is_fresh_within_window(self) -> None:
        receipt = _build_basic_receipt(
            issued_at="2026-04-17T12:00:00Z",
            freshness_sla_seconds=3600,
        )
        assert receipt.is_fresh(now=datetime(2026, 4, 17, 12, 30, tzinfo=UTC)) is True

    def test_is_stale_after_window(self) -> None:
        receipt = _build_basic_receipt(
            issued_at="2026-04-17T12:00:00Z",
            freshness_sla_seconds=3600,
        )
        assert receipt.is_fresh(now=datetime(2026, 4, 17, 14, 0, tzinfo=UTC)) is False

    def test_is_settled_after_settlement_window(self) -> None:
        receipt = _build_basic_receipt(
            issued_at="2026-04-17T12:00:00Z",
            settlement_window_seconds=3600,
        )
        assert receipt.is_settled(now=datetime(2026, 4, 17, 11, 0, tzinfo=UTC)) is False
        assert receipt.is_settled(now=datetime(2026, 4, 17, 13, 1, tzinfo=UTC)) is True

    def test_freshness_returns_false_on_unparseable_timestamp(self) -> None:
        receipt = _build_basic_receipt(
            issued_at="2026-04-17T12:00:00Z",
            freshness_sla_seconds=3600,
        )
        # Manually construct a receipt with a bogus issued_at to exercise the branch
        broken = replace(
            receipt,
            issued_at="not-a-date",
        )
        assert broken.is_fresh() is False
        assert broken.is_settled() is False


class TestJsonRoundtrip:
    def test_roundtrip_preserves_signature_and_fields(self) -> None:
        receipt = _build_basic_receipt(
            cruxset={
                "cruxset_id": "crxset_a",
                "checksum": "deadbeef",
                "cruxes": [{"crux_id": "c1"}],
            },
            dissent=(
                DissentEntry(agent_id="bob", statement="risky", confidence=0.6),
                DissentEntry(agent_id="carol", statement="overspecified"),
            ),
            reputation_deltas_applied=(
                ReputationDelta(agent_id="alice", domain="debate_position", delta=0.05),
            ),
            freshness_sla_seconds=600,
            settlement_window_seconds=3600,
            provenance={"debate_id": "d1", "arena_run_id": "r1"},
        )
        payload = receipt.to_json()
        roundtrip = AgentReceipt.from_json(payload)
        assert roundtrip == receipt
        assert roundtrip.verify_signature(verification_key=ISSUER_PUBLIC_KEY) is True

    def test_roundtrip_without_optional_fields(self) -> None:
        receipt = _build_basic_receipt()
        roundtrip = AgentReceipt.from_json(receipt.to_json())
        assert roundtrip == receipt

    def test_from_json_tolerates_missing_optional_fields(self) -> None:
        # Build a minimal payload by hand; from_json should populate defaults
        receipt = _build_basic_receipt()
        payload = receipt.to_json()
        del payload["dissent"]
        del payload["reputation_deltas_applied"]
        del payload["provenance"]
        del payload["cruxset"]
        rebuilt = AgentReceipt.from_json(payload)
        assert rebuilt.dissent == ()
        assert rebuilt.reputation_deltas_applied == ()
        assert rebuilt.provenance == {}
        assert rebuilt.cruxset is None
        assert rebuilt.verify_signature(verification_key=ISSUER_PUBLIC_KEY) is True


class TestForwardCompatibility:
    def test_ignores_unknown_top_level_fields_when_loading(self) -> None:
        receipt = _build_basic_receipt()
        payload = receipt.to_json()
        payload["future_field_we_do_not_know"] = {"speculative": True}
        rebuilt = AgentReceipt.from_json(payload)
        # Round-trip works; the unknown field is dropped (the contract is
        # explicit about what we serialize), but loading does not crash.
        assert rebuilt.receipt_id == receipt.receipt_id
        # Signature stays valid because the unknown field was not part of
        # the canonical payload that produced the original signature.
        assert rebuilt.verify_signature(verification_key=ISSUER_PUBLIC_KEY) is True
