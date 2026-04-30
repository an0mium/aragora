"""Tests for the Gauntlet -> Epistemic CruxReceipt bridge.

Round 2026-04-30c — Phase D. Closes the receipt-lineage break documented in
docs/plans/2026-04-28-dialectical-runtime-integration-audit.md.
"""

from __future__ import annotations

from typing import Any

import pytest

from aragora.epistemic.crux_receipt import CruxEntry, CruxReceipt as EpistemicCruxReceipt
from aragora.epistemic.gauntlet_crux_bridge import (
    from_gauntlet_receipt,
    km_crux_ingestion_enabled,
)
from aragora.gauntlet.receipt_models import CruxReceipt as GauntletCruxReceipt


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _gauntlet_crux_dict(
    *,
    claim_id: str = "claim.x",
    crux_score: float = 0.85,
    uncertainty_score: float = 0.6,
    contesting_agents: list[str] | None = None,
    affected_claims: list[str] | None = None,
    resolution_impact: float = 0.5,
) -> dict[str, Any]:
    """A gauntlet-shaped crux dict matching CruxClaim.to_dict()."""
    return {
        "claim_id": claim_id,
        "statement": f"statement of {claim_id}",
        "author": "alice",
        "crux_score": crux_score,
        "influence_score": 0.7,
        "disagreement_score": 0.8,
        "uncertainty_score": uncertainty_score,
        "centrality_score": 0.4,
        "affected_claims": affected_claims or [],
        "contesting_agents": contesting_agents or ["alice", "bob"],
        "resolution_impact": resolution_impact,
    }


def _gauntlet_receipt(
    *,
    cruxes: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    recommended_focus: list[str] | None = None,
    resolution_strategies: list[dict[str, Any]] | None = None,
    raw_claims_hash: str = "abc123",
) -> GauntletCruxReceipt:
    # Use ``is None`` rather than ``or`` so callers can pass ``[]``/``{}``
    # to override defaults without falling back to the default value.
    return GauntletCruxReceipt(
        receipt_id="crux-deadbeef",
        debate_id="debate.42",
        question="Should we ship?",
        timestamp="2026-04-30T03:00:00+00:00",
        agents=["alice", "bob"],
        rounds=3,
        cruxes=cruxes if cruxes is not None else [_gauntlet_crux_dict()],
        convergence_barrier=0.65,
        counterfactuals=[{"if": "X", "then": "Y"}],
        recommended_focus=(
            recommended_focus if recommended_focus is not None else ["focus.1", "focus.2"]
        ),
        resolution_strategies=(
            resolution_strategies
            if resolution_strategies is not None
            else [{"strategy": "consult"}]
        ),
        raw_claims_hash=raw_claims_hash,
        metadata=metadata if metadata is not None else {"source": "test"},
    )


# ---------------------------------------------------------------------------
# Conversion correctness
# ---------------------------------------------------------------------------


class TestEntryConversion:
    """Each gauntlet crux dict converts to the right CruxEntry shape."""

    def test_field_mapping(self) -> None:
        g = _gauntlet_receipt()
        e = from_gauntlet_receipt(g)
        assert isinstance(e, EpistemicCruxReceipt)
        assert len(e.cruxes) == 1
        entry = e.cruxes[0]
        assert isinstance(entry, CruxEntry)
        # claim_id -> crux_id, crux_score -> load_bearing_score, others by name.
        assert entry.crux_id == "claim.x"
        assert entry.statement == "statement of claim.x"
        assert entry.load_bearing_score == 0.85
        assert entry.uncertainty_score == 0.6
        assert entry.contesting_agents == ["alice", "bob"]
        assert entry.affected_claims == []
        assert entry.resolution_impact == 0.5

    def test_multiple_cruxes(self) -> None:
        g = _gauntlet_receipt(
            cruxes=[
                _gauntlet_crux_dict(claim_id="a"),
                _gauntlet_crux_dict(claim_id="b"),
                _gauntlet_crux_dict(claim_id="c"),
            ]
        )
        e = from_gauntlet_receipt(g)
        assert [c.crux_id for c in e.cruxes] == ["a", "b", "c"]

    def test_empty_cruxes(self) -> None:
        g = _gauntlet_receipt(cruxes=[])
        e = from_gauntlet_receipt(g)
        assert e.cruxes == []

    def test_missing_optional_fields_default_safely(self) -> None:
        g = _gauntlet_receipt(
            cruxes=[
                {
                    "claim_id": "minimal",
                    "statement": "minimal claim",
                    # no other fields
                }
            ]
        )
        e = from_gauntlet_receipt(g)
        entry = e.cruxes[0]
        assert entry.crux_id == "minimal"
        assert entry.load_bearing_score == 0.0
        assert entry.uncertainty_score == 0.0
        assert entry.contesting_agents == []
        assert entry.affected_claims == []
        assert entry.resolution_impact == 0.0


class TestReceiptShape:
    """Top-level receipt fields map correctly."""

    def test_basic_fields_carry_over(self) -> None:
        g = _gauntlet_receipt()
        e = from_gauntlet_receipt(g)
        assert e.debate_id == "debate.42"
        assert e.question == "Should we ship?"
        assert e.agents == ["alice", "bob"]
        assert e.rounds == 3
        assert e.convergence_barrier == 0.65
        assert e.counterfactuals == [{"if": "X", "then": "Y"}]

    def test_receipt_id_minted_fresh_by_default(self) -> None:
        g = _gauntlet_receipt()
        e = from_gauntlet_receipt(g)
        # Fresh epistemic-style id, NOT the gauntlet id.
        assert e.receipt_id != "crux-deadbeef"
        assert e.receipt_id.startswith("crux_rcpt_")
        assert len(e.receipt_id) == len("crux_rcpt_") + 16

    def test_receipt_id_preserved_when_requested(self) -> None:
        g = _gauntlet_receipt()
        e = from_gauntlet_receipt(g, preserve_receipt_id=True)
        assert e.receipt_id == "crux-deadbeef"

    def test_checksum_recomputed_for_epistemic_shape(self) -> None:
        g = _gauntlet_receipt()
        e = from_gauntlet_receipt(g)
        assert isinstance(e.checksum, str)
        # 64-char SHA-256 hex per the epistemic schema (gauntlet's checksum
        # is a property, truncated to 16 chars; we recompute for full length).
        assert len(e.checksum) == 64
        # All hex characters.
        assert all(c in "0123456789abcdef" for c in e.checksum)

    def test_checksum_changes_when_cruxes_change(self) -> None:
        g1 = _gauntlet_receipt(cruxes=[_gauntlet_crux_dict(claim_id="a")])
        g2 = _gauntlet_receipt(cruxes=[_gauntlet_crux_dict(claim_id="b")])
        e1 = from_gauntlet_receipt(g1, preserve_receipt_id=True)
        e2 = from_gauntlet_receipt(g2, preserve_receipt_id=True)
        # Same receipt_id but different cruxes -> different checksum.
        assert e1.receipt_id == e2.receipt_id
        assert e1.checksum != e2.checksum


class TestMetadataPreservation:
    """Gauntlet-only fields are carried into metadata as gauntlet_* keys."""

    def test_original_metadata_preserved(self) -> None:
        g = _gauntlet_receipt(metadata={"foo": "bar", "tag": "x"})
        e = from_gauntlet_receipt(g)
        assert e.metadata["foo"] == "bar"
        assert e.metadata["tag"] == "x"

    def test_gauntlet_receipt_id_preserved_in_metadata(self) -> None:
        g = _gauntlet_receipt()
        e = from_gauntlet_receipt(g)  # default: fresh id minted
        assert e.metadata["gauntlet_receipt_id"] == "crux-deadbeef"

    def test_gauntlet_timestamp_preserved(self) -> None:
        g = _gauntlet_receipt()
        e = from_gauntlet_receipt(g)
        assert e.metadata["gauntlet_timestamp"] == "2026-04-30T03:00:00+00:00"

    def test_recommended_focus_preserved(self) -> None:
        g = _gauntlet_receipt(recommended_focus=["a", "b"])
        e = from_gauntlet_receipt(g)
        assert e.metadata["gauntlet_recommended_focus"] == ["a", "b"]

    def test_resolution_strategies_preserved(self) -> None:
        g = _gauntlet_receipt(resolution_strategies=[{"strategy": "consult-domain-expert"}])
        e = from_gauntlet_receipt(g)
        assert e.metadata["gauntlet_resolution_strategies"] == [
            {"strategy": "consult-domain-expert"}
        ]

    def test_raw_claims_hash_preserved(self) -> None:
        g = _gauntlet_receipt(raw_claims_hash="hash-xyz")
        e = from_gauntlet_receipt(g)
        assert e.metadata["gauntlet_raw_claims_hash"] == "hash-xyz"

    def test_user_provided_metadata_takes_precedence_over_gauntlet_provenance(self) -> None:
        """If the gauntlet receipt's own metadata already has a gauntlet_* key,
        the bridge does not overwrite it (uses ``setdefault``)."""
        g = _gauntlet_receipt(metadata={"gauntlet_receipt_id": "user-override"})
        e = from_gauntlet_receipt(g)
        assert e.metadata["gauntlet_receipt_id"] == "user-override"


# ---------------------------------------------------------------------------
# KM ingestion flag (default off)
# ---------------------------------------------------------------------------


class TestKmIngestionFlag:
    """``km_crux_ingestion_enabled`` reflects ARAGORA_KM_CRUX_INGESTION_ENABLED."""

    def test_default_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARAGORA_KM_CRUX_INGESTION_ENABLED", raising=False)
        assert km_crux_ingestion_enabled() is False

    def test_explicit_off_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for val in ("0", "false", "no", "off", ""):
            monkeypatch.setenv("ARAGORA_KM_CRUX_INGESTION_ENABLED", val)
            assert km_crux_ingestion_enabled() is False, f"{val!r} should be off"

    def test_explicit_on_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for val in ("1", "true", "yes", "on", "TRUE"):
            monkeypatch.setenv("ARAGORA_KM_CRUX_INGESTION_ENABLED", val)
            assert km_crux_ingestion_enabled() is True, f"{val!r} should be on"

    def test_monkeypatch_sets_flag_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Per the audit, no ``enable_*`` helper exists; callers set the env
        var directly (in tests via ``monkeypatch.setenv``)."""
        monkeypatch.delenv("ARAGORA_KM_CRUX_INGESTION_ENABLED", raising=False)
        assert km_crux_ingestion_enabled() is False
        monkeypatch.setenv("ARAGORA_KM_CRUX_INGESTION_ENABLED", "1")
        assert km_crux_ingestion_enabled() is True

    def test_bridge_construction_does_not_check_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Construction is always safe — flag gates side effects only.

        from_gauntlet_receipt itself never reads the flag; the flag is a
        contract for callers about whether they may dispatch KM ingestion.
        """
        monkeypatch.delenv("ARAGORA_KM_CRUX_INGESTION_ENABLED", raising=False)
        g = _gauntlet_receipt()
        # Should not raise, regardless of flag.
        e = from_gauntlet_receipt(g)
        assert isinstance(e, EpistemicCruxReceipt)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


class TestPublicSurface:
    def test_exported_from_aragora_epistemic(self) -> None:
        import aragora.epistemic as ep

        assert hasattr(ep, "from_gauntlet_receipt")
        assert hasattr(ep, "km_crux_ingestion_enabled")
        assert "from_gauntlet_receipt" in ep.__all__
        assert "km_crux_ingestion_enabled" in ep.__all__


# ---------------------------------------------------------------------------
# End-to-end: bridged receipt is shape-compatible with the KM adapter
# ---------------------------------------------------------------------------


class TestKmAdapterCompatibility:
    """The bridged receipt is a real ``aragora.epistemic.CruxReceipt`` and
    can be fed to the KM adapter without manual coercion."""

    def test_bridged_receipt_is_epistemic_class(self) -> None:
        """isinstance check — the adapter declares :class:`CruxReceipt` as its
        ingest type, so the bridged receipt must satisfy that."""
        from aragora.epistemic.crux_receipt import CruxReceipt as Target

        g = _gauntlet_receipt()
        e = from_gauntlet_receipt(g)
        assert isinstance(e, Target)

    def test_bridged_receipt_to_dict_serialisable(self) -> None:
        """The KM adapter calls ``crux.crux_id`` per entry; verify the
        bridged shape supports the expected attribute access pattern."""
        g = _gauntlet_receipt(
            cruxes=[
                _gauntlet_crux_dict(claim_id="c1"),
                _gauntlet_crux_dict(claim_id="c2"),
            ]
        )
        e = from_gauntlet_receipt(g)
        # KM adapter's `for crux in receipt.cruxes: crux.crux_id` pattern.
        ids = [c.crux_id for c in e.cruxes]
        assert ids == ["c1", "c2"]
        # to_dict round-trips (caller may serialise for logging).
        d = e.to_dict()
        assert d["debate_id"] == "debate.42"
        assert len(d["cruxes"]) == 2
