"""Tests for FusionMixin integration in adapter implementations."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter
from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter
from aragora.knowledge.mound.adapters.evidence_adapter import EvidenceAdapter
from aragora.knowledge.mound.ops.fusion import AdapterValidation, FusedValidation, FusionStrategy


def _make_fusion_result(item_id: str, confidence: float) -> FusedValidation:
    validation = AdapterValidation(
        adapter_name="consensus",
        item_id=item_id,
        confidence=confidence,
        is_valid=True,
    )
    return FusedValidation(
        item_id=item_id,
        fused_confidence=confidence,
        is_valid=True,
        strategy_used=FusionStrategy.WEIGHTED_AVERAGE,
        source_validations=[validation],
        participating_adapters=["consensus"],
    )


class TestBeliefAdapterFusion:
    """BeliefAdapter fusion integration tests."""

    def test_extract_fusible_data_includes_crux_metadata(self):
        adapter = BeliefAdapter()
        km_item = {
            "id": "belief-1",
            "confidence": 0.9,
            "metadata": {
                "is_crux": True,
                "crux_score": 0.42,
                "centrality": 0.11,
                "sources": ["debate-1"],
            },
        }

        fusible = adapter._extract_fusible_data(km_item)

        assert fusible is not None
        assert fusible["confidence"] == 0.9
        assert fusible["is_crux"] is True
        assert fusible["crux_score"] == 0.42
        assert fusible["centrality"] == 0.11
        assert fusible["is_valid"] is True
        assert fusible["sources"] == ["debate-1"]

    def test_apply_fusion_result_updates_record_and_cache(self):
        adapter = BeliefAdapter()
        record_id = "belief-123"
        record = {"id": record_id, "confidence": 0.4, "metadata": {}}
        adapter._beliefs[record_id] = {"confidence": 0.4}

        fusion_result = _make_fusion_result(record_id, 0.87)

        applied = adapter._apply_fusion_result(
            record,
            fusion_result,
            metadata={"reason": "fusion"},
        )

        assert applied is True
        assert record["confidence"] == 0.87
        assert record["km_fused"] is True
        assert record["km_fused_confidence"] == 0.87
        assert record["metadata"]["fusion_metadata"]["reason"] == "fusion"
        assert adapter._beliefs[record_id]["confidence"] == 0.87
        assert adapter._beliefs[record_id]["km_fused"] is True


class TestEvidenceAdapterFusion:
    """EvidenceAdapter fusion integration tests."""

    def test_extract_fusible_data_uses_reliability_and_quality(self):
        adapter = EvidenceAdapter(store=MagicMock())
        km_item = {
            "id": "ev-1",
            "metadata": {
                "reliability_score": 0.75,
                "quality_score": 0.82,
                "sources": ["source-a"],
            },
        }

        fusible = adapter._extract_fusible_data(km_item)

        assert fusible is not None
        assert fusible["confidence"] == 0.75
        assert fusible["reliability"] == 0.75
        assert fusible["quality"] == 0.82
        assert fusible["is_valid"] is True
        assert fusible["sources"] == ["source-a"]

    def test_apply_fusion_result_updates_record(self):
        adapter = EvidenceAdapter(store=MagicMock())
        record_id = "ev-123"
        record = {"id": record_id, "reliability_score": 0.2, "metadata": {}}
        fusion_result = _make_fusion_result(record_id, 0.66)

        applied = adapter._apply_fusion_result(
            record,
            fusion_result,
            metadata={"trace": "km"},
        )

        assert applied is True
        assert record["reliability_score"] == 0.66
        assert record["km_fused"] is True
        assert record["km_fused_confidence"] == 0.66
        assert record["fusion_metadata"]["trace"] == "km"


class TestContinuumAdapterFusion:
    """ContinuumAdapter fusion integration tests."""

    def test_extract_fusible_data_reads_tier_and_importance(self):
        adapter = ContinuumAdapter(MagicMock())
        km_item = {
            "id": "cm-1",
            "confidence": 0.55,
            "importance": 0.72,
            "metadata": {"tier": "S"},
        }

        fusible = adapter._extract_fusible_data(km_item)

        assert fusible is not None
        assert fusible["confidence"] == 0.55
        assert fusible["tier"] == "S"
        assert fusible["importance"] == 0.72
        assert fusible["is_valid"] is True

    def test_apply_fusion_result_updates_continuum(self):
        continuum = MagicMock()
        adapter = ContinuumAdapter(continuum)
        record = SimpleNamespace(id="cm-123", metadata={"tier": "A"})
        fusion_result = _make_fusion_result("cm-123", 0.78)

        applied = adapter._apply_fusion_result(
            record,
            fusion_result,
            metadata={"origin": "fusion"},
        )

        assert applied is True
        continuum.update.assert_called_once()
        call_args = continuum.update.call_args
        assert call_args.args[0] == "cm-123"
        assert call_args.kwargs["importance"] == 0.78
        updated_metadata = call_args.kwargs["metadata"]
        assert updated_metadata["km_fused"] is True
        assert updated_metadata["km_fused_confidence"] == 0.78
        assert updated_metadata["fusion_metadata"]["origin"] == "fusion"
