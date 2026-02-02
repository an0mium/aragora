"""Tests for EvidenceAdapter — bridges EvidenceStore to Knowledge Mound.

Note: Error handling paths are covered in test_evidence_adapter_errors.py.
This file covers the happy-path functionality.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from aragora.knowledge.mound.adapters.evidence_adapter import (
    EvidenceAdapter,
    EvidenceAdapterError,
    EvidenceNotFoundError,
    EvidenceSearchResult,
    EvidenceStoreUnavailableError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.search_evidence = MagicMock(return_value=[])
    store.get_evidence = MagicMock(return_value=None)
    store.get_evidence_by_hash = MagicMock(return_value=None)
    store.save_evidence = MagicMock(return_value="ev_123")
    store.get_stats = MagicMock(return_value={"total": 100})
    store.get_debate_evidence = MagicMock(return_value=[])
    store.mark_used_in_consensus = MagicMock()
    store.update_evidence = MagicMock()
    return store


@pytest.fixture
def adapter(mock_store):
    return EvidenceAdapter(store=mock_store)


# =============================================================================
# Initialization
# =============================================================================


class TestEvidenceAdapterInit:
    def test_init(self, mock_store):
        a = EvidenceAdapter(store=mock_store)
        assert a.evidence_store is mock_store
        assert a.adapter_name == "evidence"

    def test_init_no_store(self):
        a = EvidenceAdapter(store=None)
        assert a.evidence_store is None

    def test_init_with_options(self, mock_store):
        cb = MagicMock()
        a = EvidenceAdapter(
            store=mock_store,
            enable_dual_write=True,
            event_callback=cb,
        )
        assert a._enable_dual_write is True
        assert a._event_callback is cb


# =============================================================================
# search_by_topic
# =============================================================================


class TestSearchByTopic:
    def test_basic_search(self, adapter, mock_store):
        results = [{"id": "1", "source": "web"}]
        mock_store.search_evidence.return_value = results

        found = adapter.search_by_topic("contract law", limit=5)

        mock_store.search_evidence.assert_called_once_with(
            query="contract law", limit=5, min_reliability=0.0
        )
        assert found == results

    def test_filter_by_source(self, adapter, mock_store):
        mock_store.search_evidence.return_value = [
            {"id": "1", "source": "github"},
            {"id": "2", "source": "web"},
        ]

        found = adapter.search_by_topic("test", source="github")
        assert len(found) == 1
        assert found[0]["source"] == "github"

    def test_min_reliability(self, adapter, mock_store):
        adapter.search_by_topic("q", min_reliability=0.8)
        mock_store.search_evidence.assert_called_once_with(query="q", limit=10, min_reliability=0.8)


# =============================================================================
# search_similar
# =============================================================================


class TestSearchSimilar:
    def test_hash_lookup_found(self, adapter, mock_store):
        existing = {"id": "ev_dup", "content": "test"}
        mock_store.get_evidence_by_hash.return_value = existing

        result = adapter.search_similar("test content")
        assert result == [existing]

    def test_hash_lookup_miss_falls_to_text(self, adapter, mock_store):
        mock_store.get_evidence_by_hash.return_value = None
        mock_store.search_evidence.return_value = [{"id": "1"}]

        result = adapter.search_similar("some long content for search")
        assert len(result) == 1

    def test_no_hash_support_falls_to_text(self, adapter, mock_store):
        del mock_store.get_evidence_by_hash
        mock_store.search_evidence.return_value = []

        result = adapter.search_similar("content")
        assert result == []


# =============================================================================
# get
# =============================================================================


class TestGet:
    def test_get_found(self, adapter, mock_store):
        mock_store.get_evidence.return_value = {"id": "123", "snippet": "test"}
        result = adapter.get("123")
        assert result["id"] == "123"

    def test_get_strips_prefix(self, adapter, mock_store):
        mock_store.get_evidence.return_value = {"id": "123"}
        adapter.get("ev_123")
        mock_store.get_evidence.assert_called_once_with("123")

    def test_get_not_found(self, adapter, mock_store):
        mock_store.get_evidence.return_value = None
        assert adapter.get("missing") is None


# =============================================================================
# to_knowledge_item
# =============================================================================


class TestToKnowledgeItem:
    def test_high_reliability(self, adapter):
        ev = {
            "id": "e1",
            "snippet": "Evidence text",
            "reliability_score": 0.95,
            "source": "github",
            "title": "Test Evidence",
            "url": "https://example.com",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-02T00:00:00+00:00",
        }
        item = adapter.to_knowledge_item(ev)

        assert item.id == "ev_e1"
        assert item.content == "Evidence text"
        assert item.metadata["reliability_score"] == 0.95
        assert item.metadata["source"] == "github"

    def test_confidence_mapping(self, adapter):
        """Maps reliability score to confidence level."""
        for score, expected in [
            (0.95, "verified"),
            (0.75, "high"),
            (0.55, "medium"),
            (0.35, "low"),
            (0.1, "unverified"),
        ]:
            ev = {"id": "e1", "snippet": "test", "reliability_score": score}
            item = adapter.to_knowledge_item(ev)
            assert expected in str(item.confidence).lower(), (
                f"Expected {expected} for score {score}"
            )

    def test_quality_scores_json_parsing(self, adapter):
        import json

        ev = {
            "id": "e1",
            "snippet": "test",
            "quality_scores_json": json.dumps({"relevance": 0.8, "accuracy": 0.9}),
        }
        item = adapter.to_knowledge_item(ev)
        assert item.metadata["quality_scores"]["relevance"] == 0.8

    def test_invalid_quality_scores_json(self, adapter):
        ev = {"id": "e1", "snippet": "test", "quality_scores_json": "not valid json"}
        item = adapter.to_knowledge_item(ev)
        assert item.metadata["quality_scores"] == {}

    def test_enriched_metadata_json(self, adapter):
        import json

        ev = {
            "id": "e1",
            "snippet": "test",
            "enriched_metadata_json": json.dumps({"extra": "data"}),
        }
        item = adapter.to_knowledge_item(ev)
        assert item.metadata["enriched"]["extra"] == "data"

    def test_timestamp_parsing(self, adapter):
        ev = {
            "id": "e1",
            "snippet": "test",
            "created_at": "2026-01-15T10:30:00Z",
            "updated_at": "2026-01-16T12:00:00Z",
        }
        item = adapter.to_knowledge_item(ev)
        assert item.created_at.year == 2026

    def test_missing_timestamps(self, adapter):
        ev = {"id": "e1", "snippet": "test"}
        item = adapter.to_knowledge_item(ev)
        # Should use datetime.now() as fallback
        assert isinstance(item.created_at, datetime)

    def test_default_reliability(self, adapter):
        ev = {"id": "e1", "snippet": "test"}
        item = adapter.to_knowledge_item(ev)
        assert item.metadata["reliability_score"] == 0.5


# =============================================================================
# store
# =============================================================================


class TestStore:
    def test_store_evidence(self, adapter, mock_store):
        result = adapter.store(
            evidence_id="ev_new",
            source="web",
            title="New Evidence",
            snippet="Content here",
            url="https://example.com",
            reliability_score=0.8,
        )
        assert result == "ev_123"
        mock_store.save_evidence.assert_called_once()

    def test_store_with_debate_id(self, adapter, mock_store):
        adapter.store(
            evidence_id="ev_new",
            source="web",
            title="t",
            snippet="s",
            debate_id="debate-1",
        )
        call_kwargs = mock_store.save_evidence.call_args.kwargs
        assert call_kwargs["debate_id"] == "debate-1"


# =============================================================================
# mark_used_in_consensus
# =============================================================================


class TestMarkUsedInConsensus:
    def test_marks_evidence(self, adapter, mock_store):
        adapter.mark_used_in_consensus("ev_1", "debate-1")
        mock_store.mark_used_in_consensus.assert_called_once_with("debate-1", ["ev_1"])

    def test_no_method_on_store(self, adapter, mock_store):
        del mock_store.mark_used_in_consensus
        # Should not raise
        adapter.mark_used_in_consensus("ev_1", "debate-1")


# =============================================================================
# update_reliability_from_km
# =============================================================================


class TestUpdateReliabilityFromKM:
    @pytest.mark.asyncio
    async def test_updates_reliability(self, adapter, mock_store):
        mock_store.get_evidence.return_value = {
            "id": "123",
            "reliability_score": 0.5,
        }

        await adapter.update_reliability_from_km(
            "123",
            {"confidence": 0.9, "validation_count": 5},
        )

        mock_store.update_evidence.assert_called_once()
        call_kwargs = mock_store.update_evidence.call_args
        new_reliability = (
            call_kwargs[1]["reliability_score"]
            if call_kwargs[1]
            else call_kwargs.kwargs["reliability_score"]
        )
        # 0.5 * (1-0.5) + 0.9 * 0.5 = 0.25 + 0.45 = 0.7
        assert 0.69 < new_reliability < 0.71

    @pytest.mark.asyncio
    async def test_strips_ev_prefix(self, adapter, mock_store):
        mock_store.get_evidence.return_value = {"id": "123", "reliability_score": 0.5}

        await adapter.update_reliability_from_km(
            "ev_123",
            {"confidence": 0.8, "validation_count": 1},
        )
        mock_store.get_evidence.assert_called_with("123")

    @pytest.mark.asyncio
    async def test_evidence_not_found(self, adapter, mock_store):
        mock_store.get_evidence.return_value = None

        with pytest.raises(EvidenceNotFoundError):
            await adapter.update_reliability_from_km("missing", {"confidence": 0.8})


# =============================================================================
# get_stats and get_debate_evidence
# =============================================================================


class TestStatsAndDebateEvidence:
    def test_get_stats(self, adapter, mock_store):
        assert adapter.get_stats() == {"total": 100}

    def test_get_debate_evidence(self, adapter, mock_store):
        mock_store.get_debate_evidence.return_value = [{"id": "1"}]
        result = adapter.get_debate_evidence("debate-1")
        assert len(result) == 1


# =============================================================================
# SemanticSearchMixin methods
# =============================================================================


class TestSemanticSearchMixin:
    def test_get_record_by_id(self, adapter, mock_store):
        mock_store.get_evidence.return_value = {"id": "123"}
        result = adapter._get_record_by_id("ev_123")
        mock_store.get_evidence.assert_called_with("123")
        assert result["id"] == "123"

    def test_get_record_by_id_no_store(self):
        a = EvidenceAdapter(store=None)
        assert a._get_record_by_id("123") is None

    def test_record_to_dict_from_dict(self, adapter):
        record = {"id": "1", "content": "test"}
        d = adapter._record_to_dict(record, similarity=0.8)
        assert d["id"] == "1"
        assert d["similarity"] == 0.8

    def test_record_to_dict_from_object(self, adapter):
        record = MagicMock()
        record.id = "1"
        record.content = "test"
        record.source = "web"
        record.reliability = 0.9
        record.quality = 0.8
        record.topics = ["law"]
        record.metadata = {"k": "v"}
        d = adapter._record_to_dict(record, similarity=0.7)
        assert d["id"] == "1"
        assert d["similarity"] == 0.7

    def test_extract_record_id(self, adapter):
        assert adapter._extract_record_id("ev_123") == "123"
        assert adapter._extract_record_id("456") == "456"


# =============================================================================
# FusionMixin methods
# =============================================================================


class TestFusionMixin:
    def test_get_fusion_sources(self, adapter):
        sources = adapter._get_fusion_sources()
        assert "consensus" in sources
        assert "elo" in sources

    def test_extract_fusible_data(self, adapter):
        km_item = {
            "id": "item-1",
            "confidence": 0.8,
            "metadata": {"reliability_score": 0.9, "quality_score": 0.7},
        }
        data = adapter._extract_fusible_data(km_item)
        assert data["confidence"] == 0.8
        assert data["reliability"] == 0.9
        assert data["is_valid"] is True

    def test_apply_fusion_result_dict(self, adapter):
        record = {"id": "1", "reliability_score": 0.5}
        fusion = MagicMock(fused_confidence=0.85)

        result = adapter._apply_fusion_result(record, fusion)
        assert result is True
        assert record["reliability_score"] == 0.85
        assert record["km_fused"] is True

    def test_apply_fusion_result_no_confidence(self, adapter):
        record = {"id": "1"}
        fusion = MagicMock(spec=[])  # no fused_confidence attribute

        result = adapter._apply_fusion_result(record, fusion)
        assert result is False

    def test_apply_fusion_result_object(self, adapter):
        record = MagicMock()
        record.metadata = {}
        fusion = MagicMock(fused_confidence=0.9)

        result = adapter._apply_fusion_result(record, fusion)
        assert result is True


# =============================================================================
# EvidenceSearchResult dataclass
# =============================================================================


class TestEvidenceSearchResult:
    def test_defaults(self):
        r = EvidenceSearchResult(evidence={"id": "1"})
        assert r.relevance_score == 0.0
        assert r.matched_topics == []

    def test_with_values(self):
        r = EvidenceSearchResult(
            evidence={"id": "1"},
            relevance_score=0.9,
            matched_topics=["law", "contracts"],
        )
        assert r.matched_topics == ["law", "contracts"]
