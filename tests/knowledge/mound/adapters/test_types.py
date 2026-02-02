"""Tests for shared adapter type definitions (_types.py)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aragora.knowledge.mound.adapters._types import (
    BatchSyncResult,
    SearchResult,
    SyncResult,
    ValidationResult,
    ValidationSyncResult,
)


# =============================================================================
# SyncResult
# =============================================================================


class TestSyncResult:
    def test_defaults(self):
        r = SyncResult()
        assert r.records_synced == 0
        assert r.records_skipped == 0
        assert r.records_failed == 0
        assert r.errors == []
        assert r.duration_ms == 0.0

    def test_init(self):
        r = SyncResult(
            records_synced=10,
            records_skipped=3,
            records_failed=2,
            errors=["err1"],
            duration_ms=45.5,
        )
        assert r.records_synced == 10
        assert r.records_skipped == 3
        assert r.records_failed == 2
        assert r.errors == ["err1"]
        assert r.duration_ms == 45.5

    def test_total_processed(self):
        r = SyncResult(records_synced=5, records_skipped=3, records_failed=2)
        assert r.total_processed == 10

    def test_success_rate_normal(self):
        r = SyncResult(records_synced=8, records_skipped=1, records_failed=1)
        assert r.success_rate == 80.0

    def test_success_rate_zero_total(self):
        r = SyncResult()
        assert r.success_rate == 100.0

    def test_success_rate_zero_synced(self):
        r = SyncResult(records_skipped=5, records_failed=5)
        assert r.success_rate == 0.0

    def test_success_rate_all_synced(self):
        r = SyncResult(records_synced=10)
        assert r.success_rate == 100.0

    def test_to_dict(self):
        r = SyncResult(records_synced=5, errors=["e1"], duration_ms=10.0)
        d = r.to_dict()
        assert d["records_synced"] == 5
        assert d["records_skipped"] == 0
        assert d["records_failed"] == 0
        assert d["errors"] == ["e1"]
        assert d["duration_ms"] == 10.0

    def test_errors_none_becomes_list(self):
        r = SyncResult(errors=None)
        assert r.errors == []

    def test_errors_explicit_list_preserved(self):
        r = SyncResult(errors=["a", "b"])
        assert r.errors == ["a", "b"]


# =============================================================================
# ValidationSyncResult
# =============================================================================


class TestValidationSyncResult:
    def test_defaults(self):
        r = ValidationSyncResult()
        assert r.records_analyzed == 0
        assert r.records_updated == 0
        assert r.records_skipped == 0
        assert r.errors == []
        assert r.duration_ms == 0.0

    def test_init(self):
        r = ValidationSyncResult(
            records_analyzed=20,
            records_updated=15,
            records_skipped=5,
            errors=["timeout"],
            duration_ms=100.0,
        )
        assert r.records_analyzed == 20
        assert r.records_updated == 15

    def test_to_dict(self):
        r = ValidationSyncResult(records_analyzed=10, records_updated=7)
        d = r.to_dict()
        assert d["records_analyzed"] == 10
        assert d["records_updated"] == 7
        assert d["records_skipped"] == 0
        assert d["errors"] == []

    def test_errors_none_becomes_list(self):
        r = ValidationSyncResult(errors=None)
        assert r.errors == []


# =============================================================================
# SearchResult
# =============================================================================


class TestSearchResult:
    def test_defaults(self):
        r = SearchResult(item="test")
        assert r.item == "test"
        assert r.relevance_score == 0.0
        assert r.similarity == 0.0
        assert r.matched_fields == []

    def test_with_values(self):
        r = SearchResult(
            item={"id": "1"},
            relevance_score=0.9,
            similarity=0.85,
            matched_fields=["title", "body"],
        )
        assert r.item == {"id": "1"}
        assert r.relevance_score == 0.9
        assert r.matched_fields == ["title", "body"]

    def test_to_dict_item_with_to_dict(self):
        mock_item = MagicMock()
        mock_item.to_dict.return_value = {"id": "x", "content": "y"}
        r = SearchResult(item=mock_item, relevance_score=0.7)
        d = r.to_dict()
        assert d["item"] == {"id": "x", "content": "y"}
        assert d["relevance_score"] == 0.7

    def test_to_dict_item_without_to_dict(self):
        r = SearchResult(item="plain string", similarity=0.5)
        d = r.to_dict()
        assert d["item"] == "plain string"
        assert d["similarity"] == 0.5

    def test_to_dict_dict_item(self):
        r = SearchResult(item={"key": "val"}, matched_fields=["key"])
        d = r.to_dict()
        assert d["item"] == {"key": "val"}
        assert d["matched_fields"] == ["key"]

    def test_generic_typing(self):
        """SearchResult should work with different item types."""
        r1: SearchResult[str] = SearchResult(item="hello")
        r2: SearchResult[dict] = SearchResult(item={"k": "v"})
        r3: SearchResult[int] = SearchResult(item=42)
        assert r1.item == "hello"
        assert r2.item == {"k": "v"}
        assert r3.item == 42


# =============================================================================
# ValidationResult
# =============================================================================


class TestValidationResult:
    def test_defaults(self):
        r = ValidationResult(source_id="src-1")
        assert r.source_id == "src-1"
        assert r.confidence == 0.0
        assert r.recommendation == "keep"
        assert r.adjustment == 0.0
        assert r.reason == ""
        assert r.metadata == {}

    def test_with_values(self):
        r = ValidationResult(
            source_id="src-1",
            confidence=0.9,
            recommendation="boost",
            adjustment=0.1,
            reason="High cross-debate success",
            metadata={"debates": 5},
        )
        assert r.confidence == 0.9
        assert r.recommendation == "boost"
        assert r.adjustment == 0.1

    def test_should_apply_high_confidence_non_keep(self):
        r = ValidationResult(source_id="x", confidence=0.8, recommendation="boost")
        assert r.should_apply() is True

    def test_should_apply_low_confidence(self):
        r = ValidationResult(source_id="x", confidence=0.5, recommendation="boost")
        assert r.should_apply() is False

    def test_should_apply_keep_with_adjustment(self):
        r = ValidationResult(source_id="x", confidence=0.8, recommendation="keep", adjustment=0.5)
        assert r.should_apply() is True

    def test_should_apply_keep_no_adjustment(self):
        r = ValidationResult(source_id="x", confidence=0.8, recommendation="keep", adjustment=0.0)
        assert r.should_apply() is False

    def test_should_apply_custom_threshold(self):
        r = ValidationResult(source_id="x", confidence=0.6, recommendation="penalize")
        assert r.should_apply(min_confidence=0.5) is True
        assert r.should_apply(min_confidence=0.7) is False

    def test_to_dict(self):
        r = ValidationResult(
            source_id="s1",
            confidence=0.85,
            recommendation="boost",
            adjustment=10.0,
            reason="test",
            metadata={"k": "v"},
        )
        d = r.to_dict()
        assert d["source_id"] == "s1"
        assert d["confidence"] == 0.85
        assert d["recommendation"] == "boost"
        assert d["adjustment"] == 10.0
        assert d["reason"] == "test"
        assert d["metadata"] == {"k": "v"}


# =============================================================================
# BatchSyncResult
# =============================================================================


class TestBatchSyncResult:
    def test_defaults(self):
        r = BatchSyncResult()
        assert r.adapter_results == {}
        assert r.total_synced == 0
        assert r.total_failed == 0
        assert r.duration_ms == 0.0

    def test_add_result(self):
        r = BatchSyncResult()
        sr = SyncResult(records_synced=5, records_failed=1)
        r.add_result("consensus", sr)

        assert "consensus" in r.adapter_results
        assert r.total_synced == 5
        assert r.total_failed == 1

    def test_add_multiple_results(self):
        r = BatchSyncResult()
        r.add_result("consensus", SyncResult(records_synced=5, records_failed=1))
        r.add_result("evidence", SyncResult(records_synced=3, records_failed=0))

        assert r.total_synced == 8
        assert r.total_failed == 1
        assert len(r.adapter_results) == 2

    def test_to_dict(self):
        r = BatchSyncResult(duration_ms=200.0)
        r.add_result("elo", SyncResult(records_synced=10))

        d = r.to_dict()
        assert "elo" in d["adapter_results"]
        assert d["adapter_results"]["elo"]["records_synced"] == 10
        assert d["total_synced"] == 10
        assert d["total_failed"] == 0
        assert d["duration_ms"] == 200.0

    def test_to_dict_empty(self):
        r = BatchSyncResult()
        d = r.to_dict()
        assert d["adapter_results"] == {}
        assert d["total_synced"] == 0
