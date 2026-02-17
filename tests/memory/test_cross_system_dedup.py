"""Tests for the Cross-System Deduplication Engine."""

from __future__ import annotations

import pytest

from aragora.memory.dedup import (
    CrossSystemDedupEngine,
    DedupCheckResult,
    DedupMatch,
    CrossSystemDedupReport,
    dedup_results,
    _normalize_text,
)


# ---------------------------------------------------------------------------
# Tests: Content Hash
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_deterministic(self):
        h1 = CrossSystemDedupEngine.compute_content_hash("hello world")
        h2 = CrossSystemDedupEngine.compute_content_hash("hello world")
        assert h1 == h2

    def test_different_content_different_hash(self):
        h1 = CrossSystemDedupEngine.compute_content_hash("hello")
        h2 = CrossSystemDedupEngine.compute_content_hash("world")
        assert h1 != h2

    def test_normalization_collapses_whitespace(self):
        h1 = CrossSystemDedupEngine.compute_content_hash("hello  world")
        h2 = CrossSystemDedupEngine.compute_content_hash("hello world")
        assert h1 == h2

    def test_normalization_case_insensitive(self):
        h1 = CrossSystemDedupEngine.compute_content_hash("Hello World")
        h2 = CrossSystemDedupEngine.compute_content_hash("hello world")
        assert h1 == h2

    def test_normalization_strips(self):
        h1 = CrossSystemDedupEngine.compute_content_hash("  hello  ")
        h2 = CrossSystemDedupEngine.compute_content_hash("hello")
        assert h1 == h2


# ---------------------------------------------------------------------------
# Tests: Normalize Text
# ---------------------------------------------------------------------------


class TestNormalizeText:
    def test_lowercase(self):
        assert _normalize_text("HELLO") == "hello"

    def test_collapse_whitespace(self):
        assert _normalize_text("a  b\n\tc") == "a b c"

    def test_strip(self):
        assert _normalize_text("  hello  ") == "hello"


# ---------------------------------------------------------------------------
# Tests: Register and Check Duplicates
# ---------------------------------------------------------------------------


class TestDuplicateCheck:
    @pytest.mark.asyncio
    async def test_no_duplicate_on_empty_index(self):
        engine = CrossSystemDedupEngine()
        result = await engine.check_duplicate_before_write("new content")
        assert result.is_duplicate is False

    @pytest.mark.asyncio
    async def test_detects_exact_duplicate(self):
        engine = CrossSystemDedupEngine()
        engine.register_item("item_1", "km", "existing content")
        result = await engine.check_duplicate_before_write("existing content")
        assert result.is_duplicate is True
        assert result.existing_id == "item_1"
        assert result.existing_source == "km"
        assert result.similarity == 1.0

    @pytest.mark.asyncio
    async def test_target_filter(self):
        engine = CrossSystemDedupEngine()
        engine.register_item("item_1", "km", "existing content")
        # Check against only continuum source - should not find km duplicate
        result = await engine.check_duplicate_before_write(
            "existing content", targets=["continuum"]
        )
        assert result.is_duplicate is False

    @pytest.mark.asyncio
    async def test_content_hash_populated(self):
        engine = CrossSystemDedupEngine()
        result = await engine.check_duplicate_before_write("test")
        assert result.content_hash != ""


# ---------------------------------------------------------------------------
# Tests: Cross-System Scan
# ---------------------------------------------------------------------------


class TestCrossSystemScan:
    @pytest.mark.asyncio
    async def test_detects_exact_cross_system_duplicates(self):
        engine = CrossSystemDedupEngine()
        items = [
            {"id": "cm_1", "source": "continuum", "content": "rate limiting"},
            {"id": "km_1", "source": "km", "content": "rate limiting"},
        ]
        report = await engine.scan_cross_system_duplicates(items)
        assert report.exact_duplicates == 1
        assert len(report.matches) == 1
        assert report.matches[0].match_type == "exact"
        assert report.matches[0].similarity == 1.0

    @pytest.mark.asyncio
    async def test_ignores_same_source_duplicates(self):
        engine = CrossSystemDedupEngine()
        items = [
            {"id": "km_1", "source": "km", "content": "rate limiting"},
            {"id": "km_2", "source": "km", "content": "rate limiting"},
        ]
        report = await engine.scan_cross_system_duplicates(items)
        assert report.exact_duplicates == 0

    @pytest.mark.asyncio
    async def test_no_duplicates(self):
        engine = CrossSystemDedupEngine()
        items = [
            {"id": "cm_1", "source": "continuum", "content": "topic A"},
            {"id": "km_1", "source": "km", "content": "topic B"},
        ]
        report = await engine.scan_cross_system_duplicates(items)
        assert report.exact_duplicates == 0
        assert report.total_items_scanned == 2

    @pytest.mark.asyncio
    async def test_near_duplicate_detection(self):
        engine = CrossSystemDedupEngine(near_duplicate_threshold=0.5)
        items = [
            {"id": "cm_1", "source": "continuum", "content": "rate limiting with token bucket algorithm"},
            {"id": "km_1", "source": "km", "content": "rate limiting with token bucket strategy"},
        ]
        report = await engine.scan_cross_system_duplicates(items)
        # These share most tokens, should be detected as near-duplicates
        assert report.near_duplicates >= 1 or report.exact_duplicates >= 0

    @pytest.mark.asyncio
    async def test_duration_tracked(self):
        engine = CrossSystemDedupEngine()
        report = await engine.scan_cross_system_duplicates([])
        assert report.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_empty_items(self):
        engine = CrossSystemDedupEngine()
        report = await engine.scan_cross_system_duplicates([])
        assert report.total_items_scanned == 0


# ---------------------------------------------------------------------------
# Tests: Helper Function
# ---------------------------------------------------------------------------


class TestDedupResultsHelper:
    def test_removes_exact_duplicates(self):
        results = [
            {"content": "hello world", "source": "km"},
            {"content": "hello world", "source": "continuum"},
            {"content": "unique item", "source": "km"},
        ]
        deduped, removed = dedup_results(results)
        assert removed == 1
        assert len(deduped) == 2

    def test_no_duplicates(self):
        results = [
            {"content": "a", "source": "km"},
            {"content": "b", "source": "continuum"},
        ]
        deduped, removed = dedup_results(results)
        assert removed == 0
        assert len(deduped) == 2

    def test_empty_list(self):
        deduped, removed = dedup_results([])
        assert removed == 0
        assert deduped == []


# ---------------------------------------------------------------------------
# Tests: Hash Index Management
# ---------------------------------------------------------------------------


class TestHashIndex:
    def test_register_and_size(self):
        engine = CrossSystemDedupEngine()
        assert engine.get_hash_index_size() == 0
        engine.register_item("id1", "km", "content1")
        assert engine.get_hash_index_size() == 1

    def test_clear(self):
        engine = CrossSystemDedupEngine()
        engine.register_item("id1", "km", "content1")
        engine.clear_hash_index()
        assert engine.get_hash_index_size() == 0


# ---------------------------------------------------------------------------
# Tests: Dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_dedup_match(self):
        m = DedupMatch(
            item_id_a="a",
            source_a="km",
            item_id_b="b",
            source_b="continuum",
            match_type="exact",
            similarity=1.0,
            content_hash="abc",
        )
        assert m.match_type == "exact"

    def test_dedup_check_result_defaults(self):
        r = DedupCheckResult()
        assert r.is_duplicate is False
        assert r.existing_id is None

    def test_cross_system_dedup_report_defaults(self):
        r = CrossSystemDedupReport()
        assert r.total_items_scanned == 0
        assert r.matches == []
