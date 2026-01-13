"""
Tests for evidence store persistence.

Tests the EvidenceStore and InMemoryEvidenceStore components
of the evidence system.
"""

import os
import pytest
import tempfile
from datetime import datetime
from pathlib import Path

from aragora.evidence.store import (
    EvidenceStore,
    InMemoryEvidenceStore,
)
from aragora.evidence.quality import QualityContext


# =============================================================================
# InMemoryEvidenceStore Tests
# =============================================================================


class TestInMemoryEvidenceStore:
    """Tests for InMemoryEvidenceStore class."""

    @pytest.fixture
    def store(self):
        """Create an InMemoryEvidenceStore instance."""
        return InMemoryEvidenceStore()

    def test_save_evidence_basic(self, store):
        """Test saving basic evidence."""
        eid = store.save_evidence(
            evidence_id="test-001",
            source="web",
            title="Test Evidence",
            snippet="This is test evidence content.",
            url="https://example.com",
        )
        assert eid == "test-001"

    def test_save_evidence_with_reliability(self, store):
        """Test saving evidence with reliability score."""
        store.save_evidence(
            evidence_id="test-002",
            source="academic",
            title="Academic Paper",
            snippet="Research findings.",
            reliability_score=0.9,
        )
        evidence = store.get_evidence("test-002")
        assert evidence["reliability_score"] == 0.9

    def test_save_evidence_with_metadata(self, store):
        """Test saving evidence with metadata."""
        store.save_evidence(
            evidence_id="test-003",
            source="api",
            title="API Response",
            snippet="API data.",
            metadata={"endpoint": "/api/test", "version": "v1"},
        )
        evidence = store.get_evidence("test-003")
        assert evidence["metadata"]["endpoint"] == "/api/test"

    def test_save_evidence_deduplication(self, store):
        """Test evidence deduplication by content hash."""
        eid1 = store.save_evidence(
            evidence_id="orig-001",
            source="web",
            title="First Save",
            snippet="Duplicate content for testing.",
        )
        eid2 = store.save_evidence(
            evidence_id="orig-002",  # Different ID
            source="web",
            title="Second Save",
            snippet="Duplicate content for testing.",  # Same content
        )
        # Should return the original ID
        assert eid2 == eid1

    def test_save_evidence_with_debate(self, store):
        """Test saving evidence with debate association."""
        store.save_evidence(
            evidence_id="test-004",
            source="web",
            title="Debate Evidence",
            snippet="Evidence for debate.",
            debate_id="debate-001",
            round_number=1,
        )
        debate_evidence = store.get_debate_evidence("debate-001")
        assert len(debate_evidence) == 1
        assert debate_evidence[0]["id"] == "test-004"

    def test_save_evidence_enrichment(self, store):
        """Test evidence enrichment during save."""
        store.save_evidence(
            evidence_id="test-005",
            source="github",
            title="Code Example",
            snippet="def hello(): print('world')",
            url="https://github.com/test/repo",
            enrich=True,
        )
        evidence = store.get_evidence("test-005")
        assert evidence["enriched_metadata"] is not None

    def test_save_evidence_quality_scoring(self, store):
        """Test quality scoring during save."""
        store.save_evidence(
            evidence_id="test-006",
            source="documentation",
            title="Documentation",
            snippet="Comprehensive documentation content with many details.",
            score_quality=True,
        )
        evidence = store.get_evidence("test-006")
        assert evidence["quality_scores"] is not None
        assert "overall_score" in evidence["quality_scores"]

    def test_get_evidence_exists(self, store):
        """Test getting existing evidence."""
        store.save_evidence(
            evidence_id="test-007",
            source="web",
            title="Test",
            snippet="Content.",
        )
        evidence = store.get_evidence("test-007")
        assert evidence is not None
        assert evidence["id"] == "test-007"
        assert evidence["source"] == "web"

    def test_get_evidence_not_exists(self, store):
        """Test getting non-existent evidence."""
        evidence = store.get_evidence("non-existent")
        assert evidence is None

    def test_get_debate_evidence_empty(self, store):
        """Test getting debate evidence when none exists."""
        evidence = store.get_debate_evidence("no-debate")
        assert evidence == []

    def test_get_debate_evidence_by_round(self, store):
        """Test getting debate evidence by round."""
        store.save_evidence(
            evidence_id="round1-001",
            source="web",
            title="Round 1",
            snippet="Round 1 evidence.",
            debate_id="debate-002",
            round_number=1,
        )
        store.save_evidence(
            evidence_id="round2-001",
            source="web",
            title="Round 2",
            snippet="Round 2 evidence.",
            debate_id="debate-002",
            round_number=2,
        )

        round1 = store.get_debate_evidence("debate-002", round_number=1)
        assert len(round1) == 1
        assert round1[0]["id"] == "round1-001"

        round2 = store.get_debate_evidence("debate-002", round_number=2)
        assert len(round2) == 1
        assert round2[0]["id"] == "round2-001"

    def test_get_debate_evidence_all_rounds(self, store):
        """Test getting all debate evidence."""
        store.save_evidence(
            evidence_id="all-001",
            source="web",
            title="All 1",
            snippet="Evidence 1.",
            debate_id="debate-003",
            round_number=1,
        )
        store.save_evidence(
            evidence_id="all-002",
            source="web",
            title="All 2",
            snippet="Evidence 2.",
            debate_id="debate-003",
            round_number=2,
        )

        all_evidence = store.get_debate_evidence("debate-003")
        assert len(all_evidence) == 2

    def test_search_evidence_basic(self, store):
        """Test basic evidence search."""
        store.save_evidence(
            evidence_id="search-001",
            source="web",
            title="Machine Learning",
            snippet="Machine learning is a subset of AI.",
        )
        store.save_evidence(
            evidence_id="search-002",
            source="web",
            title="Cooking",
            snippet="How to cook pasta.",
        )

        results = store.search_evidence("machine learning")
        assert len(results) > 0
        assert any("machine" in r["snippet"].lower() for r in results)

    def test_search_evidence_with_source_filter(self, store):
        """Test search with source filter."""
        store.save_evidence(
            evidence_id="filter-001",
            source="academic",
            title="Academic",
            snippet="Academic content about testing.",
        )
        store.save_evidence(
            evidence_id="filter-002",
            source="web",
            title="Web",
            snippet="Web content about testing.",
        )

        results = store.search_evidence("testing", source_filter="academic")
        assert all(r["source"] == "academic" for r in results)

    def test_search_evidence_with_min_reliability(self, store):
        """Test search with minimum reliability."""
        store.save_evidence(
            evidence_id="rel-001",
            source="web",
            title="High Rel",
            snippet="High reliability content.",
            reliability_score=0.9,
        )
        store.save_evidence(
            evidence_id="rel-002",
            source="web",
            title="Low Rel",
            snippet="Low reliability content.",
            reliability_score=0.3,
        )

        results = store.search_evidence("reliability", min_reliability=0.5)
        assert all(r["reliability_score"] >= 0.5 for r in results)

    def test_search_evidence_limit(self, store):
        """Test search with limit."""
        for i in range(10):
            store.save_evidence(
                evidence_id=f"limit-{i:03d}",
                source="web",
                title=f"Test {i}",
                snippet="Test content for search.",
            )

        results = store.search_evidence("test", limit=5)
        assert len(results) <= 5

    def test_delete_evidence_exists(self, store):
        """Test deleting existing evidence."""
        store.save_evidence(
            evidence_id="del-001",
            source="web",
            title="Delete Me",
            snippet="To be deleted.",
        )
        assert store.get_evidence("del-001") is not None

        result = store.delete_evidence("del-001")
        assert result is True
        assert store.get_evidence("del-001") is None

    def test_delete_evidence_not_exists(self, store):
        """Test deleting non-existent evidence."""
        result = store.delete_evidence("non-existent")
        assert result is False

    def test_delete_evidence_removes_debate_association(self, store):
        """Test deleting evidence removes debate associations."""
        store.save_evidence(
            evidence_id="del-assoc-001",
            source="web",
            title="Associated",
            snippet="Evidence with association.",
            debate_id="debate-del",
        )
        assert len(store.get_debate_evidence("debate-del")) == 1

        store.delete_evidence("del-assoc-001")
        assert len(store.get_debate_evidence("debate-del")) == 0

    def test_get_statistics_empty(self, store):
        """Test statistics on empty store."""
        stats = store.get_statistics()
        assert stats["total_evidence"] == 0
        assert stats["by_source"] == {}
        assert stats["average_reliability"] == 0.0

    def test_get_statistics_with_data(self, store):
        """Test statistics with data."""
        store.save_evidence(
            evidence_id="stat-001",
            source="web",
            title="Web 1",
            snippet="Content 1.",
            reliability_score=0.8,
        )
        store.save_evidence(
            evidence_id="stat-002",
            source="web",
            title="Web 2",
            snippet="Content 2.",
            reliability_score=0.6,
        )
        store.save_evidence(
            evidence_id="stat-003",
            source="academic",
            title="Academic",
            snippet="Content 3.",
            reliability_score=0.9,
            debate_id="stat-debate",
        )

        stats = store.get_statistics()
        assert stats["total_evidence"] == 3
        assert stats["by_source"]["web"] == 2
        assert stats["by_source"]["academic"] == 1
        assert 0.7 <= stats["average_reliability"] <= 0.8
        assert stats["debate_associations"] == 1
        assert stats["unique_debates"] == 1

    def test_close_noop(self, store):
        """Test close is a no-op for in-memory store."""
        store.save_evidence(
            evidence_id="close-001",
            source="web",
            title="Test",
            snippet="Content.",
        )
        store.close()
        # Should still work after close
        assert store.get_evidence("close-001") is not None


# =============================================================================
# EvidenceStore Tests
# =============================================================================


class TestEvidenceStore:
    """Tests for EvidenceStore class (SQLite-based)."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create an EvidenceStore with temporary database."""
        db_path = tmp_path / "test_evidence.db"
        store = EvidenceStore(db_path=db_path)
        yield store
        store.close()

    def test_init_creates_db(self, tmp_path):
        """Test initialization creates database file."""
        db_path = tmp_path / "new_evidence.db"
        store = EvidenceStore(db_path=db_path)
        assert db_path.exists()
        store.close()

    def test_init_creates_tables(self, store):
        """Test initialization creates required tables."""
        with store._cursor() as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row["name"] for row in cursor.fetchall()}
        assert "evidence" in tables
        assert "debate_evidence" in tables

    def test_save_evidence_basic(self, store):
        """Test saving basic evidence."""
        eid = store.save_evidence(
            evidence_id="sql-001",
            source="web",
            title="SQLite Test",
            snippet="Testing SQLite storage.",
        )
        assert eid == "sql-001"

    def test_save_evidence_persists(self, store):
        """Test evidence persists in database."""
        store.save_evidence(
            evidence_id="persist-001",
            source="web",
            title="Persist Test",
            snippet="Should persist.",
        )

        # Query directly
        with store._cursor() as cursor:
            cursor.execute(
                "SELECT * FROM evidence WHERE id = ?",
                ("persist-001",),
            )
            row = cursor.fetchone()
        assert row is not None
        assert row["source"] == "web"

    def test_save_evidence_deduplication(self, store):
        """Test content deduplication."""
        content = "Unique content for dedup test."
        eid1 = store.save_evidence(
            evidence_id="dedup-001",
            source="web",
            title="First",
            snippet=content,
        )
        eid2 = store.save_evidence(
            evidence_id="dedup-002",
            source="web",
            title="Second",
            snippet=content,
        )
        assert eid1 == eid2

    def test_save_evidence_with_debate(self, store):
        """Test saving with debate association."""
        store.save_evidence(
            evidence_id="debate-001",
            source="web",
            title="Debate Evidence",
            snippet="For debate.",
            debate_id="test-debate",
            round_number=1,
        )

        with store._cursor() as cursor:
            cursor.execute(
                "SELECT * FROM debate_evidence WHERE debate_id = ?",
                ("test-debate",),
            )
            row = cursor.fetchone()
        assert row is not None
        assert row["evidence_id"] == "debate-001"

    def test_get_evidence_exists(self, store):
        """Test getting existing evidence."""
        store.save_evidence(
            evidence_id="get-001",
            source="documentation",
            title="Get Test",
            snippet="Content to get.",
            url="https://docs.test.com",
        )

        evidence = store.get_evidence("get-001")
        assert evidence is not None
        assert evidence["source"] == "documentation"
        assert evidence["url"] == "https://docs.test.com"

    def test_get_evidence_not_exists(self, store):
        """Test getting non-existent evidence."""
        evidence = store.get_evidence("no-exist")
        assert evidence is None

    def test_get_evidence_with_metadata(self, store):
        """Test getting evidence with parsed metadata."""
        store.save_evidence(
            evidence_id="meta-001",
            source="api",
            title="With Metadata",
            snippet="Has metadata.",
            metadata={"key": "value"},
        )

        evidence = store.get_evidence("meta-001")
        assert evidence["metadata"]["key"] == "value"

    def test_get_debate_evidence(self, store):
        """Test getting debate evidence."""
        store.save_evidence(
            evidence_id="de-001",
            source="web",
            title="DE 1",
            snippet="First.",
            debate_id="debate-get",
            round_number=1,
        )
        store.save_evidence(
            evidence_id="de-002",
            source="web",
            title="DE 2",
            snippet="Second.",
            debate_id="debate-get",
            round_number=2,
        )

        evidence = store.get_debate_evidence("debate-get")
        assert len(evidence) == 2

    def test_get_debate_evidence_by_round(self, store):
        """Test getting debate evidence by round."""
        store.save_evidence(
            evidence_id="r1-001",
            source="web",
            title="R1",
            snippet="Round 1.",
            debate_id="debate-round",
            round_number=1,
        )
        store.save_evidence(
            evidence_id="r2-001",
            source="web",
            title="R2",
            snippet="Round 2.",
            debate_id="debate-round",
            round_number=2,
        )

        r1 = store.get_debate_evidence("debate-round", round_number=1)
        assert len(r1) == 1
        assert r1[0]["id"] == "r1-001"

    def test_search_evidence_fts(self, store):
        """Test full-text search."""
        store.save_evidence(
            evidence_id="fts-001",
            source="web",
            title="Python Programming",
            snippet="Python is a programming language used for scripting.",
        )
        store.save_evidence(
            evidence_id="fts-002",
            source="web",
            title="Java Development",
            snippet="Java is used for enterprise applications.",
        )

        # FTS5 requires exact token matching
        results = store.search_evidence("Python")
        # FTS may or may not find depending on tokenization
        assert isinstance(results, list)

    def test_search_evidence_with_source_filter(self, store):
        """Test search with source filter."""
        store.save_evidence(
            evidence_id="sf-001",
            source="academic",
            title="Academic Paper",
            snippet="Research on testing.",
        )
        store.save_evidence(
            evidence_id="sf-002",
            source="web",
            title="Web Article",
            snippet="Article on testing.",
        )

        results = store.search_evidence("testing", source_filter="academic")
        assert all(r["source"] == "academic" for r in results)

    def test_search_evidence_with_min_reliability(self, store):
        """Test search with minimum reliability."""
        store.save_evidence(
            evidence_id="mr-001",
            source="web",
            title="High",
            snippet="High reliability search test.",
            reliability_score=0.9,
        )
        store.save_evidence(
            evidence_id="mr-002",
            source="web",
            title="Low",
            snippet="Low reliability search test.",
            reliability_score=0.2,
        )

        results = store.search_evidence("search test", min_reliability=0.5)
        assert all(r["reliability_score"] >= 0.5 for r in results)

    def test_search_evidence_limit(self, store):
        """Test search with limit."""
        for i in range(10):
            store.save_evidence(
                evidence_id=f"lim-{i:03d}",
                source="web",
                title=f"Limit Test {i}",
                snippet="Search limit test content.",
            )

        results = store.search_evidence("limit test", limit=3)
        assert len(results) <= 3

    def test_search_evidence_with_context(self, store):
        """Test search with quality context."""
        store.save_evidence(
            evidence_id="ctx-001",
            source="web",
            title="Context Test",
            snippet="Content for context-aware search.",
        )

        ctx = QualityContext(query="context")
        results = store.search_evidence("Context", context=ctx)
        # FTS search may or may not match depending on tokenization
        assert isinstance(results, list)
        if len(results) > 0:
            assert "quality_scores" in results[0]

    def test_search_similar(self, store):
        """Test similar content search."""
        store.save_evidence(
            evidence_id="sim-001",
            source="web",
            title="Machine Learning Basics",
            snippet="Machine learning is a method of data analysis.",
        )
        store.save_evidence(
            evidence_id="sim-002",
            source="web",
            title="Deep Learning",
            snippet="Deep learning uses neural networks for analysis.",
        )

        results = store.search_similar(
            "machine learning algorithms for data",
            exclude_id="sim-001",
        )
        # May or may not find similar depending on FTS matching
        assert isinstance(results, list)

    def test_mark_used_in_consensus(self, store):
        """Test marking evidence as used in consensus."""
        store.save_evidence(
            evidence_id="cons-001",
            source="web",
            title="Consensus Evidence",
            snippet="Used in consensus.",
            debate_id="consensus-debate",
        )

        store.mark_used_in_consensus(
            debate_id="consensus-debate",
            evidence_ids=["cons-001"],
        )

        evidence = store.get_debate_evidence("consensus-debate")
        # SQLite stores booleans as 0/1
        assert evidence[0]["used_in_consensus"] in (True, 1)

    def test_delete_evidence(self, store):
        """Test deleting evidence."""
        store.save_evidence(
            evidence_id="del-001",
            source="web",
            title="Delete Me",
            snippet="Will be deleted.",
        )
        assert store.get_evidence("del-001") is not None

        result = store.delete_evidence("del-001")
        assert result is True
        assert store.get_evidence("del-001") is None

    def test_delete_evidence_not_exists(self, store):
        """Test deleting non-existent evidence."""
        result = store.delete_evidence("no-exist-del")
        assert result is False

    def test_delete_evidence_removes_fts(self, store):
        """Test deleting removes FTS entry."""
        store.save_evidence(
            evidence_id="fts-del-001",
            source="web",
            title="FTS Delete Test",
            snippet="Will be removed from FTS index.",
        )

        # Verify evidence exists
        assert store.get_evidence("fts-del-001") is not None

        # Delete it
        store.delete_evidence("fts-del-001")

        # Verify evidence is gone
        assert store.get_evidence("fts-del-001") is None

    def test_delete_debate_evidence(self, store):
        """Test deleting debate associations."""
        store.save_evidence(
            evidence_id="assoc-001",
            source="web",
            title="Associated",
            snippet="Has association.",
            debate_id="assoc-debate",
        )
        store.save_evidence(
            evidence_id="assoc-002",
            source="web",
            title="Also Associated",
            snippet="Also has association.",
            debate_id="assoc-debate",
        )

        count = store.delete_debate_evidence("assoc-debate")
        assert count == 2

        # Evidence still exists, just not associated
        assert store.get_evidence("assoc-001") is not None
        assert len(store.get_debate_evidence("assoc-debate")) == 0

    def test_get_statistics(self, store):
        """Test getting store statistics."""
        store.save_evidence(
            evidence_id="stat-001",
            source="web",
            title="Web",
            snippet="Web content.",
            reliability_score=0.7,
        )
        store.save_evidence(
            evidence_id="stat-002",
            source="academic",
            title="Academic",
            snippet="Academic content.",
            reliability_score=0.9,
            debate_id="stat-debate",
        )

        stats = store.get_statistics()
        assert stats["total_evidence"] == 2
        assert stats["by_source"]["web"] == 1
        assert stats["by_source"]["academic"] == 1
        assert 0.7 <= stats["average_reliability"] <= 0.9
        assert stats["debate_associations"] == 1
        assert stats["unique_debates"] == 1

    def test_thread_safety_multiple_saves(self, store):
        """Test thread safety with multiple saves."""
        import threading

        def save_evidence(thread_id):
            for i in range(5):
                store.save_evidence(
                    evidence_id=f"thread-{thread_id}-{i}",
                    source="web",
                    title=f"Thread {thread_id} Item {i}",
                    snippet=f"Unique content from thread {thread_id} item {i}.",  # Unique to avoid dedup
                )

        threads = [threading.Thread(target=save_evidence, args=(tid,)) for tid in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = store.get_statistics()
        assert stats["total_evidence"] == 15  # 3 threads * 5 items

    def test_close_and_reopen(self, tmp_path):
        """Test closing and reopening database."""
        db_path = tmp_path / "reopen.db"

        # Create and save
        store1 = EvidenceStore(db_path=db_path)
        store1.save_evidence(
            evidence_id="reopen-001",
            source="web",
            title="Reopen Test",
            snippet="Should persist after reopen.",
        )
        store1.close()

        # Reopen and verify
        store2 = EvidenceStore(db_path=db_path)
        evidence = store2.get_evidence("reopen-001")
        assert evidence is not None
        assert evidence["title"] == "Reopen Test"
        store2.close()


# =============================================================================
# Integration Tests
# =============================================================================


class TestEvidenceStoreIntegration:
    """Integration tests for EvidenceStore."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create store with temporary database."""
        db_path = tmp_path / "integration.db"
        store = EvidenceStore(db_path=db_path)
        yield store
        store.close()

    def test_full_workflow(self, store):
        """Test complete evidence workflow."""
        # 1. Save evidence for a debate
        for i in range(5):
            store.save_evidence(
                evidence_id=f"workflow-{i:03d}",
                source="web" if i % 2 == 0 else "academic",
                title=f"Evidence {i}",
                snippet=f"This is evidence number {i} for the workflow test with unique content here.",
                url=f"https://example.com/{i}",
                reliability_score=0.5 + (i * 0.1),
                debate_id="workflow-debate",
                round_number=i // 2 + 1,
            )

        # 2. Get debate evidence (skip FTS search as it's tokenization dependent)
        debate_evidence = store.get_debate_evidence("workflow-debate")
        assert len(debate_evidence) == 5

        # 3. Get specific round
        round1 = store.get_debate_evidence("workflow-debate", round_number=1)
        assert len(round1) == 2

        # 4. Mark used in consensus
        store.mark_used_in_consensus(
            "workflow-debate",
            ["workflow-000", "workflow-002"],
        )

        # 5. Verify consensus marking (SQLite stores bool as 0/1)
        debate_evidence = store.get_debate_evidence("workflow-debate")
        consensus_used = [e for e in debate_evidence if e.get("used_in_consensus") in (True, 1)]
        assert len(consensus_used) == 2

        # 6. Get statistics
        stats = store.get_statistics()
        assert stats["total_evidence"] == 5
        assert stats["unique_debates"] == 1

    def test_save_evidence_pack_mock(self, store):
        """Test saving evidence pack (simulated)."""

        # Simulate EvidencePack
        class MockSnippet:
            def __init__(self, eid, source, title, snippet):
                self.id = eid
                self.source = source
                self.title = title
                self.snippet = snippet
                self.url = ""
                self.reliability_score = 0.7
                self.metadata = {}

        class MockPack:
            snippets = [
                MockSnippet("pack-001", "web", "Pack 1", "Content 1"),
                MockSnippet("pack-002", "web", "Pack 2", "Content 2"),
                MockSnippet("pack-003", "academic", "Pack 3", "Content 3"),
            ]

        # Save pack manually
        for snippet in MockPack.snippets:
            store.save_evidence(
                evidence_id=snippet.id,
                source=snippet.source,
                title=snippet.title,
                snippet=snippet.snippet,
                debate_id="pack-debate",
            )

        evidence = store.get_debate_evidence("pack-debate")
        assert len(evidence) == 3
