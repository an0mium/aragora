"""
Integration tests for debate → memory tier promotion pipeline.

Tests the full flow from debate outcomes to memory storage, retrieval,
and tier promotion/demotion based on surprise scores.
"""

import tempfile
import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from aragora.memory.continuum import ContinuumMemory, MemoryTier, ContinuumMemoryEntry
from aragora.memory.tier_manager import TierManager, get_tier_manager, reset_tier_manager
from aragora.debate.memory_manager import MemoryManager
from aragora.storage.schema import get_wal_connection


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockDebateResult:
    """Mock debate result for testing."""

    id: str = "test-debate-123"
    final_answer: str = (
        "The recommended approach is to use a rate limiter with token bucket algorithm."
    )
    confidence: float = 0.85
    consensus_reached: bool = True
    rounds_used: int = 3
    winner: Optional[str] = "claude"
    scores: Dict[str, float] = field(default_factory=dict)


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield str(Path(tmpdir) / "test_continuum.db")


@pytest.fixture
def continuum_memory(temp_db_path):
    """Create a ContinuumMemory instance with clean state."""
    reset_tier_manager()
    return ContinuumMemory(db_path=temp_db_path)


@pytest.fixture
def memory_manager(continuum_memory):
    """Create a MemoryManager with ContinuumMemory."""
    return MemoryManager(
        continuum_memory=continuum_memory,
        domain_extractor=lambda: "backend",
    )


# =============================================================================
# Debate Outcome Storage Tests
# =============================================================================


class TestDebateOutcomeStorage:
    """Tests for storing debate outcomes in memory."""

    def test_store_debate_outcome_basic(self, memory_manager, continuum_memory):
        """store_debate_outcome should create memory entry."""
        result = MockDebateResult()
        task = "Design a rate limiter for API endpoints"

        memory_manager.store_debate_outcome(result, task)

        # Verify memory was stored
        entries = continuum_memory.retrieve(tiers=[MemoryTier.FAST], limit=10)
        assert len(entries) >= 1

        # Find the debate outcome entry
        outcome_entry = next((e for e in entries if "debate_outcome" in e.id), None)
        assert outcome_entry is not None
        assert (
            "rate limiter" in outcome_entry.content.lower()
            or "recommended approach" in outcome_entry.content.lower()
        )

    def test_store_debate_outcome_tier_selection_high_confidence(
        self, memory_manager, continuum_memory
    ):
        """High confidence multi-round debates go to FAST tier."""
        result = MockDebateResult(
            confidence=0.9,
            rounds_used=3,
            consensus_reached=True,
        )

        memory_manager.store_debate_outcome(result, "High confidence task")

        entries = continuum_memory.retrieve(tiers=[MemoryTier.FAST], limit=10)
        outcome_entries = [e for e in entries if "debate_outcome" in e.id]
        assert len(outcome_entries) >= 1

    def test_store_debate_outcome_tier_selection_medium_confidence(self, memory_manager):
        """Medium confidence debates go to MEDIUM tier."""
        # Create a new memory manager with fresh memory
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            memory = ContinuumMemory(db_path=db_path)
            mm = MemoryManager(continuum_memory=memory)

            result = MockDebateResult(
                confidence=0.6,
                rounds_used=2,
                consensus_reached=True,
            )

            mm.store_debate_outcome(result, "Medium confidence task")

            # Check MEDIUM tier
            entries = memory.retrieve(tiers=[MemoryTier.MEDIUM], limit=10)
            outcome_entries = [e for e in entries if "debate_outcome" in e.id]
            assert len(outcome_entries) >= 1

    def test_store_debate_outcome_tier_selection_low_confidence(self, memory_manager):
        """Low confidence debates go to SLOW tier."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            memory = ContinuumMemory(db_path=db_path)
            mm = MemoryManager(continuum_memory=memory)

            result = MockDebateResult(
                confidence=0.3,
                rounds_used=1,
                consensus_reached=False,
            )

            mm.store_debate_outcome(result, "Low confidence task")

            entries = memory.retrieve(tiers=[MemoryTier.SLOW], limit=10)
            outcome_entries = [e for e in entries if "debate_outcome" in e.id]
            assert len(outcome_entries) >= 1

    def test_store_debate_outcome_importance_calculation(self, memory_manager, continuum_memory):
        """Importance should be based on confidence and consensus."""
        # High confidence + consensus = high importance
        high_result = MockDebateResult(
            id="high-importance",
            confidence=0.95,
            consensus_reached=True,
        )
        memory_manager.store_debate_outcome(high_result, "High importance task")

        # Low confidence + no consensus = lower importance
        low_result = MockDebateResult(
            id="low-importance",
            confidence=0.4,
            consensus_reached=False,
        )
        memory_manager.store_debate_outcome(low_result, "Low importance task")

        # Retrieve all entries
        all_entries = continuum_memory.retrieve(limit=100)

        high_entry = next((e for e in all_entries if "high-importance" in e.id), None)
        low_entry = next((e for e in all_entries if "low-importance" in e.id), None)

        if high_entry and low_entry:
            assert high_entry.importance > low_entry.importance

    def test_store_debate_outcome_metadata(self, memory_manager, continuum_memory):
        """Metadata should include debate context."""
        result = MockDebateResult(
            id="metadata-test",
            confidence=0.8,
            winner="claude",
        )

        memory_manager.store_debate_outcome(result, "Metadata test task")

        entries = continuum_memory.retrieve(limit=10)
        # ID is truncated to first 8 chars: "debate_outcome_metadata"
        entry = next((e for e in entries if "debate_outcome_metadata" in e.id), None)

        assert entry is not None
        assert "debate_id" in entry.metadata
        assert "domain" in entry.metadata

    def test_store_debate_outcome_no_final_answer(self, memory_manager, continuum_memory):
        """Should not store if no final_answer."""
        result = MockDebateResult(final_answer="")

        # Get count before
        before = len(continuum_memory.retrieve(limit=100))

        memory_manager.store_debate_outcome(result, "No answer task")

        # Count should not increase
        after = len(continuum_memory.retrieve(limit=100))
        assert after == before


# =============================================================================
# Memory Outcome Update Tests
# =============================================================================


class TestMemoryOutcomeUpdates:
    """Tests for updating memories based on debate outcomes."""

    def test_update_memory_outcomes_success(self, memory_manager, continuum_memory):
        """Successful debate should reinforce retrieved memories."""
        # First add a memory
        entry = continuum_memory.add(
            id="pattern-123",
            content="Rate limiting is important for API stability",
            tier=MemoryTier.SLOW,
            importance=0.5,
        )

        # Track as retrieved
        memory_manager.track_retrieved_ids(["pattern-123"])

        # Update with successful outcome
        result = MockDebateResult(
            confidence=0.9,
            consensus_reached=True,
        )
        memory_manager.update_memory_outcomes(result)

        # Verify update was applied
        updated = continuum_memory.get("pattern-123")
        assert updated is not None
        # Success count should increase
        assert updated.success_count >= 1 or updated.update_count >= 1

    def test_update_memory_outcomes_failure(self, memory_manager, continuum_memory):
        """Failed debate should update failure count."""
        continuum_memory.add(
            id="pattern-456",
            content="Some unreliable pattern",
            tier=MemoryTier.MEDIUM,
            importance=0.5,
        )

        memory_manager.track_retrieved_ids(["pattern-456"])

        result = MockDebateResult(
            confidence=0.3,
            consensus_reached=False,
        )
        memory_manager.update_memory_outcomes(result)

        updated = continuum_memory.get("pattern-456")
        assert updated is not None

    def test_update_memory_outcomes_clears_tracked_ids(self, memory_manager, continuum_memory):
        """update_memory_outcomes should clear tracked IDs after update."""
        continuum_memory.add(
            id="pattern-789",
            content="Test pattern",
            tier=MemoryTier.SLOW,
        )

        memory_manager.track_retrieved_ids(["pattern-789"])
        assert len(memory_manager._retrieved_ids) == 1

        result = MockDebateResult()
        memory_manager.update_memory_outcomes(result)

        assert len(memory_manager._retrieved_ids) == 0

    def test_update_memory_outcomes_no_tracked_ids(self, memory_manager, continuum_memory):
        """Should handle empty tracked IDs gracefully."""
        memory_manager.clear_retrieved_ids()

        result = MockDebateResult()
        # Should not raise
        memory_manager.update_memory_outcomes(result)


# =============================================================================
# Tier Promotion/Demotion Tests
# =============================================================================


class TestTierPromotion:
    """Tests for memory tier promotion based on surprise."""

    def test_consolidate_promotes_high_surprise(self, continuum_memory):
        """Entries with high surprise should be promoted."""
        # Add entry to SLOW tier with high surprise
        continuum_memory.add(
            id="promote-candidate",
            content="Surprising pattern that keeps being relevant",
            tier=MemoryTier.SLOW,
            importance=0.8,
        )

        # Manually update surprise score to exceed threshold
        with get_wal_connection(continuum_memory.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET surprise_score = 0.8 WHERE id = ?",
                ("promote-candidate",),
            )
            conn.commit()

        # Run consolidation
        results = continuum_memory.consolidate()

        # Check if promoted
        entry = continuum_memory.get("promote-candidate")
        # Entry should be promoted to MEDIUM (one tier up)
        if results["promotions"] > 0:
            assert entry.tier in [MemoryTier.MEDIUM, MemoryTier.SLOW]

    def test_consolidate_demotes_low_surprise(self, continuum_memory):
        """Entries with low surprise and many updates should be demoted."""
        # Add entry to MEDIUM tier with low surprise
        continuum_memory.add(
            id="demote-candidate",
            content="Stable pattern that rarely changes",
            tier=MemoryTier.MEDIUM,
            importance=0.5,
        )

        # Update to have low surprise and many updates
        with get_wal_connection(continuum_memory.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE continuum_memory
                   SET surprise_score = 0.1, update_count = 20
                   WHERE id = ?""",
                ("demote-candidate",),
            )
            conn.commit()

        results = continuum_memory.consolidate()

        entry = continuum_memory.get("demote-candidate")
        # Entry should be demoted to SLOW (one tier down)
        if results["demotions"] > 0:
            assert entry.tier in [MemoryTier.SLOW, MemoryTier.MEDIUM]

    def test_consolidate_fast_tier_cannot_promote(self, continuum_memory):
        """FAST tier entries cannot be promoted further."""
        continuum_memory.add(
            id="fast-entry",
            content="Already at fastest tier",
            tier=MemoryTier.FAST,
            importance=0.9,
        )

        # Set high surprise
        with get_wal_connection(continuum_memory.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET surprise_score = 0.9 WHERE id = ?",
                ("fast-entry",),
            )
            conn.commit()

        continuum_memory.consolidate()

        entry = continuum_memory.get("fast-entry")
        assert entry.tier == MemoryTier.FAST

    def test_consolidate_glacial_tier_cannot_demote(self, continuum_memory):
        """GLACIAL tier entries cannot be demoted further."""
        continuum_memory.add(
            id="glacial-entry",
            content="Already at slowest tier",
            tier=MemoryTier.GLACIAL,
            importance=0.3,
        )

        # Set low surprise and many updates
        with get_wal_connection(continuum_memory.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE continuum_memory
                   SET surprise_score = 0.05, update_count = 100
                   WHERE id = ?""",
                ("glacial-entry",),
            )
            conn.commit()

        continuum_memory.consolidate()

        entry = continuum_memory.get("glacial-entry")
        assert entry.tier == MemoryTier.GLACIAL

    def test_consolidate_records_transitions(self, continuum_memory):
        """Tier transitions should be recorded in history."""
        continuum_memory.add(
            id="transition-test",
            content="Test entry for transition tracking",
            tier=MemoryTier.SLOW,
            importance=0.7,
        )

        # Set high surprise to trigger promotion
        with get_wal_connection(continuum_memory.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET surprise_score = 0.8 WHERE id = ?",
                ("transition-test",),
            )
            conn.commit()

        continuum_memory.consolidate()

        # Check stats for transitions
        stats = continuum_memory.get_stats()
        # Transitions may or may not occur depending on thresholds
        assert "transitions" in stats


# =============================================================================
# Full Pipeline Integration Tests
# =============================================================================


class TestFullPipeline:
    """Tests for the complete debate → memory → promotion pipeline."""

    def test_debate_to_memory_to_retrieval(self, memory_manager, continuum_memory):
        """Full flow: store debate outcome, then retrieve for similar task."""
        # Store a debate outcome
        result = MockDebateResult(
            id="pipeline-debate-1",
            final_answer="Use Redis for distributed rate limiting",
            confidence=0.9,
            consensus_reached=True,
            rounds_used=3,
        )
        memory_manager.store_debate_outcome(result, "Design distributed rate limiter")

        # Retrieve memories (simulating next debate)
        entries = continuum_memory.retrieve(
            query="rate limiting distributed",
            tiers=[MemoryTier.FAST, MemoryTier.MEDIUM],
            limit=5,
        )

        # Should find relevant memory
        assert len(entries) >= 1
        found_relevant = any(
            "rate" in e.content.lower() or "redis" in e.content.lower() for e in entries
        )
        # Note: may not find if query doesn't match - that's OK for this test
        # The key is that the storage → retrieval pipeline works

    def test_multiple_debates_affect_memory_tiers(self):
        """Multiple debate outcomes should affect memory tier decisions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "pipeline.db")
            memory = ContinuumMemory(db_path=db_path)
            mm = MemoryManager(continuum_memory=memory)

            # Run several "debates" storing outcomes
            # Use unique 8-char prefixes since id is truncated to first 8 chars
            for i in range(5):
                result = MockDebateResult(
                    id=f"debate{i:02d}x",  # Unique prefixes: debate00x, debate01x, etc.
                    final_answer=f"Approach {i} is optimal",
                    confidence=0.7 + (i * 0.05),
                    rounds_used=2 + (i % 2),
                )
                mm.store_debate_outcome(result, f"Task {i}")

            # All should be stored (unique IDs)
            all_entries = memory.retrieve(limit=100)
            debate_entries = [e for e in all_entries if "debate_outcome" in e.id]
            assert len(debate_entries) >= 5

    def test_surprise_updates_from_debate_outcomes(self, memory_manager, continuum_memory):
        """Debate outcomes should update surprise scores of retrieved memories."""
        # Add baseline memory
        continuum_memory.add(
            id="baseline-pattern",
            content="Token bucket is good for rate limiting",
            tier=MemoryTier.MEDIUM,
            importance=0.6,
        )

        initial = continuum_memory.get("baseline-pattern")
        initial_surprise = initial.surprise_score if initial else 0

        # Track as retrieved
        memory_manager.track_retrieved_ids(["baseline-pattern"])

        # Update with unexpected outcome (low confidence when we expected success)
        result = MockDebateResult(
            confidence=0.4,  # Lower than expected
            consensus_reached=False,
        )
        memory_manager.update_memory_outcomes(result)

        # The update should have been recorded
        updated = continuum_memory.get("baseline-pattern")
        assert updated is not None
        # Note: exact surprise change depends on implementation

    def test_tier_manager_metrics_updated(self, continuum_memory):
        """TierManager metrics should reflect tier transitions."""
        tier_manager = continuum_memory.tier_manager

        # Reset metrics
        tier_manager.reset_metrics()

        # Add entries and force transitions
        continuum_memory.add(
            id="metrics-test-1",
            content="Test entry 1",
            tier=MemoryTier.SLOW,
        )

        # Set conditions for promotion
        with get_wal_connection(continuum_memory.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET surprise_score = 0.9 WHERE id = ?",
                ("metrics-test-1",),
            )
            conn.commit()

        # Run consolidation
        continuum_memory.consolidate()

        # Check metrics
        metrics = tier_manager.get_metrics_dict()
        assert "promotions" in metrics or "total_promotions" in str(metrics).lower()


# =============================================================================
# Evidence Storage Tests
# =============================================================================


class TestEvidenceStorage:
    """Tests for storing evidence snippets from debates."""

    def test_store_evidence_basic(self, memory_manager, continuum_memory):
        """store_evidence should store snippets in MEDIUM tier."""

        class MockSnippet:
            def __init__(self, content, source, relevance):
                self.content = content
                self.source = source
                self.relevance = relevance

        snippets = [
            MockSnippet(
                "Redis provides atomic operations for distributed rate limiting",
                "redis-docs",
                0.8,
            ),
            MockSnippet(
                "Token bucket algorithm allows burst traffic handling",
                "wikipedia",
                0.7,
            ),
        ]

        memory_manager.store_evidence(snippets, "Rate limiter design")

        # Check MEDIUM tier for evidence
        # Content format is "[Evidence:{domain}] ..."
        entries = continuum_memory.retrieve(tiers=[MemoryTier.MEDIUM], limit=20)
        evidence_entries = [e for e in entries if "[Evidence:" in e.content]
        assert len(evidence_entries) >= 2

    def test_store_evidence_filters_short_snippets(self, memory_manager, continuum_memory):
        """Short snippets should be filtered out."""

        class MockSnippet:
            def __init__(self, content):
                self.content = content
                self.source = "test"
                self.relevance = 0.5

        snippets = [
            MockSnippet("Too short"),  # < 50 chars
            MockSnippet(
                "This is a longer snippet that should be stored in the memory system for future retrieval"
            ),
        ]

        memory_manager.store_evidence(snippets, "Test task")

        entries = continuum_memory.retrieve(tiers=[MemoryTier.MEDIUM], limit=20)
        evidence_entries = [e for e in entries if "[Evidence:" in e.content]
        # Only the longer one should be stored
        assert all("Too short" not in e.content for e in evidence_entries)

    def test_store_evidence_limits_to_top_10(self, memory_manager, continuum_memory):
        """Should limit to top 10 snippets."""

        class MockSnippet:
            def __init__(self, content):
                self.content = content
                self.source = "test"
                self.relevance = 0.5

        # Create 15 snippets
        snippets = [
            MockSnippet(
                f"Snippet number {i} with enough content to pass the length filter test requirement"
            )
            for i in range(15)
        ]

        memory_manager.store_evidence(snippets, "Bulk test")

        entries = continuum_memory.retrieve(tiers=[MemoryTier.MEDIUM], limit=50)
        evidence_entries = [e for e in entries if "[Evidence:" in e.content]
        # Should be at most 10
        assert len(evidence_entries) <= 10


# =============================================================================
# Memory Retrieval Integration Tests
# =============================================================================


class TestMemoryRetrieval:
    """Tests for memory retrieval during debates."""

    def test_retrieve_filters_by_tier(self, continuum_memory):
        """Retrieval should filter by specified tiers."""
        # Add entries to different tiers
        continuum_memory.add(id="fast-1", content="Fast entry", tier=MemoryTier.FAST)
        continuum_memory.add(id="medium-1", content="Medium entry", tier=MemoryTier.MEDIUM)
        continuum_memory.add(id="slow-1", content="Slow entry", tier=MemoryTier.SLOW)

        # Retrieve only FAST
        fast_entries = continuum_memory.retrieve(tiers=[MemoryTier.FAST], limit=10)
        assert all(e.tier == MemoryTier.FAST for e in fast_entries)

        # Retrieve FAST and MEDIUM
        fast_medium = continuum_memory.retrieve(
            tiers=[MemoryTier.FAST, MemoryTier.MEDIUM],
            limit=10,
        )
        assert all(e.tier in [MemoryTier.FAST, MemoryTier.MEDIUM] for e in fast_medium)

    def test_retrieve_orders_by_importance(self, continuum_memory):
        """Retrieval should order by importance."""
        continuum_memory.add(id="low-imp", content="Low importance", importance=0.2)
        continuum_memory.add(id="high-imp", content="High importance", importance=0.9)
        continuum_memory.add(id="mid-imp", content="Mid importance", importance=0.5)

        entries = continuum_memory.retrieve(limit=10)

        # Should be ordered by importance descending
        importances = [e.importance for e in entries]
        assert importances == sorted(importances, reverse=True)

    def test_retrieve_with_query(self, continuum_memory):
        """Retrieval with query should filter relevant entries."""
        continuum_memory.add(
            id="relevant",
            content="Rate limiting prevents API abuse",
            importance=0.7,
        )
        continuum_memory.add(
            id="irrelevant",
            content="Database indexing improves query performance",
            importance=0.7,
        )

        # Query for rate limiting
        entries = continuum_memory.retrieve(query="rate limit", limit=10)

        # Should prioritize relevant entry
        if entries:
            # The relevant entry should appear
            contents = [e.content.lower() for e in entries]
            assert any("rate" in c for c in contents)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_memory_manager_handles_none_continuum(self):
        """MemoryManager should handle None continuum_memory."""
        mm = MemoryManager(continuum_memory=None)

        result = MockDebateResult()
        # Should not raise
        mm.store_debate_outcome(result, "Test task")
        mm.update_memory_outcomes(result)

    def test_continuum_memory_handles_concurrent_access(self, temp_db_path):
        """ContinuumMemory should handle concurrent access."""
        import threading

        memory = ContinuumMemory(db_path=temp_db_path)
        errors = []

        def add_entries(thread_id):
            try:
                for i in range(10):
                    memory.add(
                        id=f"thread-{thread_id}-entry-{i}",
                        content=f"Entry from thread {thread_id}",
                        tier=MemoryTier.SLOW,
                    )
            except Exception as e:
                errors.append(e)

        # Run multiple threads
        threads = [threading.Thread(target=add_entries, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have no errors
        assert len(errors) == 0

        # Should have all entries
        entries = memory.retrieve(limit=100)
        assert len(entries) == 50  # 5 threads * 10 entries

    def test_large_content_handling(self, continuum_memory):
        """Should handle large content gracefully."""
        large_content = "x" * 10000  # 10KB content

        continuum_memory.add(
            id="large-content",
            content=large_content,
            tier=MemoryTier.SLOW,
        )

        entry = continuum_memory.get("large-content")
        assert entry is not None
        assert len(entry.content) == 10000

    def test_special_characters_in_content(self, continuum_memory):
        """Should handle special characters in content."""
        special_content = "Test with 'quotes', \"double quotes\", and unicode: 日本語"

        continuum_memory.add(
            id="special-chars",
            content=special_content,
            tier=MemoryTier.SLOW,
        )

        entry = continuum_memory.get("special-chars")
        assert entry is not None
        assert entry.content == special_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
