"""
Tests for RLM Knowledge Helpers.

Covers:
- KnowledgeItem and KnowledgeREPLContext dataclasses
- Context loading from KnowledgeMound
- Knowledge retrieval functions (get_facts, get_claims, get_evidence)
- Filtering and search functions (filter_by_confidence, search_knowledge)
- Graph traversal (get_related)
- Grouping and partitioning (group_by_source, partition_by_topic)
- RLM primitives (RLM_M, FINAL)
- Helper injection (get_knowledge_helpers)
- Error handling and graceful degradation
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.rlm.knowledge_helpers import (
    FINAL,
    KnowledgeItem,
    KnowledgeREPLContext,
    RLM_M,
    _to_knowledge_item,
    _truncate_content,
    filter_by_confidence,
    get_claims,
    get_evidence,
    get_facts,
    get_item,
    get_knowledge_helpers,
    get_related,
    group_by_source,
    load_knowledge_context,
    partition_by_topic,
    search_knowledge,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_fact():
    """Sample fact knowledge item."""
    return KnowledgeItem(
        id="fact-001",
        content="Rate limiting prevents API abuse by restricting request frequency.",
        source="fact",
        confidence=0.9,
        created_at="2024-06-15T12:00:00Z",
        metadata={"category": "api", "verified": True},
        relationships=["fact-002", "claim-001"],
    )


@pytest.fixture
def sample_claim():
    """Sample claim knowledge item."""
    return KnowledgeItem(
        id="claim-001",
        content="The current rate limiter implementation has a bug in token bucket refill.",
        source="claim",
        confidence=0.6,
        created_at="2024-06-15T13:00:00Z",
        metadata={"validated": False, "debated": True},
        relationships=["fact-001"],
    )


@pytest.fixture
def sample_evidence():
    """Sample evidence knowledge item."""
    return KnowledgeItem(
        id="evidence-001",
        content="Test results show 5% of requests fail under high load.",
        source="evidence",
        confidence=0.85,
        created_at="2024-06-15T14:00:00Z",
        metadata={"source_type": "document", "test_run": "TR-456"},
        relationships=["claim-001"],
    )


@pytest.fixture
def sample_context(sample_fact, sample_claim, sample_evidence):
    """Sample KnowledgeREPLContext with test data."""
    all_items = [sample_fact, sample_claim, sample_evidence]
    by_id = {item.id: item for item in all_items}
    relationships = {item.id: item.relationships for item in all_items}

    return KnowledgeREPLContext(
        workspace_id="ws-test-123",
        facts=[sample_fact],
        claims=[sample_claim],
        evidence=[sample_evidence],
        all_items=all_items,
        by_id=by_id,
        relationships=relationships,
        total_items=3,
        avg_confidence=0.783,  # (0.9 + 0.6 + 0.85) / 3
    )


@pytest.fixture
def mock_mound():
    """Mock KnowledgeMound with test data."""
    mound = MagicMock()
    mound.get_facts.return_value = [
        {"id": "fact-001", "content": "Test fact", "confidence": 0.9},
        {"id": "fact-002", "content": "Another fact about security", "confidence": 0.8},
    ]
    mound.get_claims.return_value = [
        {"id": "claim-001", "content": "Test claim", "confidence": 0.6, "metadata": {"validated": True}},
    ]
    mound.get_evidence.return_value = [
        {"id": "evidence-001", "content": "Test evidence", "confidence": 0.85, "metadata": {"source_type": "document"}},
    ]
    return mound


# ---------------------------------------------------------------------------
# Test KnowledgeItem Dataclass
# ---------------------------------------------------------------------------


class TestKnowledgeItem:
    """Tests for KnowledgeItem dataclass."""

    def test_knowledge_item_creation(self, sample_fact):
        """KnowledgeItem should have all expected fields."""
        assert sample_fact.id == "fact-001"
        assert "rate limiting" in sample_fact.content.lower()
        assert sample_fact.source == "fact"
        assert sample_fact.confidence == 0.9
        assert sample_fact.created_at == "2024-06-15T12:00:00Z"
        assert "category" in sample_fact.metadata
        assert len(sample_fact.relationships) == 2

    def test_knowledge_item_defaults(self):
        """KnowledgeItem should have sensible defaults."""
        item = KnowledgeItem(
            id="test-id",
            content="Test content",
            source="fact",
            confidence=0.5,
            created_at="",
        )
        assert item.metadata == {}
        assert item.relationships == []


# ---------------------------------------------------------------------------
# Test KnowledgeREPLContext Dataclass
# ---------------------------------------------------------------------------


class TestKnowledgeREPLContext:
    """Tests for KnowledgeREPLContext dataclass."""

    def test_context_creation(self, sample_context):
        """KnowledgeREPLContext should have all expected fields."""
        assert sample_context.workspace_id == "ws-test-123"
        assert len(sample_context.facts) == 1
        assert len(sample_context.claims) == 1
        assert len(sample_context.evidence) == 1
        assert len(sample_context.all_items) == 3
        assert sample_context.total_items == 3
        assert 0.78 <= sample_context.avg_confidence <= 0.79

    def test_context_by_id_lookup(self, sample_context):
        """Context should support item lookup by ID."""
        assert "fact-001" in sample_context.by_id
        assert sample_context.by_id["fact-001"].source == "fact"

    def test_context_relationships_graph(self, sample_context):
        """Context should have relationship graph."""
        assert "fact-001" in sample_context.relationships
        assert "claim-001" in sample_context.relationships["fact-001"]


# ---------------------------------------------------------------------------
# Test Context Loading
# ---------------------------------------------------------------------------


class TestLoadKnowledgeContext:
    """Tests for load_knowledge_context function."""

    def test_load_knowledge_context_basic(self, mock_mound):
        """load_knowledge_context should create context from mound."""
        context = load_knowledge_context(mock_mound, "ws-123", limit=100)

        assert context.workspace_id == "ws-123"
        assert len(context.facts) == 2
        assert len(context.claims) == 1
        assert len(context.evidence) == 1
        assert context.total_items == 4

    def test_load_knowledge_context_calculates_avg_confidence(self, mock_mound):
        """load_knowledge_context should calculate average confidence."""
        context = load_knowledge_context(mock_mound, "ws-123")

        # (0.9 + 0.8 + 0.6 + 0.85) / 4 = 0.7875
        assert 0.78 <= context.avg_confidence <= 0.79

    def test_load_knowledge_context_builds_by_id(self, mock_mound):
        """load_knowledge_context should build by_id lookup."""
        context = load_knowledge_context(mock_mound, "ws-123")

        assert "fact-001" in context.by_id
        assert "claim-001" in context.by_id
        assert "evidence-001" in context.by_id

    def test_load_knowledge_context_graceful_degradation(self):
        """load_knowledge_context should handle missing methods gracefully."""
        mound = MagicMock(spec=[])  # Empty spec - no methods

        context = load_knowledge_context(mound, "ws-123")

        assert context.workspace_id == "ws-123"
        assert context.total_items == 0
        assert context.avg_confidence == 0.0

    def test_load_knowledge_context_handles_none_results(self):
        """load_knowledge_context should handle None from mound methods."""
        mound = MagicMock()
        mound.get_facts.return_value = None
        mound.get_claims.return_value = None
        mound.get_evidence.return_value = None

        context = load_knowledge_context(mound, "ws-123")

        assert context.total_items == 0


# ---------------------------------------------------------------------------
# Test Knowledge Retrieval Functions
# ---------------------------------------------------------------------------


class TestGetFacts:
    """Tests for get_facts function."""

    def test_get_facts_all(self, sample_context):
        """get_facts should return all facts when no filters."""
        facts = get_facts(sample_context)
        assert len(facts) == 1
        assert facts[0].source == "fact"

    def test_get_facts_with_query(self):
        """get_facts should filter by query string."""
        fact1 = KnowledgeItem(id="f1", content="Rate limiting is important", source="fact", confidence=0.9, created_at="")
        fact2 = KnowledgeItem(id="f2", content="Security best practices", source="fact", confidence=0.8, created_at="")
        context = KnowledgeREPLContext(
            workspace_id="ws",
            facts=[fact1, fact2],
            claims=[],
            evidence=[],
            all_items=[fact1, fact2],
            by_id={"f1": fact1, "f2": fact2},
            relationships={},
            total_items=2,
            avg_confidence=0.85,
        )

        results = get_facts(context, query="rate")
        assert len(results) == 1
        assert results[0].id == "f1"

    def test_get_facts_with_min_confidence(self, sample_context):
        """get_facts should filter by minimum confidence."""
        results = get_facts(sample_context, min_confidence=0.95)
        assert len(results) == 0

        results = get_facts(sample_context, min_confidence=0.8)
        assert len(results) == 1


class TestGetClaims:
    """Tests for get_claims function."""

    def test_get_claims_all(self, sample_context):
        """get_claims should return all claims when no filters."""
        claims = get_claims(sample_context)
        assert len(claims) == 1
        assert claims[0].source == "claim"

    def test_get_claims_validated_only(self):
        """get_claims should filter by validation status."""
        claim1 = KnowledgeItem(id="c1", content="Claim 1", source="claim", confidence=0.7, created_at="", metadata={"validated": True})
        claim2 = KnowledgeItem(id="c2", content="Claim 2", source="claim", confidence=0.5, created_at="", metadata={"validated": False})
        context = KnowledgeREPLContext(
            workspace_id="ws",
            facts=[],
            claims=[claim1, claim2],
            evidence=[],
            all_items=[claim1, claim2],
            by_id={"c1": claim1, "c2": claim2},
            relationships={},
            total_items=2,
            avg_confidence=0.6,
        )

        results = get_claims(context, validated_only=True)
        assert len(results) == 1
        assert results[0].id == "c1"


class TestGetEvidence:
    """Tests for get_evidence function."""

    def test_get_evidence_all(self, sample_context):
        """get_evidence should return all evidence when no filters."""
        evidence = get_evidence(sample_context)
        assert len(evidence) == 1
        assert evidence[0].source == "evidence"

    def test_get_evidence_by_source_type(self, sample_context):
        """get_evidence should filter by source type."""
        results = get_evidence(sample_context, source_type="document")
        assert len(results) == 1

        results = get_evidence(sample_context, source_type="debate")
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Test Filtering Functions
# ---------------------------------------------------------------------------


class TestFilterByConfidence:
    """Tests for filter_by_confidence function."""

    def test_filter_by_min_confidence(self, sample_context):
        """filter_by_confidence should filter by minimum."""
        results = filter_by_confidence(sample_context.all_items, min_confidence=0.8)
        assert len(results) == 2  # fact (0.9) and evidence (0.85)

    def test_filter_by_max_confidence(self, sample_context):
        """filter_by_confidence should filter by maximum."""
        results = filter_by_confidence(sample_context.all_items, max_confidence=0.7)
        assert len(results) == 1  # claim (0.6)

    def test_filter_by_confidence_range(self, sample_context):
        """filter_by_confidence should filter by range."""
        results = filter_by_confidence(sample_context.all_items, min_confidence=0.7, max_confidence=0.9)
        assert len(results) == 2  # fact (0.9) and evidence (0.85)


class TestSearchKnowledge:
    """Tests for search_knowledge function."""

    def test_search_knowledge_basic(self, sample_context):
        """search_knowledge should find items matching pattern."""
        results = search_knowledge(sample_context, "rate")
        # Both fact-001 ("Rate limiting") and claim-001 ("rate limiter") match
        assert len(results) == 2
        result_ids = {r.id for r in results}
        assert "fact-001" in result_ids
        assert "claim-001" in result_ids

    def test_search_knowledge_case_insensitive(self, sample_context):
        """search_knowledge should be case insensitive by default."""
        results_lower = search_knowledge(sample_context, "rate")
        results_upper = search_knowledge(sample_context, "RATE")
        assert len(results_lower) == len(results_upper)

    def test_search_knowledge_regex(self, sample_context):
        """search_knowledge should support regex patterns."""
        results = search_knowledge(sample_context, r"api|test")
        assert len(results) >= 1

    def test_search_knowledge_case_sensitive(self):
        """search_knowledge should support case sensitive search."""
        item = KnowledgeItem(id="i1", content="API endpoint", source="fact", confidence=0.9, created_at="")
        context = KnowledgeREPLContext(
            workspace_id="ws",
            facts=[item],
            claims=[],
            evidence=[],
            all_items=[item],
            by_id={"i1": item},
            relationships={},
            total_items=1,
            avg_confidence=0.9,
        )

        results_insensitive = search_knowledge(context, "api", case_insensitive=True)
        results_sensitive = search_knowledge(context, "api", case_insensitive=False)

        assert len(results_insensitive) == 1
        assert len(results_sensitive) == 0  # "api" doesn't match "API"


# ---------------------------------------------------------------------------
# Test Graph Traversal
# ---------------------------------------------------------------------------


class TestGetRelated:
    """Tests for get_related function."""

    def test_get_related_depth_1(self, sample_context):
        """get_related should find directly related items."""
        related = get_related(sample_context, "fact-001", depth=1)
        # fact-001 relates to fact-002 and claim-001, but fact-002 not in context
        assert any(item.id == "claim-001" for item in related)

    def test_get_related_depth_2(self, sample_context):
        """get_related should follow relationships up to specified depth."""
        related = get_related(sample_context, "evidence-001", depth=2)
        # evidence-001 -> claim-001 -> fact-001
        ids = [item.id for item in related]
        assert "claim-001" in ids or "fact-001" in ids

    def test_get_related_nonexistent_id(self, sample_context):
        """get_related should return empty list for nonexistent ID."""
        related = get_related(sample_context, "nonexistent", depth=1)
        assert related == []


class TestGetItem:
    """Tests for get_item function."""

    def test_get_item_found(self, sample_context):
        """get_item should return item when found."""
        item = get_item(sample_context, "fact-001")
        assert item is not None
        assert item.id == "fact-001"

    def test_get_item_not_found(self, sample_context):
        """get_item should return None when not found."""
        item = get_item(sample_context, "nonexistent")
        assert item is None


# ---------------------------------------------------------------------------
# Test Grouping and Partitioning
# ---------------------------------------------------------------------------


class TestGroupBySource:
    """Tests for group_by_source function."""

    def test_group_by_source(self, sample_context):
        """group_by_source should group items by source type."""
        grouped = group_by_source(sample_context)

        assert "fact" in grouped
        assert "claim" in grouped
        assert "evidence" in grouped
        assert len(grouped["fact"]) == 1
        assert len(grouped["claim"]) == 1
        assert len(grouped["evidence"]) == 1


class TestPartitionByTopic:
    """Tests for partition_by_topic function."""

    def test_partition_by_topic(self, sample_context):
        """partition_by_topic should partition items by topic keywords."""
        partitions = partition_by_topic(sample_context, ["rate", "test", "security"])

        assert "rate" in partitions
        assert "test" in partitions
        assert "other" in partitions
        assert len(partitions["rate"]) >= 1  # fact-001 contains "rate"

    def test_partition_by_topic_other_category(self):
        """partition_by_topic should put non-matching items in 'other'."""
        item = KnowledgeItem(id="i1", content="Unrelated content", source="fact", confidence=0.9, created_at="")
        context = KnowledgeREPLContext(
            workspace_id="ws",
            facts=[item],
            claims=[],
            evidence=[],
            all_items=[item],
            by_id={"i1": item},
            relationships={},
            total_items=1,
            avg_confidence=0.9,
        )

        partitions = partition_by_topic(context, ["api", "security"])
        assert len(partitions["other"]) == 1


# ---------------------------------------------------------------------------
# Test RLM Primitives
# ---------------------------------------------------------------------------


class TestRLM_M:
    """Tests for RLM_M primitive."""

    def test_rlm_m_empty_subset(self):
        """RLM_M should handle empty subset."""
        result = RLM_M("What are the facts?", subset=[])
        assert "No knowledge items" in result

    def test_rlm_m_none_subset(self):
        """RLM_M should handle None subset."""
        result = RLM_M("What are the facts?", subset=None)
        assert "No knowledge items" in result

    def test_rlm_m_summary_query(self, sample_context):
        """RLM_M should provide summary for summary queries."""
        result = RLM_M("Summarize the findings", subset=sample_context.all_items)
        assert "synthesis" in result.lower() or "items" in result.lower()

    def test_rlm_m_verification_query(self, sample_context):
        """RLM_M should focus on high-confidence items for verification queries."""
        result = RLM_M("What are the verified facts?", subset=sample_context.all_items)
        assert "confidence" in result.lower() or "high" in result.lower() or "0." in result

    def test_rlm_m_validation_query(self, sample_context):
        """RLM_M should focus on low-confidence claims for validation queries."""
        result = RLM_M("What claims need validation?", subset=sample_context.all_items)
        # Should mention validation or claims
        assert "claim" in result.lower() or "validation" in result.lower() or "0." in result


class TestFINAL:
    """Tests for FINAL primitive."""

    def test_final_returns_answer(self):
        """FINAL should return the answer."""
        result = FINAL("The answer is 42")
        assert result == "The answer is 42"

    def test_final_preserves_type(self):
        """FINAL should preserve answer type."""
        assert FINAL("string") == "string"
        # FINAL always returns string in this implementation


# ---------------------------------------------------------------------------
# Test Helper Injection
# ---------------------------------------------------------------------------


class TestGetKnowledgeHelpers:
    """Tests for get_knowledge_helpers function."""

    def test_get_knowledge_helpers_basic(self):
        """get_knowledge_helpers should return dictionary of helpers."""
        helpers = get_knowledge_helpers()

        assert isinstance(helpers, dict)
        assert "KnowledgeItem" in helpers
        assert "KnowledgeREPLContext" in helpers
        assert "load_knowledge_context" in helpers
        assert "get_facts" in helpers
        assert "get_claims" in helpers
        assert "get_evidence" in helpers
        assert "filter_by_confidence" in helpers
        assert "search_knowledge" in helpers
        assert "get_related" in helpers

    def test_get_knowledge_helpers_excludes_rlm_by_default(self):
        """get_knowledge_helpers should exclude RLM primitives by default."""
        helpers = get_knowledge_helpers()

        assert "RLM_M" not in helpers
        assert "FINAL" not in helpers

    def test_get_knowledge_helpers_includes_rlm_when_requested(self):
        """get_knowledge_helpers should include RLM primitives when requested."""
        helpers = get_knowledge_helpers(include_rlm_primitives=True)

        assert "RLM_M" in helpers
        assert "FINAL" in helpers
        assert helpers["RLM_M"] == RLM_M
        assert helpers["FINAL"] == FINAL

    def test_get_knowledge_helpers_with_mound(self, mock_mound):
        """get_knowledge_helpers should add km_load when mound provided."""
        helpers = get_knowledge_helpers(mound=mock_mound)

        assert "km_load" in helpers
        assert callable(helpers["km_load"])

    def test_km_load_convenience_function(self, mock_mound):
        """km_load should load context for workspace."""
        helpers = get_knowledge_helpers(mound=mock_mound)
        context = helpers["km_load"]("ws-test")

        assert context.workspace_id == "ws-test"
        assert len(context.all_items) > 0


# ---------------------------------------------------------------------------
# Test Internal Helper Functions
# ---------------------------------------------------------------------------


class TestToKnowledgeItem:
    """Tests for _to_knowledge_item helper."""

    def test_from_dict(self):
        """_to_knowledge_item should convert dict to KnowledgeItem."""
        data = {
            "id": "test-id",
            "content": "Test content",
            "confidence": 0.8,
            "created_at": "2024-01-01",
            "metadata": {"key": "value"},
            "relationships": ["rel-1"],
        }
        item = _to_knowledge_item(data, "fact")

        assert item.id == "test-id"
        assert item.content == "Test content"
        assert item.confidence == 0.8
        assert item.source == "fact"

    def test_from_dict_with_text_field(self):
        """_to_knowledge_item should use 'text' field as fallback for content."""
        data = {"id": "test-id", "text": "Test text"}
        item = _to_knowledge_item(data, "claim")

        assert item.content == "Test text"

    def test_from_object_with_model_dump(self):
        """_to_knowledge_item should handle objects with model_dump."""
        mock_obj = MagicMock()
        mock_obj.model_dump.return_value = {"id": "obj-id", "content": "Object content", "confidence": 0.7}
        item = _to_knowledge_item(mock_obj, "evidence")

        assert item.id == "obj-id"
        assert item.content == "Object content"

    def test_from_object_with_dict(self):
        """_to_knowledge_item should handle objects with __dict__."""

        class SimpleObject:
            def __init__(self):
                self.id = "simple-id"
                self.content = "Simple content"
                self.confidence = 0.6

        obj = SimpleObject()
        item = _to_knowledge_item(obj, "fact")

        assert item.id == "simple-id"
        assert item.content == "Simple content"

    def test_from_primitive(self):
        """_to_knowledge_item should handle primitive values."""
        item = _to_knowledge_item("Just a string", "claim")

        assert item.content == "Just a string"
        assert item.confidence == 0.5  # Default


class TestTruncateContent:
    """Tests for _truncate_content helper."""

    def test_no_truncation_needed(self):
        """_truncate_content should not truncate short content."""
        content = "Short"
        result = _truncate_content(content, 100)
        assert result == "Short"

    def test_truncation_at_word_boundary(self):
        """_truncate_content should truncate at word boundary."""
        content = "This is a longer piece of content that needs truncation"
        result = _truncate_content(content, 30)
        assert len(result) <= 33  # 30 + "..."
        assert result.endswith("...")

    def test_truncation_exact_length(self):
        """_truncate_content should handle exact length."""
        content = "Exactly ten"
        result = _truncate_content(content, 11)
        assert result == "Exactly ten"
