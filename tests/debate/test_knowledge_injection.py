"""Tests for DebateKnowledgeInjector -- the Receipt -> KM -> Next Debate pipeline.

Verifies that:
1. Query returns relevant past receipts when KM has matching data
2. Query returns empty list when no relevant receipts found
3. format_for_injection produces markdown with past decisions
4. inject_into_prompt appends context to base prompt
5. Respects max_relevant_receipts limit
6. Filters by min_relevance_score
7. Handles ImportError gracefully when KM not available
8. Handles empty KM response
9. Token budget truncation works
"""

from __future__ import annotations

from typing import Any

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from aragora.debate.knowledge_injection import (
    DebateKnowledgeInjector,
    KnowledgeInjectionConfig,
    PastDebateKnowledge,
)

# Patch target: the source module that the lazy import reads from.
_KM_GET = "aragora.knowledge.mound.get_knowledge_mound"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_knowledge_item(
    receipt_id: str = "rcpt_001",
    task: str = "Design a rate limiter",
    confidence: float = 0.85,
    verdict: str = "APPROVED",
    consensus_reached: bool = True,
    final_answer: str = "Token bucket algorithm",
    key_insights: list[str] | None = None,
    dissenting_views: list[str] | None = None,
    tags: list[str] | None = None,
    content: str | None = None,
) -> MagicMock:
    """Create a mock KnowledgeItem matching the unified types schema."""
    item = MagicMock()
    item.id = f"item_{receipt_id}"
    item.content = content or (
        f"Decision Receipt: {verdict}\n\n"
        f"Input: {task}\n\n"
        f"Confidence: {confidence:.0%}\n"
        f"Risk Level: low\n"
    )
    item.metadata = {
        "receipt_id": receipt_id,
        "task": task,
        "verdict": verdict,
        "confidence": confidence,
        "consensus_reached": consensus_reached,
        "final_answer": final_answer,
        "key_insights": key_insights or [],
        "dissenting_views": dissenting_views or [],
        "tags": tags or ["decision_receipt", "summary"],
    }
    item.importance = 0.8
    item.created_at = datetime.now(timezone.utc)
    return item


def _make_query_result(items: list[MagicMock] | None = None) -> MagicMock:
    """Create a mock QueryResult."""
    result = MagicMock()
    result.items = items or []
    result.total_count = len(result.items)
    return result


# ---------------------------------------------------------------------------
# Test PastDebateKnowledge dataclass
# ---------------------------------------------------------------------------

class TestPastDebateKnowledge:
    def test_defaults(self):
        k = PastDebateKnowledge(
            debate_id="d1",
            task="Test",
            final_answer="Answer",
            confidence=0.9,
            consensus_reached=True,
            relevance_score=0.8,
        )
        assert k.key_insights == []
        assert k.dissenting_views == []

    def test_fields(self):
        k = PastDebateKnowledge(
            debate_id="d1",
            task="Rate limiter",
            final_answer="Token bucket",
            confidence=0.85,
            consensus_reached=True,
            relevance_score=0.95,
            key_insights=["Use sliding window"],
            dissenting_views=["Leaky bucket is simpler"],
        )
        assert k.debate_id == "d1"
        assert k.confidence == 0.85
        assert len(k.key_insights) == 1
        assert len(k.dissenting_views) == 1


# ---------------------------------------------------------------------------
# Test KnowledgeInjectionConfig
# ---------------------------------------------------------------------------

class TestKnowledgeInjectionConfig:
    def test_defaults(self):
        config = KnowledgeInjectionConfig()
        assert config.enable_injection is True
        assert config.max_relevant_receipts == 3
        assert config.min_relevance_score == 0.3
        assert config.include_confidence is True
        assert config.include_dissenting_views is True
        assert config.max_context_tokens == 500

    def test_custom_values(self):
        config = KnowledgeInjectionConfig(
            enable_injection=False,
            max_relevant_receipts=5,
            min_relevance_score=0.5,
        )
        assert config.enable_injection is False
        assert config.max_relevant_receipts == 5
        assert config.min_relevance_score == 0.5


# ---------------------------------------------------------------------------
# Test query_relevant_knowledge
# ---------------------------------------------------------------------------

class TestQueryRelevantKnowledge:
    """Test 1: Query returns relevant past receipts when KM has matching data."""

    @pytest.mark.asyncio
    async def test_returns_receipts_from_km(self):
        items = [
            _make_knowledge_item(receipt_id="rcpt_001", task="Design rate limiter"),
            _make_knowledge_item(receipt_id="rcpt_002", task="Auth system design"),
        ]
        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=_make_query_result(items))

        injector = DebateKnowledgeInjector()

        with patch(_KM_GET, return_value=mock_mound):
            result = await injector.query_relevant_knowledge("Design a rate limiter")

        assert len(result) == 2
        assert result[0].debate_id == "rcpt_001"
        assert result[0].relevance_score >= result[1].relevance_score

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_matches(self):
        """Test 2: Query returns empty list when no relevant receipts found."""
        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=_make_query_result([]))

        injector = DebateKnowledgeInjector()

        with patch(_KM_GET, return_value=mock_mound):
            result = await injector.query_relevant_knowledge("Totally unrelated task")

        assert result == []

    @pytest.mark.asyncio
    async def test_respects_max_relevant_receipts(self):
        """Test 5: Respects max_relevant_receipts limit."""
        items = [
            _make_knowledge_item(receipt_id=f"rcpt_{i:03d}", task=f"Task {i}")
            for i in range(10)
        ]
        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=_make_query_result(items))

        config = KnowledgeInjectionConfig(max_relevant_receipts=2)
        injector = DebateKnowledgeInjector(config)

        with patch(_KM_GET, return_value=mock_mound):
            result = await injector.query_relevant_knowledge("Some task")

        assert len(result) <= 2

    @pytest.mark.asyncio
    async def test_filters_by_min_relevance_score(self):
        """Test 6: Filters by min_relevance_score."""
        items = [
            _make_knowledge_item(receipt_id=f"rcpt_{i:03d}", task=f"Task {i}")
            for i in range(10)
        ]
        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=_make_query_result(items))

        # Very high threshold: only first few items should survive
        config = KnowledgeInjectionConfig(
            min_relevance_score=0.9,
            max_relevant_receipts=10,
        )
        injector = DebateKnowledgeInjector(config)

        with patch(_KM_GET, return_value=mock_mound):
            result = await injector.query_relevant_knowledge("Some task")

        # Position-based scoring: only items near the top have relevance >= 0.9
        assert len(result) < 10
        for item in result:
            assert item.relevance_score >= 0.9

    @pytest.mark.asyncio
    async def test_handles_import_error_gracefully(self):
        """Test 7: Handles ImportError gracefully when KM not available."""
        injector = DebateKnowledgeInjector()

        # Temporarily remove the KM module so the lazy import fails
        import sys
        km_modules = {
            k: v for k, v in sys.modules.items() if k.startswith("aragora.knowledge.mound")
        }
        for k in km_modules:
            sys.modules[k] = None  # type: ignore[assignment]
        try:
            result = await injector.query_relevant_knowledge("Test task")
        finally:
            # Restore
            for k, v in km_modules.items():
                sys.modules[k] = v

        assert result == []

    @pytest.mark.asyncio
    async def test_handles_empty_km_response(self):
        """Test 8: Handles empty KM response."""
        mock_result = MagicMock()
        mock_result.items = []

        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=mock_result)

        injector = DebateKnowledgeInjector()

        with patch(_KM_GET, return_value=mock_mound):
            result = await injector.query_relevant_knowledge("Test")

        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_injection_disabled(self):
        config = KnowledgeInjectionConfig(enable_injection=False)
        injector = DebateKnowledgeInjector(config)
        result = await injector.query_relevant_knowledge("Test task")
        assert result == []

    @pytest.mark.asyncio
    async def test_handles_km_query_exception(self):
        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(side_effect=RuntimeError("connection lost"))

        injector = DebateKnowledgeInjector()

        with patch(_KM_GET, return_value=mock_mound):
            result = await injector.query_relevant_knowledge("Test")

        assert result == []

    @pytest.mark.asyncio
    async def test_handles_get_mound_exception(self):
        injector = DebateKnowledgeInjector()

        with patch(_KM_GET, side_effect=RuntimeError("init failed")):
            result = await injector.query_relevant_knowledge("Test")

        assert result == []

    @pytest.mark.asyncio
    async def test_domain_boost(self):
        """Domain matching boosts relevance score."""
        items = [
            _make_knowledge_item(
                receipt_id="rcpt_eng",
                task="Eng task",
                tags=["decision_receipt", "engineering"],
            ),
        ]
        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=_make_query_result(items))

        injector = DebateKnowledgeInjector()

        with patch(_KM_GET, return_value=mock_mound):
            result = await injector.query_relevant_knowledge("Build API", domain="engineering")

        assert len(result) == 1
        # With domain boost, relevance should be capped to 1.0
        assert result[0].relevance_score == 1.0

    @pytest.mark.asyncio
    async def test_extracts_task_from_content_fallback(self):
        """When metadata has no task, falls back to content parsing."""
        item = _make_knowledge_item(receipt_id="rcpt_fb", task="")
        item.content = "Decision Receipt: APPROVED\n\nInput: Build a cache layer\n\nConfidence: 85%"
        item.metadata["task"] = ""

        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=_make_query_result([item]))

        injector = DebateKnowledgeInjector()

        with patch(_KM_GET, return_value=mock_mound):
            result = await injector.query_relevant_knowledge("Cache design")

        assert len(result) == 1
        assert "Build a cache layer" in result[0].task

    @pytest.mark.asyncio
    async def test_extracts_verdict_as_final_answer_fallback(self):
        """When no final_answer, falls back to verdict."""
        item = _make_knowledge_item(receipt_id="rcpt_v", final_answer="")
        item.metadata["final_answer"] = ""

        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=_make_query_result([item]))

        injector = DebateKnowledgeInjector()

        with patch(_KM_GET, return_value=mock_mound):
            result = await injector.query_relevant_knowledge("Test")

        assert len(result) == 1
        assert "APPROVED" in result[0].final_answer


# ---------------------------------------------------------------------------
# Test format_for_injection
# ---------------------------------------------------------------------------

class TestFormatForInjection:
    """Test 3: format_for_injection produces markdown with past decisions."""

    def test_produces_markdown(self):
        knowledge = [
            PastDebateKnowledge(
                debate_id="d1",
                task="Design rate limiter",
                final_answer="Token bucket algorithm",
                confidence=0.85,
                consensus_reached=True,
                relevance_score=0.9,
                key_insights=["Sliding window works better at scale"],
                dissenting_views=["Leaky bucket is simpler"],
            ),
        ]
        injector = DebateKnowledgeInjector()
        result = injector.format_for_injection(knowledge)

        assert "## Relevant Past Decisions" in result
        assert "**Design rate limiter**" in result
        assert "(confidence: 0.85)" in result
        assert "- Decision: Token bucket algorithm" in result
        assert "- Key insight: Sliding window works better at scale" in result
        assert "- Dissenting view: Leaky bucket is simpler" in result

    def test_empty_knowledge_returns_empty_string(self):
        injector = DebateKnowledgeInjector()
        assert injector.format_for_injection([]) == ""

    def test_excludes_dissenting_views_when_disabled(self):
        knowledge = [
            PastDebateKnowledge(
                debate_id="d1",
                task="Test",
                final_answer="Answer",
                confidence=0.9,
                consensus_reached=True,
                relevance_score=0.8,
                dissenting_views=["Counterpoint"],
            ),
        ]
        config = KnowledgeInjectionConfig(include_dissenting_views=False)
        injector = DebateKnowledgeInjector(config)
        result = injector.format_for_injection(knowledge)

        assert "Dissenting view" not in result

    def test_excludes_confidence_when_disabled(self):
        knowledge = [
            PastDebateKnowledge(
                debate_id="d1",
                task="Test",
                final_answer="Answer",
                confidence=0.9,
                consensus_reached=True,
                relevance_score=0.8,
            ),
        ]
        config = KnowledgeInjectionConfig(include_confidence=False)
        injector = DebateKnowledgeInjector(config)
        result = injector.format_for_injection(knowledge)

        assert "confidence:" not in result

    def test_token_budget_truncation(self):
        """Test 9: Token budget truncation works."""
        # Create knowledge that will exceed a small token budget
        knowledge = [
            PastDebateKnowledge(
                debate_id=f"d{i}",
                task=f"Very long task description number {i} " * 10,
                final_answer=f"Detailed answer {i} " * 20,
                confidence=0.85,
                consensus_reached=True,
                relevance_score=0.9 - (i * 0.1),
                key_insights=[f"Insight {i} " * 10],
                dissenting_views=[f"Dissent {i} " * 10],
            )
            for i in range(5)
        ]
        config = KnowledgeInjectionConfig(max_context_tokens=50)  # Very small budget
        injector = DebateKnowledgeInjector(config)
        result = injector.format_for_injection(knowledge)

        # 50 tokens * 4 chars = 200 chars max
        assert len(result) <= 200 + 10  # small margin for trailing "..."
        assert result.endswith("...")

    def test_multiple_items(self):
        knowledge = [
            PastDebateKnowledge(
                debate_id="d1",
                task="First task",
                final_answer="First answer",
                confidence=0.9,
                consensus_reached=True,
                relevance_score=0.95,
            ),
            PastDebateKnowledge(
                debate_id="d2",
                task="Second task",
                final_answer="Second answer",
                confidence=0.7,
                consensus_reached=False,
                relevance_score=0.8,
            ),
        ]
        injector = DebateKnowledgeInjector()
        result = injector.format_for_injection(knowledge)

        assert "First task" in result
        assert "Second task" in result
        assert "First answer" in result
        assert "Second answer" in result

    def test_uses_debate_id_when_no_task(self):
        knowledge = [
            PastDebateKnowledge(
                debate_id="d123",
                task="",
                final_answer="Answer",
                confidence=0.8,
                consensus_reached=True,
                relevance_score=0.7,
            ),
        ]
        injector = DebateKnowledgeInjector()
        result = injector.format_for_injection(knowledge)

        assert "**d123**" in result


# ---------------------------------------------------------------------------
# Test inject_into_prompt
# ---------------------------------------------------------------------------

class TestInjectIntoPrompt:
    """Test 4: inject_into_prompt appends context to base prompt."""

    @pytest.mark.asyncio
    async def test_appends_context_to_prompt(self):
        items = [
            _make_knowledge_item(
                receipt_id="rcpt_001",
                task="Rate limiter design",
                final_answer="Token bucket",
                key_insights=["Use sliding window"],
            ),
        ]
        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=_make_query_result(items))

        injector = DebateKnowledgeInjector()
        base = "You are debating rate limiter design."

        with patch(_KM_GET, return_value=mock_mound):
            result = await injector.inject_into_prompt(base, "Rate limiter design")

        assert result.startswith(base)
        assert "## Relevant Past Decisions" in result
        assert "Rate limiter design" in result

    @pytest.mark.asyncio
    async def test_returns_base_prompt_when_no_knowledge(self):
        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=_make_query_result([]))

        injector = DebateKnowledgeInjector()
        base = "You are debating."

        with patch(_KM_GET, return_value=mock_mound):
            result = await injector.inject_into_prompt(base, "Unknown topic")

        assert result == base

    @pytest.mark.asyncio
    async def test_returns_base_when_disabled(self):
        config = KnowledgeInjectionConfig(enable_injection=False)
        injector = DebateKnowledgeInjector(config)
        base = "You are debating."
        result = await injector.inject_into_prompt(base, "Test")
        assert result == base

    @pytest.mark.asyncio
    async def test_passes_domain_to_query(self):
        items = [_make_knowledge_item()]
        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=_make_query_result(items))

        injector = DebateKnowledgeInjector()

        with patch(_KM_GET, return_value=mock_mound):
            result = await injector.inject_into_prompt(
                "Base prompt", "Design API", domain="engineering"
            )

        assert "## Relevant Past Decisions" in result
