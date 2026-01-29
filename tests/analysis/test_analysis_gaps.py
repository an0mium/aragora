"""
Tests for previously untested analysis modules:
- aragora.analysis.email_priority (EmailPriorityAnalyzer, EmailFeedbackLearner, dataclasses)
- aragora.analysis.nl_query (DocumentQueryEngine, QueryConfig, QueryResult, QueryMode, etc.)
- aragora.analysis.codebase.models (VulnerabilityFinding, DependencyInfo, ScanResult, etc.)
- aragora.analysis.codebase.metrics (CodeMetricsAnalyzer, PythonAnalyzer, ComplexityVisitor, etc.)
"""

from __future__ import annotations

import ast
import asyncio
import textwrap
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# email_priority imports
# ---------------------------------------------------------------------------
from aragora.analysis.email_priority import (
    EmailFeedbackLearner,
    EmailPriorityAnalyzer,
    EmailPriorityScore,
    UserEmailPreferences,
)

# ---------------------------------------------------------------------------
# nl_query imports
# ---------------------------------------------------------------------------
from aragora.analysis.nl_query import (
    AnswerConfidence,
    Citation,
    DocumentQueryEngine,
    QueryConfig,
    QueryMode,
    QueryResult,
    StreamingChunk,
)

# ---------------------------------------------------------------------------
# codebase.models imports
# ---------------------------------------------------------------------------
from aragora.analysis.codebase.models import (
    CodeMetric,
    DependencyInfo,
    HotspotFinding,
    MetricType,
    ScanResult,
    SecretFinding,
    SecretsScanResult,
    SecretType,
    VulnerabilityFinding,
    VulnerabilityReference,
    VulnerabilitySeverity,
    VulnerabilitySource,
)

# ---------------------------------------------------------------------------
# codebase.metrics imports
# ---------------------------------------------------------------------------
from aragora.analysis.codebase.metrics import (
    CodeMetricsAnalyzer,
    ComplexityVisitor,
    DuplicateBlock,
    DuplicateDetector,
    FileMetrics,
    FunctionMetrics,
    MetricsReport,
    PythonAnalyzer,
    TypeScriptAnalyzer,
)


# ==========================================================================
# EmailPriorityScore dataclass
# ==========================================================================


class TestEmailPriorityScore:
    def test_creation(self):
        score = EmailPriorityScore(
            email_id="e1", score=0.85, reason="Important sender", factors={"sender": 1.0}
        )
        assert score.email_id == "e1"
        assert score.score == 0.85
        assert score.reason == "Important sender"
        assert score.factors == {"sender": 1.0}

    def test_to_dict(self):
        score = EmailPriorityScore(
            email_id="e2", score=0.5, reason="Neutral", factors={"urgency": 0.5}
        )
        d = score.to_dict()
        assert d["email_id"] == "e2"
        assert d["score"] == 0.5
        assert d["reason"] == "Neutral"
        assert d["factors"] == {"urgency": 0.5}

    def test_default_factors(self):
        score = EmailPriorityScore(email_id="e3", score=0.0, reason="")
        assert score.factors == {}
        assert score.to_dict()["factors"] == {}


# ==========================================================================
# UserEmailPreferences dataclass
# ==========================================================================


class TestUserEmailPreferences:
    def test_creation_defaults(self):
        prefs = UserEmailPreferences(user_id="u1")
        assert prefs.user_id == "u1"
        assert prefs.important_senders == []
        assert prefs.important_domains == []
        assert prefs.important_keywords == []
        assert prefs.low_priority_senders == []
        assert prefs.low_priority_keywords == []
        assert prefs.interaction_weights == {}

    def test_to_dict(self):
        prefs = UserEmailPreferences(
            user_id="u2",
            important_senders=["boss@work.com"],
            important_domains=["work.com"],
            important_keywords=["deadline"],
            low_priority_senders=["spam@promo.com"],
            low_priority_keywords=["unsubscribe"],
            interaction_weights={"replied": 0.9},
        )
        d = prefs.to_dict()
        assert d["user_id"] == "u2"
        assert "boss@work.com" in d["important_senders"]
        assert d["interaction_weights"] == {"replied": 0.9}


# ==========================================================================
# EmailPriorityAnalyzer
# ==========================================================================


class TestEmailPriorityAnalyzer:
    """Test the scoring logic of EmailPriorityAnalyzer with mocked memory/rlm."""

    def _make_analyzer(self, prefs=None):
        memory = AsyncMock()
        memory.query = AsyncMock(return_value=[])
        rlm = None
        analyzer = EmailPriorityAnalyzer(user_id="u1", memory=memory, rlm=rlm)
        if prefs:
            analyzer._preferences = prefs
        return analyzer

    @pytest.mark.asyncio
    async def test_score_email_basic(self):
        analyzer = self._make_analyzer(prefs=UserEmailPreferences(user_id="u1"))
        result = await analyzer.score_email(
            email_id="e1",
            subject="Hello",
            from_address="someone@gmail.com",
            snippet="Just saying hi",
        )
        assert isinstance(result, EmailPriorityScore)
        assert 0.0 <= result.score <= 1.0
        assert result.email_id == "e1"
        assert "sender" in result.factors
        assert "urgency" in result.factors

    @pytest.mark.asyncio
    async def test_sender_scoring_important(self):
        prefs = UserEmailPreferences(user_id="u1", important_senders=["boss@work.com"])
        analyzer = self._make_analyzer(prefs=prefs)
        result = await analyzer.score_email(
            email_id="e2",
            subject="Meeting",
            from_address="boss@work.com",
            snippet="Let's meet tomorrow",
        )
        assert result.factors["sender"] == 1.0

    @pytest.mark.asyncio
    async def test_sender_scoring_low_priority(self):
        prefs = UserEmailPreferences(user_id="u1", low_priority_senders=["spam@promo.com"])
        analyzer = self._make_analyzer(prefs=prefs)
        result = await analyzer.score_email(
            email_id="e3",
            subject="Sale!",
            from_address="spam@promo.com",
            snippet="50% off",
        )
        assert result.factors["sender"] == 0.2

    @pytest.mark.asyncio
    async def test_sender_scoring_important_domain(self):
        prefs = UserEmailPreferences(user_id="u1", important_domains=["acme.io"])
        analyzer = self._make_analyzer(prefs=prefs)
        result = await analyzer.score_email(
            email_id="e4",
            subject="Update",
            from_address="alice@acme.io",
            snippet="Project update",
        )
        assert result.factors["sender"] == 0.8

    @pytest.mark.asyncio
    async def test_gmail_signals_starred(self):
        prefs = UserEmailPreferences(user_id="u1")
        analyzer = self._make_analyzer(prefs=prefs)
        result = await analyzer.score_email(
            email_id="e5",
            subject="Test",
            from_address="x@y.com",
            snippet="test",
            is_starred=True,
        )
        assert result.factors["gmail_signals"] == 1.0

    @pytest.mark.asyncio
    async def test_gmail_signals_important_label(self):
        prefs = UserEmailPreferences(user_id="u1")
        analyzer = self._make_analyzer(prefs=prefs)
        result = await analyzer.score_email(
            email_id="e6",
            subject="Test",
            from_address="x@y.com",
            snippet="test",
            labels=["IMPORTANT"],
        )
        assert result.factors["gmail_signals"] == 0.9

    @pytest.mark.asyncio
    async def test_gmail_signals_promotions(self):
        prefs = UserEmailPreferences(user_id="u1")
        analyzer = self._make_analyzer(prefs=prefs)
        result = await analyzer.score_email(
            email_id="e7",
            subject="Test",
            from_address="x@y.com",
            snippet="test",
            labels=["CATEGORY_PROMOTIONS"],
        )
        assert result.factors["gmail_signals"] == 0.3

    @pytest.mark.asyncio
    async def test_urgency_high(self):
        prefs = UserEmailPreferences(user_id="u1")
        analyzer = self._make_analyzer(prefs=prefs)
        result = await analyzer.score_email(
            email_id="e8",
            subject="URGENT: Action Required",
            from_address="x@y.com",
            snippet="Please respond ASAP",
        )
        assert result.factors["urgency"] == 1.0

    @pytest.mark.asyncio
    async def test_urgency_medium(self):
        prefs = UserEmailPreferences(user_id="u1")
        analyzer = self._make_analyzer(prefs=prefs)
        result = await analyzer.score_email(
            email_id="e9",
            subject="Please review",
            from_address="x@y.com",
            snippet="Waiting for your response",
        )
        assert result.factors["urgency"] == 0.7

    @pytest.mark.asyncio
    async def test_urgency_low_newsletter(self):
        prefs = UserEmailPreferences(user_id="u1")
        analyzer = self._make_analyzer(prefs=prefs)
        result = await analyzer.score_email(
            email_id="e10",
            subject="Weekly Newsletter",
            from_address="news@promo.com",
            snippet="Click to unsubscribe",
        )
        assert result.factors["urgency"] == 0.2

    @pytest.mark.asyncio
    async def test_content_important_keywords(self):
        prefs = UserEmailPreferences(user_id="u1", important_keywords=["budget", "deadline"])
        analyzer = self._make_analyzer(prefs=prefs)
        result = await analyzer.score_email(
            email_id="e11",
            subject="Budget update",
            from_address="x@y.com",
            snippet="The deadline is approaching",
        )
        # Two keyword matches: 0.5 + 2*0.2 = 0.9
        assert result.factors["content"] == 0.9

    @pytest.mark.asyncio
    async def test_content_low_priority_keywords(self):
        prefs = UserEmailPreferences(user_id="u1", low_priority_keywords=["unsubscribe", "promo"])
        analyzer = self._make_analyzer(prefs=prefs)
        result = await analyzer.score_email(
            email_id="e12",
            subject="Special promo",
            from_address="x@y.com",
            snippet="Click to unsubscribe",
        )
        # Two low-priority matches reduce score
        assert result.factors["content"] < 0.5

    @pytest.mark.asyncio
    async def test_thread_scoring(self):
        prefs = UserEmailPreferences(user_id="u1")
        analyzer = self._make_analyzer(prefs=prefs)
        result = await analyzer.score_email(
            email_id="e13",
            subject="Re: Discussion",
            from_address="x@y.com",
            snippet="thread reply",
            thread_count=10,
            is_read=False,
        )
        # Unread (+0.2) + active thread (>5: +0.3) = 0.5+0.2+0.3 = 1.0
        assert result.factors["thread"] == 1.0

    @pytest.mark.asyncio
    async def test_score_clamped(self):
        prefs = UserEmailPreferences(user_id="u1")
        analyzer = self._make_analyzer(prefs=prefs)
        result = await analyzer.score_email(
            email_id="e14",
            subject="Test",
            from_address="x@y.com",
            snippet="test",
        )
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_score_batch(self):
        prefs = UserEmailPreferences(user_id="u1")
        analyzer = self._make_analyzer(prefs=prefs)
        emails = [
            {"id": "b1", "subject": "A", "from_address": "a@b.com", "snippet": "x"},
            {"id": "b2", "subject": "B", "from": "c@d.com", "snippet": "y"},
        ]
        results = await analyzer.score_batch(emails)
        assert len(results) == 2
        assert results[0].email_id == "b1"
        assert results[1].email_id == "b2"

    @pytest.mark.asyncio
    async def test_generate_reason_high_sender(self):
        prefs = UserEmailPreferences(user_id="u1", important_senders=["boss@work.com"])
        analyzer = self._make_analyzer(prefs=prefs)
        result = await analyzer.score_email(
            email_id="e15",
            subject="Update",
            from_address="boss@work.com",
            snippet="status update",
        )
        assert "important sender" in result.reason.lower() or len(result.reason) > 0

    def test_extract_sender(self):
        analyzer = self._make_analyzer()
        assert analyzer._extract_sender("replied to bob@test.com yesterday") == "bob@test.com"
        assert analyzer._extract_sender("no email here") is None

    def test_extract_keywords(self):
        analyzer = self._make_analyzer()
        keywords = analyzer._extract_keywords("The important budget deadline topic")
        # Should exclude stopwords and short words, return up to 5
        assert len(keywords) <= 5
        assert all(len(kw) > 3 for kw in keywords)

    @pytest.mark.asyncio
    async def test_get_user_preferences_from_memory(self):
        memory = AsyncMock()
        memory.query = AsyncMock(
            return_value=[
                {"content": "User replied to boss@work.com about project"},
                {"content": "important topic: budget review planning"},
            ]
        )
        analyzer = EmailPriorityAnalyzer(user_id="u1", memory=memory)
        prefs = await analyzer.get_user_preferences()
        assert prefs.user_id == "u1"
        assert "boss@work.com" in prefs.important_senders

    @pytest.mark.asyncio
    async def test_sender_corporate_domain_score(self):
        """Corporate .io/.ai domains (not gmail/yahoo) should score 0.6."""
        prefs = UserEmailPreferences(user_id="u1")
        analyzer = self._make_analyzer(prefs=prefs)
        result = await analyzer.score_email(
            email_id="e16",
            subject="Hello",
            from_address="alice@startup.io",
            snippet="hi",
        )
        assert result.factors["sender"] == 0.6


# ==========================================================================
# EmailFeedbackLearner
# ==========================================================================


class TestEmailFeedbackLearner:
    @pytest.mark.asyncio
    async def test_record_replied(self):
        memory = AsyncMock()
        memory.store = AsyncMock(return_value=None)
        learner = EmailFeedbackLearner(user_id="u1", memory=memory)
        ok = await learner.record_interaction(
            email_id="e1",
            action="replied",
            from_address="boss@work.com",
            subject="Meeting",
        )
        assert ok is True
        memory.store.assert_awaited_once()
        call_kwargs = memory.store.call_args[1]
        assert "replied" in call_kwargs["content"].lower()
        assert call_kwargs["importance"] == 0.9

    @pytest.mark.asyncio
    async def test_record_starred(self):
        memory = AsyncMock()
        memory.store = AsyncMock()
        learner = EmailFeedbackLearner(user_id="u1", memory=memory)
        ok = await learner.record_interaction("e2", "starred", "a@b.com", "Test")
        assert ok is True
        assert memory.store.call_args[1]["importance"] == 0.8

    @pytest.mark.asyncio
    async def test_record_deleted(self):
        memory = AsyncMock()
        memory.store = AsyncMock()
        learner = EmailFeedbackLearner(user_id="u1", memory=memory)
        ok = await learner.record_interaction("e3", "deleted", "a@b.com", "Spam")
        assert ok is True
        assert memory.store.call_args[1]["importance"] == 0.2

    @pytest.mark.asyncio
    async def test_record_unknown_action(self):
        memory = AsyncMock()
        learner = EmailFeedbackLearner(user_id="u1", memory=memory)
        ok = await learner.record_interaction("e4", "unknown_action", "a@b.com", "X")
        assert ok is False

    @pytest.mark.asyncio
    async def test_record_no_memory(self):
        # No memory, and import will fail in _get_memory
        learner = EmailFeedbackLearner(user_id="u1", memory=None)
        with patch(
            "aragora.analysis.email_priority.EmailFeedbackLearner._get_memory",
            new_callable=AsyncMock,
            return_value=None,
        ):
            ok = await learner.record_interaction("e5", "replied", "a@b.com", "X")
            assert ok is False

    @pytest.mark.asyncio
    async def test_consolidate_preferences(self):
        memory = AsyncMock()
        memory.consolidate = AsyncMock()
        learner = EmailFeedbackLearner(user_id="u1", memory=memory)
        ok = await learner.consolidate_preferences()
        assert ok is True
        memory.consolidate.assert_awaited_once_with("u1")

    @pytest.mark.asyncio
    async def test_consolidate_no_consolidate_method(self):
        memory = AsyncMock(spec=[])  # no attributes at all
        learner = EmailFeedbackLearner(user_id="u1", memory=memory)
        ok = await learner.consolidate_preferences()
        assert ok is False


# ==========================================================================
# nl_query dataclasses and enums
# ==========================================================================


class TestQueryMode:
    def test_enum_values(self):
        assert QueryMode.FACTUAL.value == "factual"
        assert QueryMode.ANALYTICAL.value == "analytical"
        assert QueryMode.COMPARATIVE.value == "comparative"
        assert QueryMode.SUMMARY.value == "summary"
        assert QueryMode.EXTRACTIVE.value == "extractive"


class TestAnswerConfidence:
    def test_enum_values(self):
        assert AnswerConfidence.HIGH.value == "high"
        assert AnswerConfidence.MEDIUM.value == "medium"
        assert AnswerConfidence.LOW.value == "low"
        assert AnswerConfidence.NONE.value == "none"


class TestCitation:
    def test_creation_and_to_dict(self):
        c = Citation(
            document_id="doc1",
            document_name="Contract A",
            chunk_id="ch1",
            snippet="exclusivity clause",
            page=3,
            relevance_score=0.92,
            heading_context="Section 5",
        )
        d = c.to_dict()
        assert d["document_id"] == "doc1"
        assert d["page"] == 3
        assert d["relevance_score"] == 0.92
        assert d["heading_context"] == "Section 5"

    def test_defaults(self):
        c = Citation(document_id="d", document_name="n", chunk_id="c", snippet="s")
        assert c.page is None
        assert c.relevance_score == 0.0
        assert c.heading_context == ""


class TestQueryResult:
    def _make_result(self, **overrides):
        defaults = dict(
            query_id="q1",
            question="What?",
            answer="Answer here",
            confidence=AnswerConfidence.HIGH,
            citations=[],
            query_mode=QueryMode.FACTUAL,
            chunks_searched=10,
            chunks_relevant=3,
            processing_time_ms=150,
            model_used="claude-3.5-sonnet",
        )
        defaults.update(overrides)
        return QueryResult(**defaults)

    def test_to_dict(self):
        citation = Citation(document_id="d1", document_name="Doc", chunk_id="c1", snippet="text")
        result = self._make_result(citations=[citation])
        d = result.to_dict()
        assert d["query_id"] == "q1"
        assert d["confidence"] == "high"
        assert d["query_mode"] == "factual"
        assert len(d["citations"]) == 1

    def test_has_answer_true(self):
        result = self._make_result()
        assert result.has_answer is True

    def test_has_answer_false_no_confidence(self):
        result = self._make_result(confidence=AnswerConfidence.NONE)
        assert result.has_answer is False

    def test_has_answer_false_empty_answer(self):
        result = self._make_result(answer="", confidence=AnswerConfidence.HIGH)
        assert result.has_answer is False


class TestStreamingChunk:
    def test_defaults(self):
        chunk = StreamingChunk(text="hello")
        assert chunk.is_final is False
        assert chunk.citations == []


# ==========================================================================
# DocumentQueryEngine
# ==========================================================================


class TestDocumentQueryEngine:
    def test_init_defaults(self):
        engine = DocumentQueryEngine()
        assert isinstance(engine.config, QueryConfig)
        assert engine._searcher is None

    def test_init_custom_config(self):
        config = QueryConfig(max_chunks=5, min_relevance=0.5)
        engine = DocumentQueryEngine(config=config)
        assert engine.config.max_chunks == 5
        assert engine.config.min_relevance == 0.5

    def test_detect_query_mode_factual(self):
        engine = DocumentQueryEngine()
        mode = engine._detect_query_mode("What is the contract value?")
        assert mode == QueryMode.FACTUAL

    def test_detect_query_mode_summary(self):
        engine = DocumentQueryEngine()
        mode = engine._detect_query_mode("Summarize the key points")
        assert mode == QueryMode.SUMMARY

    def test_detect_query_mode_comparative(self):
        engine = DocumentQueryEngine()
        mode = engine._detect_query_mode("Compare these two documents")
        assert mode == QueryMode.COMPARATIVE

    def test_detect_query_mode_analytical(self):
        engine = DocumentQueryEngine()
        mode = engine._detect_query_mode("Why did the revenue decline?")
        assert mode == QueryMode.ANALYTICAL

    def test_detect_query_mode_extractive(self):
        engine = DocumentQueryEngine()
        mode = engine._detect_query_mode("List all the parties mentioned")
        assert mode == QueryMode.EXTRACTIVE

    def test_expand_query(self):
        engine = DocumentQueryEngine()
        queries = engine._expand_query("What is the deadline?")
        assert len(queries) >= 1
        assert queries[0] == "What is the deadline?"
        # The expanded query should strip question words
        if len(queries) > 1:
            assert "what" not in queries[1]

    def test_expand_query_no_expansion_needed(self):
        engine = DocumentQueryEngine()
        queries = engine._expand_query("budget allocation")
        # After removing question words, nothing changes (lowercase match)
        assert queries[0] == "budget allocation"

    def test_conversation_history(self):
        engine = DocumentQueryEngine()
        engine._add_to_history("conv1", "Q1", "A1")
        engine._add_to_history("conv1", "Q2", "A2")
        ctx = engine._get_conversation_context("conv1")
        assert len(ctx) == 4  # 2 QA pairs
        assert ctx[0]["role"] == "user"
        assert ctx[1]["role"] == "assistant"

    def test_conversation_history_trimming(self):
        config = QueryConfig(max_context_turns=1)
        engine = DocumentQueryEngine(config=config)
        engine._add_to_history("conv2", "Q1", "A1")
        engine._add_to_history("conv2", "Q2", "A2")
        engine._add_to_history("conv2", "Q3", "A3")
        ctx = engine._get_conversation_context("conv2")
        # max_context_turns=1 -> keep last 2 entries (1 QA pair)
        assert len(ctx) == 2

    def test_clear_conversation(self):
        engine = DocumentQueryEngine()
        engine._add_to_history("conv3", "Q1", "A1")
        engine.clear_conversation("conv3")
        assert engine._get_conversation_context("conv3") == []

    def test_clear_nonexistent_conversation(self):
        engine = DocumentQueryEngine()
        engine.clear_conversation("nonexistent")  # should not raise

    def test_assess_confidence_no_results(self):
        engine = DocumentQueryEngine()
        assert engine._assess_confidence("answer", []) == AnswerConfidence.NONE

    def test_assess_confidence_uncertainty(self):
        engine = DocumentQueryEngine()
        mock_result = MagicMock(combined_score=0.9)
        conf = engine._assess_confidence("I couldn't find any info", [mock_result])
        assert conf == AnswerConfidence.LOW

    def test_assess_confidence_high(self):
        engine = DocumentQueryEngine()
        results = [MagicMock(combined_score=0.9), MagicMock(combined_score=0.8)]
        conf = engine._assess_confidence("The answer is X", results)
        assert conf == AnswerConfidence.HIGH

    def test_assess_confidence_medium(self):
        engine = DocumentQueryEngine()
        results = [MagicMock(combined_score=0.6), MagicMock(combined_score=0.3)]
        conf = engine._assess_confidence("The answer is X", results)
        assert conf == AnswerConfidence.MEDIUM

    def test_assess_confidence_low_scores(self):
        engine = DocumentQueryEngine()
        results = [MagicMock(combined_score=0.2), MagicMock(combined_score=0.1)]
        conf = engine._assess_confidence("The answer is X", results)
        assert conf == AnswerConfidence.LOW

    @pytest.mark.asyncio
    async def test_query_no_searcher(self):
        engine = DocumentQueryEngine(searcher=None)
        result = await engine.query("What is this?")
        assert result.confidence == AnswerConfidence.NONE
        assert "couldn't find" in result.answer.lower()
        assert result.chunks_searched == 0

    @pytest.mark.asyncio
    async def test_query_with_mock_searcher(self):
        mock_result = MagicMock()
        mock_result.chunk_id = "ch1"
        mock_result.combined_score = 0.9
        mock_result.document_id = "doc1"
        mock_result.content = "The deadline is March 15th."
        mock_result.start_page = 2
        mock_result.heading_context = "Dates"

        searcher = AsyncMock()
        searcher.search = AsyncMock(return_value=[mock_result])

        engine = DocumentQueryEngine(searcher=searcher)

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = ("March 15th", "claude-3.5-sonnet")
            result = await engine.query("What is the deadline?")

        assert result.answer == "March 15th"
        assert result.model_used == "claude-3.5-sonnet"
        assert len(result.citations) == 1
        assert result.citations[0].document_id == "doc1"

    @pytest.mark.asyncio
    async def test_compare_documents_requires_two(self):
        engine = DocumentQueryEngine()
        with pytest.raises(ValueError, match="at least 2"):
            await engine.compare_documents(document_ids=["doc1"])

    def test_build_answer_prompt_contains_question(self):
        engine = DocumentQueryEngine()
        prompt = engine._build_answer_prompt("Where is X?", "ctx", QueryMode.FACTUAL)
        assert "Where is X?" in prompt
        assert "ctx" in prompt


# ==========================================================================
# codebase.models
# ==========================================================================


class TestVulnerabilitySeverity:
    def test_from_cvss_critical(self):
        assert VulnerabilitySeverity.from_cvss(9.5) == VulnerabilitySeverity.CRITICAL

    def test_from_cvss_high(self):
        assert VulnerabilitySeverity.from_cvss(7.5) == VulnerabilitySeverity.HIGH

    def test_from_cvss_medium(self):
        assert VulnerabilitySeverity.from_cvss(5.0) == VulnerabilitySeverity.MEDIUM

    def test_from_cvss_low(self):
        assert VulnerabilitySeverity.from_cvss(2.0) == VulnerabilitySeverity.LOW

    def test_from_cvss_unknown(self):
        assert VulnerabilitySeverity.from_cvss(0.0) == VulnerabilitySeverity.UNKNOWN

    def test_from_cvss_boundary_9(self):
        assert VulnerabilitySeverity.from_cvss(9.0) == VulnerabilitySeverity.CRITICAL

    def test_from_cvss_boundary_7(self):
        assert VulnerabilitySeverity.from_cvss(7.0) == VulnerabilitySeverity.HIGH

    def test_from_cvss_boundary_4(self):
        assert VulnerabilitySeverity.from_cvss(4.0) == VulnerabilitySeverity.MEDIUM


class TestVulnerabilityFinding:
    def test_creation_minimal(self):
        v = VulnerabilityFinding(
            id="CVE-2024-001",
            title="XSS",
            description="Cross-site scripting",
            severity=VulnerabilitySeverity.HIGH,
        )
        assert v.id == "CVE-2024-001"
        assert v.fix_available is False
        assert v.cvss_score is None

    def test_to_dict_with_references(self):
        ref = VulnerabilityReference(url="https://nvd.nist.gov/1", source="NVD", tags=["patch"])
        v = VulnerabilityFinding(
            id="CVE-2024-002",
            title="SQL Injection",
            description="desc",
            severity=VulnerabilitySeverity.CRITICAL,
            cvss_score=9.8,
            references=[ref],
            cwe_ids=["CWE-89"],
            fix_available=True,
            recommended_version="2.0.1",
        )
        d = v.to_dict()
        assert d["severity"] == "critical"
        assert d["cvss_score"] == 9.8
        assert len(d["references"]) == 1
        assert d["references"][0]["tags"] == ["patch"]
        assert d["fix_available"] is True

    def test_to_dict_datetime_serialization(self):
        dt = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
        v = VulnerabilityFinding(
            id="CVE-2024-003",
            title="T",
            description="D",
            severity=VulnerabilitySeverity.LOW,
            published_at=dt,
        )
        d = v.to_dict()
        assert "2024-01-15" in d["published_at"]
        assert d["updated_at"] is None


class TestDependencyInfo:
    def test_has_vulnerabilities_false(self):
        dep = DependencyInfo(name="requests", version="2.31.0", ecosystem="pypi")
        assert dep.has_vulnerabilities is False
        assert dep.highest_severity is None

    def test_has_vulnerabilities_true(self):
        vuln = VulnerabilityFinding(
            id="CVE-1", title="t", description="d", severity=VulnerabilitySeverity.HIGH
        )
        dep = DependencyInfo(
            name="lodash", version="4.17.0", ecosystem="npm", vulnerabilities=[vuln]
        )
        assert dep.has_vulnerabilities is True
        assert dep.highest_severity == VulnerabilitySeverity.HIGH

    def test_highest_severity_critical(self):
        vulns = [
            VulnerabilityFinding(
                id="1", title="", description="", severity=VulnerabilitySeverity.LOW
            ),
            VulnerabilityFinding(
                id="2", title="", description="", severity=VulnerabilitySeverity.CRITICAL
            ),
        ]
        dep = DependencyInfo(name="x", version="1", ecosystem="npm", vulnerabilities=vulns)
        assert dep.highest_severity == VulnerabilitySeverity.CRITICAL

    def test_to_dict(self):
        dep = DependencyInfo(
            name="flask",
            version="2.3.0",
            ecosystem="pypi",
            direct=True,
            dev_dependency=False,
            license="BSD",
        )
        d = dep.to_dict()
        assert d["name"] == "flask"
        assert d["has_vulnerabilities"] is False
        assert d["highest_severity"] is None


class TestScanResult:
    def test_calculate_summary(self):
        vulns = [
            VulnerabilityFinding(
                id="1", title="", description="", severity=VulnerabilitySeverity.CRITICAL
            ),
            VulnerabilityFinding(
                id="2", title="", description="", severity=VulnerabilitySeverity.HIGH
            ),
            VulnerabilityFinding(
                id="3", title="", description="", severity=VulnerabilitySeverity.HIGH
            ),
            VulnerabilityFinding(
                id="4", title="", description="", severity=VulnerabilitySeverity.MEDIUM
            ),
        ]
        dep_with_vuln = DependencyInfo(
            name="x",
            version="1",
            ecosystem="npm",
            vulnerabilities=[vulns[0]],
        )
        dep_clean = DependencyInfo(name="y", version="2", ecosystem="npm")

        scan = ScanResult(
            scan_id="s1",
            repository="repo",
            dependencies=[dep_with_vuln, dep_clean],
            vulnerabilities=vulns,
        )
        scan.calculate_summary()

        assert scan.total_dependencies == 2
        assert scan.vulnerable_dependencies == 1
        assert scan.critical_count == 1
        assert scan.high_count == 2
        assert scan.medium_count == 1
        assert scan.low_count == 0

    def test_to_dict(self):
        scan = ScanResult(scan_id="s2", repository="repo", status="completed")
        d = scan.to_dict()
        assert d["scan_id"] == "s2"
        assert d["status"] == "completed"
        assert "summary" in d

    def test_empty_scan(self):
        scan = ScanResult(scan_id="s3", repository="repo")
        scan.calculate_summary()
        assert scan.total_dependencies == 0
        assert scan.critical_count == 0


class TestCodeMetric:
    def test_status_ok(self):
        m = CodeMetric(
            type=MetricType.COMPLEXITY,
            value=5.0,
            warning_threshold=10.0,
            error_threshold=20.0,
        )
        assert m.status == "ok"

    def test_status_warning(self):
        m = CodeMetric(
            type=MetricType.COMPLEXITY,
            value=12.0,
            warning_threshold=10.0,
            error_threshold=20.0,
        )
        assert m.status == "warning"

    def test_status_error(self):
        m = CodeMetric(
            type=MetricType.COMPLEXITY,
            value=25.0,
            warning_threshold=10.0,
            error_threshold=20.0,
        )
        assert m.status == "error"

    def test_status_no_thresholds(self):
        m = CodeMetric(type=MetricType.LINES_OF_CODE, value=1000.0)
        assert m.status == "ok"

    def test_to_dict(self):
        m = CodeMetric(
            type=MetricType.MAINTAINABILITY,
            value=75.0,
            unit="index",
            file_path="main.py",
        )
        d = m.to_dict()
        assert d["type"] == "maintainability"
        assert d["value"] == 75.0
        assert d["file_path"] == "main.py"
        assert d["status"] == "ok"


class TestHotspotFinding:
    def test_risk_score_high_complexity(self):
        h = HotspotFinding(
            file_path="f.py",
            complexity=20.0,
            change_frequency=10,
        )
        # complexity_factor = min(20/10, 1.0) = 1.0
        # change_factor = min(10/50, 1.0) = 0.2
        # risk = (1.0*0.7 + 0.2*0.3) * 100 = 76.0
        assert h.risk_score == 76.0

    def test_risk_score_zero_complexity(self):
        h = HotspotFinding(file_path="f.py", complexity=0.0, change_frequency=0)
        assert h.risk_score == 0.0

    def test_risk_score_high_change_frequency(self):
        h = HotspotFinding(file_path="f.py", complexity=5.0, change_frequency=100)
        # complexity_factor = 0.5, change_factor = 1.0
        # risk = (0.5*0.7 + 1.0*0.3)*100 = 65.0
        assert h.risk_score == pytest.approx(65.0)

    def test_to_dict(self):
        h = HotspotFinding(
            file_path="f.py",
            function_name="do_stuff",
            complexity=15.0,
            lines_of_code=50,
            change_frequency=25,
        )
        d = h.to_dict()
        assert d["file_path"] == "f.py"
        assert d["function_name"] == "do_stuff"
        assert "risk_score" in d
        assert d["risk_score"] > 0


class TestSecretFinding:
    def test_redact_short(self):
        assert SecretFinding.redact_secret("abc") == "***"
        assert SecretFinding.redact_secret("12345678") == "********"

    def test_redact_long(self):
        result = SecretFinding.redact_secret("sk_live_abcdef1234567890")
        assert result.startswith("sk_l")
        assert result.endswith("7890")
        assert "*" in result

    def test_to_dict(self):
        s = SecretFinding(
            id="sec1",
            secret_type=SecretType.AWS_ACCESS_KEY,
            file_path=".env",
            line_number=5,
            column_start=0,
            column_end=20,
            matched_text="AKIA****1234",
            context_line="AWS_KEY=AKIA****1234",
            severity=VulnerabilitySeverity.CRITICAL,
            confidence=0.95,
        )
        d = s.to_dict()
        assert d["secret_type"] == "aws_access_key"
        assert d["severity"] == "critical"
        assert d["confidence"] == 0.95


class TestSecretsScanResult:
    def test_severity_counts(self):
        secrets = [
            SecretFinding(
                id="1",
                secret_type=SecretType.AWS_ACCESS_KEY,
                file_path="f",
                line_number=1,
                column_start=0,
                column_end=10,
                matched_text="x",
                context_line="x",
                severity=VulnerabilitySeverity.CRITICAL,
                confidence=0.9,
            ),
            SecretFinding(
                id="2",
                secret_type=SecretType.GENERIC_API_KEY,
                file_path="f",
                line_number=2,
                column_start=0,
                column_end=10,
                matched_text="y",
                context_line="y",
                severity=VulnerabilitySeverity.HIGH,
                confidence=0.8,
            ),
            SecretFinding(
                id="3",
                secret_type=SecretType.GENERIC_SECRET,
                file_path="f",
                line_number=3,
                column_start=0,
                column_end=10,
                matched_text="z",
                context_line="z",
                severity=VulnerabilitySeverity.HIGH,
                confidence=0.7,
            ),
        ]
        result = SecretsScanResult(
            scan_id="ss1",
            repository="repo",
            secrets=secrets,
        )
        assert result.critical_count == 1
        assert result.high_count == 2
        assert result.medium_count == 0
        assert result.low_count == 0

    def test_to_dict_summary(self):
        result = SecretsScanResult(scan_id="ss2", repository="repo")
        d = result.to_dict()
        assert d["summary"]["total_secrets"] == 0
        assert d["summary"]["critical_count"] == 0

    def test_empty_scan(self):
        result = SecretsScanResult(scan_id="ss3", repository="repo")
        assert result.critical_count == 0
        assert result.high_count == 0


# ==========================================================================
# codebase.metrics - ComplexityVisitor
# ==========================================================================


class TestComplexityVisitor:
    def _visit_code(self, code: str) -> ComplexityVisitor:
        tree = ast.parse(textwrap.dedent(code))
        # Find the function node
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visitor = ComplexityVisitor()
                visitor.visit(node)
                return visitor
        raise ValueError("No function found in code")

    def test_simple_function(self):
        visitor = self._visit_code("""
def simple():
    return 1
""")
        assert visitor.cyclomatic == 1
        assert visitor.cognitive == 0
        assert visitor.return_count == 1

    def test_if_else(self):
        visitor = self._visit_code("""
def check(x):
    if x > 0:
        return 1
    else:
        return 0
""")
        assert visitor.cyclomatic == 2  # 1 base + 1 if
        assert visitor.return_count == 2

    def test_nested_if(self):
        visitor = self._visit_code("""
def nested(x, y):
    if x > 0:
        if y > 0:
            return 1
    return 0
""")
        # cyclomatic: 1 + 1 (outer if) + 1 (inner if) = 3
        assert visitor.cyclomatic == 3
        # cognitive: (1+0) for outer if + (1+1) for inner if = 3
        assert visitor.cognitive == 3
        assert visitor.max_nesting == 2

    def test_for_loop(self):
        visitor = self._visit_code("""
def loop(items):
    for item in items:
        print(item)
""")
        assert visitor.cyclomatic == 2  # 1 + for

    def test_while_loop(self):
        visitor = self._visit_code("""
def loop():
    while True:
        break
""")
        assert visitor.cyclomatic == 2

    def test_try_except(self):
        visitor = self._visit_code("""
def risky():
    try:
        do_thing()
    except ValueError:
        pass
    except TypeError:
        pass
""")
        assert visitor.cyclomatic == 3  # 1 + 2 except handlers

    def test_bool_op(self):
        visitor = self._visit_code("""
def check(a, b, c):
    if a and b or c:
        pass
""")
        # 1 base + 1 if + 2 bool ops (and/or each have 2 values, so 1 extra each)
        assert visitor.cyclomatic == 4

    def test_list_comprehension(self):
        visitor = self._visit_code("""
def transform(items):
    return [x for x in items]
""")
        # 1 base + 1 comprehension
        assert visitor.cyclomatic == 2
        assert visitor.cognitive == 1


# ==========================================================================
# codebase.metrics - PythonAnalyzer
# ==========================================================================


class TestPythonAnalyzer:
    def test_analyze_simple_file(self):
        code = textwrap.dedent("""\
            # A comment
            import os

            def hello():
                return "hello"

            class Foo:
                pass
        """)
        analyzer = PythonAnalyzer()
        metrics = analyzer.analyze_file("test.py", content=code)
        assert metrics.file_path == "test.py"
        assert metrics.language == "python"
        assert metrics.lines_of_comments >= 1
        assert metrics.imports >= 1
        assert metrics.classes == 1
        assert len(metrics.functions) == 1
        assert metrics.functions[0].name == "hello"

    def test_analyze_empty_file(self):
        analyzer = PythonAnalyzer()
        metrics = analyzer.analyze_file("empty.py", content="")
        assert metrics.lines_of_code == 0
        assert metrics.functions == []
        assert metrics.classes == 0

    def test_analyze_syntax_error(self):
        analyzer = PythonAnalyzer()
        metrics = analyzer.analyze_file("bad.py", content="def foo(:\n  pass")
        # Should not raise, returns partial metrics
        assert metrics.file_path == "bad.py"

    def test_function_complexity(self):
        code = textwrap.dedent("""\
            def complex_func(x, y, z=None):
                if x > 0:
                    for i in range(y):
                        if z:
                            return True
                return False
        """)
        analyzer = PythonAnalyzer()
        metrics = analyzer.analyze_file("complex.py", content=code)
        assert len(metrics.functions) == 1
        func = metrics.functions[0]
        assert func.name == "complex_func"
        assert func.cyclomatic_complexity >= 4
        assert func.parameter_count == 3
        assert func.return_count == 2

    def test_maintainability_index_calculated(self):
        code = textwrap.dedent("""\
            def a():
                if True:
                    pass

            def b():
                for i in range(10):
                    if i > 5:
                        pass
        """)
        analyzer = PythonAnalyzer()
        metrics = analyzer.analyze_file("mi.py", content=code)
        assert metrics.maintainability_index > 0
        assert metrics.maintainability_index <= 100

    def test_blank_lines_counted(self):
        code = "x = 1\n\n\ny = 2\n"
        analyzer = PythonAnalyzer()
        metrics = analyzer.analyze_file("blanks.py", content=code)
        # Trailing newline produces an extra empty line when split
        assert metrics.blank_lines >= 2


# ==========================================================================
# codebase.metrics - TypeScriptAnalyzer
# ==========================================================================


class TestTypeScriptAnalyzer:
    def test_analyze_simple_ts(self):
        code = textwrap.dedent("""\
            // A comment
            import { foo } from './bar';

            function hello() {
                return "hello";
            }

            class Foo {}
        """)
        analyzer = TypeScriptAnalyzer()
        metrics = analyzer.analyze_file("test.ts", content=code)
        assert metrics.language == "typescript"
        assert metrics.imports >= 1
        assert metrics.classes == 1
        assert metrics.lines_of_comments >= 1

    def test_analyze_js_extension(self):
        analyzer = TypeScriptAnalyzer()
        metrics = analyzer.analyze_file("test.js", content="const x = 1;")
        assert metrics.language == "javascript"

    def test_multiline_comments(self):
        code = textwrap.dedent("""\
            /*
             * Multi-line
             * comment
             */
            const x = 1;
        """)
        analyzer = TypeScriptAnalyzer()
        metrics = analyzer.analyze_file("test.ts", content=code)
        assert metrics.lines_of_comments >= 3


# ==========================================================================
# codebase.metrics - DuplicateDetector
# ==========================================================================


class TestDuplicateDetector:
    def test_no_duplicates(self):
        files = [
            ("a.py", "line1\nline2\nline3\nline4\nline5\nline6"),
            ("b.py", "other1\nother2\nother3\nother4\nother5\nother6"),
        ]
        detector = DuplicateDetector(min_lines=6)
        dupes = detector.detect_duplicates(files)
        assert len(dupes) == 0

    def test_detects_duplicates(self):
        shared_block = "\n".join([f"shared_line_{i} = {i}" for i in range(8)])
        files = [
            ("a.py", shared_block),
            ("b.py", shared_block),
        ]
        detector = DuplicateDetector(min_lines=6, min_tokens=10)
        dupes = detector.detect_duplicates(files)
        assert len(dupes) > 0

    def test_short_blocks_ignored(self):
        files = [
            ("a.py", "x\ny"),
            ("b.py", "x\ny"),
        ]
        detector = DuplicateDetector(min_lines=6)
        dupes = detector.detect_duplicates(files)
        assert len(dupes) == 0


# ==========================================================================
# codebase.metrics - MetricsReport
# ==========================================================================


class TestMetricsReport:
    def test_to_dict(self):
        report = MetricsReport(repository="testrepo", scan_id="s1")
        d = report.to_dict()
        assert d["repository"] == "testrepo"
        assert d["summary"]["total_files"] == 0
        assert d["summary"]["avg_complexity"] == 0.0
        assert d["hotspots"] == []
        assert d["duplicates"] == []
        assert d["metrics"] == []

    def test_to_dict_with_data(self):
        hotspot = HotspotFinding(file_path="f.py", complexity=15.0)
        dup = DuplicateBlock(
            hash="abc123def456",
            lines=6,
            occurrences=[("a.py", 1, 6), ("b.py", 10, 15)],
        )
        metric = CodeMetric(type=MetricType.COMPLEXITY, value=5.0)
        report = MetricsReport(
            repository="repo",
            scan_id="s2",
            total_files=2,
            hotspots=[hotspot],
            duplicates=[dup],
            metrics=[metric],
        )
        d = report.to_dict()
        assert len(d["hotspots"]) == 1
        assert len(d["duplicates"]) == 1
        assert d["duplicates"][0]["hash"] == "abc123de"  # First 8 chars
        assert len(d["metrics"]) == 1


# ==========================================================================
# codebase.metrics - CodeMetricsAnalyzer
# ==========================================================================


class TestCodeMetricsAnalyzer:
    def test_init_defaults(self):
        analyzer = CodeMetricsAnalyzer()
        assert analyzer.complexity_warning == 10
        assert analyzer.complexity_error == 20

    def test_init_custom(self):
        analyzer = CodeMetricsAnalyzer(
            complexity_warning=5,
            complexity_error=15,
            duplication_threshold=10,
        )
        assert analyzer.complexity_warning == 5
        assert analyzer.complexity_error == 15

    def test_language_extensions(self):
        assert CodeMetricsAnalyzer.LANGUAGE_EXTENSIONS[".py"] == "python"
        assert CodeMetricsAnalyzer.LANGUAGE_EXTENSIONS[".ts"] == "typescript"
        assert CodeMetricsAnalyzer.LANGUAGE_EXTENSIONS[".js"] == "javascript"

    def test_exclude_dirs(self):
        assert "node_modules" in CodeMetricsAnalyzer.EXCLUDE_DIRS
        assert "__pycache__" in CodeMetricsAnalyzer.EXCLUDE_DIRS

    def test_analyze_repository_on_temp_dir(self, tmp_path):
        # Create a small Python file
        py_file = tmp_path / "example.py"
        py_file.write_text(
            textwrap.dedent("""\
            # Example module
            import os

            def simple():
                return 1

            def moderate(x):
                if x > 0:
                    return x
                return 0

            class MyClass:
                pass
        """)
        )

        analyzer = CodeMetricsAnalyzer()
        report = analyzer.analyze_repository(str(tmp_path), scan_id="test_scan")

        assert report.repository == tmp_path.name
        assert report.scan_id == "test_scan"
        assert report.total_files == 1
        assert report.total_functions == 2
        assert report.total_classes == 1
        assert report.total_code_lines > 0
        assert report.avg_complexity >= 1.0

    def test_analyze_repository_empty(self, tmp_path):
        analyzer = CodeMetricsAnalyzer()
        report = analyzer.analyze_repository(str(tmp_path))
        assert report.total_files == 0
        assert report.total_functions == 0
        assert report.avg_complexity == 0.0

    def test_hotspot_detection(self, tmp_path):
        # Create file with a complex function
        py_file = tmp_path / "complex.py"
        code_lines = ["def mega(a, b, c, d, e):"]
        for i in range(12):
            code_lines.append(f"    if a > {i}:")
            code_lines.append(f"        b += {i}")
        code_lines.append("    return b")
        py_file.write_text("\n".join(code_lines))

        analyzer = CodeMetricsAnalyzer(complexity_warning=5)
        report = analyzer.analyze_repository(str(tmp_path))

        assert len(report.hotspots) > 0
        assert report.hotspots[0].function_name == "mega"

    def test_generate_metrics_includes_complexity(self, tmp_path):
        py_file = tmp_path / "m.py"
        py_file.write_text("def f():\n    if True:\n        pass\n")

        analyzer = CodeMetricsAnalyzer()
        report = analyzer.analyze_repository(str(tmp_path))

        metric_types = [m.type for m in report.metrics]
        assert MetricType.COMPLEXITY in metric_types
        assert MetricType.MAINTAINABILITY in metric_types
        assert MetricType.LINES_OF_CODE in metric_types

    def test_basic_analyze_other_language(self):
        code = "// comment\nint main() {\n    return 0;\n}\n\n"
        analyzer = CodeMetricsAnalyzer()
        metrics = analyzer._basic_analyze("main.c", code, "c")
        assert metrics.language == "c"
        assert metrics.lines_of_comments >= 1
        assert metrics.lines_of_code >= 2
        assert metrics.blank_lines >= 1


# ==========================================================================
# FunctionMetrics / FileMetrics dataclasses
# ==========================================================================


class TestFunctionMetrics:
    def test_creation(self):
        fm = FunctionMetrics(
            name="foo",
            file_path="test.py",
            start_line=1,
            end_line=5,
            cyclomatic_complexity=3,
            cognitive_complexity=2,
            parameter_count=2,
        )
        assert fm.name == "foo"
        assert fm.cyclomatic_complexity == 3
        assert fm.class_name is None


class TestFileMetrics:
    def test_defaults(self):
        fm = FileMetrics(file_path="test.py", language="python")
        assert fm.lines_of_code == 0
        assert fm.functions == []
        assert fm.avg_complexity == 0.0
        assert fm.maintainability_index == 100.0
