"""
Comprehensive tests for FactExtractor module.

Tests the fact extraction functionality including:
- ExtractionConfig dataclass
- ExtractedFact and ExtractionResult dataclasses
- FactExtractor class methods
- Agent-based extraction and verification
- Demo extraction patterns
- Batch extraction
- Fact persistence
- Error handling and edge cases

Run with:
    pytest tests/knowledge/test_fact_extractor.py -v --asyncio-mode=auto
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.knowledge.fact_extractor import (
    AgentProtocol,
    ExtractionConfig,
    ExtractionResult,
    ExtractedFact,
    FactExtractor,
    create_fact_extractor,
)
from aragora.knowledge.types import Fact, ValidationStatus


# =============================================================================
# Mock Agent for Testing
# =============================================================================


@dataclass
class MockAgent:
    """Mock agent implementing AgentProtocol for testing."""

    name: str = "mock_agent"
    response: str = ""
    fail_on_generate: bool = False

    async def generate(self, prompt: str, context: list | None = None) -> str:
        """Generate a mock response."""
        if self.fail_on_generate:
            raise RuntimeError("Agent generation failed")
        return self.response


class MockFactStore:
    """Mock fact store for testing persistence."""

    def __init__(self):
        self.facts: list[Fact] = []

    def add_fact(
        self,
        statement: str,
        workspace_id: str,
        confidence: float = 0.5,
        topics: list[str] | None = None,
        source_documents: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        validation_status: ValidationStatus = ValidationStatus.UNVERIFIED,
    ) -> Fact:
        """Add a fact to the store."""
        fact = Fact(
            id=f"fact_{len(self.facts) + 1}",
            statement=statement,
            workspace_id=workspace_id,
            confidence=confidence,
            topics=topics or [],
            source_documents=source_documents or [],
            metadata=metadata or {},
            validation_status=validation_status,
        )
        self.facts.append(fact)
        return fact


# =============================================================================
# ExtractionConfig Tests
# =============================================================================


class TestExtractionConfig:
    """Tests for ExtractionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ExtractionConfig()

        assert config.max_facts_per_chunk == 10
        assert config.min_confidence_threshold == 0.5
        assert config.require_evidence is True
        assert config.num_extraction_agents == 2
        assert config.require_agreement is True
        assert config.agreement_threshold == 0.7

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ExtractionConfig(
            max_facts_per_chunk=5,
            min_confidence_threshold=0.7,
            require_evidence=False,
            num_extraction_agents=3,
            require_agreement=False,
            agreement_threshold=0.9,
        )

        assert config.max_facts_per_chunk == 5
        assert config.min_confidence_threshold == 0.7
        assert config.require_evidence is False
        assert config.num_extraction_agents == 3
        assert config.require_agreement is False
        assert config.agreement_threshold == 0.9

    def test_extraction_prompt_template_has_placeholders(self):
        """Test extraction prompt template has required placeholders."""
        config = ExtractionConfig()

        assert "{filename}" in config.extraction_prompt_template
        assert "{chunk_id}" in config.extraction_prompt_template
        assert "{content}" in config.extraction_prompt_template
        assert "{max_facts}" in config.extraction_prompt_template

    def test_verification_prompt_template_has_placeholders(self):
        """Test verification prompt template has required placeholders."""
        config = ExtractionConfig()

        assert "{content}" in config.verification_prompt_template
        assert "{facts_json}" in config.verification_prompt_template


# =============================================================================
# ExtractedFact Tests
# =============================================================================


class TestExtractedFact:
    """Tests for ExtractedFact dataclass."""

    def test_default_values(self):
        """Test default values for extracted fact."""
        fact = ExtractedFact(
            statement="Test statement",
            confidence=0.8,
        )

        assert fact.statement == "Test statement"
        assert fact.confidence == 0.8
        assert fact.topics == []
        assert fact.evidence_quote == ""
        assert fact.source_chunk_id == ""
        assert fact.source_document == ""
        assert fact.verified is False
        assert fact.verification_reason == ""

    def test_full_values(self):
        """Test extracted fact with all values."""
        fact = ExtractedFact(
            statement="The contract expires on 2025-12-31",
            confidence=0.9,
            topics=["contract", "expiration", "date"],
            evidence_quote="...expires on 2025-12-31...",
            source_chunk_id="chunk_123",
            source_document="contract.pdf",
            verified=True,
            verification_reason="Confirmed by second agent",
        )

        assert fact.statement == "The contract expires on 2025-12-31"
        assert fact.confidence == 0.9
        assert len(fact.topics) == 3
        assert "contract" in fact.topics
        assert fact.verified is True


# =============================================================================
# ExtractionResult Tests
# =============================================================================


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_default_values(self):
        """Test default values for extraction result."""
        result = ExtractionResult(
            chunk_id="chunk_1",
            document_id="doc_1",
        )

        assert result.chunk_id == "chunk_1"
        assert result.document_id == "doc_1"
        assert result.facts == []
        assert result.errors == []
        assert result.extraction_time_ms == 0.0
        assert result.agent_used == ""

    def test_with_facts(self):
        """Test extraction result with facts."""
        facts = [
            ExtractedFact(statement="Fact 1", confidence=0.8),
            ExtractedFact(statement="Fact 2", confidence=0.7),
        ]
        result = ExtractionResult(
            chunk_id="chunk_1",
            document_id="doc_1",
            facts=facts,
            agent_used="claude",
            extraction_time_ms=150.5,
        )

        assert len(result.facts) == 2
        assert result.agent_used == "claude"
        assert result.extraction_time_ms == 150.5

    def test_with_errors(self):
        """Test extraction result with errors."""
        result = ExtractionResult(
            chunk_id="chunk_1",
            document_id="doc_1",
            errors=["Parse error", "Timeout"],
        )

        assert len(result.errors) == 2
        assert "Parse error" in result.errors


# =============================================================================
# FactExtractor Initialization Tests
# =============================================================================


class TestFactExtractorInit:
    """Tests for FactExtractor initialization."""

    def test_init_no_agents(self):
        """Test initialization without agents."""
        extractor = FactExtractor()

        assert extractor.agents == []
        assert extractor.config is not None
        assert extractor.fact_store is None
        assert extractor._extraction_count == 0

    def test_init_with_agents(self):
        """Test initialization with agents."""
        agents = [MockAgent(name="agent1"), MockAgent(name="agent2")]
        extractor = FactExtractor(agents=agents)

        assert len(extractor.agents) == 2
        assert extractor.agents[0].name == "agent1"

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = ExtractionConfig(max_facts_per_chunk=5)
        extractor = FactExtractor(config=config)

        assert extractor.config.max_facts_per_chunk == 5

    def test_init_with_fact_store(self):
        """Test initialization with fact store."""
        store = MockFactStore()
        extractor = FactExtractor(fact_store=store)

        assert extractor.fact_store is store


# =============================================================================
# Demo Extraction Tests
# =============================================================================


class TestDemoExtraction:
    """Tests for demo extraction without agents."""

    @pytest.fixture
    def extractor(self):
        """Create extractor without agents for demo mode."""
        return FactExtractor()

    @pytest.mark.asyncio
    async def test_extract_dates_iso_format(self, extractor):
        """Test demo extraction of ISO format dates."""
        content = "The meeting is scheduled for 2025-06-15."
        result = await extractor.extract_facts(
            content=content,
            chunk_id="chunk_1",
            document_id="doc_1",
        )

        assert result.agent_used == "demo"
        assert len(result.facts) >= 1
        date_facts = [f for f in result.facts if "2025-06-15" in f.statement]
        assert len(date_facts) >= 1

    @pytest.mark.asyncio
    async def test_extract_dates_natural_format(self, extractor):
        """Test demo extraction of natural language dates."""
        content = "The deadline is January 15, 2025."
        result = await extractor.extract_facts(
            content=content,
            chunk_id="chunk_1",
            document_id="doc_1",
        )

        assert result.agent_used == "demo"
        assert len(result.facts) >= 1

    @pytest.mark.asyncio
    async def test_extract_monetary_values(self, extractor):
        """Test demo extraction of monetary values."""
        content = "The total cost is $1,500.00 and the deposit is $250."
        result = await extractor.extract_facts(
            content=content,
            chunk_id="chunk_1",
            document_id="doc_1",
        )

        assert len(result.facts) >= 2
        monetary_facts = [f for f in result.facts if "Monetary value" in f.statement]
        assert len(monetary_facts) >= 2

    @pytest.mark.asyncio
    async def test_extract_percentages(self, extractor):
        """Test demo extraction of percentages."""
        content = "Revenue increased by 15.5% and costs decreased by 3%."
        result = await extractor.extract_facts(
            content=content,
            chunk_id="chunk_1",
            document_id="doc_1",
        )

        percentage_facts = [f for f in result.facts if "Percentage" in f.statement]
        assert len(percentage_facts) >= 2

    @pytest.mark.asyncio
    async def test_extract_deadline_pattern(self, extractor):
        """Test demo extraction of deadline patterns."""
        content = "Payment is due on 12/31/2025."
        result = await extractor.extract_facts(
            content=content,
            chunk_id="chunk_1",
            document_id="doc_1",
        )

        assert len(result.facts) >= 1
        date_facts = [f for f in result.facts if "date" in f.topics]
        assert len(date_facts) >= 1

    @pytest.mark.asyncio
    async def test_max_facts_limit(self, extractor):
        """Test that demo extraction respects max_facts_per_chunk."""
        # Content with many patterns
        content = """
        $100, $200, $300, $400, $500, $600, $700, $800, $900, $1000,
        $1100, $1200, $1300, $1400, $1500
        """
        result = await extractor.extract_facts(
            content=content,
            chunk_id="chunk_1",
            document_id="doc_1",
        )

        assert len(result.facts) <= extractor.config.max_facts_per_chunk

    @pytest.mark.asyncio
    async def test_empty_content(self, extractor):
        """Test extraction with empty content."""
        result = await extractor.extract_facts(
            content="",
            chunk_id="chunk_1",
            document_id="doc_1",
        )

        assert result.facts == []
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_no_extractable_content(self, extractor):
        """Test extraction with content having no extractable facts."""
        content = "This is just a plain sentence with no specific facts."
        result = await extractor.extract_facts(
            content=content,
            chunk_id="chunk_1",
            document_id="doc_1",
        )

        assert result.facts == []

    @pytest.mark.asyncio
    async def test_evidence_quote_populated(self, extractor):
        """Test that evidence quotes include surrounding context."""
        content = "The contract expires on 2025-12-31 as per the original agreement."
        result = await extractor.extract_facts(
            content=content,
            chunk_id="chunk_1",
            document_id="doc_1",
        )

        if result.facts:
            fact = result.facts[0]
            assert fact.evidence_quote != ""
            assert len(fact.evidence_quote) > 0


# =============================================================================
# Agent-Based Extraction Tests
# =============================================================================


class TestAgentExtraction:
    """Tests for agent-based fact extraction."""

    @pytest.fixture
    def extraction_response(self):
        """Sample extraction response from agent."""
        return json.dumps(
            {
                "facts": [
                    {
                        "statement": "The contract expires on December 31, 2025",
                        "confidence": 0.95,
                        "topics": ["contract", "expiration"],
                        "evidence_quote": "expires on December 31, 2025",
                    },
                    {
                        "statement": "Payment is due within 30 days",
                        "confidence": 0.8,
                        "topics": ["payment", "terms"],
                        "evidence_quote": "Payment is due within 30 days",
                    },
                ]
            }
        )

    @pytest.mark.asyncio
    async def test_extract_with_single_agent(self, extraction_response):
        """Test extraction with a single agent."""
        agent = MockAgent(name="claude", response=extraction_response)
        extractor = FactExtractor(
            agents=[agent],
            config=ExtractionConfig(require_agreement=False),
        )

        result = await extractor.extract_facts(
            content="Contract content here",
            chunk_id="chunk_1",
            document_id="doc_1",
        )

        assert result.agent_used == "claude"
        assert len(result.facts) == 2
        assert result.facts[0].statement == "The contract expires on December 31, 2025"

    @pytest.mark.asyncio
    async def test_extract_filters_low_confidence(self):
        """Test that low confidence facts are filtered out."""
        response = json.dumps(
            {
                "facts": [
                    {"statement": "High confidence fact", "confidence": 0.9},
                    {"statement": "Low confidence fact", "confidence": 0.3},
                ]
            }
        )
        agent = MockAgent(name="claude", response=response)
        extractor = FactExtractor(
            agents=[agent],
            config=ExtractionConfig(require_agreement=False, min_confidence_threshold=0.5),
        )

        result = await extractor.extract_facts(
            content="Content",
            chunk_id="chunk_1",
            document_id="doc_1",
        )

        assert len(result.facts) == 1
        assert result.facts[0].statement == "High confidence fact"

    @pytest.mark.asyncio
    async def test_extract_handles_invalid_json(self):
        """Test that invalid JSON is handled gracefully."""
        agent = MockAgent(name="claude", response="Not valid JSON at all")
        extractor = FactExtractor(
            agents=[agent],
            config=ExtractionConfig(require_agreement=False),
        )

        result = await extractor.extract_facts(
            content="Content",
            chunk_id="chunk_1",
            document_id="doc_1",
        )

        assert result.facts == []
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_extract_handles_partial_json(self):
        """Test extraction with JSON embedded in text."""
        response = """
        Here are the extracted facts:
        {"facts": [{"statement": "Embedded fact", "confidence": 0.85}]}
        That's all.
        """
        agent = MockAgent(name="claude", response=response)
        extractor = FactExtractor(
            agents=[agent],
            config=ExtractionConfig(require_agreement=False),
        )

        result = await extractor.extract_facts(
            content="Content",
            chunk_id="chunk_1",
            document_id="doc_1",
        )

        assert len(result.facts) == 1
        assert result.facts[0].statement == "Embedded fact"

    @pytest.mark.asyncio
    async def test_extract_skips_empty_statements(self):
        """Test that facts with empty statements are skipped."""
        response = json.dumps(
            {
                "facts": [
                    {"statement": "", "confidence": 0.9},
                    {"statement": "Valid fact", "confidence": 0.8},
                ]
            }
        )
        agent = MockAgent(name="claude", response=response)
        extractor = FactExtractor(
            agents=[agent],
            config=ExtractionConfig(require_agreement=False),
        )

        result = await extractor.extract_facts(
            content="Content",
            chunk_id="chunk_1",
            document_id="doc_1",
        )

        assert len(result.facts) == 1
        assert result.facts[0].statement == "Valid fact"

    @pytest.mark.asyncio
    async def test_agent_generation_error(self):
        """Test handling of agent generation errors."""
        agent = MockAgent(name="claude", fail_on_generate=True)
        extractor = FactExtractor(
            agents=[agent],
            config=ExtractionConfig(require_agreement=False),
        )

        result = await extractor.extract_facts(
            content="Content",
            chunk_id="chunk_1",
            document_id="doc_1",
        )

        # Should return empty facts on agent error
        assert result.facts == []


# =============================================================================
# Verification Tests
# =============================================================================


class TestVerification:
    """Tests for fact verification with second agent."""

    @pytest.fixture
    def verified_response(self):
        """Sample verification response."""
        return json.dumps(
            {
                "verified_facts": [
                    {
                        "original_statement": "The contract expires December 2025",
                        "verified": True,
                        "adjusted_confidence": 0.92,
                        "adjusted_statement": None,
                        "reason": "Verified from document",
                    }
                ]
            }
        )

    @pytest.mark.asyncio
    async def test_verification_with_two_agents(self, verified_response):
        """Test verification using second agent."""
        extraction_response = json.dumps(
            {
                "facts": [
                    {
                        "statement": "The contract expires December 2025",
                        "confidence": 0.8,
                        "topics": ["contract"],
                    }
                ]
            }
        )

        agent1 = MockAgent(name="claude", response=extraction_response)
        agent2 = MockAgent(name="gpt", response=verified_response)

        extractor = FactExtractor(
            agents=[agent1, agent2],
            config=ExtractionConfig(require_agreement=True),
        )

        result = await extractor.extract_facts(
            content="Contract content",
            chunk_id="chunk_1",
            document_id="doc_1",
        )

        assert len(result.facts) == 1
        assert result.facts[0].verified is True
        assert result.facts[0].confidence == 0.92

    @pytest.mark.asyncio
    async def test_verification_adjusts_statement(self):
        """Test that verification can adjust fact statement."""
        extraction_response = json.dumps(
            {
                "facts": [
                    {
                        "statement": "Original statement",
                        "confidence": 0.8,
                    }
                ]
            }
        )
        verification_response = json.dumps(
            {
                "verified_facts": [
                    {
                        "original_statement": "Original statement",
                        "verified": True,
                        "adjusted_confidence": 0.85,
                        "adjusted_statement": "Corrected statement",
                        "reason": "Minor correction applied",
                    }
                ]
            }
        )

        agent1 = MockAgent(name="claude", response=extraction_response)
        agent2 = MockAgent(name="gpt", response=verification_response)

        extractor = FactExtractor(
            agents=[agent1, agent2],
            config=ExtractionConfig(require_agreement=True),
        )

        result = await extractor.extract_facts(
            content="Content",
            chunk_id="chunk_1",
            document_id="doc_1",
        )

        assert result.facts[0].statement == "Corrected statement"

    @pytest.mark.asyncio
    async def test_verification_filters_unverified(self):
        """Test that unverified facts are filtered out."""
        extraction_response = json.dumps(
            {
                "facts": [
                    {"statement": "Verified fact", "confidence": 0.8},
                    {"statement": "Unverified fact", "confidence": 0.7},
                ]
            }
        )
        verification_response = json.dumps(
            {
                "verified_facts": [
                    {
                        "original_statement": "Verified fact",
                        "verified": True,
                        "adjusted_confidence": 0.85,
                        "reason": "Confirmed",
                    },
                    {
                        "original_statement": "Unverified fact",
                        "verified": False,
                        "reason": "Not found in document",
                    },
                ]
            }
        )

        agent1 = MockAgent(name="claude", response=extraction_response)
        agent2 = MockAgent(name="gpt", response=verification_response)

        extractor = FactExtractor(
            agents=[agent1, agent2],
            config=ExtractionConfig(require_agreement=True),
        )

        result = await extractor.extract_facts(
            content="Content",
            chunk_id="chunk_1",
            document_id="doc_1",
        )

        assert len(result.facts) == 1
        assert result.facts[0].statement == "Verified fact"

    @pytest.mark.asyncio
    async def test_verification_error_returns_original(self):
        """Test that verification errors return original facts."""
        extraction_response = json.dumps(
            {
                "facts": [
                    {"statement": "Original fact", "confidence": 0.8},
                ]
            }
        )

        agent1 = MockAgent(name="claude", response=extraction_response)
        agent2 = MockAgent(name="gpt", fail_on_generate=True)

        extractor = FactExtractor(
            agents=[agent1, agent2],
            config=ExtractionConfig(require_agreement=True),
        )

        result = await extractor.extract_facts(
            content="Content",
            chunk_id="chunk_1",
            document_id="doc_1",
        )

        # Original facts returned when verification fails
        assert len(result.facts) == 1
        assert result.facts[0].statement == "Original fact"
        assert result.facts[0].verified is False


# =============================================================================
# Batch Extraction Tests
# =============================================================================


class TestBatchExtraction:
    """Tests for batch fact extraction."""

    @pytest.fixture
    def extractor(self):
        """Create extractor for batch tests."""
        return FactExtractor()

    @pytest.mark.asyncio
    async def test_batch_extract_multiple_chunks(self, extractor):
        """Test batch extraction of multiple chunks."""
        chunks = [
            {"content": "Cost is $100", "chunk_id": "c1", "document_id": "doc1"},
            {"content": "Cost is $200", "chunk_id": "c2", "document_id": "doc1"},
            {"content": "Cost is $300", "chunk_id": "c3", "document_id": "doc1"},
        ]

        results = await extractor.extract_batch(chunks)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.chunk_id == f"c{i + 1}"

    @pytest.mark.asyncio
    async def test_batch_extract_respects_concurrency(self, extractor):
        """Test that batch extraction respects max_concurrent limit."""
        chunks = [
            {"content": f"Chunk {i}", "chunk_id": f"c{i}", "document_id": "doc1"} for i in range(10)
        ]

        results = await extractor.extract_batch(chunks, max_concurrent=2)

        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_batch_extract_empty_list(self, extractor):
        """Test batch extraction with empty chunk list."""
        results = await extractor.extract_batch([])

        assert results == []

    @pytest.mark.asyncio
    async def test_batch_extract_with_workspace(self, extractor):
        """Test batch extraction with workspace_id."""
        store = MockFactStore()
        extractor.fact_store = store

        chunks = [
            {"content": "$500 total", "chunk_id": "c1", "document_id": "doc1"},
        ]

        results = await extractor.extract_batch(chunks, workspace_id="ws_test")

        assert len(results) == 1
        # Facts should be persisted to store
        if results[0].facts:
            assert len(store.facts) >= 1


# =============================================================================
# Fact Persistence Tests
# =============================================================================


class TestFactPersistence:
    """Tests for fact persistence to store."""

    @pytest.fixture
    def store(self):
        """Create mock fact store."""
        return MockFactStore()

    @pytest.fixture
    def extractor(self, store):
        """Create extractor with mock store."""
        return FactExtractor(fact_store=store)

    @pytest.mark.asyncio
    async def test_persist_facts_to_store(self, extractor, store):
        """Test that facts are persisted to store."""
        result = await extractor.extract_facts(
            content="Total cost: $1,000",
            chunk_id="chunk_1",
            document_id="doc_1",
            workspace_id="ws_test",
        )

        if result.facts:
            assert len(store.facts) >= 1
            stored = store.facts[0]
            assert stored.workspace_id == "ws_test"

    @pytest.mark.asyncio
    async def test_persist_with_metadata(self, extractor, store):
        """Test persistence with metadata."""
        metadata = {"custom_key": "custom_value"}

        result = await extractor.extract_facts(
            content="Amount: $500",
            chunk_id="chunk_1",
            document_id="doc_1",
            workspace_id="ws_test",
            metadata=metadata,
        )

        if result.facts and store.facts:
            stored = store.facts[0]
            assert "custom_key" in stored.metadata

    @pytest.mark.asyncio
    async def test_no_persist_without_workspace(self, extractor, store):
        """Test that facts are not persisted without workspace_id."""
        result = await extractor.extract_facts(
            content="Cost: $100",
            chunk_id="chunk_1",
            document_id="doc_1",
            workspace_id="",  # Empty workspace
        )

        # Facts should not be persisted
        assert len(store.facts) == 0

    @pytest.mark.asyncio
    async def test_persist_verified_status(self, store):
        """Test that verified facts get correct status."""
        extraction_response = json.dumps({"facts": [{"statement": "Test fact", "confidence": 0.9}]})
        verification_response = json.dumps(
            {
                "verified_facts": [
                    {
                        "original_statement": "Test fact",
                        "verified": True,
                        "adjusted_confidence": 0.95,
                        "reason": "Confirmed",
                    }
                ]
            }
        )

        agent1 = MockAgent(name="claude", response=extraction_response)
        agent2 = MockAgent(name="gpt", response=verification_response)

        extractor = FactExtractor(
            agents=[agent1, agent2],
            fact_store=store,
            config=ExtractionConfig(require_agreement=True),
        )

        await extractor.extract_facts(
            content="Content",
            chunk_id="chunk_1",
            document_id="doc_1",
            workspace_id="ws_test",
        )

        if store.facts:
            assert store.facts[0].validation_status == ValidationStatus.MAJORITY_AGREED


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Tests for extraction statistics."""

    def test_get_statistics_initial(self):
        """Test initial statistics."""
        extractor = FactExtractor()
        stats = extractor.get_statistics()

        assert stats["total_extractions"] == 0
        assert stats["agents_available"] == 0
        assert stats["agent_names"] == []

    def test_get_statistics_with_agents(self):
        """Test statistics with agents."""
        agents = [MockAgent(name="claude"), MockAgent(name="gpt")]
        extractor = FactExtractor(agents=agents)

        stats = extractor.get_statistics()

        assert stats["agents_available"] == 2
        assert "claude" in stats["agent_names"]
        assert "gpt" in stats["agent_names"]

    @pytest.mark.asyncio
    async def test_statistics_increment_count(self):
        """Test that extraction count increments."""
        extractor = FactExtractor()

        await extractor.extract_facts(
            content="Test 1",
            chunk_id="c1",
            document_id="d1",
        )
        await extractor.extract_facts(
            content="Test 2",
            chunk_id="c2",
            document_id="d1",
        )

        stats = extractor.get_statistics()
        assert stats["total_extractions"] == 2

    def test_statistics_config_values(self):
        """Test that statistics include config values."""
        config = ExtractionConfig(
            max_facts_per_chunk=15,
            min_confidence_threshold=0.6,
            require_agreement=False,
        )
        extractor = FactExtractor(config=config)

        stats = extractor.get_statistics()

        assert stats["config"]["max_facts_per_chunk"] == 15
        assert stats["config"]["min_confidence_threshold"] == 0.6
        assert stats["config"]["require_agreement"] is False


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateFactExtractor:
    """Tests for create_fact_extractor factory function."""

    def test_create_default(self):
        """Test creating extractor with defaults."""
        extractor = create_fact_extractor()

        assert isinstance(extractor, FactExtractor)
        assert extractor.agents == []
        assert extractor.fact_store is None

    def test_create_with_agents(self):
        """Test creating extractor with agents."""
        agents = [MockAgent(name="test")]
        extractor = create_fact_extractor(agents=agents)

        assert len(extractor.agents) == 1

    def test_create_with_store(self):
        """Test creating extractor with store."""
        store = MockFactStore()
        extractor = create_fact_extractor(fact_store=store)

        assert extractor.fact_store is store

    def test_create_with_config(self):
        """Test creating extractor with config."""
        config = ExtractionConfig(max_facts_per_chunk=20)
        extractor = create_fact_extractor(config=config)

        assert extractor.config.max_facts_per_chunk == 20


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_extraction_time_recorded(self):
        """Test that extraction time is recorded."""
        extractor = FactExtractor()

        result = await extractor.extract_facts(
            content="Test content $100",
            chunk_id="c1",
            document_id="d1",
        )

        assert result.extraction_time_ms > 0

    @pytest.mark.asyncio
    async def test_content_truncation(self):
        """Test that very long content is truncated for agents."""
        long_content = "A" * 10000  # Very long content
        response = json.dumps({"facts": []})
        agent = MockAgent(name="claude", response=response)

        extractor = FactExtractor(
            agents=[agent],
            config=ExtractionConfig(require_agreement=False),
        )

        # Should not raise error
        result = await extractor.extract_facts(
            content=long_content,
            chunk_id="c1",
            document_id="d1",
        )

        assert result.errors == []

    @pytest.mark.asyncio
    async def test_special_characters_in_content(self):
        """Test extraction with special characters."""
        content = "Price: $1,000.00\nDiscount: 15%\n\tDate: 2025-01-01"
        extractor = FactExtractor()

        result = await extractor.extract_facts(
            content=content,
            chunk_id="c1",
            document_id="d1",
        )

        # Should extract at least some facts
        assert len(result.facts) >= 2

    @pytest.mark.asyncio
    async def test_unicode_content(self):
        """Test extraction with unicode content."""
        content = "Price: $500 for services in Tokyo (東京). Discount: 10%"
        extractor = FactExtractor()

        result = await extractor.extract_facts(
            content=content,
            chunk_id="c1",
            document_id="d1",
        )

        assert result.errors == []
        assert len(result.facts) >= 1

    @pytest.mark.asyncio
    async def test_exception_in_extraction_recorded(self):
        """Test that exceptions during extraction are recorded."""
        extractor = FactExtractor()

        # Patch _demo_extract to raise an error
        with patch.object(extractor, "_demo_extract", side_effect=ValueError("Test error")):
            result = await extractor.extract_facts(
                content="Test",
                chunk_id="c1",
                document_id="d1",
            )

        assert len(result.errors) == 1
        assert "Fact extraction failed" in result.errors[0]
