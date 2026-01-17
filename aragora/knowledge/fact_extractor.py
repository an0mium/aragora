"""
Fact Extractor for automatic fact extraction from document chunks.

Uses multi-agent debate to extract and verify facts from document content.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, Protocol

from aragora.knowledge.types import Fact, ValidationStatus

if TYPE_CHECKING:
    from aragora.knowledge.fact_store import FactStore, InMemoryFactStore

logger = logging.getLogger(__name__)


class AgentProtocol(Protocol):
    """Protocol for agents that can generate responses."""

    name: str

    async def generate(self, prompt: str, context: list | None = None) -> str:
        """Generate a response to the prompt."""
        ...


@dataclass
class ExtractionConfig:
    """Configuration for fact extraction."""

    # Extraction parameters
    max_facts_per_chunk: int = 10
    min_confidence_threshold: float = 0.5
    require_evidence: bool = True

    # Agent configuration
    num_extraction_agents: int = 2
    require_agreement: bool = True
    agreement_threshold: float = 0.7

    # Prompts
    extraction_prompt_template: str = """You are a fact extraction expert. Extract factual claims from the following document chunk.

Document context:
- Filename: {filename}
- Chunk ID: {chunk_id}

Content:
{content}

Extract up to {max_facts} factual claims from this content. For each fact:
1. State the fact clearly and concisely
2. Rate your confidence (0.0-1.0) based on how explicitly stated the fact is
3. List any relevant topics/categories

Respond in JSON format:
{{
    "facts": [
        {{
            "statement": "The factual claim",
            "confidence": 0.85,
            "topics": ["topic1", "topic2"],
            "evidence_quote": "The exact quote from the document supporting this fact"
        }}
    ]
}}

Focus on:
- Specific dates, numbers, and deadlines
- Named entities (people, organizations, places)
- Contractual obligations and terms
- Technical specifications
- Stated relationships between entities

Do NOT include:
- Opinions or subjective statements
- Vague or ambiguous claims
- Information that requires external context"""

    verification_prompt_template: str = """You are a fact verification expert. Review the following extracted facts for accuracy and completeness.

Original document chunk:
{content}

Extracted facts to verify:
{facts_json}

For each fact, determine:
1. Is the fact accurately stated based on the document?
2. Is the confidence rating appropriate?
3. Are the topics correctly identified?

Respond in JSON format:
{{
    "verified_facts": [
        {{
            "original_statement": "The original fact statement",
            "verified": true,
            "adjusted_confidence": 0.85,
            "adjusted_statement": "Corrected statement if needed (or null)",
            "reason": "Brief explanation of verification decision"
        }}
    ]
}}"""


@dataclass
class ExtractedFact:
    """A fact extracted from a document chunk before storage."""

    statement: str
    confidence: float
    topics: list[str] = field(default_factory=list)
    evidence_quote: str = ""
    source_chunk_id: str = ""
    source_document: str = ""
    verified: bool = False
    verification_reason: str = ""


@dataclass
class ExtractionResult:
    """Result of extracting facts from a chunk."""

    chunk_id: str
    document_id: str
    facts: list[ExtractedFact] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    extraction_time_ms: float = 0.0
    agent_used: str = ""


class FactExtractor:
    """Extract facts from document chunks using multi-agent analysis."""

    def __init__(
        self,
        agents: list[AgentProtocol] | None = None,
        config: ExtractionConfig | None = None,
        fact_store: Optional["FactStore | InMemoryFactStore"] = None,
    ):
        """Initialize the fact extractor.

        Args:
            agents: List of agents to use for extraction. If None, uses demo extraction.
            config: Extraction configuration.
            fact_store: Optional fact store for persisting extracted facts.
        """
        self.agents = agents or []
        self.config = config or ExtractionConfig()
        self.fact_store = fact_store
        self._extraction_count = 0

    async def extract_facts(
        self,
        content: str,
        chunk_id: str,
        document_id: str,
        filename: str = "",
        workspace_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ExtractionResult:
        """Extract facts from a document chunk.

        Args:
            content: The text content to extract facts from.
            chunk_id: Unique identifier for the chunk.
            document_id: Document the chunk belongs to.
            filename: Original filename for context.
            workspace_id: Workspace for the extracted facts.
            metadata: Additional metadata to attach to facts.

        Returns:
            ExtractionResult containing extracted facts.
        """
        start_time = datetime.now()
        result = ExtractionResult(
            chunk_id=chunk_id,
            document_id=document_id,
        )

        try:
            # Extract facts using available agents
            if self.agents:
                extracted = await self._extract_with_agents(
                    content=content,
                    chunk_id=chunk_id,
                    filename=filename,
                )
                result.facts = extracted
                result.agent_used = self.agents[0].name if self.agents else "none"

                # Verify with second agent if configured
                if self.config.require_agreement and len(self.agents) >= 2 and extracted:
                    result.facts = await self._verify_with_agent(
                        content=content,
                        facts=extracted,
                        agent=self.agents[1],
                    )
            else:
                # Demo extraction without agents
                result.facts = self._demo_extract(content, chunk_id, document_id)
                result.agent_used = "demo"

            # Persist facts if store is available
            if self.fact_store and workspace_id:
                await self._persist_facts(
                    facts=result.facts,
                    workspace_id=workspace_id,
                    document_id=document_id,
                    metadata=metadata,
                )

        except Exception as e:
            logger.exception(f"Fact extraction failed for chunk {chunk_id}")
            result.errors.append(str(e))

        end_time = datetime.now()
        result.extraction_time_ms = (end_time - start_time).total_seconds() * 1000
        self._extraction_count += 1

        return result

    async def extract_batch(
        self,
        chunks: list[dict[str, Any]],
        workspace_id: str = "",
        max_concurrent: int = 5,
    ) -> list[ExtractionResult]:
        """Extract facts from multiple chunks concurrently.

        Args:
            chunks: List of chunk dictionaries with 'content', 'chunk_id', 'document_id'.
            workspace_id: Workspace for the extracted facts.
            max_concurrent: Maximum concurrent extractions.

        Returns:
            List of ExtractionResult for each chunk.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def extract_with_limit(chunk: dict) -> ExtractionResult:
            async with semaphore:
                return await self.extract_facts(
                    content=chunk.get("content", ""),
                    chunk_id=chunk.get("chunk_id", ""),
                    document_id=chunk.get("document_id", ""),
                    filename=chunk.get("filename", ""),
                    workspace_id=workspace_id,
                    metadata=chunk.get("metadata"),
                )

        tasks = [extract_with_limit(chunk) for chunk in chunks]
        return await asyncio.gather(*tasks)

    async def _extract_with_agents(
        self,
        content: str,
        chunk_id: str,
        filename: str,
    ) -> list[ExtractedFact]:
        """Extract facts using the configured agents."""
        if not self.agents:
            return []

        agent = self.agents[0]
        prompt = self.config.extraction_prompt_template.format(
            content=content[:8000],  # Limit content size
            chunk_id=chunk_id,
            filename=filename or "unknown",
            max_facts=self.config.max_facts_per_chunk,
        )

        try:
            response = await agent.generate(prompt)
            return self._parse_extraction_response(response, chunk_id, filename)
        except Exception as e:
            logger.warning(f"Agent extraction failed: {e}")
            return []

    async def _verify_with_agent(
        self,
        content: str,
        facts: list[ExtractedFact],
        agent: AgentProtocol,
    ) -> list[ExtractedFact]:
        """Verify extracted facts with a second agent."""
        facts_json = json.dumps(
            [
                {
                    "statement": f.statement,
                    "confidence": f.confidence,
                    "topics": f.topics,
                }
                for f in facts
            ],
            indent=2,
        )

        prompt = self.config.verification_prompt_template.format(
            content=content[:8000],
            facts_json=facts_json,
        )

        try:
            response = await agent.generate(prompt)
            verified = self._parse_verification_response(response, facts)
            return verified
        except Exception as e:
            logger.warning(f"Verification failed: {e}")
            # Return original facts if verification fails
            return facts

    def _parse_extraction_response(
        self,
        response: str,
        chunk_id: str,
        filename: str,
    ) -> list[ExtractedFact]:
        """Parse the agent's extraction response."""
        facts = []

        # Try to extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            logger.warning("No JSON found in extraction response")
            return facts

        try:
            data = json.loads(json_match.group())
            for item in data.get("facts", []):
                if not item.get("statement"):
                    continue

                confidence = float(item.get("confidence", 0.5))
                if confidence < self.config.min_confidence_threshold:
                    continue

                fact = ExtractedFact(
                    statement=item["statement"],
                    confidence=confidence,
                    topics=item.get("topics", []),
                    evidence_quote=item.get("evidence_quote", ""),
                    source_chunk_id=chunk_id,
                    source_document=filename,
                )
                facts.append(fact)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse extraction JSON: {e}")

        return facts[: self.config.max_facts_per_chunk]

    def _parse_verification_response(
        self,
        response: str,
        original_facts: list[ExtractedFact],
    ) -> list[ExtractedFact]:
        """Parse the agent's verification response."""
        # Create lookup by statement
        fact_lookup = {f.statement: f for f in original_facts}

        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            return original_facts

        try:
            data = json.loads(json_match.group())
            verified_facts = []

            for item in data.get("verified_facts", []):
                original_statement = item.get("original_statement", "")
                original_fact = fact_lookup.get(original_statement)

                if not original_fact:
                    continue

                if item.get("verified", False):
                    # Update with verification info
                    original_fact.verified = True
                    original_fact.verification_reason = item.get("reason", "")

                    # Apply adjusted confidence if provided
                    if "adjusted_confidence" in item:
                        original_fact.confidence = float(item["adjusted_confidence"])

                    # Apply corrected statement if provided
                    if item.get("adjusted_statement"):
                        original_fact.statement = item["adjusted_statement"]

                    verified_facts.append(original_fact)

            return verified_facts if verified_facts else original_facts

        except json.JSONDecodeError:
            return original_facts

    def _demo_extract(
        self,
        content: str,
        chunk_id: str,
        document_id: str,
    ) -> list[ExtractedFact]:
        """Demo extraction without agents - extracts basic patterns."""
        facts = []

        # Extract dates
        date_patterns = [
            r"(?:expires?|due|deadline|effective)\s+(?:on\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"(\d{4}-\d{2}-\d{2})",
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}",
        ]
        for pattern in date_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                facts.append(
                    ExtractedFact(
                        statement=f"Date mentioned: {match.group()}",
                        confidence=0.6,
                        topics=["date", "timeline"],
                        evidence_quote=content[max(0, match.start() - 50) : match.end() + 50],
                        source_chunk_id=chunk_id,
                        source_document=document_id,
                    )
                )

        # Extract monetary values
        money_pattern = r"\$[\d,]+(?:\.\d{2})?"
        for match in re.finditer(money_pattern, content):
            facts.append(
                ExtractedFact(
                    statement=f"Monetary value: {match.group()}",
                    confidence=0.7,
                    topics=["financial", "amount"],
                    evidence_quote=content[max(0, match.start() - 50) : match.end() + 50],
                    source_chunk_id=chunk_id,
                    source_document=document_id,
                )
            )

        # Extract percentages
        percent_pattern = r"\d+(?:\.\d+)?%"
        for match in re.finditer(percent_pattern, content):
            facts.append(
                ExtractedFact(
                    statement=f"Percentage: {match.group()}",
                    confidence=0.7,
                    topics=["percentage", "metric"],
                    evidence_quote=content[max(0, match.start() - 50) : match.end() + 50],
                    source_chunk_id=chunk_id,
                    source_document=document_id,
                )
            )

        return facts[: self.config.max_facts_per_chunk]

    async def _persist_facts(
        self,
        facts: list[ExtractedFact],
        workspace_id: str,
        document_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Fact]:
        """Persist extracted facts to the fact store."""
        if not self.fact_store:
            return []

        stored_facts = []
        for extracted in facts:
            fact_metadata = {
                "source_chunk_id": extracted.source_chunk_id,
                "evidence_quote": extracted.evidence_quote,
                "verified": extracted.verified,
                "verification_reason": extracted.verification_reason,
                **(metadata or {}),
            }

            stored = self.fact_store.add_fact(
                statement=extracted.statement,
                workspace_id=workspace_id,
                confidence=extracted.confidence,
                topics=extracted.topics,
                source_documents=[document_id],
                metadata=fact_metadata,
                validation_status=(
                    ValidationStatus.MAJORITY_AGREED
                    if extracted.verified
                    else ValidationStatus.UNVERIFIED
                ),
            )
            stored_facts.append(stored)

        return stored_facts

    def get_statistics(self) -> dict[str, Any]:
        """Get extraction statistics."""
        return {
            "total_extractions": self._extraction_count,
            "agents_available": len(self.agents),
            "agent_names": [a.name for a in self.agents],
            "config": {
                "max_facts_per_chunk": self.config.max_facts_per_chunk,
                "min_confidence_threshold": self.config.min_confidence_threshold,
                "require_agreement": self.config.require_agreement,
            },
        }


def create_fact_extractor(
    agents: list[AgentProtocol] | None = None,
    fact_store: Optional["FactStore | InMemoryFactStore"] = None,
    config: ExtractionConfig | None = None,
) -> FactExtractor:
    """Create a fact extractor with the given configuration.

    Args:
        agents: Agents to use for extraction.
        fact_store: Store for persisting facts.
        config: Extraction configuration.

    Returns:
        Configured FactExtractor instance.
    """
    return FactExtractor(
        agents=agents,
        fact_store=fact_store,
        config=config,
    )
