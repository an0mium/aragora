"""
Knowledge Extraction from Debates.

Extracts structured knowledge from debate transcripts:
- Fact extraction from agent responses
- Claim identification and validation
- Relationship inference between concepts
- Consensus-backed knowledge promotion

Phase A2 - Knowledge Extraction & Integration
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from aragora.knowledge.mound.facade import KnowledgeMound

logger = logging.getLogger(__name__)


class ExtractionType(str, Enum):
    """Types of extracted knowledge."""

    FACT = "fact"  # Verifiable factual claim
    DEFINITION = "definition"  # Definition of a concept
    PROCEDURE = "procedure"  # How-to or process
    RELATIONSHIP = "relationship"  # Relationship between concepts
    OPINION = "opinion"  # Agent opinion (lower confidence)
    CONSENSUS = "consensus"  # Consensus-backed conclusion


class ConfidenceSource(str, Enum):
    """Sources of confidence for extracted knowledge."""

    SINGLE_AGENT = "single_agent"  # Single agent claim
    MULTIPLE_AGENTS = "multiple_agents"  # Multiple agents agree
    CONSENSUS = "consensus"  # Formal consensus reached
    VALIDATED = "validated"  # Externally validated


@dataclass
class ExtractedClaim:
    """A claim extracted from a debate."""

    id: str
    content: str
    extraction_type: ExtractionType
    source_debate_id: str
    source_agent_id: Optional[str] = None
    source_round: Optional[int] = None
    confidence: float = 0.5
    confidence_source: ConfidenceSource = ConfidenceSource.SINGLE_AGENT
    topics: List[str] = field(default_factory=list)
    supporting_agents: List[str] = field(default_factory=list)
    contradicting_agents: List[str] = field(default_factory=list)
    extracted_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def agreement_ratio(self) -> float:
        """Calculate ratio of supporting vs contradicting agents."""
        total = len(self.supporting_agents) + len(self.contradicting_agents)
        if total == 0:
            return 0.5
        return len(self.supporting_agents) / total

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "extraction_type": self.extraction_type.value,
            "source_debate_id": self.source_debate_id,
            "source_agent_id": self.source_agent_id,
            "source_round": self.source_round,
            "confidence": self.confidence,
            "confidence_source": self.confidence_source.value,
            "topics": self.topics,
            "supporting_agents": self.supporting_agents,
            "contradicting_agents": self.contradicting_agents,
            "agreement_ratio": round(self.agreement_ratio, 3),
            "extracted_at": self.extracted_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ExtractedRelationship:
    """A relationship extracted between concepts."""

    id: str
    source_concept: str
    target_concept: str
    relationship_type: str  # e.g., "is_a", "part_of", "causes", "requires"
    source_debate_id: str
    confidence: float = 0.5
    evidence: str = ""
    extracted_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "source_concept": self.source_concept,
            "target_concept": self.target_concept,
            "relationship_type": self.relationship_type,
            "source_debate_id": self.source_debate_id,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "extracted_at": self.extracted_at.isoformat(),
        }


@dataclass
class ExtractionResult:
    """Result of knowledge extraction from a debate."""

    debate_id: str
    claims: List[ExtractedClaim]
    relationships: List[ExtractedRelationship]
    topics_discovered: List[str]
    promoted_to_mound: int  # Claims added to knowledge mound
    extraction_duration_ms: float
    extracted_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "debate_id": self.debate_id,
            "claims_extracted": len(self.claims),
            "relationships_extracted": len(self.relationships),
            "topics_discovered": self.topics_discovered,
            "promoted_to_mound": self.promoted_to_mound,
            "extraction_duration_ms": round(self.extraction_duration_ms, 2),
            "extracted_at": self.extracted_at.isoformat(),
            "claims": [c.to_dict() for c in self.claims],
            "relationships": [r.to_dict() for r in self.relationships],
        }


@dataclass
class ExtractionConfig:
    """Configuration for knowledge extraction."""

    # Confidence thresholds
    min_confidence_to_extract: float = 0.3
    min_confidence_to_promote: float = 0.6
    consensus_confidence_boost: float = 0.2

    # Extraction settings
    extract_facts: bool = True
    extract_definitions: bool = True
    extract_procedures: bool = True
    extract_relationships: bool = True
    extract_opinions: bool = False  # Lower quality, off by default

    # Agreement thresholds
    min_agents_for_consensus: int = 2
    min_agreement_ratio: float = 0.6

    # Pattern matching
    fact_patterns: List[str] = field(
        default_factory=lambda: [
            r"(?:it is|is|are|was|were) (?:true that|a fact that|known that)",
            r"(?:studies show|research indicates|data shows|evidence suggests)",
            r"according to",
            r"(?:it has been|has been) (?:proven|demonstrated|shown)",
        ]
    )

    definition_patterns: List[str] = field(
        default_factory=lambda: [
            r"(?:is defined as|means|refers to|is called)",
            r"(?:the definition of|by definition)",
            r"(?:in other words|that is to say|i\.e\.|namely)",
        ]
    )

    procedure_patterns: List[str] = field(
        default_factory=lambda: [
            r"(?:to do this|the steps are|you should|first.*then)",
            r"(?:the process|procedure|method) (?:is|involves)",
            r"(?:step \d|1\.|2\.|3\.)",
        ]
    )

    relationship_patterns: List[str] = field(
        default_factory=lambda: [
            r"(\w+) (?:is a|is an|are) (?:type of|kind of|form of) (\w+)",
            r"(\w+) (?:causes|leads to|results in) (\w+)",
            r"(\w+) (?:requires|needs|depends on) (\w+)",
            r"(\w+) (?:is part of|belongs to|is included in) (\w+)",
        ]
    )


class DebateKnowledgeExtractor:
    """Extracts knowledge from debate transcripts."""

    def __init__(self, config: Optional[ExtractionConfig] = None):
        """Initialize the extractor."""
        self.config = config or ExtractionConfig()
        self._extracted_claims: Dict[str, ExtractedClaim] = {}
        self._extracted_relationships: Dict[str, ExtractedRelationship] = {}
        self._lock = asyncio.Lock()

    async def extract_from_debate(
        self,
        debate_id: str,
        messages: List[Dict[str, Any]],
        consensus_text: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> ExtractionResult:
        """Extract knowledge from a debate.

        Args:
            debate_id: ID of the debate
            messages: List of debate messages (agent responses)
            consensus_text: Optional consensus conclusion
            topic: Optional debate topic

        Returns:
            ExtractionResult with extracted knowledge
        """
        import time

        start_time = time.time()

        claims: List[ExtractedClaim] = []
        relationships: List[ExtractedRelationship] = []
        topics_discovered: Set[str] = set()

        if topic:
            topics_discovered.add(topic)

        # Track claims by content for agreement detection
        claim_supporters: Dict[str, Set[str]] = {}
        # Reserved for future contradiction tracking across agents
        # claim_contradictors: Dict[str, Set[str]] = {}

        # Process each message
        for msg in messages:
            agent_id = msg.get("agent_id") or msg.get("agent")
            content = msg.get("content") or msg.get("response") or ""
            round_num = msg.get("round")

            if not content:
                continue

            # Extract claims from this message
            msg_claims = await self._extract_claims_from_text(
                content, debate_id, agent_id, round_num
            )

            for claim in msg_claims:
                # Check if similar claim exists
                similar_key = self._normalize_for_comparison(claim.content)

                if similar_key not in claim_supporters:
                    claim_supporters[similar_key] = set()
                    claims.append(claim)

                if agent_id:
                    claim_supporters[similar_key].add(agent_id)

                # Discover topics
                topics_discovered.update(claim.topics)

            # Extract relationships
            if self.config.extract_relationships:
                msg_rels = await self._extract_relationships(content, debate_id)
                relationships.extend(msg_rels)

        # Update claims with supporter counts
        for claim in claims:
            key = self._normalize_for_comparison(claim.content)
            claim.supporting_agents = list(claim_supporters.get(key, set()))

            # Boost confidence if multiple agents agree
            if len(claim.supporting_agents) >= self.config.min_agents_for_consensus:
                claim.confidence = min(1.0, claim.confidence + 0.1)
                claim.confidence_source = ConfidenceSource.MULTIPLE_AGENTS

        # Process consensus if available
        if consensus_text:
            consensus_claims = await self._extract_claims_from_text(
                consensus_text, debate_id, None, None
            )
            for claim in consensus_claims:
                claim.extraction_type = ExtractionType.CONSENSUS
                claim.confidence = min(
                    1.0, claim.confidence + self.config.consensus_confidence_boost
                )
                claim.confidence_source = ConfidenceSource.CONSENSUS
                claims.append(claim)

        # Filter by confidence
        claims = [c for c in claims if c.confidence >= self.config.min_confidence_to_extract]

        # Store extracted knowledge
        async with self._lock:
            for claim in claims:
                self._extracted_claims[claim.id] = claim
            for rel in relationships:
                self._extracted_relationships[rel.id] = rel

        duration_ms = (time.time() - start_time) * 1000

        return ExtractionResult(
            debate_id=debate_id,
            claims=claims,
            relationships=relationships,
            topics_discovered=list(topics_discovered),
            promoted_to_mound=0,  # Updated when promoted
            extraction_duration_ms=duration_ms,
        )

    async def _extract_claims_from_text(
        self,
        text: str,
        debate_id: str,
        agent_id: Optional[str],
        round_num: Optional[int],
    ) -> List[ExtractedClaim]:
        """Extract claims from a text block."""
        import uuid

        claims: List[ExtractedClaim] = []

        # Split into sentences
        sentences = re.split(r"[.!?]+", text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue

            extraction_type = self._classify_sentence(sentence)
            if extraction_type is None:
                continue

            # Skip opinions if disabled
            if extraction_type == ExtractionType.OPINION and not self.config.extract_opinions:
                continue

            # Calculate base confidence
            confidence = self._calculate_confidence(sentence, extraction_type)

            # Extract topics from sentence
            topics = self._extract_topics(sentence)

            claim = ExtractedClaim(
                id=str(uuid.uuid4()),
                content=sentence,
                extraction_type=extraction_type,
                source_debate_id=debate_id,
                source_agent_id=agent_id,
                source_round=round_num,
                confidence=confidence,
                topics=topics,
            )
            claims.append(claim)

        return claims

    def _classify_sentence(self, sentence: str) -> Optional[ExtractionType]:
        """Classify a sentence by extraction type."""
        sentence_lower = sentence.lower()

        # Check fact patterns
        if self.config.extract_facts:
            for pattern in self.config.fact_patterns:
                if re.search(pattern, sentence_lower):
                    return ExtractionType.FACT

        # Check definition patterns
        if self.config.extract_definitions:
            for pattern in self.config.definition_patterns:
                if re.search(pattern, sentence_lower):
                    return ExtractionType.DEFINITION

        # Check procedure patterns
        if self.config.extract_procedures:
            for pattern in self.config.procedure_patterns:
                if re.search(pattern, sentence_lower):
                    return ExtractionType.PROCEDURE

        # Default to opinion if contains opinion markers
        opinion_markers = ["i think", "i believe", "in my opinion", "it seems", "probably"]
        for marker in opinion_markers:
            if marker in sentence_lower:
                return ExtractionType.OPINION

        return None

    def _calculate_confidence(
        self,
        sentence: str,
        extraction_type: ExtractionType,
    ) -> float:
        """Calculate confidence for an extracted claim."""
        base_confidence = 0.5

        # Type-based adjustment
        if extraction_type == ExtractionType.CONSENSUS:
            base_confidence = 0.8
        elif extraction_type == ExtractionType.FACT:
            base_confidence = 0.6
        elif extraction_type == ExtractionType.DEFINITION:
            base_confidence = 0.7
        elif extraction_type == ExtractionType.PROCEDURE:
            base_confidence = 0.6
        elif extraction_type == ExtractionType.OPINION:
            base_confidence = 0.3

        # Hedging words reduce confidence
        hedges = ["maybe", "possibly", "might", "could", "perhaps"]
        for hedge in hedges:
            if hedge in sentence.lower():
                base_confidence -= 0.1

        # Certainty markers increase confidence
        certainty = ["definitely", "certainly", "always", "never", "proven"]
        for cert in certainty:
            if cert in sentence.lower():
                base_confidence += 0.1

        return max(0.1, min(1.0, base_confidence))

    def _extract_topics(self, sentence: str) -> List[str]:
        """Extract topic keywords from a sentence."""
        # Simple noun phrase extraction
        # In production, would use NLP library
        words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", sentence)

        # Filter common words
        stop_words = {"The", "This", "That", "These", "Those", "It", "They", "We", "You"}
        topics = [w for w in words if w not in stop_words]

        return topics[:5]  # Limit to 5 topics

    async def _extract_relationships(
        self,
        text: str,
        debate_id: str,
    ) -> List[ExtractedRelationship]:
        """Extract relationships between concepts."""
        import uuid

        relationships: List[ExtractedRelationship] = []

        for pattern in self.config.relationship_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    source = match.group(1)
                    target = match.group(2)

                    # Determine relationship type
                    rel_type = self._infer_relationship_type(match.group(0))

                    rel = ExtractedRelationship(
                        id=str(uuid.uuid4()),
                        source_concept=source,
                        target_concept=target,
                        relationship_type=rel_type,
                        source_debate_id=debate_id,
                        confidence=0.6,
                        evidence=match.group(0),
                    )
                    relationships.append(rel)

        return relationships

    def _infer_relationship_type(self, text: str) -> str:
        """Infer relationship type from text."""
        text_lower = text.lower()

        if "is a" in text_lower or "is an" in text_lower or "type of" in text_lower:
            return "is_a"
        if "causes" in text_lower or "leads to" in text_lower:
            return "causes"
        if "requires" in text_lower or "needs" in text_lower:
            return "requires"
        if "part of" in text_lower or "belongs to" in text_lower:
            return "part_of"

        return "related_to"

    def _normalize_for_comparison(self, text: str) -> str:
        """Normalize text for similarity comparison."""
        # Remove punctuation, lowercase, sort words
        words = re.findall(r"\w+", text.lower())
        return " ".join(sorted(words))

    async def promote_to_mound(
        self,
        mound: "KnowledgeMound",
        workspace_id: str,
        claims: Optional[List[ExtractedClaim]] = None,
        min_confidence: Optional[float] = None,
    ) -> int:
        """Promote extracted claims to Knowledge Mound.

        Args:
            mound: KnowledgeMound instance
            workspace_id: Workspace to add to
            claims: Specific claims to promote (uses stored if None)
            min_confidence: Minimum confidence to promote

        Returns:
            Number of claims promoted
        """
        min_conf = min_confidence or self.config.min_confidence_to_promote

        if claims is None:
            async with self._lock:
                claims = [c for c in self._extracted_claims.values() if c.confidence >= min_conf]

        promoted = 0
        for claim in claims:
            if claim.confidence < min_conf:
                continue

            try:
                # Create knowledge item
                await mound.store(  # type: ignore[misc,call-arg]
                    workspace_id=workspace_id,
                    content=claim.content,
                    topics=claim.topics,
                    metadata={
                        "extraction_type": claim.extraction_type.value,
                        "source_debate_id": claim.source_debate_id,
                        "source_agent_id": claim.source_agent_id,
                        "supporting_agents": claim.supporting_agents,
                        "agreement_ratio": claim.agreement_ratio,
                    },
                    confidence=claim.confidence,
                )
                promoted += 1
            except Exception as e:
                logger.warning(f"Failed to promote claim {claim.id}: {e}")

        return promoted

    def get_stats(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        by_type: Dict[str, int] = {}
        total_confidence = 0.0

        for claim in self._extracted_claims.values():
            by_type[claim.extraction_type.value] = by_type.get(claim.extraction_type.value, 0) + 1
            total_confidence += claim.confidence

        return {
            "total_claims": len(self._extracted_claims),
            "total_relationships": len(self._extracted_relationships),
            "by_type": by_type,
            "average_confidence": total_confidence / len(self._extracted_claims)
            if self._extracted_claims
            else 0.0,
        }


class ExtractionMixin:
    """Mixin for knowledge extraction operations on KnowledgeMound."""

    _extractor: Optional[DebateKnowledgeExtractor] = None

    def _get_extractor(self) -> DebateKnowledgeExtractor:
        """Get or create extractor."""
        if self._extractor is None:
            self._extractor = DebateKnowledgeExtractor()
        return self._extractor

    async def extract_from_debate(
        self,
        debate_id: str,
        messages: List[Dict[str, Any]],
        consensus_text: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> ExtractionResult:
        """Extract knowledge from a debate."""
        return await self._get_extractor().extract_from_debate(
            debate_id, messages, consensus_text, topic
        )

    async def promote_extracted_knowledge(
        self,
        workspace_id: str,
        claims: Optional[List[ExtractedClaim]] = None,
        min_confidence: float = 0.6,
    ) -> int:
        """Promote extracted claims to Knowledge Mound."""
        return await self._get_extractor().promote_to_mound(
            self,  # type: ignore[arg-type]
            workspace_id,
            claims,
            min_confidence,  # type: ignore[arg-type]
        )

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return self._get_extractor().get_stats()


# Singleton instance
_extractor: Optional[DebateKnowledgeExtractor] = None


def get_debate_extractor() -> DebateKnowledgeExtractor:
    """Get the global debate knowledge extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = DebateKnowledgeExtractor()
    return _extractor
