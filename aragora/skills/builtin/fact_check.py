"""
Fact Check Skill.

Provides claim verification capabilities against the Knowledge Mound
and external fact-checking sources.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..base import (
    Skill,
    SkillCapability,
    SkillContext,
    SkillManifest,
    SkillResult,
)

logger = logging.getLogger(__name__)


class VerificationStatus(str, Enum):
    """Status of claim verification."""

    VERIFIED = "verified"  # Claim is supported by evidence
    REFUTED = "refuted"  # Claim is contradicted by evidence
    PARTIALLY_TRUE = "partially_true"  # Some aspects are true, some false
    UNVERIFIABLE = "unverifiable"  # Cannot be verified with available sources
    NEEDS_CONTEXT = "needs_context"  # Claim requires additional context
    OPINION = "opinion"  # Claim is subjective/opinion


@dataclass
class VerificationEvidence:
    """Evidence supporting or refuting a claim."""

    source: str
    content: str
    relevance: float
    supports_claim: bool | None  # True=supports, False=refutes, None=neutral
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "content": self.content,
            "relevance": self.relevance,
            "supports_claim": self.supports_claim,
            "url": self.url,
            "metadata": self.metadata,
        }


@dataclass
class VerificationResult:
    """Result of claim verification."""

    claim: str
    status: VerificationStatus
    confidence: float
    explanation: str
    evidence: list[VerificationEvidence] = field(default_factory=list)
    related_claims: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "status": self.status.value,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "evidence": [e.to_dict() for e in self.evidence],
            "related_claims": self.related_claims,
        }


class FactCheckSkill(Skill):
    """
    Skill for verifying claims against knowledge sources.

    Verification pipeline:
    1. Parse and normalize the claim
    2. Query Knowledge Mound for relevant consensus
    3. Search external fact-checking databases
    4. Cross-reference with debate history
    5. Generate verification result with confidence
    """

    def __init__(
        self,
        min_confidence_threshold: float = 0.5,
        max_evidence_items: int = 5,
    ):
        """
        Initialize fact check skill.

        Args:
            min_confidence_threshold: Minimum confidence to report a verdict
            max_evidence_items: Maximum evidence items to return
        """
        self._min_confidence = min_confidence_threshold
        self._max_evidence = max_evidence_items

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="fact_check",
            version="1.0.0",
            description="Verify claims against knowledge base and fact-checking sources",
            capabilities=[
                SkillCapability.KNOWLEDGE_QUERY,
                SkillCapability.EXTERNAL_API,
            ],
            input_schema={
                "claim": {
                    "type": "string",
                    "description": "The claim to verify",
                    "required": True,
                },
                "context": {
                    "type": "string",
                    "description": "Additional context for the claim",
                },
                "sources": {
                    "type": "array",
                    "description": "Sources to check: knowledge_mound, web, debates",
                    "default": ["knowledge_mound", "web"],
                },
                "detailed": {
                    "type": "boolean",
                    "description": "Include detailed evidence breakdown",
                    "default": True,
                },
            },
            tags=["fact-check", "verification", "truth"],
            debate_compatible=True,
            requires_debate_context=False,
            max_execution_time_seconds=60.0,
            rate_limit_per_minute=20,
        )

    async def execute(
        self,
        input_data: dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        """Execute fact check."""
        claim = input_data.get("claim", "").strip()
        if not claim:
            return SkillResult.create_failure(
                "Claim is required",
                error_code="missing_claim",
            )

        claim_context = input_data.get("context", "")
        sources = input_data.get("sources", ["knowledge_mound", "web"])
        detailed = input_data.get("detailed", True)

        try:
            # Analyze claim type
            claim_type = self._analyze_claim_type(claim)
            if claim_type == "opinion":
                return SkillResult.create_success(
                    VerificationResult(
                        claim=claim,
                        status=VerificationStatus.OPINION,
                        confidence=0.9,
                        explanation="This appears to be a subjective opinion rather than a verifiable claim.",
                    ).to_dict()
                )

            # Collect evidence from various sources
            all_evidence: list[VerificationEvidence] = []

            if "knowledge_mound" in sources:
                km_evidence = await self._check_knowledge_mound(claim, claim_context)
                all_evidence.extend(km_evidence)

            if "web" in sources:
                web_evidence = await self._check_web_sources(claim)
                all_evidence.extend(web_evidence)

            if "debates" in sources:
                debate_evidence = await self._check_debate_history(claim, context)
                all_evidence.extend(debate_evidence)

            # Analyze evidence and determine verdict
            result = self._analyze_evidence(claim, all_evidence)

            # Format response
            response = result.to_dict()
            if not detailed:
                response["evidence"] = response["evidence"][: self._max_evidence]

            return SkillResult.create_success(response)

        except (RuntimeError, ValueError, OSError) as e:
            logger.exception(f"Fact check failed: {e}")
            return SkillResult.create_failure(f"Fact check failed: {e}")

    def _analyze_claim_type(self, claim: str) -> str:
        """Analyze what type of claim this is."""
        claim_lower = claim.lower()

        # Opinion indicators
        opinion_indicators = [
            "i think",
            "i believe",
            "in my opinion",
            "i feel",
            "should",
            "better than",
            "worst",
            "best",
            "beautiful",
            "ugly",
            "good",
            "bad",
            "like",
            "love",
            "hate",
        ]
        if any(indicator in claim_lower for indicator in opinion_indicators):
            return "opinion"

        # Statistical/factual indicators
        factual_indicators = [
            "percent",
            "%",
            "million",
            "billion",
            "studies show",
            "according to",
            "research",
            "data",
            "statistics",
            "was",
            "is",
            "are",
            "were",
            "happened",
            "occurred",
        ]
        if any(indicator in claim_lower for indicator in factual_indicators):
            return "factual"

        return "general"

    async def _check_knowledge_mound(self, claim: str, context: str) -> list[VerificationEvidence]:
        """Check claim against Knowledge Mound."""
        evidence: list[VerificationEvidence] = []

        try:
            from aragora.knowledge.mound import get_knowledge_mound

            mound = get_knowledge_mound()
            if not mound:
                return evidence

            # Query consensus
            if hasattr(mound, "query_consensus"):
                query = f"{claim} {context}".strip()
                results = await mound.query_consensus(
                    query=query,
                    limit=5,
                    min_confidence=0.3,
                )

                for r in results:
                    position = getattr(r, "position", str(r))
                    confidence = getattr(r, "confidence", 0.5)

                    # Determine if this supports or refutes the claim
                    supports = self._determine_support(claim, position)

                    evidence.append(
                        VerificationEvidence(
                            source="knowledge_mound_consensus",
                            content=position,
                            relevance=confidence,
                            supports_claim=supports,
                            metadata={
                                "debate_id": getattr(r, "debate_id", None),
                                "topic": getattr(r, "topic", None),
                            },
                        )
                    )

            # Query evidence records
            if hasattr(mound, "query_evidence"):
                results = await mound.query_evidence(
                    query=claim,
                    limit=5,
                    min_relevance=0.3,
                )

                for r in results:
                    evidence.append(
                        VerificationEvidence(
                            source="knowledge_mound_evidence",
                            content=getattr(r, "claim", str(r)),
                            relevance=getattr(r, "relevance", 0.5),
                            supports_claim=None,  # Neutral evidence
                            url=getattr(r, "url", None),
                        )
                    )

        except ImportError:
            logger.debug("Knowledge Mound not available")
        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.warning(f"Knowledge Mound query error: {e}")

        return evidence

    async def _check_web_sources(self, claim: str) -> list[VerificationEvidence]:
        """Check claim against web sources using existing web search skill."""
        evidence = []

        try:
            # Use web search to find relevant sources
            from .web_search import WebSearchSkill

            search_skill = WebSearchSkill()
            from ..base import SkillContext

            search_context = SkillContext()

            # Search for the claim
            result = await search_skill.execute(
                {"query": f'fact check "{claim}"', "max_results": 5},
                search_context,
            )

            if result.success and result.data:
                for item in result.data.get("results", []):
                    evidence.append(
                        VerificationEvidence(
                            source="web_search",
                            content=item.get("snippet", ""),
                            relevance=item.get("relevance_score", 0.5),
                            supports_claim=None,
                            url=item.get("url"),
                            metadata={"title": item.get("title")},
                        )
                    )

        except ImportError:
            logger.debug("Web search skill not available")
        except (RuntimeError, OSError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Web search error: {e}")

        return evidence

    async def _check_debate_history(
        self, claim: str, context: SkillContext
    ) -> list[VerificationEvidence]:
        """Check claim against historical debate outcomes."""
        evidence: list[VerificationEvidence] = []

        if not context.debate_context:
            return evidence

        try:
            # Check if there's relevant debate history
            from aragora.memory.consensus import ConsensusMemory

            memory = ConsensusMemory()

            # Search for related debates
            if hasattr(memory, "search"):
                results = await memory.search(claim, limit=3)
                for r in results:
                    evidence.append(
                        VerificationEvidence(
                            source="debate_history",
                            content=getattr(r, "consensus", str(r)),
                            relevance=getattr(r, "relevance", 0.5),
                            supports_claim=None,
                            metadata={
                                "debate_id": getattr(r, "debate_id", None),
                                "participants": getattr(r, "participants", []),
                            },
                        )
                    )

        except ImportError:
            logger.debug("Consensus memory not available")
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(f"Debate history check error: {e}")

        return evidence

    def _determine_support(self, claim: str, evidence_text: str) -> bool | None:
        """Determine if evidence supports, refutes, or is neutral to claim."""
        if not evidence_text:
            return None

        claim_lower = claim.lower()
        evidence_lower = evidence_text.lower()

        # Simple heuristic: check for contradiction indicators
        contradiction_words = [
            "however",
            "but",
            "false",
            "incorrect",
            "wrong",
            "not true",
            "debunked",
            "myth",
            "actually",
            "contrary",
        ]

        support_words = [
            "confirmed",
            "verified",
            "true",
            "correct",
            "accurate",
            "indeed",
            "supports",
            "evidence shows",
        ]

        # Check for explicit contradiction
        for word in contradiction_words:
            if word in evidence_lower:
                # Check if it's in context of the claim
                return False

        # Check for explicit support
        for word in support_words:
            if word in evidence_lower:
                return True

        # Check semantic similarity (simple word overlap)
        claim_words = set(claim_lower.split())
        evidence_words = set(evidence_lower.split())
        overlap = len(claim_words & evidence_words) / max(len(claim_words), 1)

        if overlap > 0.5:
            return True  # High overlap suggests support

        return None  # Neutral

    def _analyze_evidence(
        self, claim: str, evidence: list[VerificationEvidence]
    ) -> VerificationResult:
        """Analyze collected evidence and determine verdict."""
        if not evidence:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.UNVERIFIABLE,
                confidence=0.3,
                explanation="No relevant evidence found to verify this claim.",
                evidence=[],
            )

        # Count supporting and refuting evidence
        supporting = [e for e in evidence if e.supports_claim is True]
        refuting = [e for e in evidence if e.supports_claim is False]
        neutral = [e for e in evidence if e.supports_claim is None]

        # Calculate weighted scores
        support_score = sum(e.relevance for e in supporting)
        refute_score = sum(e.relevance for e in refuting)
        total_score = support_score + refute_score + sum(e.relevance for e in neutral)

        if total_score == 0:
            total_score = 1  # Avoid division by zero

        # Determine verdict
        if support_score > refute_score * 2:
            status = VerificationStatus.VERIFIED
            confidence = min(0.9, support_score / total_score + 0.3)
            explanation = f"Found {len(supporting)} supporting evidence items. The claim appears to be accurate based on available sources."
        elif refute_score > support_score * 2:
            status = VerificationStatus.REFUTED
            confidence = min(0.9, refute_score / total_score + 0.3)
            explanation = f"Found {len(refuting)} contradicting evidence items. The claim appears to be inaccurate or misleading."
        elif supporting and refuting:
            status = VerificationStatus.PARTIALLY_TRUE
            confidence = 0.5
            explanation = f"Mixed evidence found: {len(supporting)} supporting, {len(refuting)} refuting. The claim may be partially true or require more context."
        elif neutral and not supporting and not refuting:
            status = VerificationStatus.NEEDS_CONTEXT
            confidence = 0.4
            explanation = "Found related information but cannot definitively verify or refute. Additional context may be needed."
        else:
            status = VerificationStatus.UNVERIFIABLE
            confidence = 0.3
            explanation = "Unable to find sufficient evidence to verify this claim."

        # Sort evidence by relevance
        sorted_evidence = sorted(evidence, key=lambda e: e.relevance, reverse=True)

        return VerificationResult(
            claim=claim,
            status=status,
            confidence=round(confidence, 2),
            explanation=explanation,
            evidence=sorted_evidence[: self._max_evidence],
        )


# Skill instance for registration
SKILLS = [FactCheckSkill()]
