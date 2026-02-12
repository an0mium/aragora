"""
Essay Synthesis Pipeline - Transform conversations into structured essays.

This pipeline enables users to:
1. Ingest conversation exports from ChatGPT/Claude
2. Extract intellectual claims and positions
3. Cluster claims by topic/theme
4. Run multi-agent debate to stress-test positions
5. Find scholarly attribution for claims
6. Synthesize into a structured long-form essay

Usage:
    from aragora.pipelines.essay_synthesis import EssaySynthesisPipeline

    pipeline = EssaySynthesisPipeline()

    # Load conversation exports
    pipeline.load_conversations("/path/to/exports")

    # Extract and cluster claims
    claims = pipeline.extract_all_claims()
    clusters = pipeline.cluster_claims(claims)

    # Stress-test claims via debate
    debate_results = await pipeline.stress_test_claims(claims)

    # Find scholarly attribution
    attributed_claims = await pipeline.find_attribution(claims)

    # Synthesize essay
    essay = await pipeline.synthesize_essay(
        title="On Systems and Narratives",
        clusters=clusters,
        attributed_claims=attributed_claims,
    )
"""

from __future__ import annotations

__all__ = [
    "EssaySynthesisPipeline",
    "TopicCluster",
    "AttributedClaim",
    "EssayOutline",
    "EssaySection",
    "SynthesisConfig",
]

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from aragora.connectors.conversation_ingestor import (
    ConversationIngestorConnector,
    ClaimExtraction,
    Conversation,
    ConversationExport,
)
from aragora.connectors.base import Evidence

logger = logging.getLogger(__name__)


@dataclass
class SynthesisConfig:
    """Configuration for essay synthesis."""

    # Claim extraction
    min_claim_length: int = 50
    max_claims_per_conversation: int = 20
    claim_confidence_threshold: float = 0.5

    # Topic clustering
    min_cluster_size: int = 3
    max_clusters: int = 20
    similarity_threshold: float = 0.3

    # Attribution
    max_sources_per_claim: int = 5
    min_attribution_confidence: float = 0.6
    academic_source_priority: bool = True

    # Essay structure
    target_word_count: int = 50000  # ~100 pages
    section_word_target: int = 5000
    max_sections: int = 15
    include_counterarguments: bool = True
    include_methodology: bool = True


@dataclass
class TopicCluster:
    """A cluster of related claims around a topic."""

    id: str
    name: str
    description: str
    keywords: list[str]
    claims: list[ClaimExtraction]
    coherence_score: float = 0.0
    representative_claim: ClaimExtraction | None = None
    related_clusters: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def claim_count(self) -> int:
        return len(self.claims)

    @property
    def average_confidence(self) -> float:
        if not self.claims:
            return 0.0
        return sum(c.confidence for c in self.claims) / len(self.claims)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "keywords": self.keywords,
            "claim_count": self.claim_count,
            "average_confidence": self.average_confidence,
            "coherence_score": self.coherence_score,
            "representative_claim": self.representative_claim.to_dict() if self.representative_claim else None,
            "claims": [c.to_dict() for c in self.claims],
            "related_clusters": self.related_clusters,
            "metadata": self.metadata,
        }


@dataclass
class AttributedClaim:
    """A claim with scholarly attribution."""

    claim: ClaimExtraction
    sources: list[Evidence]
    attribution_confidence: float
    supporting_quotes: list[str] = field(default_factory=list)
    contradicting_sources: list[Evidence] = field(default_factory=list)
    synthesis_notes: str = ""
    scholarly_context: str = ""

    def to_dict(self) -> dict:
        return {
            "claim": self.claim.to_dict(),
            "sources": [{"id": s.id, "title": s.title, "content": s.content[:500]} for s in self.sources],
            "attribution_confidence": self.attribution_confidence,
            "supporting_quotes": self.supporting_quotes,
            "contradicting_source_count": len(self.contradicting_sources),
            "synthesis_notes": self.synthesis_notes,
            "scholarly_context": self.scholarly_context,
        }


@dataclass
class EssaySection:
    """A section of the synthesized essay."""

    id: str
    title: str
    level: int  # 1 = chapter, 2 = section, 3 = subsection
    content: str
    word_count: int
    claims_referenced: list[str]  # Claim IDs
    sources_cited: list[str]  # Source IDs
    subsections: list["EssaySection"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "level": self.level,
            "word_count": self.word_count,
            "claims_referenced": self.claims_referenced,
            "sources_cited": self.sources_cited,
            "subsections": [s.to_dict() for s in self.subsections],
        }


@dataclass
class EssayOutline:
    """Outline structure for the essay."""

    title: str
    thesis: str
    sections: list[EssaySection]
    target_word_count: int
    bibliography: list[Evidence] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    @property
    def total_words(self) -> int:
        def count_section(section: EssaySection) -> int:
            return section.word_count + sum(count_section(s) for s in section.subsections)

        return sum(count_section(s) for s in self.sections)

    @property
    def section_count(self) -> int:
        def count_sections(section: EssaySection) -> int:
            return 1 + sum(count_sections(s) for s in section.subsections)

        return sum(count_sections(s) for s in self.sections)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "thesis": self.thesis,
            "total_words": self.total_words,
            "section_count": self.section_count,
            "sections": [s.to_dict() for s in self.sections],
            "bibliography_count": len(self.bibliography),
            "generated_at": self.generated_at.isoformat(),
            "metadata": self.metadata,
        }


class EssaySynthesisPipeline:
    """
    Pipeline for synthesizing essays from conversation exports.

    Integrates conversation ingestion, claim extraction, topic clustering,
    multi-agent debate, and scholarly attribution to produce long-form essays.
    """

    def __init__(
        self,
        config: SynthesisConfig | None = None,
        connectors: dict[str, Any] | None = None,
    ):
        self.config = config or SynthesisConfig()
        self.ingestor = ConversationIngestorConnector()
        self.connectors = connectors or {}

        # State
        self._claims: list[ClaimExtraction] = []
        self._clusters: list[TopicCluster] = []
        self._attributed_claims: dict[str, AttributedClaim] = {}

    # =========================================================================
    # Conversation Loading
    # =========================================================================

    def load_conversations(self, path: str | Path) -> list[ConversationExport]:
        """
        Load conversation exports from a file or directory.

        Args:
            path: Path to export file or directory

        Returns:
            List of loaded ConversationExports
        """
        path = Path(path)
        if path.is_file():
            return [self.ingestor.load_export(path)]
        elif path.is_dir():
            return self.ingestor.load_directory(path)
        else:
            raise FileNotFoundError(f"Path not found: {path}")

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about loaded conversations."""
        stats = self.ingestor.get_statistics()
        stats["claims_extracted"] = len(self._claims)
        stats["clusters_created"] = len(self._clusters)
        stats["claims_attributed"] = len(self._attributed_claims)
        return stats

    # =========================================================================
    # Claim Extraction
    # =========================================================================

    def extract_all_claims(
        self,
        custom_patterns: list[str] | None = None,
    ) -> list[ClaimExtraction]:
        """
        Extract claims from all loaded conversations.

        Args:
            custom_patterns: Additional regex patterns for extraction

        Returns:
            List of extracted claims
        """
        claims = self.ingestor.extract_claims(
            min_length=self.config.min_claim_length,
            patterns=custom_patterns,
            include_assistant=self.config.include_counterarguments,
        )

        # Filter by confidence threshold
        claims = [c for c in claims if c.confidence >= self.config.claim_confidence_threshold]

        # Deduplicate similar claims
        claims = self._deduplicate_claims(claims)

        self._claims = claims
        logger.info(f"Extracted {len(claims)} unique claims")
        return claims

    def _deduplicate_claims(self, claims: list[ClaimExtraction]) -> list[ClaimExtraction]:
        """Remove duplicate or near-duplicate claims."""
        seen_content = set()
        unique_claims = []

        for claim in claims:
            # Normalize for comparison
            normalized = re.sub(r"\s+", " ", claim.claim.lower().strip())

            # Simple duplicate check
            if normalized not in seen_content:
                seen_content.add(normalized)
                unique_claims.append(claim)

        return unique_claims

    # =========================================================================
    # Topic Clustering
    # =========================================================================

    def cluster_claims(
        self,
        claims: list[ClaimExtraction] | None = None,
    ) -> list[TopicCluster]:
        """
        Cluster claims by topic using keyword overlap.

        This is a simple keyword-based clustering. For production,
        consider using embeddings (e.g., sentence-transformers).

        Args:
            claims: Claims to cluster (uses extracted claims if not provided)

        Returns:
            List of TopicClusters
        """
        claims = claims or self._claims
        if not claims:
            raise ValueError("No claims to cluster. Run extract_all_claims first.")

        # Extract keywords from each claim
        claim_keywords = {}
        for claim in claims:
            keywords = self._extract_keywords(claim.claim)
            claim_keywords[claim.claim] = keywords

        # Build keyword -> claims mapping
        keyword_claims: dict[str, list[ClaimExtraction]] = {}
        for claim in claims:
            for keyword in claim_keywords[claim.claim]:
                if keyword not in keyword_claims:
                    keyword_claims[keyword] = []
                keyword_claims[keyword].append(claim)

        # Find clusters around high-frequency keywords
        sorted_keywords = sorted(keyword_claims.items(), key=lambda x: -len(x[1]))

        clusters: list[TopicCluster] = []
        clustered_claims: set[str] = set()

        for keyword, keyword_claim_list in sorted_keywords[:self.config.max_clusters]:
            # Skip if most claims already clustered
            unclustered = [c for c in keyword_claim_list if c.claim not in clustered_claims]
            if len(unclustered) < self.config.min_cluster_size:
                continue

            # Create cluster
            cluster = TopicCluster(
                id=f"cluster_{len(clusters):03d}",
                name=keyword.title(),
                description=f"Claims related to {keyword}",
                keywords=self._expand_keywords(keyword, claim_keywords),
                claims=unclustered,
                coherence_score=self._calculate_coherence(unclustered, claim_keywords),
            )

            # Set representative claim (highest confidence)
            if unclustered:
                cluster.representative_claim = max(unclustered, key=lambda c: c.confidence)

            clusters.append(cluster)
            clustered_claims.update(c.claim for c in unclustered)

        # Find cluster relationships
        self._link_related_clusters(clusters)

        self._clusters = clusters
        logger.info(f"Created {len(clusters)} topic clusters")
        return clusters

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract keywords from text."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "that", "this", "these", "those", "it", "its", "i", "you", "we", "they",
            "what", "which", "who", "how", "when", "where", "why", "think", "believe",
            "about", "more", "some", "any", "just", "only", "also", "very", "really",
        }

        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        return {w for w in words if w not in stopwords}

    def _expand_keywords(
        self,
        primary_keyword: str,
        claim_keywords: dict[str, set[str]],
    ) -> list[str]:
        """Find related keywords that co-occur with primary keyword."""
        co_occurrence: dict[str, int] = {}

        for claim_text, keywords in claim_keywords.items():
            if primary_keyword in keywords:
                for kw in keywords:
                    if kw != primary_keyword:
                        co_occurrence[kw] = co_occurrence.get(kw, 0) + 1

        # Return top co-occurring keywords
        sorted_keywords = sorted(co_occurrence.items(), key=lambda x: -x[1])
        return [primary_keyword] + [k for k, _ in sorted_keywords[:9]]

    def _calculate_coherence(
        self,
        claims: list[ClaimExtraction],
        claim_keywords: dict[str, set[str]],
    ) -> float:
        """Calculate coherence score for a cluster."""
        if len(claims) < 2:
            return 0.0

        # Average pairwise keyword overlap
        total_overlap = 0.0
        pairs = 0

        for i, c1 in enumerate(claims):
            for c2 in claims[i + 1 :]:
                kw1 = claim_keywords.get(c1.claim, set())
                kw2 = claim_keywords.get(c2.claim, set())
                if kw1 and kw2:
                    overlap = len(kw1 & kw2) / len(kw1 | kw2)
                    total_overlap += overlap
                    pairs += 1

        return total_overlap / pairs if pairs > 0 else 0.0

    def _link_related_clusters(self, clusters: list[TopicCluster]) -> None:
        """Find and link related clusters."""
        for i, c1 in enumerate(clusters):
            kw1 = set(c1.keywords)
            for c2 in clusters[i + 1 :]:
                kw2 = set(c2.keywords)
                overlap = len(kw1 & kw2) / len(kw1 | kw2) if kw1 | kw2 else 0

                if overlap >= self.config.similarity_threshold:
                    c1.related_clusters.append(c2.id)
                    c2.related_clusters.append(c1.id)

    # =========================================================================
    # Attribution Finding
    # =========================================================================

    async def find_attribution(
        self,
        claims: list[ClaimExtraction] | None = None,
        connectors: list[str] | None = None,
    ) -> list[AttributedClaim]:
        """
        Find scholarly attribution for claims.

        Searches academic and authoritative sources to find supporting
        or contradicting evidence for each claim.

        Args:
            claims: Claims to attribute (uses extracted claims if not provided)
            connectors: List of connector names to use (e.g., ["arxiv", "semantic_scholar"])

        Returns:
            List of AttributedClaims with sources
        """
        claims = claims or self._claims
        if not claims:
            raise ValueError("No claims to attribute. Run extract_all_claims first.")

        # Default connectors for academic attribution
        connector_names = connectors or ["arxiv", "semantic_scholar", "crossref"]

        attributed = []
        for claim in claims:
            # Generate search queries from claim
            queries = self._generate_search_queries(claim)

            # Search across connectors
            all_evidence = []
            for query in queries[:3]:  # Limit queries per claim
                for conn_name in connector_names:
                    if conn_name in self.connectors:
                        try:
                            results = await self.connectors[conn_name].search(
                                query, limit=self.config.max_sources_per_claim
                            )
                            all_evidence.extend(results)
                        except Exception as e:
                            logger.debug(f"Search failed for {conn_name}: {e}")

            # Score and filter evidence
            scored_evidence = self._score_evidence_relevance(claim, all_evidence)
            supporting = [e for e, score in scored_evidence if score > 0]
            contradicting = [e for e, score in scored_evidence if score < 0]

            # Calculate attribution confidence
            confidence = self._calculate_attribution_confidence(supporting, contradicting)

            attributed_claim = AttributedClaim(
                claim=claim,
                sources=supporting[: self.config.max_sources_per_claim],
                attribution_confidence=confidence,
                contradicting_sources=contradicting[:3],
            )

            attributed.append(attributed_claim)
            self._attributed_claims[claim.claim] = attributed_claim

        logger.info(f"Attributed {len(attributed)} claims")
        return attributed

    def _generate_search_queries(self, claim: ClaimExtraction) -> list[str]:
        """Generate search queries from a claim."""
        queries = []

        # Direct claim as query
        queries.append(claim.claim[:200])

        # Extract key phrases
        keywords = list(self._extract_keywords(claim.claim))
        if len(keywords) >= 3:
            queries.append(" ".join(keywords[:5]))

        # Topic + claim type
        if claim.topics:
            queries.append(f"{claim.topics[0]} {claim.claim_type}")

        return queries

    def _score_evidence_relevance(
        self,
        claim: ClaimExtraction,
        evidence: list[Evidence],
    ) -> list[tuple[Evidence, float]]:
        """Score evidence relevance to claim."""
        claim_keywords = self._extract_keywords(claim.claim)

        scored = []
        for e in evidence:
            if not e.content:
                continue

            evidence_keywords = self._extract_keywords(e.content)
            overlap = len(claim_keywords & evidence_keywords)

            # Simple relevance score
            score = overlap / max(len(claim_keywords), 1) if claim_keywords else 0
            score *= e.confidence  # Weight by source confidence

            scored.append((e, score))

        return sorted(scored, key=lambda x: -abs(x[1]))

    def _calculate_attribution_confidence(
        self,
        supporting: list[Evidence],
        contradicting: list[Evidence],
    ) -> float:
        """Calculate confidence in attribution."""
        if not supporting:
            return 0.0

        # Base confidence from number of sources
        base = min(len(supporting) / 3, 1.0) * 0.5

        # Average source confidence
        avg_confidence = sum(e.confidence for e in supporting) / len(supporting)

        # Penalty for contradicting sources
        contradiction_penalty = min(len(contradicting) / 5, 0.3)

        return max(0.0, min(1.0, base + avg_confidence * 0.5 - contradiction_penalty))

    # =========================================================================
    # Essay Synthesis
    # =========================================================================

    async def generate_outline(
        self,
        title: str,
        thesis: str | None = None,
        clusters: list[TopicCluster] | None = None,
    ) -> EssayOutline:
        """
        Generate essay outline from clusters.

        Args:
            title: Essay title
            thesis: Main thesis statement
            clusters: Topic clusters (uses created clusters if not provided)

        Returns:
            EssayOutline with sections
        """
        clusters = clusters or self._clusters
        if not clusters:
            raise ValueError("No clusters available. Run cluster_claims first.")

        # Sort clusters by coherence and claim count
        sorted_clusters = sorted(
            clusters,
            key=lambda c: (c.coherence_score * 0.3 + (c.claim_count / 20) * 0.7),
            reverse=True,
        )

        # Generate thesis if not provided
        if not thesis:
            thesis = self._generate_thesis(sorted_clusters)

        # Create sections from clusters
        sections = []
        words_per_section = self.config.target_word_count // min(
            len(sorted_clusters), self.config.max_sections
        )

        for i, cluster in enumerate(sorted_clusters[: self.config.max_sections]):
            section = EssaySection(
                id=f"section_{i:03d}",
                title=cluster.name,
                level=1,
                content="",  # To be filled during synthesis
                word_count=words_per_section,
                claims_referenced=[c.claim[:50] for c in cluster.claims[:10]],
                sources_cited=[],
            )

            # Add subsections for claim types
            claim_types = set(c.claim_type for c in cluster.claims)
            for claim_type in claim_types:
                type_claims = [c for c in cluster.claims if c.claim_type == claim_type]
                if len(type_claims) >= 2:
                    subsection = EssaySection(
                        id=f"section_{i:03d}_{claim_type}",
                        title=f"{claim_type.title()}s",
                        level=2,
                        content="",
                        word_count=words_per_section // max(len(claim_types), 1),
                        claims_referenced=[c.claim[:50] for c in type_claims[:5]],
                        sources_cited=[],
                    )
                    section.subsections.append(subsection)

            sections.append(section)

        # Add methodology section if configured
        if self.config.include_methodology:
            sections.insert(
                0,
                EssaySection(
                    id="section_methodology",
                    title="Methodology",
                    level=1,
                    content="",
                    word_count=self.config.section_word_target // 2,
                    claims_referenced=[],
                    sources_cited=[],
                ),
            )

        # Collect all sources for bibliography
        all_sources = []
        for claim_text, attr_claim in self._attributed_claims.items():
            all_sources.extend(attr_claim.sources)

        # Deduplicate sources
        seen_ids = set()
        unique_sources = []
        for source in all_sources:
            if source.id not in seen_ids:
                seen_ids.add(source.id)
                unique_sources.append(source)

        outline = EssayOutline(
            title=title,
            thesis=thesis,
            sections=sections,
            target_word_count=self.config.target_word_count,
            bibliography=unique_sources,
        )

        logger.info(f"Generated outline with {outline.section_count} sections")
        return outline

    def _generate_thesis(self, clusters: list[TopicCluster]) -> str:
        """Generate thesis statement from clusters."""
        if not clusters:
            return "This essay explores themes from personal intellectual discourse."

        # Use top cluster keywords
        top_keywords = []
        for cluster in clusters[:3]:
            top_keywords.extend(cluster.keywords[:2])

        if len(top_keywords) >= 3:
            return (
                f"This essay argues for a synthesis of {top_keywords[0]}, "
                f"{top_keywords[1]}, and {top_keywords[2]} as interconnected aspects "
                f"of a coherent worldview emerging from systematic reflection."
            )

        return "This essay synthesizes themes from extended intellectual dialogue."

    def export_for_synthesis(self, outline: EssayOutline) -> dict[str, Any]:
        """
        Export all data needed for essay synthesis.

        This creates a comprehensive package that can be used by
        a language model or human writer to synthesize the essay.

        Returns:
            Dict containing outline, claims, clusters, and sources
        """
        return {
            "outline": outline.to_dict(),
            "claims": [c.to_dict() for c in self._claims],
            "clusters": [c.to_dict() for c in self._clusters],
            "attributed_claims": {k: v.to_dict() for k, v in self._attributed_claims.items()},
            "statistics": self.get_statistics(),
            "config": {
                "target_word_count": self.config.target_word_count,
                "include_counterarguments": self.config.include_counterarguments,
                "include_methodology": self.config.include_methodology,
            },
        }


# Convenience function
def create_essay_pipeline(
    config: SynthesisConfig | None = None,
    **kwargs,
) -> EssaySynthesisPipeline:
    """Create an essay synthesis pipeline with optional configuration."""
    return EssaySynthesisPipeline(config=config, **kwargs)
