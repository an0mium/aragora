"""
Audit-Evidence Adapter.

Connects the document audit system to the evidence collection system,
enabling automatic evidence gathering for audit findings.

This provides:
- Automatic evidence collection for findings
- Evidence-backed confidence scoring
- Source attribution for audit reports
- Cross-reference validation

Usage:
    from aragora.audit.evidence_adapter import FindingEvidenceCollector

    collector = FindingEvidenceCollector()
    enriched_finding = await collector.enrich_finding(finding, document_content)

    # Or batch enrichment
    enriched_findings = await collector.enrich_findings_batch(findings, documents)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class EvidenceSource:
    """A source of evidence supporting a finding."""

    source_id: str
    source_type: str  # "document", "external", "cross_reference"
    title: str
    snippet: str
    location: str  # Page, line, chunk reference
    relevance_score: float
    reliability_score: float
    url: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "title": self.title,
            "snippet": self.snippet,
            "location": self.location,
            "relevance_score": self.relevance_score,
            "reliability_score": self.reliability_score,
            "url": self.url,
            "metadata": self.metadata,
        }


@dataclass
class EvidenceEnrichment:
    """Evidence enrichment data for a finding."""

    finding_id: str
    sources: list[EvidenceSource]
    original_confidence: float
    adjusted_confidence: float
    evidence_summary: str
    collected_at: datetime = field(default_factory=datetime.utcnow)
    collection_time_ms: int = 0

    @property
    def has_strong_evidence(self) -> bool:
        """Check if there is strong supporting evidence."""
        if not self.sources:
            return False
        avg_relevance = sum(s.relevance_score for s in self.sources) / len(self.sources)
        return avg_relevance > 0.7 and len(self.sources) >= 2

    @property
    def confidence_boost(self) -> float:
        """Get the confidence adjustment from evidence."""
        return self.adjusted_confidence - self.original_confidence

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "finding_id": self.finding_id,
            "sources": [s.to_dict() for s in self.sources],
            "original_confidence": self.original_confidence,
            "adjusted_confidence": self.adjusted_confidence,
            "evidence_summary": self.evidence_summary,
            "collected_at": self.collected_at.isoformat(),
            "collection_time_ms": self.collection_time_ms,
            "has_strong_evidence": self.has_strong_evidence,
        }


@dataclass
class EvidenceConfig:
    """Configuration for evidence collection."""

    # Collection settings
    max_sources_per_finding: int = 5
    min_relevance_threshold: float = 0.3
    enable_external_sources: bool = True
    enable_cross_reference: bool = True

    # Confidence adjustment
    evidence_weight: float = 0.3  # Weight of evidence in confidence calculation
    strong_evidence_boost: float = 0.15  # Boost for strong evidence
    weak_evidence_penalty: float = 0.1  # Penalty for contradictory evidence

    # Search settings
    search_window: int = 500  # Characters around finding to search
    max_parallel_searches: int = 5


class FindingEvidenceCollector:
    """
    Collects and associates evidence with audit findings.

    Enriches findings with supporting evidence from:
    - The source document itself (context)
    - Cross-references in other documents
    - External sources (when enabled)
    """

    def __init__(
        self,
        config: Optional[EvidenceConfig] = None,
        evidence_collector: Optional[Any] = None,  # EvidenceCollector
    ):
        """
        Initialize the finding evidence collector.

        Args:
            config: Evidence collection configuration
            evidence_collector: Optional EvidenceCollector instance
        """
        self.config = config or EvidenceConfig()
        self._evidence_collector = evidence_collector

    async def _get_evidence_collector(self):
        """Lazily initialize the evidence collector."""
        if self._evidence_collector is None:
            try:
                from aragora.evidence.collector import EvidenceCollector

                self._evidence_collector = EvidenceCollector()
            except ImportError:
                logger.warning("Evidence collector not available")
                return None
        return self._evidence_collector

    async def enrich_finding(
        self,
        finding: Any,  # AuditFinding
        document_content: Optional[str] = None,
        related_documents: Optional[dict[str, str]] = None,
    ) -> EvidenceEnrichment:
        """
        Enrich a finding with supporting evidence.

        Args:
            finding: The AuditFinding to enrich
            document_content: Content of the source document
            related_documents: Dict mapping doc IDs to content

        Returns:
            EvidenceEnrichment with collected sources
        """
        import time

        start_time = time.time()

        sources: list[EvidenceSource] = []
        finding_id = getattr(finding, "id", f"finding_{hashlib.sha256(finding.title.encode()).hexdigest()[:8]}")

        # 1. Collect evidence from the source document
        if document_content:
            doc_sources = self._extract_document_evidence(
                finding=finding,
                content=document_content,
                document_id=finding.document_id,
            )
            sources.extend(doc_sources)

        # 2. Cross-reference related documents
        if related_documents and self.config.enable_cross_reference:
            cross_sources = await self._collect_cross_references(
                finding=finding,
                documents=related_documents,
            )
            sources.extend(cross_sources)

        # 3. Collect external evidence (if enabled)
        if self.config.enable_external_sources:
            external_sources = await self._collect_external_evidence(finding)
            sources.extend(external_sources)

        # Rank and limit sources
        sources = self._rank_sources(sources, finding)
        sources = sources[: self.config.max_sources_per_finding]

        # Calculate adjusted confidence
        original_confidence = finding.confidence
        adjusted_confidence = self._calculate_adjusted_confidence(
            original=original_confidence,
            sources=sources,
        )

        # Generate evidence summary
        summary = self._generate_evidence_summary(finding, sources)

        collection_time = int((time.time() - start_time) * 1000)

        return EvidenceEnrichment(
            finding_id=finding_id,
            sources=sources,
            original_confidence=original_confidence,
            adjusted_confidence=adjusted_confidence,
            evidence_summary=summary,
            collection_time_ms=collection_time,
        )

    async def enrich_findings_batch(
        self,
        findings: Sequence[Any],  # Sequence[AuditFinding]
        documents: dict[str, str],  # doc_id -> content
        max_concurrent: int = 5,
    ) -> dict[str, EvidenceEnrichment]:
        """
        Enrich multiple findings in parallel.

        Args:
            findings: List of findings to enrich
            documents: Dict mapping document IDs to content
            max_concurrent: Maximum concurrent enrichment tasks

        Returns:
            Dict mapping finding IDs to enrichments
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results: dict[str, EvidenceEnrichment] = {}

        async def enrich_with_limit(finding: Any) -> tuple[str, EvidenceEnrichment]:
            async with semaphore:
                finding_id = getattr(finding, "id", str(hash(finding.title)))
                doc_content = documents.get(finding.document_id)

                # Get related documents (excluding the source)
                related = {
                    k: v for k, v in documents.items() if k != finding.document_id
                }

                enrichment = await self.enrich_finding(
                    finding=finding,
                    document_content=doc_content,
                    related_documents=related,
                )
                return finding_id, enrichment

        tasks = [enrich_with_limit(f) for f in findings]
        completed = await asyncio.gather(*tasks)

        for finding_id, enrichment in completed:
            results[finding_id] = enrichment

        return results

    def _extract_document_evidence(
        self,
        finding: Any,
        content: str,
        document_id: str,
    ) -> list[EvidenceSource]:
        """Extract evidence from the source document."""
        sources = []

        # Get the evidence text location if available
        evidence_text = getattr(finding, "evidence_text", "")
        evidence_location = getattr(finding, "evidence_location", "")

        # Search for the evidence in the document
        if evidence_text and evidence_text in content:
            # Find the position
            pos = content.find(evidence_text)
            if pos >= 0:
                # Get extended context
                start = max(0, pos - self.config.search_window // 2)
                end = min(len(content), pos + len(evidence_text) + self.config.search_window // 2)
                context = content[start:end]

                sources.append(
                    EvidenceSource(
                        source_id=f"doc_{document_id}_{pos}",
                        source_type="document",
                        title=f"Document: {document_id}",
                        snippet=context,
                        location=evidence_location or f"Position {pos}",
                        relevance_score=0.95,  # High relevance - exact match
                        reliability_score=0.90,  # Source document is reliable
                        metadata={
                            "document_id": document_id,
                            "position": pos,
                            "exact_match": True,
                        },
                    )
                )

        # Look for related content using finding keywords
        keywords = self._extract_keywords(finding.title + " " + finding.description)
        for keyword in keywords[:3]:
            occurrences = self._find_keyword_occurrences(content, keyword)
            for occ in occurrences[:2]:  # Max 2 per keyword
                if occ["text"] not in [s.snippet for s in sources]:  # Avoid duplicates
                    sources.append(
                        EvidenceSource(
                            source_id=f"doc_{document_id}_{occ['pos']}",
                            source_type="document",
                            title=f"Related context: '{keyword}'",
                            snippet=occ["text"],
                            location=f"Position {occ['pos']}",
                            relevance_score=0.70,
                            reliability_score=0.85,
                            metadata={
                                "document_id": document_id,
                                "keyword": keyword,
                            },
                        )
                    )

        return sources

    async def _collect_cross_references(
        self,
        finding: Any,
        documents: dict[str, str],
    ) -> list[EvidenceSource]:
        """Find cross-references in related documents."""
        sources = []

        # Extract key terms from the finding
        keywords = self._extract_keywords(finding.title + " " + finding.description)

        for doc_id, content in documents.items():
            # Search for keyword matches
            for keyword in keywords[:3]:
                occurrences = self._find_keyword_occurrences(content, keyword)
                if occurrences:
                    occ = occurrences[0]  # Best match
                    sources.append(
                        EvidenceSource(
                            source_id=f"xref_{doc_id}_{occ['pos']}",
                            source_type="cross_reference",
                            title=f"Cross-reference: {doc_id}",
                            snippet=occ["text"],
                            location=f"Document {doc_id}, position {occ['pos']}",
                            relevance_score=0.60,  # Lower than direct evidence
                            reliability_score=0.80,
                            metadata={
                                "source_doc": doc_id,
                                "keyword": keyword,
                            },
                        )
                    )

        return sources

    async def _collect_external_evidence(
        self,
        finding: Any,
    ) -> list[EvidenceSource]:
        """Collect evidence from external sources."""
        sources = []

        collector = await self._get_evidence_collector()
        if collector is None:
            return sources

        try:
            # Build search query from finding
            query = f"{finding.title} {finding.category}"

            # Collect evidence
            pack = await collector.collect_evidence(
                task=query,
                enabled_connectors=["web", "github"],
            )

            # Convert to EvidenceSource objects
            for snippet in pack.snippets[:3]:  # Limit external sources
                sources.append(
                    EvidenceSource(
                        source_id=f"ext_{snippet.id}",
                        source_type="external",
                        title=snippet.title,
                        snippet=snippet.snippet[:500],
                        location=snippet.url or snippet.source,
                        relevance_score=snippet.combined_score * 0.8,  # Discount external
                        reliability_score=snippet.reliability_score,
                        url=snippet.url,
                        metadata={
                            "source": snippet.source,
                            "fetched_at": snippet.fetched_at.isoformat(),
                        },
                    )
                )

        except Exception as e:
            logger.warning(f"Failed to collect external evidence: {e}")

        return sources

    def _rank_sources(
        self,
        sources: list[EvidenceSource],
        finding: Any,
    ) -> list[EvidenceSource]:
        """Rank sources by relevance and reliability."""

        def score(source: EvidenceSource) -> float:
            # Combined score with source type weighting
            type_weights = {
                "document": 1.0,  # Direct evidence from source doc
                "cross_reference": 0.8,
                "external": 0.6,
            }
            type_weight = type_weights.get(source.source_type, 0.5)

            return (
                source.relevance_score * 0.5
                + source.reliability_score * 0.3
                + type_weight * 0.2
            )

        return sorted(sources, key=score, reverse=True)

    def _calculate_adjusted_confidence(
        self,
        original: float,
        sources: list[EvidenceSource],
    ) -> float:
        """Calculate confidence adjusted by evidence."""
        if not sources:
            return original

        # Calculate evidence strength
        avg_relevance = sum(s.relevance_score for s in sources) / len(sources)
        avg_reliability = sum(s.reliability_score for s in sources) / len(sources)

        evidence_strength = avg_relevance * avg_reliability

        # Calculate adjustment
        if evidence_strength > 0.7 and len(sources) >= 2:
            # Strong evidence boost
            adjustment = self.config.strong_evidence_boost
        elif evidence_strength > 0.4:
            # Moderate evidence - small boost
            adjustment = evidence_strength * self.config.evidence_weight
        else:
            # Weak evidence - potential penalty
            adjustment = -self.config.weak_evidence_penalty * (1 - evidence_strength)

        # Apply adjustment
        adjusted = original + adjustment

        # Clamp to valid range
        return max(0.0, min(1.0, adjusted))

    def _generate_evidence_summary(
        self,
        finding: Any,
        sources: list[EvidenceSource],
    ) -> str:
        """Generate a summary of the collected evidence."""
        if not sources:
            return "No supporting evidence collected."

        doc_sources = [s for s in sources if s.source_type == "document"]
        xref_sources = [s for s in sources if s.source_type == "cross_reference"]
        ext_sources = [s for s in sources if s.source_type == "external"]

        parts = []

        if doc_sources:
            parts.append(f"{len(doc_sources)} source(s) from the document")

        if xref_sources:
            parts.append(f"{len(xref_sources)} cross-reference(s)")

        if ext_sources:
            parts.append(f"{len(ext_sources)} external source(s)")

        summary = f"Evidence collected: {', '.join(parts)}."

        # Add top source snippet
        if sources:
            top_source = sources[0]
            snippet_preview = top_source.snippet[:100] + "..." if len(top_source.snippet) > 100 else top_source.snippet
            summary += f" Top evidence: \"{snippet_preview}\""

        return summary

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text."""
        import re

        words = re.findall(r"\b\w+\b", text.lower())

        # Remove stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "can", "shall", "this", "that",
            "these", "those", "it", "its", "from", "about",
        }

        keywords = [w for w in words if len(w) > 3 and w not in stop_words]

        # Return unique keywords
        return list(dict.fromkeys(keywords))[:10]

    def _find_keyword_occurrences(
        self,
        content: str,
        keyword: str,
        max_occurrences: int = 3,
    ) -> list[dict[str, Any]]:
        """Find occurrences of a keyword with surrounding context."""
        import re

        occurrences = []
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)

        for match in pattern.finditer(content):
            pos = match.start()
            # Get context window
            start = max(0, pos - self.config.search_window // 2)
            end = min(len(content), pos + len(keyword) + self.config.search_window // 2)
            context = content[start:end]

            occurrences.append({
                "pos": pos,
                "text": context,
                "keyword": keyword,
            })

            if len(occurrences) >= max_occurrences:
                break

        return occurrences


# Convenience function
async def enrich_finding_with_evidence(
    finding: Any,
    document_content: Optional[str] = None,
    config: Optional[EvidenceConfig] = None,
) -> EvidenceEnrichment:
    """
    Quick function to enrich a finding with evidence.

    Args:
        finding: The AuditFinding to enrich
        document_content: Optional source document content
        config: Optional evidence configuration

    Returns:
        EvidenceEnrichment with collected sources
    """
    collector = FindingEvidenceCollector(config=config)
    return await collector.enrich_finding(finding, document_content)


__all__ = [
    "FindingEvidenceCollector",
    "EvidenceConfig",
    "EvidenceEnrichment",
    "EvidenceSource",
    "enrich_finding_with_evidence",
]
