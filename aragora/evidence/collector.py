"""
Evidence Collector.

Auto-collects citations and snippets from existing connectors
to provide factual grounding for debates.
"""

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

from aragora.connectors.base import Connector
from aragora.core import Message, Environment
from aragora.reasoning.provenance import ProvenanceManager


@dataclass
class EvidenceSnippet:
    """A piece of evidence from a connector."""

    id: str
    source: str  # "local_docs", "github", etc.
    title: str
    snippet: str
    url: str = ""
    reliability_score: float = 0.5  # 0-1, based on source trustworthiness
    metadata: Dict[str, Any] = field(default_factory=dict)
    fetched_at: datetime = field(default_factory=datetime.now)

    @property
    def freshness_score(self) -> float:
        """Calculate freshness score (1.0 = very fresh, 0.0 = stale).

        Evidence degrades over time:
        - < 1 hour: 1.0 (very fresh)
        - 1-24 hours: 0.9-0.7
        - 1-7 days: 0.7-0.5
        - > 7 days: 0.5-0.3
        """
        age_seconds = (datetime.now() - self.fetched_at).total_seconds()
        age_hours = age_seconds / 3600

        if age_hours < 1:
            return 1.0
        elif age_hours < 24:
            return 0.9 - (age_hours / 24) * 0.2
        elif age_hours < 168:  # 7 days
            return 0.7 - ((age_hours - 24) / 144) * 0.2
        else:
            return max(0.3, 0.5 - (age_hours - 168) / 720 * 0.2)

    @property
    def combined_score(self) -> float:
        """Combined reliability and freshness score."""
        return self.reliability_score * 0.7 + self.freshness_score * 0.3

    def to_text_block(self) -> str:
        """Format as a text block for debate context."""
        freshness_indicator = "🟢" if self.freshness_score > 0.8 else "🟡" if self.freshness_score > 0.5 else "🔴"
        return f"""EVID-{self.id}:
Source: {self.source} ({self.reliability_score:.1f} reliability, {freshness_indicator} {self.freshness_score:.1f} fresh)
Title: {self.title}
Snippet: {self.snippet[:500]}{"..." if len(self.snippet) > 500 else ""}
URL: {self.url}
---"""

    def to_citation(self) -> str:
        """Format as an academic-style citation.

        Returns a formatted citation string like:
        [1] Title. Source (reliability: 0.9). URL
        """
        url_part = f" {self.url}" if self.url else ""
        return f"[{self.id}] {self.title}. {self.source.title()} (reliability: {self.reliability_score:.1f}).{url_part}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "source": self.source,
            "title": self.title,
            "snippet": self.snippet,
            "url": self.url,
            "reliability_score": self.reliability_score,
            "freshness_score": self.freshness_score,
            "combined_score": self.combined_score,
            "fetched_at": self.fetched_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class EvidencePack:
    """A collection of evidence snippets for a debate."""

    topic_keywords: List[str]
    snippets: List[EvidenceSnippet]
    search_timestamp: datetime = field(default_factory=datetime.now)
    total_searched: int = 0

    @property
    def average_reliability(self) -> float:
        """Average reliability score across all snippets."""
        if not self.snippets:
            return 0.0
        return sum(s.reliability_score for s in self.snippets) / len(self.snippets)

    @property
    def average_freshness(self) -> float:
        """Average freshness score across all snippets."""
        if not self.snippets:
            return 0.0
        return sum(s.freshness_score for s in self.snippets) / len(self.snippets)

    def to_context_string(self) -> str:
        """Convert to a formatted context string for debate."""
        if not self.snippets:
            return "No relevant evidence found."

        header = f"EVIDENCE PACK (collected {self.search_timestamp.isoformat()}):\n"
        header += f"Search terms: {', '.join(self.topic_keywords)}\n"
        header += f"Total sources searched: {self.total_searched}\n"
        header += f"Quality: {self.average_reliability:.1%} reliability, {self.average_freshness:.1%} fresh\n\n"

        evidence_blocks = [snippet.to_text_block() for snippet in self.snippets]
        return header + "\n".join(evidence_blocks) + "\n\nEND EVIDENCE PACK\n"

    def to_bibliography(self) -> str:
        """Format all evidence as an academic bibliography.

        Returns numbered citation list for appending to debate output.
        """
        if not self.snippets:
            return ""

        lines = ["## References\n"]
        for i, snippet in enumerate(self.snippets, 1):
            lines.append(f"{i}. {snippet.to_citation()}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "topic_keywords": self.topic_keywords,
            "snippets": [s.to_dict() for s in self.snippets],
            "search_timestamp": self.search_timestamp.isoformat(),
            "total_searched": self.total_searched,
            "average_reliability": self.average_reliability,
            "average_freshness": self.average_freshness,
        }


class EvidenceCollector:
    """Collects evidence from multiple connectors for debate grounding."""

    def __init__(self, connectors: Optional[Dict[str, Connector]] = None):
        self.connectors = connectors or {}
        self.provenance_manager = ProvenanceManager()
        self.max_snippets_per_connector = 3
        self.max_total_snippets = 8
        self.snippet_max_length = 1000

    def add_connector(self, name: str, connector: Connector):
        """Add a connector for evidence collection."""
        self.connectors[name] = connector

    async def collect_evidence(self, task: str, enabled_connectors: List[str] = None) -> EvidencePack:
        """Collect evidence relevant to the task."""
        if enabled_connectors is None:
            enabled_connectors = list(self.connectors.keys())

        all_snippets = []
        total_searched = 0

        # NEW: First, fetch any explicit URLs mentioned in the task
        explicit_urls = self._extract_urls(task)
        if explicit_urls:
            logger.info(f"Fetching {len(explicit_urls)} explicit URL(s): {explicit_urls}")
            if "web" in self.connectors:
                web_connector = self.connectors["web"]
                for url in explicit_urls:
                    try:
                        # Normalize URL
                        full_url = url if url.startswith(('http://', 'https://')) else f'https://{url}'
                        if hasattr(web_connector, 'fetch_url'):
                            evidence = await web_connector.fetch_url(full_url)
                            if evidence and getattr(evidence, 'confidence', 0) > 0:
                                snippet = EvidenceSnippet(
                                    id=f"url_{hashlib.sha256(full_url.encode()).hexdigest()[:12]}",
                                    source="direct_url",
                                    title=getattr(evidence, 'title', full_url),
                                    snippet=self._truncate_snippet(evidence.content),
                                    url=full_url,
                                    reliability_score=0.9,  # High reliability for direct URL fetch
                                    metadata={"fetched_directly": True, "original_url": url}
                                )
                                all_snippets.append(snippet)
                                total_searched += 1
                                logger.debug(f"Fetched: {full_url} ({len(evidence.content)} chars)")
                    except Exception as e:
                        logger.warning(f"Failed to fetch {url}: {e}")

        # Extract keywords from task for search
        keywords = self._extract_keywords(task)
        logger.info(f"Searching for keywords: {keywords}")

        # Search all enabled connectors concurrently
        search_tasks = []
        for connector_name in enabled_connectors:
            if connector_name in self.connectors:
                connector = self.connectors[connector_name]
                search_tasks.append(self._search_connector(connector_name, connector, keywords))

        if search_tasks:
            results = await asyncio.gather(*search_tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Connector search error: {result}")
                else:
                    connector_snippets, searched_count = result
                    all_snippets.extend(connector_snippets)
                    total_searched += searched_count

        # Rank and limit snippets
        ranked_snippets = self._rank_snippets(all_snippets, keywords)[:self.max_total_snippets]

        # Record in provenance (optional - method may not exist yet)
        if hasattr(self.provenance_manager, 'record_evidence_use'):
            for snippet in ranked_snippets:
                self.provenance_manager.record_evidence_use(snippet.id, task, "debate_context")

        return EvidencePack(
            topic_keywords=keywords,
            snippets=ranked_snippets,
            total_searched=total_searched
        )

    async def _search_connector(
        self,
        connector_name: str,
        connector: Connector,
        keywords: List[str]
    ) -> Tuple[List[EvidenceSnippet], int]:
        """Search a single connector and return snippets."""
        try:
            # Build search query from keywords
            query = " ".join(keywords[:3])  # Use top 3 keywords

            # Call connector search (assuming it has a search method)
            if hasattr(connector, 'search'):
                results = await connector.search(query, limit=self.max_snippets_per_connector)
            else:
                # Fallback for connectors without search
                results = []

            snippets = []
            for i, result in enumerate(results[:self.max_snippets_per_connector]):
                # Handle both Evidence objects (from WebConnector) and dict results (from other connectors)
                if hasattr(result, 'title'):  # Evidence object
                    snippet = EvidenceSnippet(
                        id=f"{connector_name}_{result.id}",
                        source=connector_name,
                        title=result.title,
                        snippet=self._truncate_snippet(result.content),
                        url=result.url or '',
                        reliability_score=self._calculate_reliability_from_evidence(connector_name, result),
                        metadata=result.metadata
                    )
                else:  # Dict result from other connectors
                    snippet = EvidenceSnippet(
                        id=f"{connector_name}_{i}",
                        source=connector_name,
                        title=result.get('title', result.get('name', 'Unknown')),
                        snippet=self._truncate_snippet(result.get('content', result.get('text', ''))),
                        url=result.get('url', ''),
                        reliability_score=self._calculate_reliability(connector_name, result),
                        metadata=result
                    )
                snippets.append(snippet)

            return snippets, len(results)

        except Exception as e:
            logger.warning(f"Error searching {connector_name}: {e}")
            return [], 0

    def _extract_urls(self, task: str) -> List[str]:
        """Extract explicit URLs and domain references from task description."""
        patterns = [
            r'https?://[^\s)<>\[\]]+',  # Full URLs (http/https)
            r'www\.[^\s)<>\[\]]+',  # www URLs
            r'\b([a-zA-Z0-9][-a-zA-Z0-9]*\.(?:com|org|net|io|ai|dev|app|co|edu|gov))\b',  # Domain names
        ]
        urls = []
        for pattern in patterns:
            matches = re.findall(pattern, task, re.IGNORECASE)
            urls.extend(matches)
        # Deduplicate while preserving order
        seen = set()
        unique_urls = []
        for url in urls:
            url_lower = url.lower()
            if url_lower not in seen:
                seen.add(url_lower)
                unique_urls.append(url)
        return unique_urls

    def _extract_keywords(self, task: str) -> List[str]:
        """Extract search keywords from task description."""
        # Simple keyword extraction - split and filter
        words = re.findall(r'\b\w+\b', task.lower())

        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall'}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]

        # Get unique keywords, prioritize nouns/important terms
        unique_keywords = list(set(keywords))

        # Boost keywords that appear in task title or are technical
        boosted = []
        for keyword in unique_keywords:
            if any(char in keyword for char in ['#', 'ai', 'tech', 'data', 'system', 'code']):
                boosted.extend([keyword] * 2)  # Duplicate for higher weight
            else:
                boosted.append(keyword)

        return boosted[:5]  # Top 5 keywords

    def _truncate_snippet(self, text: str) -> str:
        """Truncate snippet to max length, trying to end at sentence boundary."""
        if len(text) <= self.snippet_max_length:
            return text

        truncated = text[:self.snippet_max_length]

        # Try to find sentence end
        last_sentence_end = max(
            truncated.rfind('. '),
            truncated.rfind('! '),
            truncated.rfind('? ')
        )

        if last_sentence_end > self.snippet_max_length * 0.7:  # If we can keep most of it
            return truncated[:last_sentence_end + 1]

        return truncated + "..."

    def _calculate_reliability(self, connector_name: str, result: Dict[str, Any]) -> float:
        """Calculate reliability score based on source and metadata."""
        base_scores = {
            'github': 0.8,  # Code/docs from GitHub
            'local_docs': 0.9,  # Local documentation
            'web_search': 0.6,  # General web results
            'academic': 0.9,  # Academic sources
        }

        base_score = base_scores.get(connector_name, 0.5)

        # Adjust based on metadata
        if result.get('verified', False):
            base_score += 0.1
        if result.get('recent', False):
            base_score += 0.05
        if len(result.get('content', '')) > 1000:  # Substantial content
            base_score += 0.05

        return min(1.0, base_score)

    def _calculate_reliability_from_evidence(self, connector_name: str, evidence) -> float:
        """Calculate reliability score from Evidence object."""
        base_scores = {
            'github': 0.8,
            'local_docs': 0.9,
            'web': 0.6,  # WebConnector uses 'web' as source
            'academic': 0.9,
        }

        base_score = base_scores.get(connector_name, 0.5)

        # Use evidence authority and confidence
        base_score = (base_score + evidence.authority + evidence.confidence) / 3.0

        # Adjust based on content length
        if len(evidence.content) > 1000:
            base_score += 0.05

        return min(1.0, base_score)

    def _rank_snippets(self, snippets: List[EvidenceSnippet], keywords: List[str]) -> List[EvidenceSnippet]:
        """Rank snippets by relevance, reliability, and freshness."""
        def score_snippet(snippet: EvidenceSnippet) -> float:
            relevance_score = 0
            text_lower = (snippet.title + " " + snippet.snippet).lower()

            for keyword in keywords:
                if keyword.lower() in text_lower:
                    relevance_score += 1

            # Boost for keyword matches in title
            title_lower = snippet.title.lower()
            for keyword in keywords:
                if keyword.lower() in title_lower:
                    relevance_score += 0.5

            # Normalize relevance (max ~5 keywords)
            relevance_normalized = min(1.0, relevance_score / 5)

            # Combined scoring: relevance (50%), reliability (35%), freshness (15%)
            return (
                relevance_normalized * 0.50 +
                snippet.reliability_score * 0.35 +
                snippet.freshness_score * 0.15
            )

        return sorted(snippets, key=score_snippet, reverse=True)