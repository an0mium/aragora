"""
Evidence Collector.

Auto-collects citations and snippets from existing connectors
to provide factual grounding for debates.
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

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

    def to_text_block(self) -> str:
        """Format as a text block for debate context."""
        return f"""EVID-{self.id}:
Source: {self.source} ({self.reliability_score:.1f} reliability)
Title: {self.title}
Snippet: {self.snippet[:500]}{"..." if len(self.snippet) > 500 else ""}
URL: {self.url}
---"""


@dataclass
class EvidencePack:
    """A collection of evidence snippets for a debate."""

    topic_keywords: List[str]
    snippets: List[EvidenceSnippet]
    search_timestamp: datetime = field(default_factory=datetime.now)
    total_searched: int = 0

    def to_context_string(self) -> str:
        """Convert to a formatted context string for debate."""
        if not self.snippets:
            return "No relevant evidence found."

        header = f"EVIDENCE PACK (collected {self.search_timestamp.isoformat()}):\n"
        header += f"Search terms: {', '.join(self.topic_keywords)}\n"
        header += f"Total sources searched: {self.total_searched}\n\n"

        evidence_blocks = [snippet.to_text_block() for snippet in self.snippets]
        return header + "\n".join(evidence_blocks) + "\n\nEND EVIDENCE PACK\n"


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

        # Extract keywords from task for search
        keywords = self._extract_keywords(task)
        print(f"Evidence collection: searching for keywords {keywords}")

        all_snippets = []
        total_searched = 0

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
                    print(f"Connector search error: {result}")
                else:
                    connector_snippets, searched_count = result
                    all_snippets.extend(connector_snippets)
                    total_searched += searched_count

        # Rank and limit snippets
        ranked_snippets = self._rank_snippets(all_snippets, keywords)[:self.max_total_snippets]

        # Record in provenance
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
            print(f"Error searching {connector_name}: {e}")
            return [], 0

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

    def _rank_snippets(self, snippets: List[EvidenceSnippet], keywords: List[str]) -> List[EvidenceSnippet]:
        """Rank snippets by relevance to keywords and reliability."""
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

            # Combine relevance and reliability
            return relevance_score * 0.7 + snippet.reliability_score * 0.3

        return sorted(snippets, key=score_snippet, reverse=True)