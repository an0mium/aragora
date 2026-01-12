"""
Evidence Metadata Enrichment.

Enriches evidence snippets with additional metadata including:
- Source type classification
- Provenance tracking (author, publication date, etc.)
- Confidence scoring
- Temporal context
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class SourceType(str, Enum):
    """Classification of evidence source types."""

    WEB = "web"                    # General web content
    ACADEMIC = "academic"          # Academic papers, journals
    DOCUMENTATION = "documentation"  # Technical documentation
    NEWS = "news"                  # News articles
    SOCIAL = "social"              # Social media, forums
    CODE = "code"                  # Code repositories
    API = "api"                    # API responses
    DATABASE = "database"          # Database records
    LOCAL = "local"                # Local files
    UNKNOWN = "unknown"            # Unclassified


@dataclass
class Provenance:
    """Provenance information for evidence."""

    author: Optional[str] = None
    organization: Optional[str] = None
    publication_date: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    url: Optional[str] = None
    doi: Optional[str] = None  # Digital Object Identifier
    isbn: Optional[str] = None
    version: Optional[str] = None
    license: Optional[str] = None
    citation_count: Optional[int] = None
    peer_reviewed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "author": self.author,
            "organization": self.organization,
            "publication_date": self.publication_date.isoformat() if self.publication_date else None,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "url": self.url,
            "doi": self.doi,
            "isbn": self.isbn,
            "version": self.version,
            "license": self.license,
            "citation_count": self.citation_count,
            "peer_reviewed": self.peer_reviewed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Provenance":
        """Create from dictionary."""
        pub_date = data.get("publication_date")
        last_mod = data.get("last_modified")

        return cls(
            author=data.get("author"),
            organization=data.get("organization"),
            publication_date=datetime.fromisoformat(pub_date) if pub_date else None,
            last_modified=datetime.fromisoformat(last_mod) if last_mod else None,
            url=data.get("url"),
            doi=data.get("doi"),
            isbn=data.get("isbn"),
            version=data.get("version"),
            license=data.get("license"),
            citation_count=data.get("citation_count"),
            peer_reviewed=data.get("peer_reviewed", False),
        )


@dataclass
class EnrichedMetadata:
    """Enriched metadata for evidence snippets."""

    source_type: SourceType = SourceType.UNKNOWN
    provenance: Provenance = field(default_factory=Provenance)
    confidence: float = 0.5  # 0-1, confidence in the evidence
    timestamp: datetime = field(default_factory=datetime.now)
    language: str = "en"
    word_count: int = 0
    has_citations: bool = False
    has_code: bool = False
    has_data: bool = False
    topics: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)  # Named entities
    content_hash: str = ""  # For deduplication

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_type": self.source_type.value,
            "provenance": self.provenance.to_dict(),
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "language": self.language,
            "word_count": self.word_count,
            "has_citations": self.has_citations,
            "has_code": self.has_code,
            "has_data": self.has_data,
            "topics": self.topics,
            "entities": self.entities,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnrichedMetadata":
        """Create from dictionary."""
        return cls(
            source_type=SourceType(data.get("source_type", "unknown")),
            provenance=Provenance.from_dict(data.get("provenance", {})),
            confidence=data.get("confidence", 0.5),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
            language=data.get("language", "en"),
            word_count=data.get("word_count", 0),
            has_citations=data.get("has_citations", False),
            has_code=data.get("has_code", False),
            has_data=data.get("has_data", False),
            topics=data.get("topics", []),
            entities=data.get("entities", []),
            content_hash=data.get("content_hash", ""),
        )


class MetadataEnricher:
    """Enriches evidence with additional metadata."""

    # Domain patterns for source type classification
    ACADEMIC_DOMAINS = {
        "arxiv.org", "scholar.google.com", "pubmed.ncbi.nlm.nih.gov",
        "doi.org", "researchgate.net", "academia.edu", "jstor.org",
        "ieee.org", "acm.org", "springer.com", "nature.com", "science.org",
    }

    DOCUMENTATION_DOMAINS = {
        "docs.python.org", "developer.mozilla.org", "docs.microsoft.com",
        "readthedocs.io", "readthedocs.org", "devdocs.io", "man7.org",
        "cppreference.com", "docs.oracle.com", "docs.aws.amazon.com",
    }

    NEWS_DOMAINS = {
        "reuters.com", "bbc.com", "bbc.co.uk", "nytimes.com", "wsj.com",
        "theguardian.com", "washingtonpost.com", "cnn.com", "apnews.com",
        "news.ycombinator.com", "techcrunch.com", "arstechnica.com",
    }

    SOCIAL_DOMAINS = {
        "twitter.com", "x.com", "reddit.com", "stackoverflow.com",
        "medium.com", "dev.to", "linkedin.com", "quora.com",
        "news.ycombinator.com", "lobste.rs",
    }

    CODE_DOMAINS = {
        "github.com", "gitlab.com", "bitbucket.org", "gist.github.com",
        "codepen.io", "jsfiddle.net", "replit.com", "codesandbox.io",
    }

    # Patterns for content analysis
    CITATION_PATTERNS = [
        r'\[\d+\]',  # [1], [2], etc.
        r'\(\w+,?\s*\d{4}\)',  # (Author, 2024) or (Author 2024)
        r'et\s+al\.',  # et al.
        r'doi:\s*[\d.\/\-]+',  # DOI references
        r'arXiv:\d+\.\d+',  # arXiv references
    ]

    CODE_PATTERNS = [
        r'```[\w]*\n',  # Markdown code blocks
        r'def\s+\w+\s*\(',  # Python function definitions
        r'function\s+\w+\s*\(',  # JavaScript functions
        r'class\s+\w+',  # Class definitions
        r'import\s+\w+',  # Import statements
        r'const\s+\w+\s*=',  # Const declarations
        r'let\s+\w+\s*=',  # Let declarations
    ]

    DATA_PATTERNS = [
        r'\d+%',  # Percentages
        r'\$[\d,]+',  # Dollar amounts
        r'\d+\s*(million|billion|trillion)',  # Large numbers
        r'table\s+\d+',  # Table references
        r'figure\s+\d+',  # Figure references
        r'\d+\.\d+\s*(ms|s|kb|mb|gb)',  # Measurements
    ]

    def __init__(self):
        """Initialize the metadata enricher."""
        self._compiled_citation_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.CITATION_PATTERNS
        ]
        self._compiled_code_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.CODE_PATTERNS
        ]
        self._compiled_data_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.DATA_PATTERNS
        ]

    def enrich(
        self,
        content: str,
        url: Optional[str] = None,
        source: Optional[str] = None,
        existing_metadata: Optional[Dict[str, Any]] = None,
    ) -> EnrichedMetadata:
        """Enrich content with metadata.

        Args:
            content: The text content to analyze
            url: Optional URL of the source
            source: Optional source identifier
            existing_metadata: Optional existing metadata to incorporate

        Returns:
            EnrichedMetadata with enriched information
        """
        metadata = EnrichedMetadata()

        # Compute content hash
        metadata.content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Classify source type
        metadata.source_type = self._classify_source_type(url, source, content)

        # Extract provenance
        metadata.provenance = self._extract_provenance(url, existing_metadata)

        # Analyze content
        metadata.word_count = len(content.split())
        metadata.has_citations = self._has_citations(content)
        metadata.has_code = self._has_code(content)
        metadata.has_data = self._has_data(content)

        # Calculate confidence based on source type and content quality
        metadata.confidence = self._calculate_confidence(metadata, content)

        # Extract topics (simple keyword extraction)
        metadata.topics = self._extract_topics(content)

        # Extract named entities (simple pattern matching)
        metadata.entities = self._extract_entities(content)

        # Incorporate existing metadata
        if existing_metadata:
            self._merge_existing_metadata(metadata, existing_metadata)

        return metadata

    def _classify_source_type(
        self,
        url: Optional[str],
        source: Optional[str],
        content: str,
    ) -> SourceType:
        """Classify the source type based on URL, source name, and content."""
        if url:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower().replace("www.", "")

                if any(d in domain for d in self.ACADEMIC_DOMAINS):
                    return SourceType.ACADEMIC
                if any(d in domain for d in self.DOCUMENTATION_DOMAINS):
                    return SourceType.DOCUMENTATION
                if any(d in domain for d in self.NEWS_DOMAINS):
                    return SourceType.NEWS
                if any(d in domain for d in self.SOCIAL_DOMAINS):
                    return SourceType.SOCIAL
                if any(d in domain for d in self.CODE_DOMAINS):
                    return SourceType.CODE
            except Exception:
                pass

        # Check source name
        if source:
            source_lower = source.lower()
            if source_lower in ("local", "local_docs", "file"):
                return SourceType.LOCAL
            if source_lower in ("api", "rest", "graphql"):
                return SourceType.API
            if source_lower in ("database", "db", "sql"):
                return SourceType.DATABASE
            if source_lower in ("github", "gitlab", "code"):
                return SourceType.CODE

        # Analyze content for hints
        if self._has_code(content) and not self._has_citations(content):
            return SourceType.CODE
        if self._has_citations(content):
            return SourceType.ACADEMIC

        return SourceType.WEB

    def _extract_provenance(
        self,
        url: Optional[str],
        existing_metadata: Optional[Dict[str, Any]],
    ) -> Provenance:
        """Extract provenance information."""
        provenance = Provenance(url=url)

        if existing_metadata:
            # Extract author
            for key in ("author", "authors", "creator", "by"):
                if key in existing_metadata:
                    value = existing_metadata[key]
                    if isinstance(value, list):
                        provenance.author = ", ".join(str(v) for v in value)
                    else:
                        provenance.author = str(value)
                    break

            # Extract organization
            for key in ("organization", "org", "publisher", "site_name"):
                if key in existing_metadata:
                    provenance.organization = str(existing_metadata[key])
                    break

            # Extract dates
            for key in ("published", "date", "publication_date", "created"):
                if key in existing_metadata:
                    provenance.publication_date = self._parse_date(existing_metadata[key])
                    break

            for key in ("modified", "updated", "last_modified"):
                if key in existing_metadata:
                    provenance.last_modified = self._parse_date(existing_metadata[key])
                    break

            # Extract DOI
            if "doi" in existing_metadata:
                provenance.doi = str(existing_metadata["doi"])

            # Extract version
            if "version" in existing_metadata:
                provenance.version = str(existing_metadata["version"])

        return provenance

    def _parse_date(self, value: Any) -> Optional[datetime]:
        """Parse a date from various formats."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            # Try common formats
            formats = [
                "%Y-%m-%d",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%d/%m/%Y",
                "%m/%d/%Y",
                "%B %d, %Y",
                "%b %d, %Y",
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
        return None

    def _has_citations(self, content: str) -> bool:
        """Check if content has citations."""
        return any(p.search(content) for p in self._compiled_citation_patterns)

    def _has_code(self, content: str) -> bool:
        """Check if content has code."""
        return any(p.search(content) for p in self._compiled_code_patterns)

    def _has_data(self, content: str) -> bool:
        """Check if content has data/statistics."""
        return any(p.search(content) for p in self._compiled_data_patterns)

    def _calculate_confidence(self, metadata: EnrichedMetadata, content: str) -> float:
        """Calculate confidence score based on metadata and content quality."""
        score = 0.5  # Base score

        # Source type adjustments
        source_scores = {
            SourceType.ACADEMIC: 0.2,
            SourceType.DOCUMENTATION: 0.15,
            SourceType.CODE: 0.1,
            SourceType.NEWS: 0.05,
            SourceType.LOCAL: 0.1,
            SourceType.WEB: 0.0,
            SourceType.SOCIAL: -0.1,
            SourceType.UNKNOWN: -0.1,
        }
        score += source_scores.get(metadata.source_type, 0.0)

        # Provenance adjustments
        if metadata.provenance.author:
            score += 0.05
        if metadata.provenance.organization:
            score += 0.05
        if metadata.provenance.doi:
            score += 0.1
        if metadata.provenance.peer_reviewed:
            score += 0.15
        if metadata.provenance.publication_date:
            # More recent is better
            age_days = (datetime.now() - metadata.provenance.publication_date).days
            if age_days < 30:
                score += 0.1
            elif age_days < 365:
                score += 0.05
            elif age_days > 1095:  # > 3 years
                score -= 0.05

        # Content quality adjustments
        if metadata.has_citations:
            score += 0.1
        if metadata.has_data:
            score += 0.05
        if metadata.word_count > 500:
            score += 0.05  # Substantial content
        if metadata.word_count < 50:
            score -= 0.1  # Too brief

        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))

    def _extract_topics(self, content: str) -> List[str]:
        """Extract topic keywords from content."""
        # Simple keyword extraction - find capitalized phrases and technical terms
        topics = []

        # Find capitalized multi-word phrases
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', content)
        topics.extend(capitalized[:5])

        # Find technical terms (camelCase, snake_case, etc.)
        technical = re.findall(r'\b[a-z]+(?:[A-Z][a-z]+)+\b', content)  # camelCase
        topics.extend(technical[:3])

        technical_snake = re.findall(r'\b[a-z]+_[a-z_]+\b', content)  # snake_case
        topics.extend(technical_snake[:3])

        # Deduplicate
        return list(dict.fromkeys(topics))[:10]

    def _extract_entities(self, content: str) -> List[str]:
        """Extract named entities from content."""
        entities = []

        # Find capitalized words that might be names/organizations
        # Skip common words at sentence starts
        potential_entities = re.findall(r'(?<=[.!?]\s)[A-Z][a-z]+|(?<=\s)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', content)

        # Filter out common words
        common_words = {"The", "This", "That", "These", "Those", "It", "They", "We", "You", "I"}
        entities = [e for e in potential_entities if e not in common_words and len(e) > 2]

        # Deduplicate and limit
        return list(dict.fromkeys(entities))[:10]

    def _merge_existing_metadata(
        self,
        metadata: EnrichedMetadata,
        existing: Dict[str, Any],
    ) -> None:
        """Merge existing metadata into enriched metadata."""
        # Don't overwrite computed values, but add missing information
        if "language" in existing and existing["language"]:
            metadata.language = existing["language"]

        if "topics" in existing and existing["topics"]:
            existing_topics = existing["topics"]
            if isinstance(existing_topics, list):
                metadata.topics = list(dict.fromkeys(existing_topics + metadata.topics))[:10]

        if "entities" in existing and existing["entities"]:
            existing_entities = existing["entities"]
            if isinstance(existing_entities, list):
                metadata.entities = list(dict.fromkeys(existing_entities + metadata.entities))[:10]


def enrich_evidence_snippet(
    snippet: Any,  # EvidenceSnippet
    enricher: Optional[MetadataEnricher] = None,
) -> EnrichedMetadata:
    """Convenience function to enrich an EvidenceSnippet.

    Args:
        snippet: An EvidenceSnippet object
        enricher: Optional MetadataEnricher instance

    Returns:
        EnrichedMetadata for the snippet
    """
    if enricher is None:
        enricher = MetadataEnricher()

    return enricher.enrich(
        content=snippet.snippet,
        url=snippet.url,
        source=snippet.source,
        existing_metadata=snippet.metadata,
    )
