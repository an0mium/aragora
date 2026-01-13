"""
Scholarly Citation Grounding - Evidence-backed verdicts with academic rigor.

Inspired by Heavy3.ai's Deep Audit which delivers "verdicts with scholarly references".
This module provides:

1. ScholarlyEvidence: Structured representation of academic/authoritative sources
2. CitationExtractor: Extract citation-worthy claims from agent responses
3. CitationVerifier: Validate citations against known sources
4. CitationFormatter: Format citations in standard academic styles
5. GroundedVerdict: Verdict + supporting citations
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any
import re
import hashlib


class CitationType(Enum):
    """Type of citation source."""

    ACADEMIC_PAPER = "academic_paper"
    BOOK = "book"
    CONFERENCE = "conference"
    PREPRINT = "preprint"
    DOCUMENTATION = "documentation"
    OFFICIAL_SOURCE = "official_source"  # Government, standards bodies
    NEWS_ARTICLE = "news_article"
    BLOG_POST = "blog_post"
    CODE_REPOSITORY = "code_repository"
    DATASET = "dataset"
    WEB_PAGE = "web_page"
    INTERNAL_DEBATE = "internal_debate"  # Reference to prior aragora debate
    UNKNOWN = "unknown"


class CitationQuality(Enum):
    """Quality level of a citation."""

    PEER_REVIEWED = "peer_reviewed"  # Highest quality
    AUTHORITATIVE = "authoritative"  # Official but not peer-reviewed
    REPUTABLE = "reputable"  # Well-known source
    MIXED = "mixed"  # Some quality indicators
    UNVERIFIED = "unverified"  # Unknown quality
    QUESTIONABLE = "questionable"  # Quality concerns


@dataclass
class ScholarlyEvidence:
    """
    A piece of scholarly evidence supporting a claim.

    Designed for rigorous citation tracking in high-stakes debates.
    """

    id: str = ""

    # Source identification
    citation_type: CitationType = CitationType.UNKNOWN
    title: str = ""
    authors: list[str] = field(default_factory=list)
    publication: str = ""  # Journal, conference, publisher
    year: Optional[int] = None
    url: Optional[str] = None
    doi: Optional[str] = None

    # Content
    excerpt: str = ""  # Relevant quote or summary
    relevance_score: float = 0.0  # 0-1, how relevant to claim
    page_numbers: Optional[str] = None

    # Quality assessment
    quality: CitationQuality = CitationQuality.UNVERIFIED
    peer_reviewed: bool = False
    impact_factor: Optional[float] = None  # For academic sources
    citation_count: Optional[int] = None

    # Linking
    claim_id: Optional[str] = None  # Which claim this supports
    debate_id: Optional[str] = None

    # Metadata
    retrieved_at: datetime = field(default_factory=datetime.now)
    verified: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            # Generate deterministic ID from content
            content = f"{self.title}:{','.join(self.authors)}:{self.year}"
            self.id = hashlib.sha256(content.encode()).hexdigest()[:12]

    def format_apa(self) -> str:
        """Format citation in APA style."""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."

        year_str = f"({self.year})" if self.year else "(n.d.)"

        if self.citation_type == CitationType.ACADEMIC_PAPER:
            return f"{authors_str} {year_str}. {self.title}. {self.publication}."
        elif self.citation_type == CitationType.BOOK:
            return f"{authors_str} {year_str}. {self.title}. {self.publication}."
        elif self.citation_type == CitationType.WEB_PAGE:
            return f"{authors_str} {year_str}. {self.title}. Retrieved from {self.url}"
        else:
            return f"{authors_str} {year_str}. {self.title}."

    def format_inline(self) -> str:
        """Format as inline citation."""
        if self.authors:
            first_parts = self.authors[0].split()
            first_author = first_parts[-1] if first_parts else "Unknown"  # Last name
            if len(self.authors) > 2:
                return f"({first_author} et al., {self.year or 'n.d.'})"
            elif len(self.authors) == 2:
                second_parts = self.authors[1].split()
                second_author = second_parts[-1] if second_parts else "Unknown"
                return f"({first_author} & {second_author}, {self.year or 'n.d.'})"
            else:
                return f"({first_author}, {self.year or 'n.d.'})"
        else:
            return f"({self.title[:20]}..., {self.year or 'n.d.'})"

    def quality_score(self) -> float:
        """Calculate overall quality score (0-1)."""
        score = 0.0

        # Base score by type
        type_scores = {
            CitationType.ACADEMIC_PAPER: 0.9,
            CitationType.BOOK: 0.8,
            CitationType.CONFERENCE: 0.85,
            CitationType.PREPRINT: 0.6,
            CitationType.DOCUMENTATION: 0.7,
            CitationType.OFFICIAL_SOURCE: 0.8,
            CitationType.NEWS_ARTICLE: 0.4,
            CitationType.BLOG_POST: 0.3,
            CitationType.CODE_REPOSITORY: 0.5,
            CitationType.WEB_PAGE: 0.3,
            CitationType.UNKNOWN: 0.1,
        }
        score = type_scores.get(self.citation_type, 0.2)

        # Adjust for quality indicators
        if self.peer_reviewed:
            score = min(1.0, score + 0.1)
        if self.doi:
            score = min(1.0, score + 0.05)
        if self.citation_count and self.citation_count > 100:
            score = min(1.0, score + 0.05)
        if self.verified:
            score = min(1.0, score + 0.05)

        return score

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON transmission."""
        return {
            "id": self.id,
            "citation_type": self.citation_type.value,
            "title": self.title,
            "authors": self.authors,
            "publication": self.publication,
            "year": self.year,
            "url": self.url,
            "doi": self.doi,
            "excerpt": self.excerpt,
            "relevance_score": self.relevance_score,
            "quality": self.quality.value if self.quality else None,
            "quality_score": self.quality_score(),
            "peer_reviewed": self.peer_reviewed,
        }


@dataclass
class CitedClaim:
    """A claim with supporting citations."""

    claim_text: str
    claim_id: str = ""
    citations: list[ScholarlyEvidence] = field(default_factory=list)
    confidence: float = 0.0
    grounding_score: float = 0.0  # How well-grounded in citations

    def __post_init__(self):
        if not self.claim_id:
            self.claim_id = hashlib.sha256(self.claim_text.encode()).hexdigest()[:12]

        # Calculate grounding score
        if self.citations:
            avg_quality = sum(c.quality_score() for c in self.citations) / len(self.citations)
            avg_relevance = sum(c.relevance_score for c in self.citations) / len(self.citations)
            self.grounding_score = (avg_quality + avg_relevance) / 2

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON transmission."""
        return {
            "claim_text": self.claim_text,
            "claim_id": self.claim_id,
            "citations": [c.to_dict() for c in self.citations],
            "confidence": self.confidence,
            "grounding_score": self.grounding_score,
        }


@dataclass
class GroundedVerdict:
    """
    A verdict grounded in scholarly evidence.

    Heavy3-inspired: "Delivers verdicts with scholarly references."
    """

    verdict: str
    confidence: float
    claims: list[CitedClaim] = field(default_factory=list)
    all_citations: list[ScholarlyEvidence] = field(default_factory=list)
    grounding_score: float = 0.0  # Overall evidence grounding

    def __post_init__(self):
        # Collect all citations
        if self.claims and not self.all_citations:
            seen = set()
            for claim in self.claims:
                for citation in claim.citations:
                    if citation.id not in seen:
                        self.all_citations.append(citation)
                        seen.add(citation.id)

        # Calculate overall grounding
        if self.claims:
            self.grounding_score = sum(c.grounding_score for c in self.claims) / len(self.claims)

    def format_bibliography(self) -> str:
        """Generate formatted bibliography."""
        if not self.all_citations:
            return "No citations."

        lines = ["References:", ""]
        for i, citation in enumerate(sorted(self.all_citations, key=lambda c: c.format_apa()), 1):
            lines.append(f"[{i}] {citation.format_apa()}")

        return "\n".join(lines)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "GROUNDED VERDICT",
            "=" * 60,
            "",
            f"Verdict: {self.verdict[:500]}",
            f"Confidence: {self.confidence:.0%}",
            f"Evidence Grounding: {self.grounding_score:.0%}",
            "",
            f"Supported by {len(self.all_citations)} citations:",
        ]

        for citation in self.all_citations[:5]:
            lines.append(f"  - {citation.format_inline()}: {citation.title[:50]}...")

        if len(self.all_citations) > 5:
            lines.append(f"  ... and {len(self.all_citations) - 5} more")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON transmission."""
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "claims": [c.to_dict() for c in self.claims],
            "all_citations": [c.to_dict() for c in self.all_citations],
            "grounding_score": self.grounding_score,
        }


class CitationExtractor:
    """
    Extracts citation-worthy claims from agent responses.

    Identifies statements that should be backed by evidence.
    """

    # Patterns that indicate claims needing citation
    CLAIM_PATTERNS = [
        r"research shows",
        r"studies have found",
        r"according to",
        r"evidence suggests",
        r"data indicates",
        r"\d+% of",
        r"proven to",
        r"scientifically",
        r"experts agree",
        r"best practice",
        r"industry standard",
    ]

    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.CLAIM_PATTERNS]

    def extract_claims(self, text: str) -> list[str]:
        """Extract sentences that contain claim patterns."""
        claims = []
        sentences = re.split(r"[.!?]+", text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            for pattern in self.patterns:
                if pattern.search(sentence):
                    claims.append(sentence)
                    break

        return claims

    def identify_citation_needs(self, text: str) -> list[dict]:
        """Identify claims and their citation needs."""
        claims = self.extract_claims(text)
        needs = []

        for claim in claims:
            # Estimate how critical citation is
            priority = "medium"
            if any(word in claim.lower() for word in ["proven", "data", "research", "%"]):
                priority = "high"
            elif any(word in claim.lower() for word in ["suggests", "may", "could"]):
                priority = "low"

            needs.append(
                {
                    "claim": claim,
                    "priority": priority,
                    "suggested_source_types": self._suggest_source_types(claim),
                }
            )

        return needs

    def _suggest_source_types(self, claim: str) -> list[CitationType]:
        """Suggest appropriate source types for a claim."""
        claim_lower = claim.lower()

        if any(word in claim_lower for word in ["research", "study", "scientist"]):
            return [CitationType.ACADEMIC_PAPER, CitationType.PREPRINT]
        elif any(word in claim_lower for word in ["code", "programming", "software"]):
            return [CitationType.DOCUMENTATION, CitationType.CODE_REPOSITORY]
        elif any(word in claim_lower for word in ["law", "regulation", "standard"]):
            return [CitationType.OFFICIAL_SOURCE, CitationType.DOCUMENTATION]
        else:
            return [CitationType.ACADEMIC_PAPER, CitationType.OFFICIAL_SOURCE]


class CitationStore:
    """
    Stores and retrieves citations for reuse across debates.

    Maintains a knowledge base of verified citations.
    """

    def __init__(self):
        self.citations: dict[str, ScholarlyEvidence] = {}
        self.claim_to_citations: dict[str, list[str]] = {}  # claim_id -> [citation_ids]

    def add(self, citation: ScholarlyEvidence) -> str:
        """Add a citation to the store."""
        self.citations[citation.id] = citation
        return citation.id

    def get(self, citation_id: str) -> Optional[ScholarlyEvidence]:
        """Get a citation by ID."""
        return self.citations.get(citation_id)

    def find_for_claim(self, claim_text: str, limit: int = 5) -> list[ScholarlyEvidence]:
        """Find relevant citations for a claim."""
        # Simple keyword matching - could be enhanced with embeddings
        claim_words = set(claim_text.lower().split())
        scored = []

        for citation in self.citations.values():
            title_words = set(citation.title.lower().split())
            excerpt_words = set(citation.excerpt.lower().split())
            overlap = len(claim_words & (title_words | excerpt_words))
            if overlap > 0:
                scored.append((overlap, citation))

        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:limit]]

    def link_claim_to_citation(self, claim_id: str, citation_id: str):
        """Link a claim to a supporting citation."""
        if claim_id not in self.claim_to_citations:
            self.claim_to_citations[claim_id] = []
        if citation_id not in self.claim_to_citations[claim_id]:
            self.claim_to_citations[claim_id].append(citation_id)

    def get_citations_for_claim(self, claim_id: str) -> list[ScholarlyEvidence]:
        """Get all citations linked to a claim."""
        citation_ids = self.claim_to_citations.get(claim_id, [])
        return [self.citations[cid] for cid in citation_ids if cid in self.citations]


def create_citation_from_url(url: str, title: str = "", excerpt: str = "") -> ScholarlyEvidence:
    """Create a citation from a URL with automatic type detection."""
    citation_type = CitationType.WEB_PAGE
    quality = CitationQuality.UNVERIFIED

    # Detect type from URL
    url_lower = url.lower()
    if "arxiv.org" in url_lower:
        citation_type = CitationType.PREPRINT
        quality = CitationQuality.REPUTABLE
    elif "doi.org" in url_lower or "pubmed" in url_lower:
        citation_type = CitationType.ACADEMIC_PAPER
        quality = CitationQuality.PEER_REVIEWED
    elif "github.com" in url_lower:
        citation_type = CitationType.CODE_REPOSITORY
        quality = CitationQuality.REPUTABLE
    elif any(domain in url_lower for domain in [".gov", ".edu"]):
        citation_type = CitationType.OFFICIAL_SOURCE
        quality = CitationQuality.AUTHORITATIVE
    elif "docs." in url_lower or "documentation" in url_lower:
        citation_type = CitationType.DOCUMENTATION
        quality = CitationQuality.REPUTABLE

    return ScholarlyEvidence(
        citation_type=citation_type,
        title=title,
        url=url,
        excerpt=excerpt,
        quality=quality,
    )
