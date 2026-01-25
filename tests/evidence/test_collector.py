"""
Tests for the aragora.evidence.collector module.

Tests EvidenceSnippet, EvidencePack, and EvidenceCollector classes.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.evidence.collector import (
    DEFAULT_ALLOWED_DOMAINS,
    EvidenceCollector,
    EvidencePack,
    EvidenceSnippet,
)


# =============================================================================
# EvidenceSnippet Tests
# =============================================================================


class TestEvidenceSnippet:
    """Tests for EvidenceSnippet dataclass."""

    def test_basic_creation(self):
        """Should create snippet with required fields."""
        snippet = EvidenceSnippet(
            id="test-001",
            source="web",
            title="Test Title",
            snippet="Test content here.",
        )
        assert snippet.id == "test-001"
        assert snippet.source == "web"
        assert snippet.title == "Test Title"
        assert snippet.snippet == "Test content here."
        assert snippet.url == ""
        assert snippet.reliability_score == 0.5

    def test_creation_with_all_fields(self):
        """Should create snippet with all optional fields."""
        now = datetime.now()
        snippet = EvidenceSnippet(
            id="test-002",
            source="academic",
            title="Full Test",
            snippet="Full content.",
            url="https://example.com/doc",
            reliability_score=0.9,
            metadata={"author": "Test Author"},
            fetched_at=now,
        )
        assert snippet.url == "https://example.com/doc"
        assert snippet.reliability_score == 0.9
        assert snippet.metadata["author"] == "Test Author"
        assert snippet.fetched_at == now


class TestEvidenceSnippetFreshnessScore:
    """Tests for EvidenceSnippet.freshness_score property."""

    def test_very_fresh_under_1_hour(self):
        """Evidence less than 1 hour old should have score 1.0."""
        snippet = EvidenceSnippet(
            id="fresh-001",
            source="web",
            title="Fresh",
            snippet="Just fetched.",
            fetched_at=datetime.now() - timedelta(minutes=30),
        )
        assert snippet.freshness_score == 1.0

    def test_moderately_fresh_1_to_24_hours(self):
        """Evidence 1-24 hours old should have score 0.7-0.9."""
        snippet = EvidenceSnippet(
            id="mod-001",
            source="web",
            title="Moderate",
            snippet="Few hours old.",
            fetched_at=datetime.now() - timedelta(hours=12),
        )
        score = snippet.freshness_score
        assert 0.7 <= score <= 0.9

    def test_stale_1_to_7_days(self):
        """Evidence 1-7 days old should have score 0.5-0.7."""
        snippet = EvidenceSnippet(
            id="stale-001",
            source="web",
            title="Stale",
            snippet="Days old.",
            fetched_at=datetime.now() - timedelta(days=3),
        )
        score = snippet.freshness_score
        assert 0.5 <= score <= 0.7

    def test_very_stale_over_7_days(self):
        """Evidence over 7 days old should have score 0.3-0.5."""
        snippet = EvidenceSnippet(
            id="old-001",
            source="web",
            title="Old",
            snippet="Week old.",
            fetched_at=datetime.now() - timedelta(days=14),
        )
        score = snippet.freshness_score
        assert 0.3 <= score <= 0.5

    def test_minimum_score_floor(self):
        """Very old evidence should not go below 0.3."""
        snippet = EvidenceSnippet(
            id="ancient-001",
            source="web",
            title="Ancient",
            snippet="Very old.",
            fetched_at=datetime.now() - timedelta(days=365),
        )
        assert snippet.freshness_score >= 0.3


class TestEvidenceSnippetCombinedScore:
    """Tests for EvidenceSnippet.combined_score property."""

    def test_combined_score_calculation(self):
        """Combined score should weight reliability 70%, freshness 30%."""
        snippet = EvidenceSnippet(
            id="combined-001",
            source="web",
            title="Combined",
            snippet="Test.",
            reliability_score=0.8,
            fetched_at=datetime.now(),  # freshness_score = 1.0
        )
        # 0.8 * 0.7 + 1.0 * 0.3 = 0.56 + 0.30 = 0.86
        expected = 0.8 * 0.7 + 1.0 * 0.3
        assert abs(snippet.combined_score - expected) < 0.01


class TestEvidenceSnippetFormatting:
    """Tests for EvidenceSnippet formatting methods."""

    @pytest.fixture
    def sample_snippet(self):
        """Create a sample snippet for formatting tests."""
        return EvidenceSnippet(
            id="fmt-001",
            source="github",
            title="Sample Repository",
            snippet="This is a sample repository with code examples.",
            url="https://github.com/user/repo",
            reliability_score=0.85,
            fetched_at=datetime.now(),
        )

    def test_to_text_block(self, sample_snippet):
        """Should format as readable text block."""
        block = sample_snippet.to_text_block()
        assert "EVID-fmt-001" in block
        assert "github" in block
        assert "0.85" in block or "0.8" in block
        assert "Sample Repository" in block
        assert sample_snippet.snippet in block
        assert sample_snippet.url in block

    def test_to_text_block_truncates_long_snippet(self):
        """Should truncate snippets over 500 characters."""
        long_content = "x" * 600
        snippet = EvidenceSnippet(
            id="long-001",
            source="web",
            title="Long",
            snippet=long_content,
        )
        block = snippet.to_text_block()
        assert "..." in block
        assert len(block) < len(long_content) + 200  # Some room for formatting

    def test_to_text_block_freshness_indicators(self):
        """Should show correct freshness indicator emoji."""
        # Fresh (green)
        fresh = EvidenceSnippet(
            id="fresh",
            source="web",
            title="Fresh",
            snippet="Fresh.",
            fetched_at=datetime.now(),
        )
        assert "🟢" in fresh.to_text_block()

        # Medium fresh (yellow)
        medium = EvidenceSnippet(
            id="medium",
            source="web",
            title="Medium",
            snippet="Medium.",
            fetched_at=datetime.now() - timedelta(days=5),
        )
        assert "🟡" in medium.to_text_block()

        # Stale (red)
        stale = EvidenceSnippet(
            id="stale",
            source="web",
            title="Stale",
            snippet="Stale.",
            fetched_at=datetime.now() - timedelta(days=30),
        )
        assert "🔴" in stale.to_text_block()

    def test_to_citation(self, sample_snippet):
        """Should format as academic-style citation."""
        citation = sample_snippet.to_citation()
        assert "[fmt-001]" in citation
        assert "Sample Repository" in citation
        assert "Github" in citation  # Title-cased source
        assert "0.8" in citation  # reliability
        assert sample_snippet.url in citation

    def test_to_citation_without_url(self):
        """Should handle citation without URL."""
        snippet = EvidenceSnippet(
            id="nourl-001",
            source="local",
            title="Local Doc",
            snippet="Local content.",
        )
        citation = snippet.to_citation()
        assert "[nourl-001]" in citation
        assert "Local Doc" in citation
        # Should not have dangling URL syntax

    def test_to_dict(self, sample_snippet):
        """Should serialize to dictionary."""
        data = sample_snippet.to_dict()
        assert data["id"] == "fmt-001"
        assert data["source"] == "github"
        assert data["title"] == "Sample Repository"
        assert data["snippet"] == sample_snippet.snippet
        assert data["url"] == "https://github.com/user/repo"
        assert data["reliability_score"] == 0.85
        assert "freshness_score" in data
        assert "combined_score" in data
        assert "fetched_at" in data
        assert "metadata" in data


# =============================================================================
# EvidencePack Tests
# =============================================================================


class TestEvidencePack:
    """Tests for EvidencePack dataclass."""

    @pytest.fixture
    def sample_snippets(self) -> List[EvidenceSnippet]:
        """Create sample snippets for pack tests."""
        return [
            EvidenceSnippet(
                id="s1",
                source="web",
                title="Web Source",
                snippet="Web content.",
                reliability_score=0.7,
                fetched_at=datetime.now(),
            ),
            EvidenceSnippet(
                id="s2",
                source="academic",
                title="Academic Source",
                snippet="Academic content.",
                reliability_score=0.9,
                fetched_at=datetime.now() - timedelta(hours=12),
            ),
        ]

    def test_basic_creation(self, sample_snippets):
        """Should create pack with snippets."""
        pack = EvidencePack(
            topic_keywords=["ai", "machine", "learning"],
            snippets=sample_snippets,
            total_searched=10,
        )
        assert len(pack.topic_keywords) == 3
        assert len(pack.snippets) == 2
        assert pack.total_searched == 10

    def test_average_reliability(self, sample_snippets):
        """Should calculate average reliability correctly."""
        pack = EvidencePack(
            topic_keywords=["test"],
            snippets=sample_snippets,
        )
        expected = (0.7 + 0.9) / 2
        assert pack.average_reliability == expected

    def test_average_reliability_empty(self):
        """Should return 0.0 for empty pack."""
        pack = EvidencePack(topic_keywords=["test"], snippets=[])
        assert pack.average_reliability == 0.0

    def test_average_freshness(self, sample_snippets):
        """Should calculate average freshness correctly."""
        pack = EvidencePack(
            topic_keywords=["test"],
            snippets=sample_snippets,
        )
        # Both snippets are relatively fresh
        assert 0.8 <= pack.average_freshness <= 1.0

    def test_average_freshness_empty(self):
        """Should return 0.0 for empty pack."""
        pack = EvidencePack(topic_keywords=["test"], snippets=[])
        assert pack.average_freshness == 0.0


class TestEvidencePackFormatting:
    """Tests for EvidencePack formatting methods."""

    @pytest.fixture
    def sample_pack(self) -> EvidencePack:
        """Create sample pack for formatting tests."""
        return EvidencePack(
            topic_keywords=["machine", "learning", "ai"],
            snippets=[
                EvidenceSnippet(
                    id="p1",
                    source="web",
                    title="Web Article",
                    snippet="An article about ML.",
                    url="https://example.com/ml",
                    reliability_score=0.8,
                ),
                EvidenceSnippet(
                    id="p2",
                    source="github",
                    title="ML Repository",
                    snippet="Code examples for ML.",
                    url="https://github.com/user/ml",
                    reliability_score=0.9,
                ),
            ],
            total_searched=5,
        )

    def test_to_context_string(self, sample_pack):
        """Should format as context string for debates."""
        context = sample_pack.to_context_string()
        assert "EVIDENCE PACK" in context
        assert "machine, learning, ai" in context
        assert "Total sources searched: 5" in context
        assert "EVID-p1" in context
        assert "EVID-p2" in context
        assert "END EVIDENCE PACK" in context

    def test_to_context_string_empty(self):
        """Should handle empty pack gracefully."""
        empty_pack = EvidencePack(topic_keywords=["test"], snippets=[])
        context = empty_pack.to_context_string()
        assert "No relevant evidence found" in context

    def test_to_bibliography(self, sample_pack):
        """Should format as academic bibliography."""
        bib = sample_pack.to_bibliography()
        assert "## References" in bib
        assert "1. " in bib
        assert "2. " in bib
        assert "[p1]" in bib
        assert "[p2]" in bib

    def test_to_bibliography_empty(self):
        """Should return empty string for empty pack."""
        empty_pack = EvidencePack(topic_keywords=["test"], snippets=[])
        assert empty_pack.to_bibliography() == ""

    def test_to_dict(self, sample_pack):
        """Should serialize to dictionary."""
        data = sample_pack.to_dict()
        assert data["topic_keywords"] == ["machine", "learning", "ai"]
        assert len(data["snippets"]) == 2
        assert "search_timestamp" in data
        assert data["total_searched"] == 5
        assert "average_reliability" in data
        assert "average_freshness" in data


# =============================================================================
# EvidenceCollector Tests - Initialization
# =============================================================================


class TestEvidenceCollectorInit:
    """Tests for EvidenceCollector initialization."""

    def test_basic_init(self):
        """Should initialize with defaults."""
        collector = EvidenceCollector()
        assert collector.connectors == {}
        assert collector.max_snippets_per_connector == 3
        assert collector.max_total_snippets == 8
        assert collector.snippet_max_length == 1000

    def test_init_with_connectors(self):
        """Should initialize with provided connectors."""
        mock_connector = MagicMock()
        collector = EvidenceCollector(connectors={"web": mock_connector})
        assert "web" in collector.connectors
        assert collector.connectors["web"] == mock_connector

    def test_init_with_allowed_domains(self):
        """Should merge custom domains with defaults."""
        collector = EvidenceCollector(allowed_domains={"custom.example.com", "trusted.org"})
        assert "custom.example.com" in collector._allowed_domains
        assert "trusted.org" in collector._allowed_domains
        # Should also have defaults
        assert "github.com" in collector._allowed_domains

    def test_init_with_consent_requirement(self):
        """Should require callback when consent is required."""
        with pytest.raises(ValueError, match="url_consent_callback is required"):
            EvidenceCollector(require_url_consent=True)

    def test_init_with_consent_callback(self):
        """Should accept consent callback."""
        callback = MagicMock(return_value=True)
        collector = EvidenceCollector(
            require_url_consent=True,
            url_consent_callback=callback,
        )
        assert collector._require_url_consent is True
        assert collector._url_consent_callback == callback

    def test_init_with_event_emitter(self):
        """Should store event emitter and loop_id."""
        emitter = MagicMock()
        collector = EvidenceCollector(event_emitter=emitter, loop_id="loop-123")
        assert collector.event_emitter == emitter
        assert collector.loop_id == "loop-123"


# =============================================================================
# EvidenceCollector Tests - SSRF Protection
# =============================================================================


class TestEvidenceCollectorSSRF:
    """Tests for SSRF protection in EvidenceCollector."""

    @pytest.fixture
    def collector(self):
        """Create collector for SSRF tests."""
        return EvidenceCollector()

    def test_safe_https_url(self, collector):
        """Should allow safe HTTPS URLs."""
        assert collector._is_safe_url("https://example.com/page") is True

    def test_safe_http_url(self, collector):
        """Should allow safe HTTP URLs."""
        assert collector._is_safe_url("http://example.com/page") is True

    def test_block_non_http_schemes(self, collector):
        """Should block non-HTTP schemes."""
        assert collector._is_safe_url("file:///etc/passwd") is False
        assert collector._is_safe_url("ftp://example.com") is False
        assert collector._is_safe_url("gopher://example.com") is False
        assert collector._is_safe_url("data:text/html,<script>") is False

    def test_block_localhost(self, collector):
        """Should block localhost variants."""
        assert collector._is_safe_url("http://localhost/") is False
        assert collector._is_safe_url("http://127.0.0.1/") is False
        assert collector._is_safe_url("http://0.0.0.0/") is False

    def test_block_private_ip_ranges(self, collector):
        """Should block private IP ranges."""
        # 10.x.x.x
        assert collector._is_safe_url("http://10.0.0.1/") is False
        # 172.16-31.x.x
        assert collector._is_safe_url("http://172.16.0.1/") is False
        assert collector._is_safe_url("http://172.31.255.255/") is False
        # 192.168.x.x
        assert collector._is_safe_url("http://192.168.1.1/") is False

    def test_block_link_local(self, collector):
        """Should block link-local addresses."""
        assert collector._is_safe_url("http://169.254.169.254/") is False

    def test_block_non_standard_ports(self, collector):
        """Should block non-standard ports."""
        assert collector._is_safe_url("http://example.com:8080/") is False
        assert collector._is_safe_url("http://example.com:22/") is False
        assert collector._is_safe_url("http://example.com:3306/") is False

    def test_allow_standard_ports(self, collector):
        """Should allow standard HTTP/HTTPS ports."""
        assert collector._is_safe_url("http://example.com:80/") is True
        assert collector._is_safe_url("https://example.com:443/") is True

    def test_allow_public_ips(self, collector):
        """Should allow public IP addresses."""
        assert collector._is_safe_url("http://8.8.8.8/") is True
        assert collector._is_safe_url("http://1.1.1.1/") is True

    def test_handle_malformed_urls(self, collector):
        """Should handle malformed URLs safely."""
        assert collector._is_safe_url("not-a-url") is False
        assert collector._is_safe_url("") is False
        assert collector._is_safe_url("http://") is False


# =============================================================================
# EvidenceCollector Tests - Domain Allowlist
# =============================================================================


class TestEvidenceCollectorDomainAllowlist:
    """Tests for domain allowlist checking."""

    @pytest.fixture
    def collector(self):
        """Create collector for domain tests."""
        return EvidenceCollector()

    def test_allowed_domain_exact_match(self, collector):
        """Should allow exact domain matches."""
        assert collector._is_domain_allowed("github.com") is True
        assert collector._is_domain_allowed("arxiv.org") is True
        assert collector._is_domain_allowed("stackoverflow.com") is True

    def test_allowed_domain_subdomain(self, collector):
        """Should allow subdomains of allowed domains."""
        assert collector._is_domain_allowed("api.github.com") is True
        assert collector._is_domain_allowed("raw.githubusercontent.com") is True
        assert collector._is_domain_allowed("en.wikipedia.org") is True

    def test_blocked_domain(self, collector):
        """Should block non-allowlisted domains."""
        assert collector._is_domain_allowed("malicious-site.com") is False
        assert collector._is_domain_allowed("random-domain.xyz") is False

    def test_case_insensitive(self, collector):
        """Should be case insensitive."""
        assert collector._is_domain_allowed("GITHUB.COM") is True
        assert collector._is_domain_allowed("GitHub.Com") is True

    def test_default_allowed_domains(self):
        """Should include expected default domains."""
        assert "github.com" in DEFAULT_ALLOWED_DOMAINS
        assert "arxiv.org" in DEFAULT_ALLOWED_DOMAINS
        assert "wikipedia.org" in DEFAULT_ALLOWED_DOMAINS
        assert "stackoverflow.com" in DEFAULT_ALLOWED_DOMAINS
        assert "docs.python.org" in DEFAULT_ALLOWED_DOMAINS


# =============================================================================
# EvidenceCollector Tests - URL Extraction
# =============================================================================


class TestEvidenceCollectorURLExtraction:
    """Tests for URL extraction from task text."""

    @pytest.fixture
    def collector(self):
        return EvidenceCollector()

    def test_extract_https_url(self, collector):
        """Should extract HTTPS URLs."""
        task = "Check this article: https://example.com/article"
        urls = collector._extract_urls(task)
        assert "https://example.com/article" in urls

    def test_extract_http_url(self, collector):
        """Should extract HTTP URLs."""
        task = "See http://old-site.com/page for details"
        urls = collector._extract_urls(task)
        assert "http://old-site.com/page" in urls

    def test_extract_www_url(self, collector):
        """Should extract www URLs."""
        task = "Visit www.example.com/docs for more"
        urls = collector._extract_urls(task)
        assert "www.example.com/docs" in urls

    def test_extract_github_repo(self, collector):
        """Should extract GitHub repo references."""
        task = "Look at github.com/anthropics/claude for the code"
        urls = collector._extract_urls(task)
        assert any("github.com/anthropics/claude" in url for url in urls)

    def test_extract_domain_with_common_tld(self, collector):
        """Should extract domains with common TLDs."""
        task = "Check docs at anthropic.com/docs and try claude.ai"
        urls = collector._extract_urls(task)
        assert any("anthropic.com" in url for url in urls)
        assert any("claude.ai" in url for url in urls)

    def test_multiple_urls(self, collector):
        """Should extract multiple URLs."""
        task = "Compare https://site1.com and https://site2.com"
        urls = collector._extract_urls(task)
        assert len(urls) >= 2

    def test_deduplicate_urls(self, collector):
        """Should deduplicate URLs."""
        task = "Visit https://example.com twice: https://example.com"
        urls = collector._extract_urls(task)
        assert urls.count("https://example.com") == 1

    def test_no_urls(self, collector):
        """Should return empty list when no URLs present."""
        task = "Just a simple task with no links"
        urls = collector._extract_urls(task)
        assert urls == []


# =============================================================================
# EvidenceCollector Tests - Keyword Extraction
# =============================================================================


class TestEvidenceCollectorKeywordExtraction:
    """Tests for keyword extraction from task text."""

    @pytest.fixture
    def collector(self):
        return EvidenceCollector()

    def test_basic_extraction(self, collector):
        """Should extract meaningful keywords."""
        task = "Analyze machine learning algorithms"
        keywords = collector._extract_keywords(task)
        assert "machine" in keywords or "learning" in keywords or "algorithms" in keywords

    def test_remove_stop_words(self, collector):
        """Should remove common stop words."""
        task = "The quick brown fox jumps over the lazy dog"
        keywords = collector._extract_keywords(task)
        assert "the" not in keywords
        assert "over" not in keywords

    def test_filter_short_words(self, collector):
        """Should filter very short words."""
        task = "Go to AI lab for ML"
        keywords = collector._extract_keywords(task)
        # Words with 2 or fewer chars should be filtered
        assert "go" not in keywords
        assert "to" not in keywords

    def test_limit_keywords(self, collector):
        """Should limit to top 5 keywords."""
        task = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        keywords = collector._extract_keywords(task)
        assert len(keywords) <= 5

    def test_boost_technical_terms(self, collector):
        """Should boost technical terms."""
        task = "Data systems with AI technology code"
        keywords = collector._extract_keywords(task)
        # Technical terms should appear
        assert any(kw in keywords for kw in ["data", "systems", "technology", "code"])


# =============================================================================
# EvidenceCollector Tests - Snippet Truncation
# =============================================================================


class TestEvidenceCollectorSnippetTruncation:
    """Tests for snippet truncation."""

    @pytest.fixture
    def collector(self):
        return EvidenceCollector()

    def test_short_text_unchanged(self, collector):
        """Should leave short text unchanged."""
        text = "Short text."
        result = collector._truncate_snippet(text)
        assert result == text

    def test_long_text_truncated(self, collector):
        """Should truncate long text."""
        text = "x" * 2000
        result = collector._truncate_snippet(text)
        assert len(result) <= collector.snippet_max_length + 3  # +3 for "..."

    def test_truncate_at_sentence_boundary(self, collector):
        """Should try to truncate at sentence boundary."""
        text = "First sentence. " + "x" * 1500 + ". Last part."
        result = collector._truncate_snippet(text)
        # Should end with period or ellipsis
        assert result.endswith(".") or result.endswith("...")


# =============================================================================
# EvidenceCollector Tests - Snippet Ranking
# =============================================================================


class TestEvidenceCollectorSnippetRanking:
    """Tests for snippet ranking."""

    @pytest.fixture
    def collector(self):
        return EvidenceCollector()

    def test_rank_by_relevance(self, collector):
        """Should rank by keyword relevance."""
        snippets = [
            EvidenceSnippet(
                id="s1",
                source="web",
                title="Unrelated Topic",
                snippet="Nothing about the search topic.",
            ),
            EvidenceSnippet(
                id="s2",
                source="web",
                title="Machine Learning Guide",
                snippet="This covers machine learning algorithms.",
            ),
        ]
        ranked = collector._rank_snippets(snippets, ["machine", "learning"])
        assert ranked[0].id == "s2"

    def test_rank_by_reliability(self, collector):
        """Should consider reliability in ranking."""
        snippets = [
            EvidenceSnippet(
                id="s1",
                source="web",
                title="Low Trust",
                snippet="Topic content here.",
                reliability_score=0.3,
            ),
            EvidenceSnippet(
                id="s2",
                source="academic",
                title="High Trust",
                snippet="Topic content here.",
                reliability_score=0.9,
            ),
        ]
        ranked = collector._rank_snippets(snippets, ["topic"])
        # Higher reliability should rank higher when relevance is similar
        assert ranked[0].id == "s2"

    def test_title_match_boost(self, collector):
        """Should boost snippets with keyword in title."""
        snippets = [
            EvidenceSnippet(
                id="s1",
                source="web",
                title="Generic Title",
                snippet="Article about Python programming.",
            ),
            EvidenceSnippet(
                id="s2",
                source="web",
                title="Python Programming Guide",
                snippet="Generic content here.",
            ),
        ]
        ranked = collector._rank_snippets(snippets, ["python"])
        # Title match should boost s2
        assert ranked[0].id == "s2"


# =============================================================================
# EvidenceCollector Tests - GitHub URL Detection
# =============================================================================


class TestEvidenceCollectorGitHub:
    """Tests for GitHub URL detection and parsing."""

    @pytest.fixture
    def collector(self):
        return EvidenceCollector()

    def test_is_github_repo_url_valid(self, collector):
        """Should detect valid GitHub repo URLs."""
        assert collector._is_github_repo_url("https://github.com/user/repo") is True
        assert collector._is_github_repo_url("https://github.com/user/repo/") is True
        assert collector._is_github_repo_url("http://github.com/user/repo") is True

    def test_is_github_repo_url_not_repo_root(self, collector):
        """Should reject non-root GitHub URLs."""
        assert collector._is_github_repo_url("https://github.com/user/repo/tree/main") is False
        assert (
            collector._is_github_repo_url("https://github.com/user/repo/blob/main/file.py") is False
        )
        assert collector._is_github_repo_url("https://github.com/user") is False

    def test_parse_github_repo(self, collector):
        """Should parse owner and repo from URL."""
        result = collector._parse_github_repo("https://github.com/anthropics/claude")
        assert result == ("anthropics", "claude")

    def test_parse_github_repo_with_path(self, collector):
        """Should parse from URL with path."""
        result = collector._parse_github_repo("https://github.com/user/repo/tree/main")
        assert result == ("user", "repo")

    def test_parse_github_repo_invalid(self, collector):
        """Should return None for non-GitHub URLs."""
        result = collector._parse_github_repo("https://example.com/something")
        assert result is None


# =============================================================================
# EvidenceCollector Tests - Document URL Detection
# =============================================================================


class TestEvidenceCollectorDocumentURLs:
    """Tests for document URL detection."""

    @pytest.fixture
    def collector(self):
        return EvidenceCollector()

    def test_is_document_url_pdf(self, collector):
        """Should detect PDF URLs."""
        assert collector._is_document_url("https://example.com/file.pdf") is True
        assert collector._is_document_url("https://example.com/file.PDF") is True

    def test_is_document_url_office(self, collector):
        """Should detect Office document URLs."""
        assert collector._is_document_url("https://example.com/file.docx") is True
        assert collector._is_document_url("https://example.com/file.xlsx") is True
        assert collector._is_document_url("https://example.com/file.pptx") is True

    def test_is_document_url_data(self, collector):
        """Should detect data file URLs."""
        assert collector._is_document_url("https://example.com/file.csv") is True
        assert collector._is_document_url("https://example.com/file.json") is True
        assert collector._is_document_url("https://example.com/file.yaml") is True

    def test_is_document_url_not_document(self, collector):
        """Should reject non-document URLs."""
        assert collector._is_document_url("https://example.com/page") is False
        assert collector._is_document_url("https://example.com/image.png") is False


# =============================================================================
# EvidenceCollector Tests - Document Path Extraction
# =============================================================================


class TestEvidenceCollectorDocumentPaths:
    """Tests for document path extraction from task text."""

    @pytest.fixture
    def collector(self):
        return EvidenceCollector()

    def test_extract_quoted_path(self, collector):
        """Should extract quoted file paths."""
        task = 'Analyze the document "/path/to/report.pdf" for insights'
        paths = collector._extract_document_paths(task)
        assert any("report.pdf" in p for p in paths)

    def test_extract_unix_absolute_path(self, collector):
        """Should extract Unix absolute paths."""
        task = "Read /home/user/docs/analysis.docx"
        paths = collector._extract_document_paths(task)
        assert any("analysis.docx" in p for p in paths)

    def test_extract_relative_path(self, collector):
        """Should extract relative paths."""
        task = "Check ./data/results.csv for the data"
        paths = collector._extract_document_paths(task)
        assert any("results.csv" in p for p in paths)

    def test_extract_windows_path(self, collector):
        """Should extract Windows paths."""
        task = r"Open C:\Users\docs\report.xlsx"
        paths = collector._extract_document_paths(task)
        assert any("report.xlsx" in p for p in paths)

    def test_extract_bare_filename(self, collector):
        """Should extract bare filenames with extensions."""
        task = "Look at report.pdf and data.csv"
        paths = collector._extract_document_paths(task)
        assert any("report.pdf" in p for p in paths)
        assert any("data.csv" in p for p in paths)


# =============================================================================
# EvidenceCollector Tests - Claim Extraction
# =============================================================================


class TestEvidenceCollectorClaimExtraction:
    """Tests for claim extraction from text."""

    @pytest.fixture
    def collector(self):
        return EvidenceCollector()

    def test_extract_percentage_claims(self, collector):
        """Should extract claims with percentages."""
        text = "The new model achieves 95% accuracy on the benchmark."
        claims = collector.extract_claims_from_text(text)
        assert len(claims) >= 1
        assert any("95%" in c for c in claims)

    def test_extract_comparative_claims(self, collector):
        """Should extract comparative claims."""
        text = "This approach is faster than traditional methods."
        claims = collector.extract_claims_from_text(text)
        assert len(claims) >= 1
        assert any("faster than" in c for c in claims)

    def test_extract_research_claims(self, collector):
        """Should extract research-based claims."""
        text = "Studies show that regular exercise improves cognitive function."
        claims = collector.extract_claims_from_text(text)
        assert len(claims) >= 1
        assert any("Studies show" in c for c in claims)

    def test_extract_absolute_claims(self, collector):
        """Should extract absolute claims."""
        text = "All experts agree that climate change is real."
        claims = collector.extract_claims_from_text(text)
        assert len(claims) >= 1

    def test_filter_short_sentences(self, collector):
        """Should filter very short sentences."""
        text = "Yes. No. Maybe. This is a proper claim about facts."
        claims = collector.extract_claims_from_text(text)
        # Short sentences should be filtered
        for claim in claims:
            assert len(claim) >= 20

    def test_limit_claims(self, collector):
        """Should limit number of claims extracted."""
        text = ". ".join([f"Studies show result {i}" for i in range(20)])
        claims = collector.extract_claims_from_text(text)
        assert len(claims) <= 10


# =============================================================================
# EvidenceCollector Tests - Consent and Audit
# =============================================================================


class TestEvidenceCollectorConsent:
    """Tests for URL consent and audit functionality."""

    def test_consent_not_required_allows_all(self):
        """Should allow all URLs when consent not required."""
        collector = EvidenceCollector(require_url_consent=False)
        assert collector._check_url_consent("https://any-url.com") is True

    def test_consent_required_without_callback_blocks(self):
        """Should block when consent required but no callback."""
        callback = MagicMock(return_value=True)
        collector = EvidenceCollector(
            require_url_consent=True,
            url_consent_callback=callback,
        )
        # Remove callback to simulate missing
        collector._url_consent_callback = None
        assert collector._check_url_consent("https://example.com") is False

    def test_consent_callback_grants_access(self):
        """Should allow URL when callback returns True."""
        callback = MagicMock(return_value=True)
        collector = EvidenceCollector(
            require_url_consent=True,
            url_consent_callback=callback,
        )
        collector.set_org_context("org-123")
        assert collector._check_url_consent("https://example.com") is True
        callback.assert_called_once_with("https://example.com", "org-123")

    def test_consent_callback_denies_access(self):
        """Should block URL when callback returns False."""
        callback = MagicMock(return_value=False)
        collector = EvidenceCollector(
            require_url_consent=True,
            url_consent_callback=callback,
        )
        assert collector._check_url_consent("https://example.com") is False

    def test_audit_callback_called(self):
        """Should call audit callback on URL actions."""
        audit_callback = MagicMock()
        collector = EvidenceCollector(audit_callback=audit_callback)
        collector.set_org_context("org-456")

        collector._log_audit("https://example.com", "org-456", "fetch", True)

        audit_callback.assert_called_once_with("https://example.com", "org-456", "fetch", True)


# =============================================================================
# EvidenceCollector Tests - Connector Management
# =============================================================================


class TestEvidenceCollectorConnectors:
    """Tests for connector management."""

    def test_add_connector(self):
        """Should add connector to collection."""
        collector = EvidenceCollector()
        mock_connector = MagicMock()
        collector.add_connector("test", mock_connector)
        assert "test" in collector.connectors
        assert collector.connectors["test"] == mock_connector

    def test_set_org_context(self):
        """Should set organization context."""
        collector = EvidenceCollector()
        collector.set_org_context("my-org")
        assert collector._org_id == "my-org"

    def test_set_km_adapter(self):
        """Should set Knowledge Mound adapter."""
        collector = EvidenceCollector()
        mock_adapter = MagicMock()
        collector.set_km_adapter(mock_adapter)
        assert collector._km_adapter == mock_adapter


# =============================================================================
# EvidenceCollector Tests - KM Integration
# =============================================================================


class TestEvidenceCollectorKMIntegration:
    """Tests for Knowledge Mound integration."""

    def test_query_km_without_adapter(self):
        """Should return empty list without adapter."""
        collector = EvidenceCollector()
        result = collector.query_km_for_existing("test topic")
        assert result == []

    def test_query_km_with_adapter(self):
        """Should query adapter when available."""
        mock_adapter = MagicMock()
        mock_adapter.search_by_topic.return_value = [
            {"id": "km-1", "snippet": "Existing evidence", "reliability_score": 0.8}
        ]
        collector = EvidenceCollector(km_adapter=mock_adapter)

        result = collector.query_km_for_existing("test topic", limit=5, min_reliability=0.7)

        assert len(result) == 1
        assert result[0]["id"] == "km-1"
        mock_adapter.search_by_topic.assert_called_once_with(
            query="test topic", limit=5, min_reliability=0.7
        )

    def test_query_km_handles_error(self):
        """Should handle adapter errors gracefully."""
        mock_adapter = MagicMock()
        mock_adapter.search_by_topic.side_effect = Exception("KM error")
        collector = EvidenceCollector(km_adapter=mock_adapter)

        result = collector.query_km_for_existing("test topic")
        assert result == []


# =============================================================================
# EvidenceCollector Tests - Evidence Collection
# =============================================================================


class TestEvidenceCollectorCollection:
    """Tests for evidence collection."""

    @pytest.fixture
    def collector_with_mock_connector(self):
        """Create collector with mock web connector."""
        mock_connector = MagicMock()
        mock_connector.search = AsyncMock(return_value=[])
        return EvidenceCollector(connectors={"web": mock_connector})

    @pytest.mark.asyncio
    async def test_collect_evidence_empty(self, collector_with_mock_connector):
        """Should return empty pack when no evidence found."""
        pack = await collector_with_mock_connector.collect_evidence("Test task with no results")
        assert isinstance(pack, EvidencePack)
        assert len(pack.snippets) == 0

    @pytest.mark.asyncio
    async def test_collect_for_claims_empty(self, collector_with_mock_connector):
        """Should handle empty claims list."""
        pack = await collector_with_mock_connector.collect_for_claims([])
        assert isinstance(pack, EvidencePack)
        assert len(pack.snippets) == 0

    @pytest.mark.asyncio
    async def test_collect_for_claims_with_claims(self, collector_with_mock_connector):
        """Should collect evidence for claims."""
        claims = ["The model is 95% accurate", "Training takes 2 hours"]
        pack = await collector_with_mock_connector.collect_for_claims(claims)
        assert isinstance(pack, EvidencePack)


# =============================================================================
# EvidenceCollector Tests - Reliability Calculation
# =============================================================================


class TestEvidenceCollectorReliability:
    """Tests for reliability score calculation."""

    @pytest.fixture
    def collector(self):
        return EvidenceCollector()

    def test_calculate_reliability_github(self, collector):
        """GitHub sources should have 0.8 base reliability."""
        result = {"verified": False, "recent": False, "content": "short"}
        score = collector._calculate_reliability("github", result)
        assert score == 0.8

    def test_calculate_reliability_academic(self, collector):
        """Academic sources should have 0.9 base reliability."""
        result = {}
        score = collector._calculate_reliability("academic", result)
        assert score == 0.9

    def test_calculate_reliability_local_docs(self, collector):
        """Local docs should have 0.9 base reliability."""
        result = {}
        score = collector._calculate_reliability("local_docs", result)
        assert score == 0.9

    def test_calculate_reliability_web_search(self, collector):
        """Web search should have 0.6 base reliability."""
        result = {}
        score = collector._calculate_reliability("web_search", result)
        assert score == 0.6

    def test_calculate_reliability_unknown_source(self, collector):
        """Unknown sources should have 0.5 base reliability."""
        result = {}
        score = collector._calculate_reliability("unknown", result)
        assert score == 0.5

    def test_calculate_reliability_verified_boost(self, collector):
        """Verified content should boost reliability."""
        result = {"verified": True}
        score = collector._calculate_reliability("web_search", result)
        assert score == 0.7  # 0.6 + 0.1

    def test_calculate_reliability_recent_boost(self, collector):
        """Recent content should boost reliability."""
        result = {"recent": True}
        score = collector._calculate_reliability("web_search", result)
        assert score == 0.65  # 0.6 + 0.05

    def test_calculate_reliability_substantial_content_boost(self, collector):
        """Substantial content should boost reliability."""
        result = {"content": "x" * 1500}
        score = collector._calculate_reliability("web_search", result)
        assert score == 0.65  # 0.6 + 0.05

    def test_calculate_reliability_max_1(self, collector):
        """Reliability should not exceed 1.0."""
        result = {"verified": True, "recent": True, "content": "x" * 2000}
        score = collector._calculate_reliability("academic", result)
        assert score == 1.0


# =============================================================================
# EvidenceCollector Tests - Event Emission
# =============================================================================


class TestEvidenceCollectorEvents:
    """Tests for event emission."""

    def test_emit_evidence_events(self):
        """Should emit evidence_found events."""
        emitter = MagicMock()
        collector = EvidenceCollector(event_emitter=emitter, loop_id="test-loop")

        snippets = [
            EvidenceSnippet(
                id="e1",
                source="web",
                title="Test",
                snippet="Test content.",
                reliability_score=0.8,
            )
        ]

        collector._emit_evidence_events(snippets, ["test", "keywords"])

        emitter.emit.assert_called_once()
        call_args = emitter.emit.call_args
        assert call_args[0][0] == "evidence_found"
        assert call_args[1]["loop_id"] == "test-loop"

    def test_no_emit_without_emitter(self):
        """Should not fail when no emitter configured."""
        collector = EvidenceCollector()  # No emitter
        snippets = [EvidenceSnippet(id="e1", source="web", title="Test", snippet="Test.")]
        # Should not raise
        collector._emit_evidence_events(snippets, ["test"])


# =============================================================================
# EvidenceCollector Tests - Table Formatting
# =============================================================================


class TestEvidenceCollectorTableFormatting:
    """Tests for table data formatting."""

    @pytest.fixture
    def collector(self):
        return EvidenceCollector()

    def test_format_table_with_headers(self, collector):
        """Should format table with headers."""
        data = [["A", "B"], ["C", "D"]]
        headers = ["Col1", "Col2"]
        result = collector._format_table_for_snippet(data, headers)
        assert "Col1 | Col2" in result
        assert "A | B" in result
        assert "C | D" in result

    def test_format_table_without_headers(self, collector):
        """Should format table without headers."""
        data = [["A", "B"], ["C", "D"]]
        result = collector._format_table_for_snippet(data)
        assert "A | B" in result
        assert "C | D" in result

    def test_format_table_limits_rows(self, collector):
        """Should limit table to 15 rows."""
        data = [[f"Row{i}", f"Data{i}"] for i in range(20)]
        result = collector._format_table_for_snippet(data)
        assert "5 more rows" in result

    def test_format_table_truncates_cells(self, collector):
        """Should truncate long cell values."""
        data = [["x" * 50, "y" * 50]]
        result = collector._format_table_for_snippet(data)
        # Each cell should be truncated to 30 chars
        lines = result.split("\n")
        for line in lines:
            parts = line.split(" | ")
            for part in parts:
                assert len(part) <= 30
