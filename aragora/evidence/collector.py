"""
Evidence Collector.

Auto-collects citations and snippets from existing connectors
to provide factual grounding for debates.

SSRF Protection:
    URL fetching is restricted to prevent Server-Side Request Forgery attacks.
    By default, only URLs from allowlisted domains are fetched.

    Configuration via environment variables:
    - ARAGORA_URL_FETCH_ALL_ENABLED=true: Allow fetching any URL (still blocks
      private IPs and localhost for basic security).
    - ARAGORA_URL_ALLOWED_DOMAINS=domain1.com,domain2.com: Add custom domains
      to the allowlist.

    See aragora.config.settings.EvidenceSettings for details.
"""

import asyncio
import hashlib
import ipaddress
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, cast
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Default allowed domains for URL fetching (SSRF protection)
# These are well-known, trusted sources for research
DEFAULT_ALLOWED_DOMAINS: FrozenSet[str] = frozenset(
    {
        # Code/Documentation
        "github.com",
        "raw.githubusercontent.com",
        "gist.github.com",
        "gitlab.com",
        "bitbucket.org",
        "docs.python.org",
        "docs.anthropic.com",
        "platform.openai.com",
        "cloud.google.com",
        "docs.microsoft.com",
        "learn.microsoft.com",
        "developer.mozilla.org",
        # Research/Academic
        "arxiv.org",
        "wikipedia.org",
        "en.wikipedia.org",
        "scholar.google.com",
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        # Q&A/Forums
        "stackoverflow.com",
        "stackexchange.com",
        "reddit.com",
        "news.ycombinator.com",
        # News/Media
        "nytimes.com",
        "bbc.com",
        "bbc.co.uk",
        "reuters.com",
        "theguardian.com",
        # Cloud providers
        "aws.amazon.com",
        "azure.microsoft.com",
    }
)

from aragora.connectors.base import Connector
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
        freshness_indicator = (
            "🟢" if self.freshness_score > 0.8 else "🟡" if self.freshness_score > 0.5 else "🔴"
        )
        source_line = (
            f"Source: {self.source} ({self.reliability_score:.1f} reliability, "
            f"{freshness_indicator} {self.freshness_score:.1f} fresh)"
        )
        snippet_text = self.snippet[:500] + ("..." if len(self.snippet) > 500 else "")
        return f"""EVID-{self.id}:
{source_line}
Title: {self.title}
Snippet: {snippet_text}
URL: {self.url}
---"""

    def to_citation(self) -> str:
        """Format as an academic-style citation.

        Returns a formatted citation string like:
        [1] Title. Source (reliability: 0.9). URL
        """
        url_part = f" {self.url}" if self.url else ""
        source_info = f"{self.source.title()} (reliability: {self.reliability_score:.1f})"
        return f"[{self.id}] {self.title}. {source_info}.{url_part}"

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
        header += (
            f"Quality: {self.average_reliability:.1%} reliability, "
            f"{self.average_freshness:.1%} fresh\n\n"
        )

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
    """Collects evidence from multiple connectors for debate grounding.

    SSRF Protection:
        By default, URL fetching is restricted to allowlisted domains to prevent
        Server-Side Request Forgery attacks.

        Configure via environment variables:
        - ARAGORA_URL_FETCH_ALL_ENABLED=true: Allow any URL (trusted environments)
        - ARAGORA_URL_ALLOWED_DOMAINS=domain1,domain2: Extend allowlist
    """

    def __init__(
        self,
        connectors: Optional[Dict[str, Connector]] = None,
        event_emitter: Optional[Any] = None,
        loop_id: Optional[str] = None,
        allowed_domains: Optional[Set[str]] = None,
        require_url_consent: bool = False,
        url_consent_callback: Optional[Callable[[str, str], bool]] = None,
        audit_callback: Optional[Callable[[str, str, str, bool], None]] = None,
    ):
        """Initialize the evidence collector.

        Args:
            connectors: Dict of connector name to Connector instance
            event_emitter: Optional event emitter for real-time updates
            loop_id: Optional loop ID for event context
            allowed_domains: Set of allowed domains for URL fetching.
                           Merged with DEFAULT_ALLOWED_DOMAINS and settings.
            require_url_consent: If True, require explicit consent before fetching URLs.
                               When enabled, url_consent_callback must be provided.
            url_consent_callback: Callback to request user consent for URL fetching.
                                Signature: (url: str, org_id: str) -> bool
                                Returns True if consent is granted, False otherwise.
            audit_callback: Optional callback for audit logging of URL fetches.
                          Signature: (url: str, org_id: str, action: str, success: bool)
                          action is one of: "fetch", "blocked_ssrf", "blocked_domain", "blocked_consent"
        """
        self.connectors = connectors or {}
        self.provenance_manager = ProvenanceManager()
        self.max_snippets_per_connector = 3
        self.max_total_snippets = 8
        self.snippet_max_length = 1000
        self.event_emitter = event_emitter
        self.loop_id = loop_id

        # URL consent configuration
        self._require_url_consent = require_url_consent
        self._url_consent_callback = url_consent_callback
        self._audit_callback = audit_callback
        self._org_id: Optional[str] = None  # Set via set_org_context()

        if require_url_consent and url_consent_callback is None:
            raise ValueError("url_consent_callback is required when require_url_consent=True")

        # Load URL security settings
        try:
            from aragora.config.settings import get_settings

            settings = get_settings()
            self._url_fetch_all_enabled = settings.evidence.url_fetch_all_enabled
            additional_domains = settings.evidence.additional_allowed_domains
        except Exception:  # noqa: BLE001 - Settings may not be available
            # Fallback if settings not available
            self._url_fetch_all_enabled = False
            additional_domains = []

        # Build final allowlist: default + settings + explicit
        self._allowed_domains = set(DEFAULT_ALLOWED_DOMAINS)
        self._allowed_domains.update(additional_domains)
        if allowed_domains:
            self._allowed_domains.update(allowed_domains)

    def set_org_context(self, org_id: str) -> None:
        """Set the organization context for consent and audit tracking."""
        self._org_id = org_id

    def _check_url_consent(self, url: str) -> bool:
        """Check if URL fetch is allowed via consent gate.

        Returns True if:
        - Consent not required, OR
        - Consent callback returns True

        Logs audit event regardless of outcome.
        """
        org_id = self._org_id or "unknown"

        if not self._require_url_consent:
            return True

        if self._url_consent_callback is None:
            logger.warning(f"URL consent required but no callback configured. Blocking: {url}")
            self._log_audit(url, org_id, "blocked_consent", False)
            return False

        try:
            consent_granted = self._url_consent_callback(url, org_id)
            if not consent_granted:
                logger.info(f"URL consent denied for: {url}")
                self._log_audit(url, org_id, "blocked_consent", False)
            return consent_granted
        except Exception as e:
            logger.error(f"URL consent callback error: {e}")
            self._log_audit(url, org_id, "blocked_consent", False)
            return False

    def _log_audit(self, url: str, org_id: str, action: str, success: bool) -> None:
        """Log URL fetch action for audit trail."""
        if self._audit_callback:
            try:
                self._audit_callback(url, org_id, action, success)
            except Exception as e:
                logger.warning(f"Audit callback error: {e}")

    def add_connector(self, name: str, connector: Connector) -> None:
        """Add a connector for evidence collection."""
        self.connectors[name] = connector

    def _is_domain_allowed(self, domain: str) -> bool:
        """Check if domain is in the allowlist.

        Handles subdomains (e.g., api.github.com matches github.com).
        """
        domain = domain.lower()
        for allowed in self._allowed_domains:
            if domain == allowed or domain.endswith(f".{allowed}"):
                return True
        return False

    def _is_safe_url(self, url: str) -> bool:
        """Perform SSRF safety checks on a URL.

        Blocks:
        - localhost and loopback addresses
        - Private IP ranges (10.x, 172.16-31.x, 192.168.x)
        - Link-local addresses (169.254.x)
        - Non-standard ports
        - Non-HTTP(S) schemes
        """
        try:
            parsed = urlparse(url)

            # Block non-HTTP schemes
            if parsed.scheme not in ("http", "https"):
                logger.debug(f"SSRF: Blocked non-HTTP scheme: {parsed.scheme}")
                return False

            hostname = parsed.hostname
            if not hostname:
                return False

            # Block localhost variants
            if hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
                logger.debug(f"SSRF: Blocked localhost: {hostname}")
                return False

            # Block private IP ranges
            try:
                ip = ipaddress.ip_address(hostname)
                if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                    logger.debug(f"SSRF: Blocked private/reserved IP: {hostname}")
                    return False
            except ValueError:
                # Not an IP address (hostname), which is fine
                pass

            # Block non-standard ports (only allow 80 and 443)
            if parsed.port and parsed.port not in (80, 443):
                logger.debug(f"SSRF: Blocked non-standard port: {parsed.port}")
                return False

            return True

        except Exception as e:
            logger.debug(f"SSRF: URL parse error: {e}")
            return False

    def _emit_evidence_events(
        self,
        snippets: List[EvidenceSnippet],
        keywords: List[str],
    ) -> None:
        """Emit evidence_found events for real-time UI updates.

        Emits a single event containing all evidence snippets found,
        allowing the frontend to display evidence as it's collected.
        """
        if not self.event_emitter:
            return

        try:
            # Format snippets for the event
            evidence_data = {
                "keywords": keywords,
                "count": len(snippets),
                "snippets": [
                    {
                        "id": s.id,
                        "source": s.source,
                        "title": s.title,
                        "snippet": s.snippet[:300],  # Truncate for event payload
                        "url": s.url,
                        "reliability_score": s.reliability_score,
                        "freshness_score": s.freshness_score,
                    }
                    for s in snippets
                ],
            }

            self.event_emitter.emit(
                "evidence_found",
                loop_id=self.loop_id,
                data=evidence_data,
            )
            logger.debug(f"Emitted evidence_found event with {len(snippets)} snippets")
        except Exception as e:
            logger.warning(f"Failed to emit evidence_found event: {e}")

    async def collect_evidence(
        self,
        task: str,
        enabled_connectors: List[str] = None,
        fetch_urls: Optional[bool] = None,
    ) -> EvidencePack:
        """Collect evidence relevant to the task.

        Args:
            task: The task/topic to collect evidence for
            enabled_connectors: List of connector names to use
            fetch_urls: Override for URL fetching behavior.
                       - None (default): Use settings (ARAGORA_URL_FETCH_ALL_ENABLED)
                       - True: Allow any URL with safety checks
                       - False: Strict allowlist mode

        Returns:
            EvidencePack with collected evidence snippets
        """
        if enabled_connectors is None:
            enabled_connectors = list(self.connectors.keys())

        # Determine URL fetching mode (explicit parameter > settings > default)
        allow_all_urls = fetch_urls if fetch_urls is not None else self._url_fetch_all_enabled

        all_snippets = []
        total_searched = 0

        # First, fetch any explicit URLs mentioned in the task (with SSRF protection)
        explicit_urls = self._extract_urls(task)
        if explicit_urls:
            logger.info(f"Found {len(explicit_urls)} URL(s) in task: {explicit_urls}")
            if "web" in self.connectors:
                web_connector = self.connectors["web"]
                for url in explicit_urls:
                    try:
                        # Normalize URL
                        full_url = (
                            url if url.startswith(("http://", "https://")) else f"https://{url}"
                        )
                        parsed = urlparse(full_url)

                        org_id = self._org_id or "unknown"

                        # SSRF Protection: Always check basic safety first
                        if not self._is_safe_url(full_url):
                            logger.warning(f"SSRF: Blocked unsafe URL: {full_url}")
                            self._log_audit(full_url, org_id, "blocked_ssrf", False)
                            continue

                        # Then check allowlist (unless feature flag bypasses it)
                        if not allow_all_urls and not self._is_domain_allowed(parsed.netloc):
                            logger.info(f"Skipping non-allowlisted URL: {full_url}")
                            self._log_audit(full_url, org_id, "blocked_domain", False)
                            continue

                        # Consent gate: require explicit consent if configured
                        if not self._check_url_consent(full_url):
                            continue

                        # Special handling for GitHub repos: fetch README
                        if self._is_github_repo_url(full_url):
                            readme_snippet = await self._fetch_github_readme(
                                full_url, web_connector
                            )
                            if readme_snippet:
                                all_snippets.append(readme_snippet)
                                total_searched += 1
                                logger.info(
                                    f"Fetched GitHub README: {full_url} "
                                    f"({len(readme_snippet.snippet)} chars)"
                                )
                                continue  # Skip regular URL fetch for repos

                        # Regular URL fetch
                        if hasattr(web_connector, "fetch_url"):
                            evidence = await web_connector.fetch_url(full_url)
                            if evidence and getattr(evidence, "confidence", 0) > 0:
                                snippet = EvidenceSnippet(
                                    id=f"url_{hashlib.sha256(full_url.encode()).hexdigest()[:12]}",
                                    source="direct_url",
                                    title=getattr(evidence, "title", full_url),
                                    snippet=self._truncate_snippet(evidence.content),
                                    url=full_url,
                                    reliability_score=0.9,  # High reliability for direct URL fetch
                                    metadata={"fetched_directly": True, "original_url": url},
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
                if isinstance(result, BaseException):
                    logger.warning(f"Connector search error: {result}")
                else:
                    connector_snippets, searched_count = result
                    all_snippets.extend(connector_snippets)
                    total_searched += searched_count

        # Rank and limit snippets
        ranked_snippets = self._rank_snippets(all_snippets, keywords)[: self.max_total_snippets]

        # Record in provenance (optional - method may not exist yet)
        if hasattr(self.provenance_manager, "record_evidence_use"):
            for snippet in ranked_snippets:
                self.provenance_manager.record_evidence_use(snippet.id, task, "debate_context")

        # Emit evidence_found events for real-time UI updates
        if self.event_emitter and ranked_snippets:
            self._emit_evidence_events(ranked_snippets, keywords)

        return EvidencePack(
            topic_keywords=keywords, snippets=ranked_snippets, total_searched=total_searched
        )

    async def _search_connector(
        self, connector_name: str, connector: Connector, keywords: List[str]
    ) -> Tuple[List[EvidenceSnippet], int]:
        """Search a single connector and return snippets."""
        try:
            # Build search query from keywords
            query = " ".join(keywords[:3])  # Use top 3 keywords

            # Call connector search (assuming it has a search method)
            if hasattr(connector, "search"):
                results = await connector.search(query, limit=self.max_snippets_per_connector)
            else:
                # Fallback for connectors without search
                results = []

            snippets = []
            for i, result in enumerate(results[: self.max_snippets_per_connector]):
                # Handle Evidence objects (WebConnector) or dict results (others)
                if hasattr(result, "title"):  # Evidence object
                    snippet = EvidenceSnippet(
                        id=f"{connector_name}_{result.id}",
                        source=connector_name,
                        title=result.title,
                        snippet=self._truncate_snippet(result.content),
                        url=result.url or "",
                        reliability_score=self._calculate_reliability_from_evidence(
                            connector_name, result
                        ),
                        metadata=result.metadata,
                    )
                else:  # Dict result from other connectors
                    result_dict = cast(dict[str, Any], result)
                    snippet = EvidenceSnippet(
                        id=f"{connector_name}_{i}",
                        source=connector_name,
                        title=result_dict.get("title", result_dict.get("name", "Unknown")),
                        snippet=self._truncate_snippet(
                            result_dict.get("content", result_dict.get("text", ""))
                        ),
                        url=result_dict.get("url", ""),
                        reliability_score=self._calculate_reliability(connector_name, result_dict),
                        metadata=result_dict,
                    )
                snippets.append(snippet)

            return snippets, len(results)

        except Exception as e:
            logger.warning(f"Error searching {connector_name}: {e}")
            return [], 0

    def _extract_urls(self, task: str) -> List[str]:
        """Extract explicit URLs and domain references from task description.

        Enhanced detection includes:
        - Full URLs (http/https)
        - www URLs
        - Common domain TLDs
        - GitHub repo references (owner/repo or github.com/owner/repo)
        """
        urls = []

        # Pattern 1: Full URLs with scheme
        full_url_pattern = r"https?://[^\s)<>\[\]\"']+"
        urls.extend(re.findall(full_url_pattern, task, re.IGNORECASE))

        # Pattern 2: www URLs
        www_pattern = r"www\.[^\s)<>\[\]\"']+"
        urls.extend(re.findall(www_pattern, task, re.IGNORECASE))

        # Pattern 3: GitHub repos without scheme (github.com/owner/repo)
        github_pattern = r"\bgithub\.com/([\w.-]+)/([\w.-]+)"
        for match in re.finditer(github_pattern, task, re.IGNORECASE):
            full_match = match.group(0)
            # Only add if not already captured by full URL pattern
            if not any(full_match in url for url in urls):
                urls.append(f"https://{full_match}")

        # Pattern 4: Common domain TLDs
        domain_pattern = (
            r"\b([a-zA-Z0-9][-a-zA-Z0-9]*\."
            r"(?:com|org|net|io|ai|dev|app|co|edu|gov)"
            r"(?:/[^\s)<>\[\]\"']*)?)\b"
        )
        for match in re.findall(domain_pattern, task, re.IGNORECASE):
            if match and match not in urls:
                urls.append(match)

        # Deduplicate while preserving order, normalize trailing slashes
        seen = set()
        unique_urls = []
        for url in urls:
            # Normalize: strip trailing slashes for comparison
            url_normalized = url.rstrip("/").lower()
            if url_normalized not in seen:
                seen.add(url_normalized)
                unique_urls.append(url)
        return unique_urls

    def _is_github_repo_url(self, url: str) -> bool:
        """Check if URL points to a GitHub repository root."""
        pattern = r"^https?://github\.com/([\w.-]+)/([\w.-]+)/?$"
        return bool(re.match(pattern, url, re.IGNORECASE))

    def _parse_github_repo(self, url: str) -> Optional[Tuple[str, str]]:
        """Parse owner and repo from GitHub URL."""
        pattern = r"github\.com/([\w.-]+)/([\w.-]+)"
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            return match.group(1), match.group(2)
        return None

    async def _fetch_github_readme(self, url: str, web_connector: Any) -> Optional[EvidenceSnippet]:
        """Fetch README.md from a GitHub repository.

        Tries multiple branch names (main, master, develop) and README variants.

        Args:
            url: GitHub repository URL
            web_connector: Web connector with fetch_url method

        Returns:
            EvidenceSnippet with README content, or None if not found
        """
        parsed = self._parse_github_repo(url)
        if not parsed:
            return None

        owner, repo = parsed
        branches = ["main", "master", "develop"]
        readme_files = ["README.md", "readme.md", "Readme.md", "README.rst", "README.txt"]

        for branch in branches:
            for readme_file in readme_files:
                raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{readme_file}"
                try:
                    if hasattr(web_connector, "fetch_url"):
                        evidence = await web_connector.fetch_url(raw_url)
                        if evidence and getattr(evidence, "content", ""):
                            content = evidence.content
                            # Skip if it looks like a 404 page
                            if "404" in content[:100] or len(content) < 50:
                                continue

                            logger.info(
                                f"Fetched README from {owner}/{repo} ({branch}/{readme_file})"
                            )
                            return EvidenceSnippet(
                                id=f"gh_{owner}_{repo}_readme",
                                source="github_readme",
                                title=f"{owner}/{repo} README",
                                snippet=self._truncate_snippet(content),
                                url=f"https://github.com/{owner}/{repo}",
                                reliability_score=0.85,  # High reliability for official README
                                metadata={
                                    "owner": owner,
                                    "repo": repo,
                                    "branch": branch,
                                    "file": readme_file,
                                    "raw_url": raw_url,
                                },
                            )
                except Exception as e:
                    logger.debug(f"Failed to fetch {raw_url}: {e}")
                    continue

        logger.warning(f"Could not find README for {owner}/{repo}")
        return None

    def _extract_keywords(self, task: str) -> List[str]:
        """Extract search keywords from task description."""
        # Simple keyword extraction - split and filter
        words = re.findall(r"\b\w+\b", task.lower())

        # Remove stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "shall",
            "over",
            "under",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "from",
            "up",
            "down",
            "out",
            "off",
            "about",
            "between",
            "against",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
        }
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]

        # Get unique keywords, prioritize nouns/important terms
        unique_keywords = list(set(keywords))

        # Boost keywords that appear in task title or are technical
        boosted = []
        for keyword in unique_keywords:
            if any(char in keyword for char in ["#", "ai", "tech", "data", "system", "code"]):
                boosted.extend([keyword] * 2)  # Duplicate for higher weight
            else:
                boosted.append(keyword)

        return boosted[:5]  # Top 5 keywords

    def _truncate_snippet(self, text: str) -> str:
        """Truncate snippet to max length, trying to end at sentence boundary."""
        if len(text) <= self.snippet_max_length:
            return text

        truncated = text[: self.snippet_max_length]

        # Try to find sentence end
        last_sentence_end = max(truncated.rfind(". "), truncated.rfind("! "), truncated.rfind("? "))

        if last_sentence_end > self.snippet_max_length * 0.7:  # If we can keep most of it
            return truncated[: last_sentence_end + 1]

        return truncated + "..."

    def _calculate_reliability(self, connector_name: str, result: Dict[str, Any]) -> float:
        """Calculate reliability score based on source and metadata."""
        base_scores = {
            "github": 0.8,  # Code/docs from GitHub
            "local_docs": 0.9,  # Local documentation
            "web_search": 0.6,  # General web results
            "academic": 0.9,  # Academic sources
        }

        base_score = base_scores.get(connector_name, 0.5)

        # Adjust based on metadata
        if result.get("verified", False):
            base_score += 0.1
        if result.get("recent", False):
            base_score += 0.05
        if len(result.get("content", "")) > 1000:  # Substantial content
            base_score += 0.05

        return min(1.0, base_score)

    def _calculate_reliability_from_evidence(self, connector_name: str, evidence) -> float:
        """Calculate reliability score from Evidence object."""
        base_scores = {
            "github": 0.8,
            "local_docs": 0.9,
            "web": 0.6,  # WebConnector uses 'web' as source
            "academic": 0.9,
        }

        base_score = base_scores.get(connector_name, 0.5)

        # Use evidence authority and confidence
        base_score = (base_score + evidence.authority + evidence.confidence) / 3.0

        # Adjust based on content length
        if len(evidence.content) > 1000:
            base_score += 0.05

        return min(1.0, base_score)

    def _rank_snippets(
        self, snippets: List[EvidenceSnippet], keywords: List[str]
    ) -> List[EvidenceSnippet]:
        """Rank snippets by relevance, reliability, and freshness."""

        def score_snippet(snippet: EvidenceSnippet) -> float:
            relevance_score = 0.0
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
                relevance_normalized * 0.50
                + snippet.reliability_score * 0.35
                + snippet.freshness_score * 0.15
            )

        return sorted(snippets, key=score_snippet, reverse=True)

    async def collect_for_claims(
        self,
        claims: List[str],
        enabled_connectors: List[str] = None,
        max_per_claim: int = 2,
    ) -> EvidencePack:
        """Collect evidence specifically for a list of claims.

        This is used during debate rounds to refresh evidence based on
        claims that emerge from proposals and critiques.

        Args:
            claims: List of claim strings to find evidence for
            enabled_connectors: Optional list of connector names to use
            max_per_claim: Maximum snippets to collect per claim

        Returns:
            EvidencePack with evidence snippets for the claims
        """
        if enabled_connectors is None:
            enabled_connectors = list(self.connectors.keys())

        if not claims:
            return EvidencePack(
                topic_keywords=[],
                snippets=[],
                total_searched=0,
            )

        all_snippets: List[EvidenceSnippet] = []
        total_searched = 0
        all_keywords: List[str] = []

        # Process each claim
        for claim in claims[:5]:  # Limit to 5 claims to avoid API overload
            # Extract keywords from the claim
            claim_keywords = self._extract_keywords(claim)
            all_keywords.extend(claim_keywords)

            # Search connectors for this claim
            search_tasks = []
            for connector_name in enabled_connectors:
                if connector_name in self.connectors:
                    connector = self.connectors[connector_name]
                    search_tasks.append(
                        self._search_connector(connector_name, connector, claim_keywords)
                    )

            if search_tasks:
                results = await asyncio.gather(*search_tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, BaseException):
                        logger.warning(f"Connector search error for claim: {result}")
                    else:
                        connector_snippets, searched_count = result
                        # Limit per claim
                        all_snippets.extend(connector_snippets[:max_per_claim])
                        total_searched += searched_count

        # Deduplicate by snippet content hash
        seen_hashes: set = set()
        unique_snippets: List[EvidenceSnippet] = []
        for snippet in all_snippets:
            content_hash = hashlib.md5(snippet.snippet.encode(), usedforsecurity=False).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_snippets.append(snippet)

        # Rank and limit
        unique_keywords = list(set(all_keywords))
        ranked_snippets = self._rank_snippets(unique_snippets, unique_keywords)
        final_snippets = ranked_snippets[: self.max_total_snippets]

        logger.info(
            f"evidence_for_claims claims={len(claims)} snippets={len(final_snippets)} "
            f"searched={total_searched}"
        )

        return EvidencePack(
            topic_keywords=unique_keywords[:10],
            snippets=final_snippets,
            total_searched=total_searched,
        )

    def extract_claims_from_text(self, text: str) -> List[str]:
        """Extract factual claims from text that could benefit from evidence.

        Looks for:
        - Statements with numbers or statistics
        - Statements with definitive language ("is", "are", "proven")
        - Comparative statements ("better than", "faster than")
        - References to studies, research, or sources

        Args:
            text: Text to extract claims from

        Returns:
            List of claim strings
        """
        claims: List[str] = []

        # Split into sentences
        sentences = re.split(r"[.!?]\s+", text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20 or len(sentence) > 500:
                continue

            # Check for claim indicators
            claim_indicators = [
                r"\d+%",  # Percentages
                r"\d+ (times|percent|million|billion)",  # Quantitative claims
                r"(studies|research|evidence) (show|suggest|indicate)",
                r"(proven|demonstrated|established) that",
                r"(better|worse|faster|slower|more|less) than",
                r"according to",
                r"(is|are) (known|considered|recognized)",
                r"(always|never|all|none|every)",  # Absolute claims
            ]

            for pattern in claim_indicators:
                if re.search(pattern, sentence, re.IGNORECASE):
                    claims.append(sentence)
                    break

        # Deduplicate
        return list(dict.fromkeys(claims))[:10]  # Max 10 claims


__all__ = [
    "EvidenceSnippet",
    "EvidencePack",
    "EvidenceCollector",
    "DEFAULT_ALLOWED_DOMAINS",
]
