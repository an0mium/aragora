"""
Evidence Fetch Skill.

Provides evidence collection capabilities from various sources
for debate support and claim verification.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..base import (
    Skill,
    SkillCapability,
    SkillContext,
    SkillManifest,
    SkillResult,
)

logger = logging.getLogger(__name__)


class EvidenceFetchSkill(Skill):
    """
    Skill for fetching evidence from various sources.

    Supports:
    - Web page content extraction
    - Academic paper search (via Semantic Scholar)
    - News article retrieval
    - Fact-checking database queries
    """

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="evidence_fetch",
            version="1.0.0",
            description="Fetch evidence from various sources for claim verification",
            capabilities=[
                SkillCapability.EVIDENCE_COLLECTION,
                SkillCapability.WEB_FETCH,
                SkillCapability.EXTERNAL_API,
            ],
            input_schema={
                "claim": {
                    "type": "string",
                    "description": "The claim to find evidence for",
                    "required": True,
                },
                "sources": {
                    "type": "array",
                    "description": "Evidence sources to query (web, academic, news, facts)",
                    "default": ["web", "academic"],
                },
                "max_results": {
                    "type": "number",
                    "description": "Maximum results per source",
                    "default": 5,
                },
                "url": {
                    "type": "string",
                    "description": "Specific URL to fetch evidence from",
                },
            },
            tags=["evidence", "research", "verification"],
            debate_compatible=True,
            requires_debate_context=False,
            max_execution_time_seconds=60.0,
            rate_limit_per_minute=20,
        )

    async def execute(
        self,
        input_data: Dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        """Execute evidence fetch."""
        claim = input_data.get("claim", "")
        url = input_data.get("url")
        sources = input_data.get("sources", ["web", "academic"])
        max_results = input_data.get("max_results", 5)

        if not claim and not url:
            return SkillResult.create_failure(
                "Either 'claim' or 'url' is required",
                error_code="missing_input",
            )

        try:
            results: Dict[str, Any] = {
                "claim": claim,
                "sources_queried": sources,
            }

            # Fetch from specific URL if provided
            if url:
                url_evidence = await self._fetch_url(url)
                results["url_content"] = url_evidence

            # Query each requested source
            if "web" in sources and claim:
                web_results = await self._search_web(claim, max_results)
                results["web_evidence"] = web_results

            if "academic" in sources and claim:
                academic_results = await self._search_academic(claim, max_results)
                results["academic_evidence"] = academic_results

            if "news" in sources and claim:
                news_results = await self._search_news(claim, max_results)
                results["news_evidence"] = news_results

            if "facts" in sources and claim:
                fact_results = await self._check_facts(claim)
                results["fact_checks"] = fact_results

            # Calculate evidence summary
            total_evidence = (
                len(results.get("web_evidence", []))
                + len(results.get("academic_evidence", []))
                + len(results.get("news_evidence", []))
            )
            results["total_evidence_items"] = total_evidence

            return SkillResult.create_success(results)

        except Exception as e:
            logger.exception(f"Evidence fetch failed: {e}")
            return SkillResult.create_failure(f"Evidence fetch failed: {e}")

    async def _fetch_url(self, url: str) -> Dict[str, Any]:
        """Fetch and extract content from a URL."""
        try:
            import httpx
        except ImportError:
            return {"error": "httpx not installed"}

        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                content = response.text

                # Try to extract main content
                extracted = self._extract_main_content(content)

                return {
                    "url": str(response.url),
                    "status_code": response.status_code,
                    "content_type": content_type,
                    "content": extracted[:10000],  # Limit size
                    "title": self._extract_title(content),
                }

        except Exception as e:
            logger.warning(f"URL fetch error: {e}")
            return {"url": url, "error": str(e)}

    def _extract_main_content(self, html: str) -> str:
        """Extract main text content from HTML."""
        try:
            from html.parser import HTMLParser

            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text_parts: List[str] = []
                    self.skip_tags = {"script", "style", "nav", "header", "footer"}
                    self.current_skip = False

                def handle_starttag(self, tag, attrs):
                    if tag in self.skip_tags:
                        self.current_skip = True

                def handle_endtag(self, tag):
                    if tag in self.skip_tags:
                        self.current_skip = False

                def handle_data(self, data):
                    if not self.current_skip:
                        text = data.strip()
                        if text:
                            self.text_parts.append(text)

            extractor = TextExtractor()
            extractor.feed(html)
            return " ".join(extractor.text_parts)

        except Exception:
            # Fall back to basic text extraction
            import re

            text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            return " ".join(text.split())

    def _extract_title(self, html: str) -> Optional[str]:
        """Extract page title from HTML."""
        import re

        match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        return match.group(1).strip() if match else None

    async def _search_web(
        self,
        query: str,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Search the web for evidence."""
        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                search_results = ddgs.text(query, max_results=max_results)
                for r in search_results:
                    results.append(
                        {
                            "title": r.get("title", ""),
                            "url": r.get("href", ""),
                            "snippet": r.get("body", ""),
                            "source_type": "web",
                        }
                    )
            return results

        except ImportError:
            logger.debug("duckduckgo-search not installed")
            return []
        except Exception as e:
            logger.warning(f"Web search error: {e}")
            return []

    async def _search_academic(
        self,
        query: str,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Search academic papers via Semantic Scholar."""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={
                        "query": query,
                        "limit": max_results,
                        "fields": "title,abstract,url,year,citationCount,authors",
                    },
                    timeout=20.0,
                )

                if response.status_code != 200:
                    return []

                data = response.json()
                results = []
                for paper in data.get("data", []):
                    authors = paper.get("authors", [])
                    author_names = [a.get("name", "") for a in authors[:3]]

                    results.append(
                        {
                            "title": paper.get("title", ""),
                            "abstract": paper.get("abstract", "")[:500]
                            if paper.get("abstract")
                            else "",
                            "url": paper.get("url", ""),
                            "year": paper.get("year"),
                            "citation_count": paper.get("citationCount", 0),
                            "authors": author_names,
                            "source_type": "academic",
                        }
                    )
                return results

        except Exception as e:
            logger.warning(f"Academic search error: {e}")
            return []

    async def _search_news(
        self,
        query: str,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Search recent news articles."""
        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                news_results = ddgs.news(query, max_results=max_results)
                for r in news_results:
                    results.append(
                        {
                            "title": r.get("title", ""),
                            "url": r.get("url", ""),
                            "body": r.get("body", ""),
                            "date": r.get("date", ""),
                            "source": r.get("source", ""),
                            "source_type": "news",
                        }
                    )
            return results

        except ImportError:
            logger.debug("duckduckgo-search not installed")
            return []
        except Exception as e:
            logger.warning(f"News search error: {e}")
            return []

    async def _check_facts(self, claim: str) -> List[Dict[str, Any]]:
        """Check claim against fact-checking sources."""
        # This is a placeholder for fact-checking integration
        # Could integrate with ClaimBuster, Google Fact Check API, etc.
        logger.debug("Fact checking not implemented yet")
        return []


# Skill instance for registration
SKILLS = [EvidenceFetchSkill()]
