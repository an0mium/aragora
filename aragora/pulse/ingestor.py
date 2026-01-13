"""
Trending Pulse Ingestor.

Fetches real-time trending topics from social media platforms
for dynamic debate topic generation.

Production features:
- Exponential backoff with configurable retries
- Circuit breaker for failing APIs
- Proper logging (no print statements)
- Input validation
- Multiple platform support (Twitter, HackerNews, Reddit)
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import httpx

from aragora.core import Message, Environment
from aragora.exceptions import ExternalServiceError
from aragora.resilience import CircuitBreaker

logger = logging.getLogger(__name__)


@dataclass
class TrendingTopic:
    """A trending topic from social media."""

    platform: str  # "twitter", "reddit", etc.
    topic: str
    volume: int = 0  # engagement metric
    category: str = ""  # "tech", "politics", etc.
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def to_debate_prompt(self) -> str:
        """Convert to a debate-ready prompt."""
        return f"Debate the implications of trending topic: '{self.topic}' ({self.platform}, {self.volume} engagement)"


@dataclass
class TrendingTopicOutcome:
    """Records the outcome of a debate on a trending topic.

    This enables analytics on which trending topics lead to productive debates.
    """

    topic: str
    platform: str
    debate_id: str
    consensus_reached: bool
    confidence: float
    rounds_used: int = 0
    timestamp: float = field(default_factory=time.time)
    category: str = ""
    volume: int = 0  # Original volume at debate time


class PulseIngestor(ABC):
    """Abstract base class for social media ingestors."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_delay: float = 1.0,
        max_retries: int = 3,
        base_retry_delay: float = 1.0,
    ):
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        self.last_request_time = 0
        self.cache: Dict[str, List[TrendingTopic]] = {}
        self.cache_ttl = 300  # 5 minutes
        self.circuit_breaker = CircuitBreaker()

    async def _rate_limit(self):
        """Enforce rate limiting."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = now

    async def _retry_with_backoff(self, coro_factory, fallback_fn=None):
        """Execute a coroutine with exponential backoff retry.

        Args:
            coro_factory: Callable that returns a new coroutine on each call
            fallback_fn: Optional fallback function if all retries fail

        Returns:
            Result from successful coroutine or fallback
        """
        if not self.circuit_breaker.can_proceed():
            logger.debug("Circuit breaker open, using fallback")
            return fallback_fn() if fallback_fn else []

        last_error = None
        for attempt in range(self.max_retries):
            try:
                await self._rate_limit()
                result = await coro_factory()
                self.circuit_breaker.record_success()
                return result
            except Exception as e:
                last_error = e
                delay = self.base_retry_delay * (2**attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed: {e}. "
                    f"Retrying in {delay:.1f}s"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(delay)

        # All retries failed
        self.circuit_breaker.record_failure()
        logger.error(f"All {self.max_retries} retries failed: {last_error}")

        if fallback_fn:
            return fallback_fn()
        return []

    @abstractmethod
    async def fetch_trending(self, limit: int = 10) -> List[TrendingTopic]:
        """Fetch trending topics from the platform."""
        raise NotImplementedError("Subclasses must implement fetch_trending method")

    def _filter_content(
        self, topics: List[TrendingTopic], filters: Dict[str, Any]
    ) -> List[TrendingTopic]:
        """Apply content filters to remove harmful/inappropriate content."""
        filtered = []

        for topic in topics:
            # Skip if sentiment analysis indicates high toxicity (placeholder)
            if filters.get("skip_toxic", True) and self._is_toxic(topic.topic):
                continue

            # Category filtering
            if filters.get("categories") and topic.category not in filters["categories"]:
                continue

            # Volume threshold
            if filters.get("min_volume", 0) > 0 and topic.volume < filters["min_volume"]:
                continue

            filtered.append(topic)

        return filtered

    def _is_toxic(self, text: str) -> bool:
        """Enhanced toxicity check with categorized patterns.

        Uses weighted keyword matching across categories:
        - High severity: explicit hate speech, violence threats
        - Medium severity: harassment, discrimination
        - Low severity: profanity, adult content markers
        """
        text_lower = text.lower()

        # High severity - immediate reject
        high_severity = [
            "kill",
            "murder",
            "attack",
            "bomb",
            "terrorist",
            "hate crime",
            "genocide",
            "ethnic cleansing",
        ]
        if any(term in text_lower for term in high_severity):
            return True

        # Medium severity - context-dependent
        medium_severity = [
            "hate",
            "violence",
            "racist",
            "sexist",
            "homophobic",
            "slur",
            "harass",
            "threat",
            "abuse",
            "bully",
        ]
        medium_count = sum(1 for term in medium_severity if term in text_lower)
        if medium_count >= 2:
            return True

        # Low severity - adult content markers
        low_severity = ["nsfw", "explicit", "18+", "adult only", "xxx"]
        if any(term in text_lower for term in low_severity):
            return True

        return False


class TwitterIngestor(PulseIngestor):
    """Twitter/X trending topics ingestor using Twitter API v2."""

    def __init__(self, bearer_token: Optional[str] = None, **kwargs):
        super().__init__(api_key=bearer_token, **kwargs)
        self.base_url = "https://api.twitter.com/2"

    async def fetch_trending(self, limit: int = 10) -> List[TrendingTopic]:
        """Fetch trending topics from Twitter."""
        # Validate limit
        limit = max(1, min(limit, 50))

        if not self.api_key:
            logger.debug("No Twitter API key, using mock data")
            return self._mock_trending_data(limit)

        async def _fetch():
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Get trending topics for a location (WOEID 1 = worldwide)
                url = f"{self.base_url}/trends/place.json"
                params = {"id": 1}  # Worldwide
                headers = {"Authorization": f"Bearer {self.api_key}"}

                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()

                data = response.json()

                # Validate response structure
                if not isinstance(data, list) or len(data) == 0:
                    raise ValueError("Invalid Twitter API response format")
                if "trends" not in data[0]:
                    raise ValueError("Missing 'trends' key in response")

                topics = []
                for trend in data[0]["trends"][:limit]:
                    topic = TrendingTopic(
                        platform="twitter",
                        topic=trend["name"],
                        volume=trend.get("tweet_volume") or 0,
                        category=self._categorize_topic(trend["name"]),
                        raw_data=trend,
                    )
                    topics.append(topic)

                return topics

        return await self._retry_with_backoff(
            _fetch, fallback_fn=lambda: self._mock_trending_data(limit)
        )

    def _categorize_topic(self, topic: str) -> str:
        """Simple categorization based on keywords."""
        topic_lower = topic.lower()
        if any(word in topic_lower for word in ["ai", "tech", "code", "software"]):
            return "tech"
        elif any(word in topic_lower for word in ["politics", "election", "government"]):
            return "politics"
        elif any(word in topic_lower for word in ["climate", "environment", "green"]):
            return "environment"
        else:
            return "general"

    def _mock_trending_data(self, limit: int) -> List[TrendingTopic]:
        """Mock trending data for development/testing."""
        mock_topics = [
            TrendingTopic("twitter", "#AIJobDisplacement", 125000, "tech"),
            TrendingTopic("twitter", "#ClimateChange", 98000, "environment"),
            TrendingTopic("twitter", "#Election2024", 200000, "politics"),
            TrendingTopic("twitter", "#QuantumComputing", 45000, "tech"),
            TrendingTopic("twitter", "#RenewableEnergy", 78000, "environment"),
        ]
        return mock_topics[:limit]


class HackerNewsIngestor(PulseIngestor):
    """Hacker News trending stories ingestor using Algolia API (free, no auth)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = "https://hn.algolia.com/api/v1"

    async def fetch_trending(self, limit: int = 10) -> List[TrendingTopic]:
        """Fetch top stories from Hacker News."""
        limit = max(1, min(limit, 50))

        async def _fetch():
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Get front page stories sorted by popularity
                url = f"{self.base_url}/search"
                params = {
                    "tags": "front_page",
                    "hitsPerPage": limit,
                }

                response = await client.get(url, params=params)
                response.raise_for_status()

                data = response.json()

                # Validate response
                if "hits" not in data:
                    raise ValueError("Invalid HN API response format")

                topics = []
                for story in data["hits"][:limit]:
                    topic = TrendingTopic(
                        platform="hackernews",
                        topic=story.get("title", "Untitled"),
                        volume=story.get("points", 0),
                        category=self._categorize_topic(story.get("title", "")),
                        raw_data={
                            "url": story.get("url"),
                            "author": story.get("author"),
                            "num_comments": story.get("num_comments", 0),
                            "objectID": story.get("objectID"),
                        },
                    )
                    topics.append(topic)

                return topics

        return await self._retry_with_backoff(
            _fetch, fallback_fn=lambda: self._mock_trending_data(limit)
        )

    def _categorize_topic(self, title: str) -> str:
        """Categorize HN story based on title keywords."""
        title_lower = title.lower()
        if any(word in title_lower for word in ["ai", "gpt", "llm", "machine learning", "neural"]):
            return "ai"
        elif any(word in title_lower for word in ["startup", "funding", "vc", "acquisition"]):
            return "business"
        elif any(word in title_lower for word in ["rust", "python", "javascript", "go ", "code"]):
            return "programming"
        elif any(word in title_lower for word in ["security", "hack", "vulnerability", "breach"]):
            return "security"
        return "tech"

    def _mock_trending_data(self, limit: int) -> List[TrendingTopic]:
        """Mock HN data for development/testing."""
        mock_topics = [
            TrendingTopic("hackernews", "Show HN: I built an AI debate platform", 342, "ai"),
            TrendingTopic(
                "hackernews", "Why Rust is the future of systems programming", 256, "programming"
            ),
            TrendingTopic("hackernews", "The hidden costs of technical debt", 189, "tech"),
            TrendingTopic("hackernews", "OpenAI announces GPT-5 preview", 521, "ai"),
            TrendingTopic(
                "hackernews", "Startup raises $50M for quantum computing", 134, "business"
            ),
        ]
        return mock_topics[:limit]


class RedditIngestor(PulseIngestor):
    """Reddit trending posts ingestor using public JSON API (no auth required)."""

    DEFAULT_SUBREDDITS = ["technology", "programming", "science", "worldnews"]

    def __init__(self, subreddits: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.subreddits = subreddits or self.DEFAULT_SUBREDDITS
        self.base_url = "https://www.reddit.com"

    async def fetch_trending(self, limit: int = 10) -> List[TrendingTopic]:
        """Fetch hot posts from configured subreddits."""
        limit = max(1, min(limit, 50))
        per_sub_limit = max(1, limit // len(self.subreddits))

        async def _fetch():
            async with httpx.AsyncClient(timeout=10.0) as client:
                client.headers["User-Agent"] = "Aragora/1.0 (debate-platform)"

                all_topics = []
                for subreddit in self.subreddits:
                    try:
                        url = f"{self.base_url}/r/{subreddit}/hot.json"
                        params = {"limit": per_sub_limit}

                        response = await client.get(url, params=params)
                        response.raise_for_status()

                        data = response.json()

                        # Validate response
                        if "data" not in data or "children" not in data["data"]:
                            logger.warning(f"Invalid Reddit response for r/{subreddit}")
                            continue

                        for post in data["data"]["children"][:per_sub_limit]:
                            post_data = post["data"]
                            topic = TrendingTopic(
                                platform="reddit",
                                topic=post_data.get("title", "Untitled"),
                                volume=post_data.get("score", 0),
                                category=self._categorize_subreddit(subreddit),
                                raw_data={
                                    "subreddit": subreddit,
                                    "url": post_data.get("url"),
                                    "author": post_data.get("author"),
                                    "num_comments": post_data.get("num_comments", 0),
                                    "permalink": post_data.get("permalink"),
                                },
                            )
                            all_topics.append(topic)
                    except Exception as e:
                        logger.warning(f"Error fetching r/{subreddit}: {e}")
                        continue

                return all_topics[:limit]

        return await self._retry_with_backoff(
            _fetch, fallback_fn=lambda: self._mock_trending_data(limit)
        )

    def _categorize_subreddit(self, subreddit: str) -> str:
        """Map subreddit to category."""
        mapping = {
            "technology": "tech",
            "programming": "programming",
            "science": "science",
            "worldnews": "news",
            "politics": "politics",
            "askscience": "science",
            "machinelearning": "ai",
            "artificial": "ai",
        }
        return mapping.get(subreddit.lower(), "general")

    def _mock_trending_data(self, limit: int) -> List[TrendingTopic]:
        """Mock Reddit data for development/testing."""
        mock_topics = [
            TrendingTopic(
                "reddit", "Scientists discover high-temperature superconductor", 15420, "science"
            ),
            TrendingTopic("reddit", "New programming language gains traction", 8934, "programming"),
            TrendingTopic("reddit", "EU passes sweeping AI regulation", 12567, "news"),
            TrendingTopic("reddit", "Major tech company announces layoffs", 9823, "tech"),
            TrendingTopic("reddit", "Breakthrough in fusion energy announced", 18234, "science"),
        ]
        return mock_topics[:limit]


class GitHubTrendingIngestor(PulseIngestor):
    """GitHub Trending repositories ingestor using GitHub Search API.

    Uses the GitHub Search API to find recently created repositories
    with high star counts, simulating "trending" repositories.
    No authentication required for basic usage (60 requests/hour limit).
    """

    def __init__(self, access_token: Optional[str] = None, **kwargs):
        """Initialize GitHub trending ingestor.

        Args:
            access_token: Optional GitHub personal access token for higher rate limits
                          (5000 requests/hour authenticated vs 60 unauthenticated)
        """
        super().__init__(api_key=access_token, **kwargs)
        self.base_url = "https://api.github.com"
        # Set lower rate limit delay for unauthenticated requests
        if not access_token:
            self.rate_limit_delay = 2.0  # Be more conservative without auth

    async def fetch_trending(self, limit: int = 10) -> List[TrendingTopic]:
        """Fetch trending repositories from GitHub.

        Queries recently created repositories sorted by stars to simulate
        trending repositories. Uses the Search API which doesn't require auth.
        """
        limit = max(1, min(limit, 30))  # GitHub API returns max 30 per page

        async def _fetch():
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Build headers
                headers = {
                    "Accept": "application/vnd.github.v3+json",
                    "User-Agent": "Aragora/1.0 (debate-platform)",
                }
                if self.api_key:
                    headers["Authorization"] = f"token {self.api_key}"

                # Search for repositories created in the last 7 days, sorted by stars
                from datetime import datetime, timedelta

                week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

                url = f"{self.base_url}/search/repositories"
                params = {
                    "q": f"created:>{week_ago}",
                    "sort": "stars",
                    "order": "desc",
                    "per_page": limit,
                }

                response = await client.get(url, headers=headers, params=params)

                # Check for rate limiting
                if response.status_code == 403:
                    remaining = response.headers.get("X-RateLimit-Remaining", "0")
                    if remaining == "0":
                        reset_time = response.headers.get("X-RateLimit-Reset", "")
                        logger.warning(f"GitHub rate limit exceeded. Reset at: {reset_time}")
                        raise ExternalServiceError(
                            service="GitHub API",
                            reason=f"Rate limit exceeded. Reset at: {reset_time}",
                            status_code=403,
                        )

                response.raise_for_status()
                data = response.json()

                # Validate response
                if "items" not in data:
                    raise ValueError("Invalid GitHub API response format")

                topics = []
                for repo in data["items"][:limit]:
                    topic = TrendingTopic(
                        platform="github",
                        topic=f"{repo['full_name']}: {repo.get('description', 'No description')[:100]}",
                        volume=repo.get("stargazers_count", 0),
                        category=self._categorize_repo(repo),
                        raw_data={
                            "full_name": repo["full_name"],
                            "url": repo["html_url"],
                            "stars": repo.get("stargazers_count", 0),
                            "forks": repo.get("forks_count", 0),
                            "language": repo.get("language"),
                            "description": repo.get("description"),
                            "created_at": repo.get("created_at"),
                            "topics": repo.get("topics", []),
                        },
                    )
                    topics.append(topic)

                logger.debug(f"Fetched {len(topics)} trending GitHub repositories")
                return topics

        return await self._retry_with_backoff(
            _fetch, fallback_fn=lambda: self._mock_trending_data(limit)
        )

    def _categorize_repo(self, repo: Dict[str, Any]) -> str:
        """Categorize repository based on language and topics."""
        language = (repo.get("language") or "").lower()
        topics = [t.lower() for t in repo.get("topics", [])]
        description = (repo.get("description") or "").lower()

        # Check topics first (most specific)
        ai_keywords = ["machine-learning", "deep-learning", "ai", "llm", "gpt", "neural-network"]
        if any(t in topics for t in ai_keywords) or any(
            k in description for k in ["ai", "llm", "machine learning"]
        ):
            return "ai"

        web_keywords = ["react", "vue", "angular", "frontend", "web", "nextjs"]
        if any(t in topics for t in web_keywords):
            return "web"

        devops_keywords = ["docker", "kubernetes", "devops", "ci-cd", "infrastructure"]
        if any(t in topics for t in devops_keywords):
            return "devops"

        security_keywords = ["security", "pentesting", "vulnerability", "ctf"]
        if any(t in topics for t in security_keywords):
            return "security"

        # Fall back to language
        lang_categories = {
            "rust": "systems",
            "go": "systems",
            "c": "systems",
            "c++": "systems",
            "python": "programming",
            "javascript": "web",
            "typescript": "web",
        }
        if language in lang_categories:
            return lang_categories[language]

        return "programming"

    def _mock_trending_data(self, limit: int) -> List[TrendingTopic]:
        """Mock GitHub trending data for development/testing."""
        mock_topics = [
            TrendingTopic(
                "github",
                "anthropics/claude-code: Official Anthropic CLI for Claude",
                8500,
                "ai",
                raw_data={"full_name": "anthropics/claude-code", "language": "TypeScript"},
            ),
            TrendingTopic(
                "github",
                "rust-lang/cargo: The Rust package manager",
                5200,
                "systems",
                raw_data={"full_name": "rust-lang/cargo", "language": "Rust"},
            ),
            TrendingTopic(
                "github",
                "vercel/ai: Build AI-powered applications with React",
                4100,
                "ai",
                raw_data={"full_name": "vercel/ai", "language": "TypeScript"},
            ),
            TrendingTopic(
                "github",
                "kubernetes/kubernetes: Production-Grade Container Scheduling",
                3800,
                "devops",
                raw_data={"full_name": "kubernetes/kubernetes", "language": "Go"},
            ),
            TrendingTopic(
                "github",
                "fastapi/fastapi: FastAPI framework for building APIs with Python",
                3200,
                "web",
                raw_data={"full_name": "fastapi/fastapi", "language": "Python"},
            ),
        ]
        return mock_topics[:limit]


class PulseManager:
    """Manages multiple ingestors and coordinates trending topic collection."""

    def __init__(self):
        self.ingestors: Dict[str, PulseIngestor] = {}
        # Store debate outcomes for analytics
        self._outcomes: List[TrendingTopicOutcome] = []
        self._max_outcomes: int = 1000  # Rolling window

    def add_ingestor(self, name: str, ingestor: PulseIngestor):
        """Add an ingestor."""
        self.ingestors[name] = ingestor

    async def get_trending_topics(
        self,
        platforms: List[str] = None,
        limit_per_platform: int = 5,
        filters: Dict[str, Any] = None,
    ) -> List[TrendingTopic]:
        """Get trending topics from specified platforms."""
        if platforms is None:
            platforms = list(self.ingestors.keys())

        all_topics: List[TrendingTopic] = []

        # Fetch concurrently from all platforms
        tasks: List[Any] = []
        for platform in platforms:
            if platform in self.ingestors:
                tasks.append(self.ingestors[platform].fetch_trending(limit_per_platform))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, BaseException):
                    logger.warning(f"Ingestor error: {result}")
                else:
                    all_topics.extend(result)

        # Apply global filters
        if filters and platforms:
            all_topics = self.ingestors.get(platforms[0], TwitterIngestor())._filter_content(
                all_topics, filters
            )

        # Sort by volume and return top topics
        all_topics.sort(key=lambda t: t.volume, reverse=True)
        max_results = limit_per_platform * len(platforms) if platforms else limit_per_platform
        return all_topics[:max_results]

    def select_topic_for_debate(self, topics: List[TrendingTopic]) -> Optional[TrendingTopic]:
        """Select the most suitable topic for debate."""
        if not topics:
            return None

        # Prioritize diverse categories, high volume topics
        categories_seen = set()
        for topic in topics:
            if topic.category not in categories_seen:
                categories_seen.add(topic.category)
                return topic

        # Fallback to highest volume
        return max(topics, key=lambda t: t.volume)

    def record_debate_outcome(
        self,
        topic: str,
        platform: str,
        debate_id: str,
        consensus_reached: bool,
        confidence: float,
        rounds_used: int = 0,
        category: str = "",
        volume: int = 0,
    ) -> TrendingTopicOutcome:
        """Record the outcome of a debate on a trending topic.

        This enables analytics on which trending topics lead to productive debates.

        Args:
            topic: The trending topic text
            platform: Source platform (twitter, hackernews, reddit)
            debate_id: Unique debate identifier
            consensus_reached: Whether the debate reached consensus
            confidence: Final confidence score (0-1)
            rounds_used: Number of debate rounds
            category: Topic category (tech, politics, etc.)
            volume: Original engagement volume

        Returns:
            The created TrendingTopicOutcome record
        """
        outcome = TrendingTopicOutcome(
            topic=topic,
            platform=platform,
            debate_id=debate_id,
            consensus_reached=consensus_reached,
            confidence=confidence,
            rounds_used=rounds_used,
            category=category,
            volume=volume,
        )

        self._outcomes.append(outcome)

        # Trim to max size (rolling window)
        if len(self._outcomes) > self._max_outcomes:
            self._outcomes = self._outcomes[-self._max_outcomes :]

        logger.info(
            f"[pulse] Recorded debate outcome: {platform}/{topic[:50]}... "
            f"(consensus={consensus_reached}, confidence={confidence:.2f})"
        )

        return outcome

    def get_analytics(self) -> Dict[str, Any]:
        """Get analytics on trending topic debate outcomes.

        Returns:
            Dictionary with analytics data including:
            - total_debates: Total debates with trending topics
            - consensus_rate: Percentage that reached consensus
            - avg_confidence: Average confidence score
            - by_platform: Breakdown by platform
            - by_category: Breakdown by category
            - recent_outcomes: Last 10 outcomes
        """
        if not self._outcomes:
            return {
                "total_debates": 0,
                "consensus_rate": 0.0,
                "avg_confidence": 0.0,
                "by_platform": {},
                "by_category": {},
                "recent_outcomes": [],
            }

        total = len(self._outcomes)
        consensus_count = sum(1 for o in self._outcomes if o.consensus_reached)
        avg_confidence = sum(o.confidence for o in self._outcomes) / total

        # Group by platform
        by_platform: Dict[str, Dict[str, Any]] = {}
        for outcome in self._outcomes:
            if outcome.platform not in by_platform:
                by_platform[outcome.platform] = {
                    "total": 0,
                    "consensus_count": 0,
                    "confidence_sum": 0.0,
                }
            by_platform[outcome.platform]["total"] += 1
            if outcome.consensus_reached:
                by_platform[outcome.platform]["consensus_count"] += 1
            by_platform[outcome.platform]["confidence_sum"] += outcome.confidence

        # Calculate platform stats
        for platform, stats in by_platform.items():
            stats["consensus_rate"] = (
                stats["consensus_count"] / stats["total"] if stats["total"] > 0 else 0.0
            )
            stats["avg_confidence"] = (
                stats["confidence_sum"] / stats["total"] if stats["total"] > 0 else 0.0
            )
            del stats["confidence_sum"]

        # Group by category
        by_category: Dict[str, Dict[str, Any]] = {}
        for outcome in self._outcomes:
            cat = outcome.category or "general"
            if cat not in by_category:
                by_category[cat] = {
                    "total": 0,
                    "consensus_count": 0,
                    "confidence_sum": 0.0,
                }
            by_category[cat]["total"] += 1
            if outcome.consensus_reached:
                by_category[cat]["consensus_count"] += 1
            by_category[cat]["confidence_sum"] += outcome.confidence

        # Calculate category stats
        for cat, stats in by_category.items():
            stats["consensus_rate"] = (
                stats["consensus_count"] / stats["total"] if stats["total"] > 0 else 0.0
            )
            stats["avg_confidence"] = (
                stats["confidence_sum"] / stats["total"] if stats["total"] > 0 else 0.0
            )
            del stats["confidence_sum"]

        # Recent outcomes (last 10)
        recent = [
            {
                "topic": o.topic[:100],
                "platform": o.platform,
                "consensus_reached": o.consensus_reached,
                "confidence": o.confidence,
                "timestamp": o.timestamp,
            }
            for o in self._outcomes[-10:]
        ]

        return {
            "total_debates": total,
            "consensus_rate": consensus_count / total,
            "avg_confidence": avg_confidence,
            "by_platform": by_platform,
            "by_category": by_category,
            "recent_outcomes": recent,
        }
