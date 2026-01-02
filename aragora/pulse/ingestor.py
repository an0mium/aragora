"""
Trending Pulse Ingestor.

Fetches real-time trending topics from social media platforms
for dynamic debate topic generation.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import httpx

from aragora.core import Message, Environment


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


class PulseIngestor(ABC):
    """Abstract base class for social media ingestors."""

    def __init__(self, api_key: Optional[str] = None, rate_limit_delay: float = 1.0):
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        self.cache: Dict[str, List[TrendingTopic]] = {}
        self.cache_ttl = 300  # 5 minutes

    async def _rate_limit(self):
        """Enforce rate limiting."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = now

    @abstractmethod
    async def fetch_trending(self, limit: int = 10) -> List[TrendingTopic]:
        """Fetch trending topics from the platform."""
        pass

    def _filter_content(self, topics: List[TrendingTopic], filters: Dict[str, Any]) -> List[TrendingTopic]:
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
        """Simple toxicity check (placeholder - integrate with proper sentiment analysis)."""
        toxic_keywords = ["hate", "violence", "nsfw", "explicit"]
        return any(keyword in text.lower() for keyword in toxic_keywords)


class TwitterIngestor(PulseIngestor):
    """Twitter/X trending topics ingestor using Twitter API v2."""

    def __init__(self, bearer_token: Optional[str] = None, **kwargs):
        super().__init__(api_key=bearer_token, **kwargs)
        self.base_url = "https://api.twitter.com/2"

    async def fetch_trending(self, limit: int = 10) -> List[TrendingTopic]:
        """Fetch trending topics from Twitter."""
        if not self.api_key:
            # Fallback to mock data for development
            return self._mock_trending_data(limit)

        await self._rate_limit()

        try:
            async with httpx.AsyncClient() as client:
                # Get trending topics for a location (WOEID 1 = worldwide)
                url = f"{self.base_url}/trends/place.json"
                params = {"id": 1}  # Worldwide
                headers = {"Authorization": f"Bearer {self.api_key}"}

                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()

                data = response.json()
                topics = []

                for trend in data[0]["trends"][:limit]:
                    topic = TrendingTopic(
                        platform="twitter",
                        topic=trend["name"],
                        volume=trend.get("tweet_volume", 0),
                        category=self._categorize_topic(trend["name"]),
                        raw_data=trend
                    )
                    topics.append(topic)

                return topics

        except Exception as e:
            print(f"Twitter API error: {e}")
            return self._mock_trending_data(limit)

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


class PulseManager:
    """Manages multiple ingestors and coordinates trending topic collection."""

    def __init__(self):
        self.ingestors: Dict[str, PulseIngestor] = {}

    def add_ingestor(self, name: str, ingestor: PulseIngestor):
        """Add an ingestor."""
        self.ingestors[name] = ingestor

    async def get_trending_topics(
        self,
        platforms: List[str] = None,
        limit_per_platform: int = 5,
        filters: Dict[str, Any] = None
    ) -> List[TrendingTopic]:
        """Get trending topics from specified platforms."""
        if platforms is None:
            platforms = list(self.ingestors.keys())

        all_topics = []

        # Fetch concurrently from all platforms
        tasks = []
        for platform in platforms:
            if platform in self.ingestors:
                tasks.append(self.ingestors[platform].fetch_trending(limit_per_platform))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    print(f"Ingestor error: {result}")
                else:
                    all_topics.extend(result)

        # Apply global filters
        if filters:
            all_topics = self.ingestors.get(platforms[0], TwitterIngestor())._filter_content(all_topics, filters)

        # Sort by volume and return top topics
        all_topics.sort(key=lambda t: t.volume, reverse=True)
        return all_topics[:limit_per_platform * len(platforms)]

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