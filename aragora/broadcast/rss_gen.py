"""
RSS/Podcast feed generator for aragora debates.

Generates iTunes-compatible podcast feeds from debate audio broadcasts.
"""

import html
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from xml.sax.saxutils import escape

logger = logging.getLogger(__name__)


@dataclass
class PodcastConfig:
    """Configuration for the podcast feed."""

    title: str = "Aragora Debates"
    description: str = "AI agents debating technology, design, and ideas"
    author: str = "Aragora"
    email: str = "podcast@aragora.ai"
    website_url: str = "https://aragora.ai"
    feed_url: str = "https://aragora.ai/api/podcast/feed.xml"
    image_url: str = "https://aragora.ai/podcast-cover.png"
    language: str = "en-us"
    category: str = "Technology"
    subcategory: str = "Tech News"
    explicit: bool = False
    copyright: str = ""

    def __post_init__(self):
        if not self.copyright:
            self.copyright = f"Copyright {datetime.now().year} Aragora"


@dataclass
class PodcastEpisode:
    """A single podcast episode (debate broadcast)."""

    guid: str  # Unique identifier (debate_id)
    title: str
    description: str  # Short summary (500 chars max)
    content: str  # Full show notes / debate transcript summary
    audio_url: str
    pub_date: str  # ISO format or RFC 2822
    duration_seconds: int
    file_size_bytes: int = 0
    explicit: bool = False
    episode_number: Optional[int] = None
    season_number: Optional[int] = None
    agents: List[str] = field(default_factory=list)


def _escape_xml(text: str) -> str:
    """Escape text for safe XML inclusion."""
    if not text:
        return ""
    return escape(html.unescape(text))


def _escape_cdata(text: str) -> str:
    """Escape text for CDATA sections (handle ]]> sequences)."""
    if not text:
        return ""
    return text.replace("]]>", "]]]]><![CDATA[>")


def _format_rfc2822_date(iso_date: str) -> str:
    """Convert ISO date to RFC 2822 format for RSS."""
    try:
        dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        return dt.strftime("%a, %d %b %Y %H:%M:%S %z")
    except (ValueError, AttributeError):
        return datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000")


def _format_duration(seconds: int) -> str:
    """Format duration as HH:MM:SS for iTunes."""
    if not seconds or seconds < 0:
        return "00:00:00"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class PodcastFeedGenerator:
    """
    Generate iTunes-compatible podcast RSS feeds.

    Usage:
        config = PodcastConfig(title="My Podcast", ...)
        generator = PodcastFeedGenerator(config)

        episode = PodcastEpisode(
            guid="debate-123",
            title="Rate Limiter Design",
            description="Agents debate rate limiting strategies",
            ...
        )

        feed_xml = generator.generate_feed([episode])
    """

    def __init__(self, config: Optional[PodcastConfig] = None):
        """Initialize with podcast configuration."""
        self.config = config or PodcastConfig()

    def create_episode_from_debate(
        self,
        debate_id: str,
        task: str,
        agents: List[str],
        audio_url: str,
        duration_seconds: int,
        file_size_bytes: int = 0,
        created_at: Optional[str] = None,
        consensus_reached: bool = False,
        episode_number: Optional[int] = None,
    ) -> PodcastEpisode:
        """
        Create a podcast episode from debate metadata.

        Args:
            debate_id: Unique debate identifier
            task: The debate topic/question
            agents: List of participating agents
            audio_url: URL to the audio file
            duration_seconds: Audio duration
            file_size_bytes: Audio file size
            created_at: ISO timestamp of debate
            consensus_reached: Whether agents reached consensus
            episode_number: Optional episode number

        Returns:
            PodcastEpisode ready for feed inclusion
        """
        # Create title from task
        title = self._create_title(task)

        # Create description (short summary)
        description = self._create_description(task, agents, consensus_reached)

        # Create full content (show notes)
        content = self._create_content(task, agents, consensus_reached, debate_id)

        return PodcastEpisode(
            guid=debate_id,
            title=title,
            description=description,
            content=content,
            audio_url=audio_url,
            pub_date=created_at or datetime.now().isoformat(),
            duration_seconds=duration_seconds,
            file_size_bytes=file_size_bytes,
            agents=agents,
            episode_number=episode_number,
        )

    def _create_title(self, task: str, max_length: int = 100) -> str:
        """Create episode title from debate task."""
        if len(task) <= max_length:
            return task
        # Guard against invalid max_length
        if max_length <= 3:
            return task[:max_length] if max_length > 0 else ""
        truncated = task[: max_length - 3]
        # rsplit returns at least one element, but check for empty result
        parts = truncated.rsplit(" ", 1)
        prefix = parts[0] if parts and parts[0] else truncated
        return prefix + "..."

    def _create_description(
        self,
        task: str,
        agents: List[str],
        consensus_reached: bool,
        max_length: int = 500,
    ) -> str:
        """Create short episode description."""
        agent_list = ", ".join(agents[:3])
        if len(agents) > 3:
            agent_list += f" and {len(agents) - 3} more"

        consensus_text = "Consensus reached." if consensus_reached else "Debate ongoing."

        desc = f"{task}\n\nAgents: {agent_list}\n{consensus_text}"

        if len(desc) > max_length:
            desc = desc[: max_length - 3] + "..."

        return desc

    def _create_content(
        self,
        task: str,
        agents: List[str],
        consensus_reached: bool,
        debate_id: str,
    ) -> str:
        """Create full episode content (show notes)."""
        lines = [
            "<h2>Debate Topic</h2>",
            f"<p>{_escape_xml(task)}</p>",
            "<h2>Participants</h2>",
            "<ul>",
        ]

        for agent in agents:
            lines.append(f"  <li>{_escape_xml(agent)}</li>")

        lines.append("</ul>")

        if consensus_reached:
            lines.append("<p><strong>Result:</strong> Consensus reached</p>")
        else:
            lines.append("<p><strong>Result:</strong> Debate concluded without consensus</p>")

        lines.append(f"<p>Debate ID: {_escape_xml(debate_id)}</p>")

        return "\n".join(lines)

    def generate_feed(self, episodes: List[PodcastEpisode]) -> str:
        """
        Generate complete RSS/podcast feed XML.

        Args:
            episodes: List of episodes to include (most recent first)

        Returns:
            Complete RSS feed as XML string
        """
        config = self.config

        # Build channel header
        xml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<rss version="2.0"',
            '     xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"',
            '     xmlns:content="http://purl.org/rss/1.0/modules/content/"',
            '     xmlns:atom="http://www.w3.org/2005/Atom">',
            "  <channel>",
            f"    <title>{_escape_xml(config.title)}</title>",
            f"    <link>{_escape_xml(config.website_url)}</link>",
            f"    <description>{_escape_xml(config.description)}</description>",
            f"    <language>{_escape_xml(config.language)}</language>",
            f"    <copyright>{_escape_xml(config.copyright)}</copyright>",
            f"    <lastBuildDate>{_format_rfc2822_date(datetime.now().isoformat())}</lastBuildDate>",
            f'    <atom:link href="{_escape_xml(config.feed_url)}" rel="self" type="application/rss+xml"/>',
            "",
            "    <!-- iTunes-specific tags -->",
            f"    <itunes:author>{_escape_xml(config.author)}</itunes:author>",
            f"    <itunes:summary>{_escape_xml(config.description)}</itunes:summary>",
            f"    <itunes:explicit>{'yes' if config.explicit else 'no'}</itunes:explicit>",
            "    <itunes:owner>",
            f"      <itunes:name>{_escape_xml(config.author)}</itunes:name>",
            f"      <itunes:email>{_escape_xml(config.email)}</itunes:email>",
            "    </itunes:owner>",
            f'    <itunes:image href="{_escape_xml(config.image_url)}"/>',
            f'    <itunes:category text="{_escape_xml(config.category)}">',
            f'      <itunes:category text="{_escape_xml(config.subcategory)}"/>',
            "    </itunes:category>",
            "",
        ]

        # Add episodes
        for episode in episodes:
            xml_parts.extend(self._generate_episode_xml(episode))

        # Close channel and rss
        xml_parts.extend(
            [
                "  </channel>",
                "</rss>",
            ]
        )

        return "\n".join(xml_parts)

    def _generate_episode_xml(self, episode: PodcastEpisode) -> List[str]:
        """Generate XML for a single episode."""
        pub_date = _format_rfc2822_date(episode.pub_date)
        duration = _format_duration(episode.duration_seconds)

        lines = [
            "    <item>",
            f"      <title>{_escape_xml(episode.title)}</title>",
            f"      <description>{_escape_xml(episode.description)}</description>",
            f"      <content:encoded><![CDATA[{_escape_cdata(episode.content)}]]></content:encoded>",
            f"      <pubDate>{pub_date}</pubDate>",
            f'      <guid isPermaLink="false">{_escape_xml(episode.guid)}</guid>',
            f'      <enclosure url="{_escape_xml(episode.audio_url)}" '
            f'length="{episode.file_size_bytes}" type="audio/mpeg"/>',
            "",
            f"      <itunes:duration>{duration}</itunes:duration>",
            f"      <itunes:explicit>{'yes' if episode.explicit else 'no'}</itunes:explicit>",
            f"      <itunes:summary>{_escape_xml(episode.description)}</itunes:summary>",
        ]

        if episode.episode_number is not None:
            lines.append(f"      <itunes:episode>{episode.episode_number}</itunes:episode>")

        if episode.season_number is not None:
            lines.append(f"      <itunes:season>{episode.season_number}</itunes:season>")

        # Add agents as keywords
        if episode.agents:
            keywords = ", ".join(episode.agents)
            lines.append(f"      <itunes:keywords>{_escape_xml(keywords)}</itunes:keywords>")

        lines.append("    </item>")
        lines.append("")

        return lines


def create_debate_summary(
    task: str,
    agents: List[str],
    consensus_reached: bool = False,
    max_length: int = 280,
) -> str:
    """
    Create a short summary suitable for Twitter/social media.

    Args:
        task: Debate topic
        agents: Participating agents
        consensus_reached: Whether consensus was reached
        max_length: Maximum character length (280 for Twitter)

    Returns:
        Summary string within max_length
    """
    # Start with task
    summary = task

    # Add agents if space permits
    agent_suffix = f" ({', '.join(agents[:2])})"
    if len(agents) > 2:
        agent_suffix = f" ({agents[0]}, {agents[1]} +{len(agents) - 2})"

    # Add result emoji
    result_emoji = " ✅" if consensus_reached else " 🔄"

    # Build summary within limits
    full_summary = summary + agent_suffix + result_emoji

    if len(full_summary) <= max_length:
        return full_summary

    # Truncate task to fit
    available = max_length - len(agent_suffix) - len(result_emoji) - 3
    if available > 20:
        return summary[:available] + "..." + agent_suffix + result_emoji

    # Fall back to just truncated task
    return summary[: max_length - 3] + "..."
