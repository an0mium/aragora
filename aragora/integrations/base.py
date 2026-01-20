"""
Base class for chat platform integrations.

Provides common functionality for debate notifications including:
- Debate data formatting and truncation
- Rate limiting
- Session management
- Common notification patterns

All platform-specific integrations should extend BaseIntegration.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import aiohttp

from aragora.core import DebateResult

logger = logging.getLogger(__name__)


# =============================================================================
# Common Data Structures
# =============================================================================


@dataclass
class FormattedDebateData:
    """Common debate data structure for notifications."""

    debate_id: str
    question: str
    question_truncated: str
    answer: Optional[str]
    answer_truncated: Optional[str]
    total_rounds: int
    confidence: Optional[float]
    confidence_percent: Optional[str]
    agents: list[str]
    agents_display: str
    agent_count: int
    stats_line: str
    debate_url: str


@dataclass
class FormattedConsensusData:
    """Common consensus data structure for notifications."""

    debate_id: str
    answer: str
    answer_truncated: str
    confidence: float
    confidence_percent: str
    confidence_color: str  # "green"/"good" for high, "orange"/"warning" for medium
    agents: list[str]
    agents_display: str
    debate_url: str


@dataclass
class FormattedErrorData:
    """Common error data structure for notifications."""

    debate_id: str
    error: str
    error_truncated: str
    phase: Optional[str]


@dataclass
class FormattedLeaderboardData:
    """Common leaderboard data structure for notifications."""

    title: str
    domain: Optional[str]
    rankings: list[dict[str, Any]]
    leaderboard_url: str


# =============================================================================
# Base Integration Class
# =============================================================================


class BaseIntegration(ABC):
    """
    Abstract base class for chat platform integrations.

    Provides common functionality for:
    - Text formatting and truncation
    - Debate data extraction
    - Rate limiting
    - Session management

    Subclasses must implement:
    - is_configured property
    - send_message() method
    - Platform-specific notification methods
    """

    # Default URLs
    BASE_URL = "https://aragora.ai"

    # Default truncation limits
    DEFAULT_QUESTION_LIMIT = 200
    DEFAULT_ANSWER_LIMIT = 500
    DEFAULT_ERROR_LIMIT = 500
    DEFAULT_AGENTS_DISPLAY_LIMIT = 5

    def __init__(self) -> None:
        self._session: Optional[aiohttp.ClientSession] = None
        self._message_count = 0
        self._last_reset = datetime.now()

    # =========================================================================
    # Abstract Methods (must be implemented by subclasses)
    # =========================================================================

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Check if integration is properly configured."""
        pass

    @abstractmethod
    async def send_message(self, content: str, **kwargs: Any) -> bool:
        """Send a message to the platform."""
        pass

    # =========================================================================
    # Session Management
    # =========================================================================

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "BaseIntegration":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    # =========================================================================
    # Rate Limiting
    # =========================================================================

    def _check_rate_limit(self, max_per_minute: int) -> bool:
        """Check if we're within rate limits.

        Args:
            max_per_minute: Maximum messages allowed per minute

        Returns:
            True if within limits, False if rate limited
        """
        now = datetime.now()
        elapsed = (now - self._last_reset).total_seconds()

        if elapsed >= 60:
            self._message_count = 0
            self._last_reset = now

        if self._message_count >= max_per_minute:
            logger.warning(f"{self.__class__.__name__} rate limit reached")
            return False

        self._message_count += 1
        return True

    # =========================================================================
    # Text Formatting Utilities
    # =========================================================================

    @staticmethod
    def truncate_text(text: str, limit: int, suffix: str = "...") -> str:
        """Truncate text to limit, adding suffix if truncated.

        Args:
            text: Text to truncate
            limit: Maximum length
            suffix: Suffix to add if truncated

        Returns:
            Truncated text
        """
        if len(text) <= limit:
            return text
        return text[: limit - len(suffix)] + suffix

    @staticmethod
    def format_agents_list(
        agents: list[str],
        limit: int = 5,
        separator: str = ", ",
    ) -> str:
        """Format agent list with '+X more' for overflow.

        Args:
            agents: List of agent names
            limit: Maximum agents to show
            separator: Separator between agents

        Returns:
            Formatted agent string
        """
        if not agents:
            return ""

        displayed = agents[:limit]
        result = separator.join(displayed)

        if len(agents) > limit:
            result += f" +{len(agents) - limit} more"

        return result

    @staticmethod
    def format_confidence(confidence: float) -> str:
        """Format confidence as percentage string.

        Args:
            confidence: Confidence value (0-1)

        Returns:
            Formatted percentage string (e.g., "85%")
        """
        return f"{confidence:.0%}"

    @staticmethod
    def get_confidence_color(confidence: float, threshold: float = 0.8) -> str:
        """Get color indicator based on confidence level.

        Args:
            confidence: Confidence value (0-1)
            threshold: Threshold for "good" vs "warning"

        Returns:
            Color name ("green"/"good" or "orange"/"warning")
        """
        return "green" if confidence >= threshold else "orange"

    def get_debate_url(self, debate_id: str) -> str:
        """Get URL to view debate.

        Args:
            debate_id: Debate ID

        Returns:
            Full URL to debate
        """
        return f"{self.BASE_URL}/debate/{debate_id}"

    def get_leaderboard_url(self) -> str:
        """Get URL to leaderboard.

        Returns:
            Full URL to leaderboard
        """
        return f"{self.BASE_URL}/leaderboard"

    # =========================================================================
    # Data Formatting Methods
    # =========================================================================

    def format_debate_data(
        self,
        result: DebateResult,
        question_limit: int = DEFAULT_QUESTION_LIMIT,
        answer_limit: int = DEFAULT_ANSWER_LIMIT,
        agents_limit: int = DEFAULT_AGENTS_DISPLAY_LIMIT,
    ) -> FormattedDebateData:
        """Extract and format common debate data.

        Args:
            result: DebateResult to format
            question_limit: Max question length
            answer_limit: Max answer length
            agents_limit: Max agents to display

        Returns:
            FormattedDebateData with all common fields
        """
        # Build stats line
        stats_parts = [f"Rounds: {result.rounds_used}"]
        if result.confidence:
            stats_parts.append(f"Confidence: {self.format_confidence(result.confidence)}")
        if result.participants:
            stats_parts.append(f"Agents: {len(result.participants)}")
        stats_line = " | ".join(stats_parts)

        return FormattedDebateData(
            debate_id=result.debate_id,
            question=result.task,
            question_truncated=self.truncate_text(result.task, question_limit),
            answer=result.final_answer,
            answer_truncated=(
                self.truncate_text(result.final_answer, answer_limit) if result.final_answer else None
            ),
            total_rounds=result.rounds_used,
            confidence=result.confidence,
            confidence_percent=(
                self.format_confidence(result.confidence)
                if result.confidence
                else None
            ),
            agents=result.participants or [],
            agents_display=self.format_agents_list(result.participants or [], agents_limit),
            agent_count=len(result.participants) if result.participants else 0,
            stats_line=stats_line,
            debate_url=self.get_debate_url(result.debate_id),
        )

    def format_consensus_data(
        self,
        debate_id: str,
        answer: str,
        confidence: float,
        agents: Optional[list[str]] = None,
        answer_limit: int = DEFAULT_ANSWER_LIMIT,
        agents_limit: int = DEFAULT_AGENTS_DISPLAY_LIMIT,
    ) -> FormattedConsensusData:
        """Format consensus alert data.

        Args:
            debate_id: Debate ID
            answer: Consensus answer
            confidence: Confidence level (0-1)
            agents: Optional list of agreeing agents
            answer_limit: Max answer length
            agents_limit: Max agents to display

        Returns:
            FormattedConsensusData with all fields
        """
        return FormattedConsensusData(
            debate_id=debate_id,
            answer=answer,
            answer_truncated=self.truncate_text(answer, answer_limit),
            confidence=confidence,
            confidence_percent=self.format_confidence(confidence),
            confidence_color=self.get_confidence_color(confidence),
            agents=agents or [],
            agents_display=self.format_agents_list(agents or [], agents_limit),
            debate_url=self.get_debate_url(debate_id),
        )

    def format_error_data(
        self,
        debate_id: str,
        error: str,
        phase: Optional[str] = None,
        error_limit: int = DEFAULT_ERROR_LIMIT,
    ) -> FormattedErrorData:
        """Format error alert data.

        Args:
            debate_id: Debate ID
            error: Error message
            phase: Optional phase where error occurred
            error_limit: Max error message length

        Returns:
            FormattedErrorData with all fields
        """
        return FormattedErrorData(
            debate_id=debate_id,
            error=error,
            error_truncated=self.truncate_text(error, error_limit),
            phase=phase,
        )

    def format_leaderboard_data(
        self,
        rankings: list[dict[str, Any]],
        domain: Optional[str] = None,
        limit: int = 10,
    ) -> FormattedLeaderboardData:
        """Format leaderboard update data.

        Args:
            rankings: List of agent rankings
            domain: Optional domain filter
            limit: Max rankings to include

        Returns:
            FormattedLeaderboardData with all fields
        """
        title = f"LEADERBOARD UPDATE{f' ({domain})' if domain else ''}"

        return FormattedLeaderboardData(
            title=title,
            domain=domain,
            rankings=rankings[:limit],
            leaderboard_url=self.get_leaderboard_url(),
        )

    # =========================================================================
    # HTML Utilities
    # =========================================================================

    @staticmethod
    def escape_html(text: str) -> str:
        """Escape HTML special characters to prevent XSS.

        Args:
            text: Text to escape

        Returns:
            HTML-escaped text
        """
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "BaseIntegration",
    "FormattedDebateData",
    "FormattedConsensusData",
    "FormattedErrorData",
    "FormattedLeaderboardData",
]
