"""
MCP Context Tools.

Rich context fetching for chat and debate operations.
Exposes channel context, debate state, and conversation analysis as MCP tools.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def fetch_channel_context_tool(
    channel_id: str,
    platform: str = "slack",
    message_limit: int = 50,
    include_participants: bool = True,
    include_topics: bool = True,
) -> Dict[str, Any]:
    """
    Fetch context from a chat channel.

    Retrieves recent messages, participants, and detected topics
    for building debate context.

    Args:
        channel_id: ID of the channel to fetch context from
        platform: Platform type (slack, discord, telegram, teams)
        message_limit: Maximum number of messages to retrieve
        include_participants: Include list of participants
        include_topics: Include detected topics/keywords

    Returns:
        Dict with channel context including messages, participants, topics
    """
    try:
        # Import chat connector based on platform
        connector = await _get_chat_connector(platform)
        if not connector:
            return {
                "error": f"No connector available for platform: {platform}",
                "channel_id": channel_id,
            }

        # Fetch messages
        messages = await _fetch_channel_messages(connector, channel_id, message_limit)

        result = {
            "channel_id": channel_id,
            "platform": platform,
            "message_count": len(messages),
            "messages": messages,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

        # Extract participants
        if include_participants:
            participants = set()
            for msg in messages:
                if msg.get("author"):
                    participants.add(msg["author"])
            result["participants"] = list(participants)
            result["participant_count"] = len(participants)

        # Extract topics
        if include_topics:
            topics = await _extract_topics(messages)
            result["topics"] = topics

        return result

    except Exception as e:
        logger.error(f"Failed to fetch channel context: {e}")
        return {
            "error": str(e),
            "channel_id": channel_id,
            "platform": platform,
        }


async def fetch_debate_context_tool(
    debate_id: str,
    include_history: bool = True,
    include_consensus: bool = True,
    include_metrics: bool = False,
) -> Dict[str, Any]:
    """
    Fetch context for an active or completed debate.

    Retrieves debate state, history, and consensus information
    for chat integration.

    Args:
        debate_id: ID of the debate to fetch context for
        include_history: Include round history
        include_consensus: Include consensus status
        include_metrics: Include performance metrics

    Returns:
        Dict with debate context including state, history, consensus
    """
    try:
        from aragora.memory.consensus import ConsensusMemory

        # Get consensus memory
        memory = ConsensusMemory()
        await memory.initialize()

        # Look up debate
        debate = await memory.get_debate(debate_id)
        if not debate:
            return {
                "error": f"Debate not found: {debate_id}",
                "debate_id": debate_id,
            }

        result = {
            "debate_id": debate_id,
            "task": debate.get("task", ""),
            "status": debate.get("status", "unknown"),
            "created_at": debate.get("created_at", ""),
            "final_answer": debate.get("final_answer"),
        }

        if include_history:
            result["rounds"] = debate.get("rounds", [])
            result["round_count"] = len(result["rounds"])

        if include_consensus:
            result["consensus_reached"] = debate.get("consensus_reached", False)
            result["confidence"] = debate.get("confidence", 0.0)
            result["agents"] = debate.get("agents", [])

        if include_metrics:
            result["metrics"] = {
                "duration_seconds": debate.get("duration_seconds"),
                "total_tokens": debate.get("total_tokens"),
                "rounds_used": debate.get("rounds_used"),
            }

        return result

    except Exception as e:
        logger.error(f"Failed to fetch debate context: {e}")
        return {
            "error": str(e),
            "debate_id": debate_id,
        }


async def analyze_conversation_tool(
    messages: List[Dict[str, Any]],
    analyze_sentiment: bool = True,
    analyze_activity: bool = True,
    extract_questions: bool = True,
    extract_decisions: bool = True,
) -> Dict[str, Any]:
    """
    Analyze a conversation for patterns and insights.

    Extracts sentiment, activity patterns, questions, and decisions
    from a list of messages.

    Args:
        messages: List of message dicts with content, author, timestamp
        analyze_sentiment: Include sentiment analysis
        analyze_activity: Include activity patterns
        extract_questions: Extract questions from messages
        extract_decisions: Extract decisions/conclusions

    Returns:
        Dict with analysis results including sentiment, patterns, extractions
    """
    if not messages:
        return {
            "error": "No messages provided for analysis",
            "message_count": 0,
        }

    result = {
        "message_count": len(messages),
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }

    # Activity analysis
    if analyze_activity:
        activity = _analyze_activity(messages)
        result["activity"] = activity

    # Sentiment analysis
    if analyze_sentiment:
        sentiment = await _analyze_sentiment(messages)
        result["sentiment"] = sentiment

    # Question extraction
    if extract_questions:
        questions = _extract_questions(messages)
        result["questions"] = questions
        result["question_count"] = len(questions)

    # Decision extraction
    if extract_decisions:
        decisions = _extract_decisions(messages)
        result["decisions"] = decisions
        result["decision_count"] = len(decisions)

    return result


async def get_thread_context_tool(
    thread_id: str,
    platform: str = "slack",
    include_parent: bool = True,
) -> Dict[str, Any]:
    """
    Fetch context for a specific thread.

    Args:
        thread_id: ID of the thread
        platform: Platform type
        include_parent: Include parent message

    Returns:
        Dict with thread context
    """
    try:
        connector = await _get_chat_connector(platform)
        if not connector:
            return {"error": f"No connector for platform: {platform}"}

        # Fetch thread messages
        messages = []
        if hasattr(connector, "get_thread_messages"):
            messages = await connector.get_thread_messages(thread_id)

        result = {
            "thread_id": thread_id,
            "platform": platform,
            "message_count": len(messages),
            "messages": messages,
        }

        if include_parent and messages:
            result["parent_message"] = messages[0] if messages else None

        return result

    except Exception as e:
        logger.error(f"Failed to fetch thread context: {e}")
        return {"error": str(e), "thread_id": thread_id}


async def get_user_context_tool(
    user_id: str,
    platform: str = "slack",
    include_recent_messages: bool = True,
    message_limit: int = 20,
) -> Dict[str, Any]:
    """
    Fetch context for a specific user.

    Args:
        user_id: ID of the user
        platform: Platform type
        include_recent_messages: Include recent messages by user
        message_limit: Maximum messages to include

    Returns:
        Dict with user context
    """
    try:
        connector = await _get_chat_connector(platform)
        if not connector:
            return {"error": f"No connector for platform: {platform}"}

        result = {
            "user_id": user_id,
            "platform": platform,
        }

        # Get user info if available
        if hasattr(connector, "get_user_info"):
            user_info = await connector.get_user_info(user_id)
            result["user_info"] = user_info

        # Get recent messages
        if include_recent_messages and hasattr(connector, "get_user_messages"):
            messages = await connector.get_user_messages(user_id, limit=message_limit)
            result["recent_messages"] = messages
            result["message_count"] = len(messages)

        return result

    except Exception as e:
        logger.error(f"Failed to fetch user context: {e}")
        return {"error": str(e), "user_id": user_id}


# Helper functions


async def _get_chat_connector(platform: str) -> Optional[Any]:
    """Get the chat connector for a platform."""
    try:
        if platform == "slack":
            from aragora.connectors.slack import SlackConnector

            return SlackConnector()
        elif platform == "discord":
            from aragora.connectors.chat.discord import DiscordConnector

            return DiscordConnector()
        elif platform == "telegram":
            from aragora.connectors.chat.telegram import TelegramConnector

            return TelegramConnector()
        elif platform == "teams":
            from aragora.connectors.chat.teams import TeamsConnector

            return TeamsConnector()
        else:
            logger.warning(f"Unknown platform: {platform}")
            return None
    except ImportError as e:
        logger.warning(f"Could not import connector for {platform}: {e}")
        return None


async def _fetch_channel_messages(
    connector: Any,
    channel_id: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """Fetch messages from a channel."""
    if hasattr(connector, "get_channel_messages"):
        return await connector.get_channel_messages(channel_id, limit=limit)
    return []


async def _extract_topics(messages: List[Dict[str, Any]]) -> List[str]:
    """Extract topics from messages using keyword extraction."""
    # Simple keyword extraction - could be enhanced with NLP
    from collections import Counter
    import re

    words = []
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
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
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "or",
        "and",
        "but",
        "not",
        "this",
        "that",
        "it",
        "as",
        "if",
        "you",
        "your",
        "we",
        "our",
        "they",
        "their",
        "i",
        "my",
        "me",
        "he",
        "she",
        "him",
        "her",
        "what",
        "when",
        "where",
        "why",
        "how",
        "which",
    }

    for msg in messages:
        content = msg.get("content", "")
        if content:
            # Extract words
            msg_words = re.findall(r"\b[a-zA-Z]{3,}\b", content.lower())
            words.extend([w for w in msg_words if w not in stopwords])

    # Get most common
    counter = Counter(words)
    return [word for word, _ in counter.most_common(10)]


def _analyze_activity(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze activity patterns in messages."""
    if not messages:
        return {"total_messages": 0}

    # Parse timestamps
    timestamps = []
    for msg in messages:
        ts = msg.get("timestamp")
        if ts:
            try:
                if isinstance(ts, str):
                    timestamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
                elif isinstance(ts, datetime):
                    timestamps.append(ts)
            except Exception:
                pass

    activity = {
        "total_messages": len(messages),
        "unique_authors": len(set(msg.get("author", "") for msg in messages)),
    }

    if timestamps:
        timestamps.sort()
        activity["first_message"] = timestamps[0].isoformat()
        activity["last_message"] = timestamps[-1].isoformat()
        activity["duration_hours"] = (timestamps[-1] - timestamps[0]).total_seconds() / 3600

        # Activity by hour
        hours = [ts.hour for ts in timestamps]
        from collections import Counter

        hour_counts = Counter(hours)
        activity["peak_hour"] = hour_counts.most_common(1)[0][0] if hour_counts else None

    return activity


async def _analyze_sentiment(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple sentiment analysis."""
    # Simple keyword-based sentiment (could be enhanced with ML)
    positive_words = {
        "good",
        "great",
        "excellent",
        "awesome",
        "amazing",
        "love",
        "perfect",
        "wonderful",
        "fantastic",
        "agree",
        "yes",
        "thanks",
        "helpful",
        "nice",
        "best",
        "happy",
        "glad",
        "excited",
    }
    negative_words = {
        "bad",
        "terrible",
        "awful",
        "hate",
        "wrong",
        "disagree",
        "no",
        "problem",
        "issue",
        "bug",
        "error",
        "fail",
        "broken",
        "worst",
        "angry",
        "frustrated",
        "confused",
        "worried",
        "concerned",
    }

    positive_count = 0
    negative_count = 0
    neutral_count = 0

    for msg in messages:
        content = msg.get("content", "").lower()
        words = set(content.split())

        pos = len(words & positive_words)
        neg = len(words & negative_words)

        if pos > neg:
            positive_count += 1
        elif neg > pos:
            negative_count += 1
        else:
            neutral_count += 1

    total = len(messages) or 1
    return {
        "positive_ratio": positive_count / total,
        "negative_ratio": negative_count / total,
        "neutral_ratio": neutral_count / total,
        "overall": (
            "positive"
            if positive_count > negative_count + neutral_count
            else "negative"
            if negative_count > positive_count + neutral_count
            else "neutral"
        ),
    }


def _extract_questions(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract questions from messages."""
    questions = []

    for msg in messages:
        content = msg.get("content", "")
        # Simple heuristic: ends with ? or starts with question words
        if content.strip().endswith("?"):
            questions.append(
                {
                    "content": content,
                    "author": msg.get("author"),
                    "timestamp": msg.get("timestamp"),
                }
            )
        elif any(
            content.lower().startswith(w)
            for w in [
                "what",
                "why",
                "how",
                "when",
                "where",
                "who",
                "which",
                "can",
                "could",
                "would",
                "should",
            ]
        ):
            questions.append(
                {
                    "content": content,
                    "author": msg.get("author"),
                    "timestamp": msg.get("timestamp"),
                }
            )

    return questions


def _extract_decisions(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract decisions and conclusions from messages."""
    decisions = []

    decision_indicators = [
        "decided",
        "decision",
        "agreed",
        "conclusion",
        "resolved",
        "let's go with",
        "we'll do",
        "final answer",
        "the answer is",
        "consensus",
        "approved",
        "accepted",
        "moving forward with",
    ]

    for msg in messages:
        content = msg.get("content", "").lower()
        if any(indicator in content for indicator in decision_indicators):
            decisions.append(
                {
                    "content": msg.get("content", ""),
                    "author": msg.get("author"),
                    "timestamp": msg.get("timestamp"),
                }
            )

    return decisions
