"""
Rich Context Analysis for Chat Messages.

Provides topic extraction, sentiment analysis, activity pattern calculation,
and LLM-ready context formatting. Extracted from base.py for modularity.

These are pure functions that operate on ChatMessage lists without
requiring connector instance state.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .models import ChatMessage


# ==============================================================================
# Stop Words for Topic Extraction
# ==============================================================================

STOP_WORDS = {
    "the",
    "a",
    "an",
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
    "shall",
    "can",
    "need",
    "dare",
    "ought",
    "used",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "and",
    "but",
    "if",
    "or",
    "because",
    "until",
    "while",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "its",
    "our",
    "their",
}

# ==============================================================================
# Sentiment Word Lists
# ==============================================================================

POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "awesome",
    "amazing",
    "wonderful",
    "fantastic",
    "perfect",
    "love",
    "like",
    "agree",
    "yes",
    "thanks",
    "thank",
    "helpful",
    "nice",
    "well",
    "best",
    "happy",
    "glad",
}

NEGATIVE_WORDS = {
    "bad",
    "terrible",
    "awful",
    "horrible",
    "hate",
    "dislike",
    "disagree",
    "no",
    "wrong",
    "problem",
    "issue",
    "error",
    "fail",
    "broken",
    "bug",
    "frustrated",
    "annoyed",
    "worst",
    "sad",
}


# ==============================================================================
# Topic Extraction
# ==============================================================================


def extract_topics(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """
    Extract discussion topics from messages.

    Simple keyword extraction - can be extended for more sophisticated
    NLP-based topic extraction.

    Args:
        messages: List of messages to analyze

    Returns:
        List of topic dicts with topic and frequency
    """
    if not messages:
        return []

    word_counts: Counter[str] = Counter()
    for msg in messages:
        if not msg.content:
            continue
        # Extract words (lowercase, alphanumeric only)
        words = re.findall(r"\b[a-zA-Z]{3,}\b", msg.content.lower())
        significant_words = [w for w in words if w not in STOP_WORDS]
        word_counts.update(significant_words)

    # Get top topics
    topics = [
        {"topic": word, "frequency": count, "relevance": min(1.0, count / 10)}
        for word, count in word_counts.most_common(10)
        if count >= 2  # At least 2 mentions
    ]

    return topics


# ==============================================================================
# Sentiment Analysis
# ==============================================================================


def analyze_sentiment(messages: list[ChatMessage]) -> dict[str, Any]:
    """
    Basic sentiment analysis of messages.

    Simple keyword-based sentiment - can be extended for more sophisticated
    analysis using NLP models.

    Args:
        messages: List of messages to analyze

    Returns:
        Dict with sentiment metrics
    """
    if not messages:
        return {"overall": "neutral", "positive_ratio": 0.5, "scores": []}

    scores = []
    for msg in messages:
        if not msg.content:
            scores.append(0)
            continue

        words = set(msg.content.lower().split())
        pos_count = len(words & POSITIVE_WORDS)
        neg_count = len(words & NEGATIVE_WORDS)

        if pos_count > neg_count:
            scores.append(1)
        elif neg_count > pos_count:
            scores.append(-1)
        else:
            scores.append(0)

    positive_ratio = (sum(1 for s in scores if s > 0) / len(scores)) if scores else 0.5
    avg_score = sum(scores) / len(scores) if scores else 0

    if avg_score > 0.3:
        overall = "positive"
    elif avg_score < -0.3:
        overall = "negative"
    else:
        overall = "neutral"

    return {
        "overall": overall,
        "positive_ratio": round(positive_ratio, 2),
        "average_score": round(avg_score, 2),
        "message_count": len(scores),
    }


# ==============================================================================
# Activity Pattern Calculation
# ==============================================================================


def calculate_activity_patterns(messages: list[ChatMessage]) -> dict[str, Any]:
    """
    Calculate activity patterns from messages.

    Args:
        messages: List of messages to analyze

    Returns:
        Dict with activity metrics
    """
    if not messages:
        return {
            "messages_per_hour": 0,
            "most_active_participants": [],
            "peak_activity_time": None,
        }

    # Messages per participant
    participant_counts: Counter[str] = Counter()
    hour_counts: Counter[int] = Counter()

    for msg in messages:
        author_name = msg.author.display_name or msg.author.username or msg.author.id
        participant_counts[author_name] += 1

        if msg.timestamp:
            hour_counts[msg.timestamp.hour] += 1

    # Calculate messages per hour
    if messages and messages[0].timestamp and messages[-1].timestamp:
        time_span = abs((messages[0].timestamp - messages[-1].timestamp).total_seconds() / 3600)
        messages_per_hour = len(messages) / max(time_span, 1)
    else:
        messages_per_hour = len(messages)

    # Find peak activity time
    peak_hour = hour_counts.most_common(1)[0][0] if hour_counts else None

    return {
        "messages_per_hour": round(messages_per_hour, 1),
        "most_active_participants": [
            {"name": name, "message_count": count}
            for name, count in participant_counts.most_common(5)
        ],
        "peak_activity_hour": peak_hour,
        "unique_participants": len(participant_counts),
    }


# ==============================================================================
# LLM Context Formatting
# ==============================================================================


def format_context_for_llm(rich_context: dict[str, Any]) -> str:
    """
    Format rich context into an LLM-ready string.

    Args:
        rich_context: The rich context dictionary

    Returns:
        Formatted string suitable for LLM prompt injection
    """
    lines = []

    # Channel info
    channel = rich_context.get("channel", {})
    channel_name = channel.get("name") or channel.get("id", "unknown")
    lines.append(f"## Channel: {channel_name} ({channel.get('platform', 'unknown')})")
    lines.append("")

    # Participants
    participants = rich_context.get("participants", [])
    if participants:
        human_participants = [p["name"] for p in participants if not p.get("is_bot")]
        if human_participants:
            lines.append(f"**Participants:** {', '.join(human_participants[:10])}")
            if len(human_participants) > 10:
                lines.append(f"  ... and {len(human_participants) - 10} more")
        lines.append("")

    # Topics
    topics = rich_context.get("topics", [])
    if topics:
        topic_names = [t["topic"] for t in topics[:5]]
        lines.append(f"**Current topics:** {', '.join(topic_names)}")
        lines.append("")

    # Activity summary
    stats = rich_context.get("statistics", {})
    activity = rich_context.get("activity", {})
    if stats.get("message_count"):
        lines.append(
            f"**Activity:** {stats['message_count']} messages in the last "
            f"{stats.get('timespan_minutes', 60)} minutes "
            f"({activity.get('messages_per_hour', 0):.1f}/hour)"
        )
        lines.append("")

    # Recent messages summary
    messages = rich_context.get("messages", [])
    if messages:
        lines.append("**Recent discussion:**")
        # Show last 10 messages
        for msg in messages[-10:]:
            author = msg.get("author", "unknown")
            content = msg.get("content", "")
            # Truncate long messages
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"- **{author}:** {content}")
        lines.append("")

    # Sentiment if available
    sentiment = rich_context.get("sentiment")
    if sentiment and sentiment.get("overall"):
        lines.append(f"**Conversation tone:** {sentiment['overall']}")

    return "\n".join(lines)


__all__ = [
    "STOP_WORDS",
    "POSITIVE_WORDS",
    "NEGATIVE_WORDS",
    "extract_topics",
    "analyze_sentiment",
    "calculate_activity_patterns",
    "format_context_for_llm",
]
