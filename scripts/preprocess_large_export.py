#!/usr/bin/env python3
"""
Preprocess Large ChatGPT/Claude Exports

Handles exports of any size by:
1. Streaming JSON parsing (no full load into memory)
2. Filtering conversations by relevance to seed topics
3. Filtering by date range
4. Sampling to manageable size
5. Outputting filtered export for processing

Usage:
    # Basic filtering by topics
    python scripts/preprocess_large_export.py \
        --input ~/Downloads/chatgpt-export/conversations.json \
        --output filtered_conversations.json \
        --topics "AI alignment" "instrumental convergence" "evolution" "systems thinking"

    # Filter by date range
    python scripts/preprocess_large_export.py \
        --input ~/Downloads/chatgpt-export/conversations.json \
        --output filtered_conversations.json \
        --after 2024-01-01 \
        --before 2025-12-31

    # Relevance-based filtering with seed essay
    python scripts/preprocess_large_export.py \
        --input ~/Downloads/chatgpt-export/conversations.json \
        --output filtered_conversations.json \
        --seed-essay drafts/my_essay.md \
        --min-relevance 0.3

    # Sample random conversations
    python scripts/preprocess_large_export.py \
        --input ~/Downloads/chatgpt-export/conversations.json \
        --output filtered_conversations.json \
        --sample 500

    # Analyze only (no output, just statistics)
    python scripts/preprocess_large_export.py \
        --input ~/Downloads/chatgpt-export/conversations.json \
        --analyze-only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class FilterConfig:
    """Configuration for conversation filtering."""

    topics: list[str] = field(default_factory=list)
    after_date: datetime | None = None
    before_date: datetime | None = None
    min_messages: int = 2
    max_messages: int | None = None
    min_words: int = 100
    sample_size: int | None = None
    min_relevance: float = 0.0
    seed_keywords: set[str] = field(default_factory=set)
    exclude_patterns: list[str] = field(default_factory=list)
    include_only_user_heavy: bool = False  # Conversations where user wrote a lot


@dataclass
class ConversationStats:
    """Statistics about an export."""

    total_conversations: int = 0
    total_messages: int = 0
    total_words: int = 0
    conversations_by_month: dict[str, int] = field(default_factory=dict)
    avg_messages_per_conversation: float = 0.0
    avg_words_per_conversation: float = 0.0
    top_title_words: list[tuple[str, int]] = field(default_factory=list)
    date_range: tuple[str, str] | None = None
    size_distribution: dict[str, int] = field(default_factory=dict)  # small/medium/large


def stream_conversations(path: Path) -> Iterator[dict]:
    """
    Stream conversations from a large JSON file without loading all into memory.

    Handles ChatGPT export format (array of conversation objects).
    """
    with open(path, "r", encoding="utf-8") as f:
        # Skip initial whitespace and '['
        char = f.read(1)
        while char and char in " \t\n\r":
            char = f.read(1)

        if char != "[":
            # Not an array, try to load as single object
            f.seek(0)
            data = json.load(f)
            if isinstance(data, list):
                yield from data
            else:
                yield data
            return

        # Stream parse array elements
        depth = 0
        buffer = ""
        in_string = False
        escape_next = False

        while True:
            char = f.read(1)
            if not char:
                break

            if escape_next:
                buffer += char
                escape_next = False
                continue

            if char == "\\":
                buffer += char
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string
                buffer += char
                continue

            if in_string:
                buffer += char
                continue

            if char == "{":
                depth += 1
                buffer += char
            elif char == "}":
                depth -= 1
                buffer += char
                if depth == 0:
                    # Complete object
                    try:
                        conv = json.loads(buffer.strip())
                        yield conv
                    except json.JSONDecodeError:
                        pass
                    buffer = ""
            elif char == "," and depth == 0:
                # Skip comma between array elements
                buffer = ""
            elif char == "]" and depth == 0:
                # End of array
                break
            elif depth > 0:
                buffer += char


def extract_conversation_text(conv: dict) -> str:
    """Extract all text content from a conversation."""
    texts = []

    # Title
    if "title" in conv:
        texts.append(conv["title"])

    # Messages from mapping (ChatGPT format)
    if "mapping" in conv:
        for node_id, node in conv["mapping"].items():
            msg = node.get("message")
            if msg:
                content = msg.get("content", {})
                parts = content.get("parts", [])
                for part in parts:
                    if isinstance(part, str):
                        texts.append(part)

    # Messages from chat_messages (Claude format)
    if "chat_messages" in conv:
        for msg in conv["chat_messages"]:
            text = msg.get("text", msg.get("content", ""))
            if isinstance(text, str):
                texts.append(text)

    return " ".join(texts)


def extract_user_text(conv: dict) -> str:
    """Extract only user message text from a conversation."""
    texts = []

    if "mapping" in conv:
        for node_id, node in conv["mapping"].items():
            msg = node.get("message")
            if msg and msg.get("author", {}).get("role") == "user":
                parts = msg.get("content", {}).get("parts", [])
                for part in parts:
                    if isinstance(part, str):
                        texts.append(part)

    if "chat_messages" in conv:
        for msg in conv["chat_messages"]:
            if msg.get("sender") in ("human", "user"):
                text = msg.get("text", msg.get("content", ""))
                if isinstance(text, str):
                    texts.append(text)

    return " ".join(texts)


def get_conversation_date(conv: dict) -> datetime | None:
    """Extract conversation date."""
    if "create_time" in conv and conv["create_time"]:
        try:
            return datetime.fromtimestamp(conv["create_time"])
        except (ValueError, TypeError):
            pass

    if "created_at" in conv:
        try:
            return datetime.fromisoformat(conv["created_at"].replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            pass

    return None


def count_messages(conv: dict) -> int:
    """Count messages in a conversation."""
    count = 0

    if "mapping" in conv:
        for node_id, node in conv["mapping"].items():
            if node.get("message"):
                count += 1

    if "chat_messages" in conv:
        count = len(conv["chat_messages"])

    return count


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def extract_keywords(text: str, min_length: int = 4) -> set[str]:
    """Extract keywords from text."""
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "that", "this", "these", "those", "it", "its", "i", "you", "we", "they",
        "what", "which", "who", "how", "when", "where", "why", "think", "believe",
        "about", "more", "some", "any", "just", "only", "also", "very", "really",
        "like", "know", "want", "need", "make", "take", "give", "find", "tell",
        "good", "well", "much", "such", "even", "back", "way", "thing", "things",
        "then", "than", "now", "here", "there", "your", "their", "other", "first",
    }

    words = re.findall(r"\b[a-zA-Z]{%d,}\b" % min_length, text.lower())
    return {w for w in words if w not in stopwords}


def load_seed_keywords(seed_path: Path) -> set[str]:
    """Load keywords from seed essay."""
    if not seed_path.exists():
        return set()

    with open(seed_path, "r", encoding="utf-8") as f:
        content = f.read()

    return extract_keywords(content)


def calculate_relevance(conv_keywords: set[str], seed_keywords: set[str]) -> float:
    """Calculate relevance score between conversation and seed keywords."""
    if not seed_keywords or not conv_keywords:
        return 0.0

    overlap = len(conv_keywords & seed_keywords)
    # Jaccard-like but weighted towards seed coverage
    return overlap / len(seed_keywords) if seed_keywords else 0.0


def matches_topics(text: str, topics: list[str]) -> bool:
    """Check if text matches any of the topics."""
    text_lower = text.lower()
    for topic in topics:
        if topic.lower() in text_lower:
            return True
    return False


def matches_exclude_patterns(text: str, patterns: list[str]) -> bool:
    """Check if text matches any exclude pattern."""
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def filter_conversation(conv: dict, config: FilterConfig) -> tuple[bool, float]:
    """
    Check if conversation passes filters.

    Returns (passes, relevance_score)
    """
    # Date filter
    conv_date = get_conversation_date(conv)
    if config.after_date and conv_date and conv_date < config.after_date:
        return False, 0.0
    if config.before_date and conv_date and conv_date > config.before_date:
        return False, 0.0

    # Message count filter
    msg_count = count_messages(conv)
    if msg_count < config.min_messages:
        return False, 0.0
    if config.max_messages and msg_count > config.max_messages:
        return False, 0.0

    # Extract text
    full_text = extract_conversation_text(conv)
    word_count = count_words(full_text)

    # Word count filter
    if word_count < config.min_words:
        return False, 0.0

    # Exclude patterns
    if config.exclude_patterns and matches_exclude_patterns(full_text, config.exclude_patterns):
        return False, 0.0

    # Topic filter (if specified)
    if config.topics and not matches_topics(full_text, config.topics):
        return False, 0.0

    # User-heavy filter
    if config.include_only_user_heavy:
        user_text = extract_user_text(conv)
        user_words = count_words(user_text)
        if user_words < word_count * 0.3:  # User wrote less than 30%
            return False, 0.0

    # Relevance scoring
    relevance = 0.0
    if config.seed_keywords:
        conv_keywords = extract_keywords(full_text)
        relevance = calculate_relevance(conv_keywords, config.seed_keywords)
        if relevance < config.min_relevance:
            return False, relevance

    return True, relevance


def analyze_export(path: Path, max_conversations: int = 10000) -> ConversationStats:
    """Analyze export without filtering."""
    stats = ConversationStats()
    title_words: Counter = Counter()
    dates = []

    print(f"Analyzing export: {path}")
    print(f"File size: {path.stat().st_size / (1024 * 1024):.1f} MB")

    for i, conv in enumerate(stream_conversations(path)):
        if i >= max_conversations:
            print(f"  (stopped at {max_conversations} for analysis)")
            break

        stats.total_conversations += 1
        msg_count = count_messages(conv)
        stats.total_messages += msg_count

        text = extract_conversation_text(conv)
        words = count_words(text)
        stats.total_words += words

        # Size distribution
        if words < 500:
            stats.size_distribution["small (<500 words)"] = stats.size_distribution.get("small (<500 words)", 0) + 1
        elif words < 2000:
            stats.size_distribution["medium (500-2000)"] = stats.size_distribution.get("medium (500-2000)", 0) + 1
        else:
            stats.size_distribution["large (2000+)"] = stats.size_distribution.get("large (2000+)", 0) + 1

        # Date tracking
        conv_date = get_conversation_date(conv)
        if conv_date:
            month_key = conv_date.strftime("%Y-%m")
            stats.conversations_by_month[month_key] = stats.conversations_by_month.get(month_key, 0) + 1
            dates.append(conv_date)

        # Title words
        title = conv.get("title", conv.get("name", ""))
        if title:
            for word in extract_keywords(title, min_length=3):
                title_words[word] += 1

        if (i + 1) % 1000 == 0:
            print(f"  Analyzed {i + 1} conversations...")

    if stats.total_conversations > 0:
        stats.avg_messages_per_conversation = stats.total_messages / stats.total_conversations
        stats.avg_words_per_conversation = stats.total_words / stats.total_conversations

    stats.top_title_words = title_words.most_common(30)

    if dates:
        dates.sort()
        stats.date_range = (dates[0].isoformat(), dates[-1].isoformat())

    return stats


def print_stats(stats: ConversationStats):
    """Print analysis statistics."""
    print("\n" + "=" * 60)
    print("EXPORT ANALYSIS")
    print("=" * 60)

    print(f"\nTotal Conversations: {stats.total_conversations:,}")
    print(f"Total Messages: {stats.total_messages:,}")
    print(f"Total Words: {stats.total_words:,}")
    print(f"Avg Messages/Conversation: {stats.avg_messages_per_conversation:.1f}")
    print(f"Avg Words/Conversation: {stats.avg_words_per_conversation:.0f}")

    if stats.date_range:
        print(f"\nDate Range: {stats.date_range[0][:10]} to {stats.date_range[1][:10]}")

    print("\nSize Distribution:")
    for size, count in sorted(stats.size_distribution.items()):
        pct = count / stats.total_conversations * 100 if stats.total_conversations else 0
        print(f"  {size}: {count:,} ({pct:.1f}%)")

    if stats.conversations_by_month:
        print("\nConversations by Month (recent):")
        sorted_months = sorted(stats.conversations_by_month.items(), reverse=True)[:12]
        for month, count in sorted_months:
            print(f"  {month}: {count:,}")

    if stats.top_title_words:
        print("\nTop Title Keywords:")
        for word, count in stats.top_title_words[:15]:
            print(f"  {word}: {count}")


def process_export(
    input_path: Path,
    output_path: Path,
    config: FilterConfig,
    verbose: bool = False,
) -> tuple[int, int]:
    """
    Process and filter export.

    Returns (total_processed, total_kept)
    """
    kept_conversations = []
    relevance_scores = []
    total_processed = 0

    print(f"Processing: {input_path}")
    print(f"Filters: {len(config.topics)} topics, min_relevance={config.min_relevance}")

    for conv in stream_conversations(input_path):
        total_processed += 1

        passes, relevance = filter_conversation(conv, config)

        if passes:
            kept_conversations.append((conv, relevance))
            relevance_scores.append(relevance)

        if verbose and total_processed % 500 == 0:
            print(f"  Processed {total_processed}, kept {len(kept_conversations)}")

    # Apply sampling if requested
    if config.sample_size and len(kept_conversations) > config.sample_size:
        # Sort by relevance and take top + random sample
        kept_conversations.sort(key=lambda x: -x[1])

        # Keep top 50% by relevance, random sample rest
        top_count = config.sample_size // 2
        top_convs = kept_conversations[:top_count]

        remaining = kept_conversations[top_count:]
        random.shuffle(remaining)
        sampled = remaining[: config.sample_size - top_count]

        kept_conversations = top_convs + sampled
        print(f"  Sampled down to {len(kept_conversations)} conversations")

    # Sort by relevance for output
    kept_conversations.sort(key=lambda x: -x[1])

    # Write output
    output_data = [conv for conv, _ in kept_conversations]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults:")
    print(f"  Input: {total_processed:,} conversations")
    print(f"  Output: {len(kept_conversations):,} conversations")
    print(f"  Reduction: {(1 - len(kept_conversations) / total_processed) * 100:.1f}%")

    if relevance_scores:
        avg_rel = sum(relevance_scores) / len(relevance_scores)
        print(f"  Avg Relevance: {avg_rel:.3f}")

    output_size = output_path.stat().st_size / (1024 * 1024)
    print(f"  Output Size: {output_size:.1f} MB")

    return total_processed, len(kept_conversations)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess large ChatGPT/Claude exports for essay synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input conversations.json file",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output filtered JSON file",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze, don't filter or output",
    )

    # Filtering options
    parser.add_argument(
        "--topics", "-t",
        nargs="+",
        default=[],
        help="Topics to filter for (OR matching)",
    )
    parser.add_argument(
        "--seed-essay",
        type=Path,
        help="Seed essay file for relevance-based filtering",
    )
    parser.add_argument(
        "--min-relevance",
        type=float,
        default=0.0,
        help="Minimum relevance score (0-1) when using seed essay",
    )
    parser.add_argument(
        "--after",
        type=str,
        help="Only conversations after this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--before",
        type=str,
        help="Only conversations before this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--min-messages",
        type=int,
        default=2,
        help="Minimum messages per conversation",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=100,
        help="Minimum words per conversation",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Random sample to this many conversations",
    )
    parser.add_argument(
        "--exclude-patterns",
        nargs="+",
        default=[],
        help="Regex patterns to exclude conversations",
    )
    parser.add_argument(
        "--user-heavy",
        action="store_true",
        help="Only include conversations where user wrote a lot",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Analyze-only mode
    if args.analyze_only:
        stats = analyze_export(args.input)
        print_stats(stats)
        return

    # Require output for filtering
    if not args.output:
        print("Error: --output required for filtering (use --analyze-only for analysis)")
        sys.exit(1)

    # Build filter config
    config = FilterConfig(
        topics=args.topics,
        min_messages=args.min_messages,
        min_words=args.min_words,
        min_relevance=args.min_relevance,
        sample_size=args.sample,
        exclude_patterns=args.exclude_patterns,
        include_only_user_heavy=args.user_heavy,
    )

    # Parse dates
    if args.after:
        config.after_date = datetime.fromisoformat(args.after)
    if args.before:
        config.before_date = datetime.fromisoformat(args.before)

    # Load seed keywords
    if args.seed_essay:
        config.seed_keywords = load_seed_keywords(args.seed_essay)
        print(f"Loaded {len(config.seed_keywords)} seed keywords")

    # Process
    process_export(args.input, args.output, config, args.verbose)


if __name__ == "__main__":
    main()
