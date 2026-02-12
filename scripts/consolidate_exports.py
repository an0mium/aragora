#!/usr/bin/env python3
"""
Consolidate Multiple ChatGPT/Claude Exports

Merges conversation exports from multiple accounts into a single unified corpus
for essay synthesis.

Usage:
    # Point to a directory containing all your export folders/files
    python scripts/consolidate_exports.py \
        --input-dir ~/Downloads/all_exports \
        --output consolidated_conversations.json \
        --verbose

    # Or specify individual files
    python scripts/consolidate_exports.py \
        --inputs chatgpt1/conversations.json chatgpt2/conversations.json claude1.json \
        --output consolidated_conversations.json

    # With deduplication threshold
    python scripts/consolidate_exports.py \
        --input-dir ~/Downloads/all_exports \
        --output consolidated.json \
        --dedupe-threshold 0.9

Directory Structure Expected:
    all_exports/
    ├── chatgpt_account1/
    │   └── conversations.json
    ├── chatgpt_account2/
    │   └── conversations.json
    ├── claude_export1/
    │   └── conversations.json
    ├── claude_export2.json
    └── ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class ConsolidationStats:
    """Statistics about consolidation."""

    total_files_processed: int = 0
    total_conversations_found: int = 0
    duplicates_removed: int = 0
    final_conversation_count: int = 0
    conversations_by_source: dict[str, int] = field(default_factory=dict)
    conversations_by_account: dict[str, int] = field(default_factory=dict)
    date_range: tuple[str, str] | None = None
    total_words: int = 0


def detect_export_type(data: Any) -> str:
    """Detect if export is ChatGPT or Claude format."""
    if isinstance(data, list) and len(data) > 0:
        first = data[0]
        if isinstance(first, dict):
            if "mapping" in first:
                return "chatgpt"
            if "chat_messages" in first or "uuid" in first:
                return "claude"
    if isinstance(data, dict):
        if "conversations" in data:
            return "claude"
    return "unknown"


def extract_conversation_id(conv: dict, source_type: str) -> str:
    """Extract or generate a stable conversation ID."""
    if source_type == "chatgpt":
        return conv.get("id", conv.get("conversation_id", ""))
    else:  # claude
        return conv.get("uuid", conv.get("id", ""))


def extract_conversation_text(conv: dict, source_type: str) -> str:
    """Extract all text for deduplication comparison."""
    texts = []

    if source_type == "chatgpt":
        mapping = conv.get("mapping", {})
        for node_id, node in mapping.items():
            msg = node.get("message")
            if msg:
                parts = msg.get("content", {}).get("parts", [])
                for part in parts:
                    if isinstance(part, str):
                        texts.append(part)
    else:  # claude
        messages = conv.get("chat_messages", conv.get("messages", []))
        for msg in messages:
            text = msg.get("text", msg.get("content", ""))
            if isinstance(text, str):
                texts.append(text)

    return " ".join(texts)


def get_conversation_date(conv: dict, source_type: str) -> datetime | None:
    """Extract conversation creation date."""
    if source_type == "chatgpt":
        create_time = conv.get("create_time")
        if create_time:
            try:
                return datetime.fromtimestamp(create_time)
            except (ValueError, TypeError):
                pass
    else:  # claude
        created_at = conv.get("created_at")
        if created_at:
            try:
                return datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass
    return None


def compute_content_hash(text: str) -> str:
    """Compute hash of conversation content for deduplication."""
    # Normalize whitespace
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.sha256(normalized.encode()).hexdigest()[:32]


def similarity_score(text1: str, text2: str) -> float:
    """Compute simple similarity between two texts using word overlap."""
    words1 = set(re.findall(r"\b\w{4,}\b", text1.lower()))
    words2 = set(re.findall(r"\b\w{4,}\b", text2.lower()))

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def find_json_files(directory: Path) -> list[Path]:
    """Find all conversation JSON files in directory."""
    json_files = []

    for path in directory.rglob("*.json"):
        # Skip common non-conversation files
        if path.name in ("package.json", "tsconfig.json", "model_comparisons.json"):
            continue
        if "node_modules" in str(path):
            continue

        # Check if it looks like a conversation file
        try:
            with open(path, encoding="utf-8") as f:
                # Read first bit to check structure
                start = f.read(1000)
                if '"mapping"' in start or '"chat_messages"' in start or '"conversations"' in start:
                    json_files.append(path)
                elif start.strip().startswith("["):
                    # Could be an array of conversations
                    json_files.append(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

    return json_files


def load_conversations_from_file(
    path: Path,
    account_label: str | None = None,
) -> tuple[list[dict], str, str]:
    """
    Load conversations from a JSON file.

    Returns (conversations, source_type, account_label)
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    source_type = detect_export_type(data)

    # Extract conversations list
    if isinstance(data, list):
        conversations = data
    elif isinstance(data, dict) and "conversations" in data:
        conversations = data["conversations"]
    else:
        conversations = [data]

    # Generate account label if not provided
    if not account_label:
        account_label = path.parent.name if path.name == "conversations.json" else path.stem

    # Add metadata to each conversation
    for conv in conversations:
        conv["_source_file"] = str(path)
        conv["_source_type"] = source_type
        conv["_account_label"] = account_label

    return conversations, source_type, account_label


def deduplicate_conversations(
    conversations: list[dict],
    threshold: float = 0.9,
    verbose: bool = False,
) -> tuple[list[dict], int]:
    """
    Remove duplicate conversations based on content similarity.

    Returns (deduplicated_conversations, num_removed)
    """
    if threshold <= 0:
        return conversations, 0

    # Group by content hash for exact duplicates
    hash_groups: dict[str, list[dict]] = defaultdict(list)

    for conv in conversations:
        source_type = conv.get("_source_type", "unknown")
        text = extract_conversation_text(conv, source_type)
        content_hash = compute_content_hash(text)
        hash_groups[content_hash].append(conv)

    # Keep one from each exact duplicate group (prefer most recent)
    unique_convs = []
    exact_dupes_removed = 0

    def get_sortable_date(c):
        """Get a timezone-naive datetime for sorting."""
        date = get_conversation_date(c, c.get("_source_type", "unknown"))
        if date is None:
            return datetime.min
        # Strip timezone info for comparison
        if date.tzinfo is not None:
            return date.replace(tzinfo=None)
        return date

    for hash_val, group in hash_groups.items():
        if len(group) > 1:
            # Sort by date, keep most recent
            group.sort(
                key=get_sortable_date,
                reverse=True,
            )
            exact_dupes_removed += len(group) - 1
            if verbose and len(group) > 1:
                title = group[0].get("title", group[0].get("name", "Untitled"))[:50]
                print(f"  Exact duplicate: '{title}' ({len(group)} copies)")

        unique_convs.append(group[0])

    # For near-duplicates, use similarity comparison (expensive, so only if threshold < 1.0)
    near_dupes_removed = 0
    if threshold < 1.0:
        # Extract text for all conversations
        conv_texts = []
        for conv in unique_convs:
            source_type = conv.get("_source_type", "unknown")
            text = extract_conversation_text(conv, source_type)
            conv_texts.append((conv, text))

        # Find near-duplicates (O(n^2) but typically small n)
        to_remove = set()
        for i, (conv1, text1) in enumerate(conv_texts):
            if i in to_remove:
                continue
            for j, (conv2, text2) in enumerate(conv_texts[i + 1 :], start=i + 1):
                if j in to_remove:
                    continue

                sim = similarity_score(text1, text2)
                if sim >= threshold:
                    # Keep the one with more content
                    if len(text1) >= len(text2):
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break

                    if verbose:
                        title1 = conv1.get("title", conv1.get("name", ""))[:30]
                        title2 = conv2.get("title", conv2.get("name", ""))[:30]
                        print(f"  Near duplicate ({sim:.2f}): '{title1}' ~ '{title2}'")

        near_dupes_removed = len(to_remove)
        unique_convs = [conv for i, (conv, _) in enumerate(conv_texts) if i not in to_remove]

    total_removed = exact_dupes_removed + near_dupes_removed
    return unique_convs, total_removed


def consolidate_exports(
    input_paths: list[Path],
    dedupe_threshold: float = 0.9,
    verbose: bool = False,
) -> tuple[list[dict], ConsolidationStats]:
    """
    Consolidate multiple export files into a single list.

    Returns (consolidated_conversations, stats)
    """
    stats = ConsolidationStats()
    all_conversations = []
    all_dates = []

    for path in input_paths:
        if not path.exists():
            print(f"Warning: Path not found: {path}")
            continue

        try:
            conversations, source_type, account = load_conversations_from_file(path)
            stats.total_files_processed += 1
            stats.total_conversations_found += len(conversations)

            # Track by source type
            stats.conversations_by_source[source_type] = (
                stats.conversations_by_source.get(source_type, 0) + len(conversations)
            )

            # Track by account
            stats.conversations_by_account[account] = (
                stats.conversations_by_account.get(account, 0) + len(conversations)
            )

            # Collect dates
            for conv in conversations:
                date = get_conversation_date(conv, source_type)
                if date:
                    all_dates.append(date)

            all_conversations.extend(conversations)

            if verbose:
                print(f"  Loaded {len(conversations)} from {path.name} ({source_type})")

        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")

    # Deduplicate
    if verbose:
        print(f"\nDeduplicating {len(all_conversations)} conversations...")

    deduped, removed = deduplicate_conversations(
        all_conversations,
        threshold=dedupe_threshold,
        verbose=verbose,
    )

    stats.duplicates_removed = removed
    stats.final_conversation_count = len(deduped)

    # Calculate total words
    for conv in deduped:
        source_type = conv.get("_source_type", "unknown")
        text = extract_conversation_text(conv, source_type)
        stats.total_words += len(text.split())

    # Date range - normalize to naive datetimes for sorting
    if all_dates:
        normalized_dates = []
        for d in all_dates:
            if d.tzinfo is not None:
                normalized_dates.append(d.replace(tzinfo=None))
            else:
                normalized_dates.append(d)
        normalized_dates.sort()
        stats.date_range = (normalized_dates[0].isoformat()[:10], normalized_dates[-1].isoformat()[:10])

    return deduped, stats


def print_stats(stats: ConsolidationStats):
    """Print consolidation statistics."""
    print("\n" + "=" * 60)
    print("CONSOLIDATION RESULTS")
    print("=" * 60)

    print(f"\nFiles Processed: {stats.total_files_processed}")
    print(f"Total Conversations Found: {stats.total_conversations_found:,}")
    print(f"Duplicates Removed: {stats.duplicates_removed:,}")
    print(f"Final Conversation Count: {stats.final_conversation_count:,}")
    print(f"Total Words: {stats.total_words:,}")

    if stats.date_range:
        print(f"Date Range: {stats.date_range[0]} to {stats.date_range[1]}")

    print("\nBy Source Type:")
    for source, count in sorted(stats.conversations_by_source.items()):
        print(f"  {source}: {count:,}")

    print("\nBy Account:")
    for account, count in sorted(stats.conversations_by_account.items(), key=lambda x: -x[1]):
        print(f"  {account}: {count:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate multiple ChatGPT/Claude exports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input-dir", "-d",
        type=Path,
        help="Directory containing export folders/files",
    )
    parser.add_argument(
        "--inputs", "-i",
        type=Path,
        nargs="+",
        help="Individual export files to consolidate",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output consolidated JSON file",
    )
    parser.add_argument(
        "--dedupe-threshold",
        type=float,
        default=0.9,
        help="Similarity threshold for deduplication (0-1, default 0.9)",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Skip deduplication",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Gather input files
    input_paths = []

    if args.input_dir:
        if not args.input_dir.exists():
            print(f"Error: Input directory not found: {args.input_dir}")
            sys.exit(1)
        input_paths.extend(find_json_files(args.input_dir))

    if args.inputs:
        input_paths.extend(args.inputs)

    if not input_paths:
        print("Error: No input files specified. Use --input-dir or --inputs")
        sys.exit(1)

    print(f"Found {len(input_paths)} potential export files")

    # Consolidate
    dedupe_threshold = 0 if args.no_dedupe else args.dedupe_threshold
    consolidated, stats = consolidate_exports(
        input_paths,
        dedupe_threshold=dedupe_threshold,
        verbose=args.verbose,
    )

    # Write output
    # Remove internal metadata before saving
    for conv in consolidated:
        conv.pop("_source_file", None)
        # Keep _source_type and _account_label for traceability

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2, default=str)

    print_stats(stats)

    output_size = args.output.stat().st_size / (1024 * 1024)
    print(f"\nOutput written to: {args.output}")
    print(f"Output size: {output_size:.1f} MB")


if __name__ == "__main__":
    main()
