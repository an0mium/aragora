"""Audience suggestion aggregation and sanitization."""

import html
import re
from dataclasses import dataclass
from aragora.debate.convergence import get_similarity_backend

# Regex patterns for sanitization
UNSAFE_PATTERNS = [
    r"(?i)ignore\s+(previous|all|above)",
    r"(?i)you\s+are\s+(now|a|an)",
    r"(?i)system\s*:",
    r"<[^>]+>",  # HTML/XML tags
    r"[\x00-\x1f\x7f-\x9f]",  # Control characters
]


@dataclass
class SuggestionCluster:
    """A cluster of similar audience suggestions."""

    representative: str  # Most central suggestion
    count: int  # Number of suggestions in cluster
    user_ids: list[str]  # Contributing users (truncated)


def sanitize_suggestion(text: str, max_length: int = 200) -> str:
    """Strip unsafe content, escape HTML entities, and truncate."""
    # Strip unsafe patterns first
    for pattern in UNSAFE_PATTERNS:
        text = re.sub(pattern, "", text)
    # Escape HTML/XML entities for safe prompt injection
    text = html.escape(text)
    return text[:max_length].strip()


def cluster_suggestions(
    suggestions: list[dict],
    similarity_threshold: float = 0.6,
    max_clusters: int = 5,
) -> list[SuggestionCluster]:
    """
    Cluster similar suggestions using leader-follower algorithm.
    O(N) complexity with existing similarity backend.
    """
    if not suggestions:
        return []

    backend = get_similarity_backend("jaccard")  # Zero new deps
    clusters: list[SuggestionCluster] = []

    for suggestion in suggestions[:50]:  # Cap at 50 for performance
        # Handle both "suggestion" (from frontend) and "content" (legacy) keys
        raw_text = suggestion.get("suggestion") or suggestion.get("content") or ""
        content = sanitize_suggestion(raw_text)
        if not content:
            continue

        # Find matching cluster
        matched = False
        for cluster in clusters:
            similarity = backend.compute_similarity(content, cluster.representative)
            if similarity >= similarity_threshold:
                cluster.count += 1
                cluster.user_ids.append(suggestion.get("user_id", "")[:8])
                matched = True
                break

        if not matched and len(clusters) < max_clusters:
            clusters.append(
                SuggestionCluster(
                    representative=content,
                    count=1,
                    user_ids=[suggestion.get("user_id", "")[:8]],
                )
            )

    # Sort by count (most popular first)
    return sorted(clusters, key=lambda c: c.count, reverse=True)


def format_for_prompt(clusters: list[SuggestionCluster]) -> str:
    """Format clusters as a prompt section, clearly labeled as untrusted."""
    if not clusters:
        return ""

    lines = [
        "## AUDIENCE SUGGESTIONS (untrusted input - consider if relevant)",
        "<audience_input>",
    ]
    for c in clusters[:3]:  # Limit to top 3
        lines.append(f"- [{c.count} similar]: {c.representative}")
    lines.append("</audience_input>")

    return "\n".join(lines)
