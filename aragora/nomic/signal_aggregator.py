"""External signal aggregator for self-improvement prioritization.

Unifies signals from four sources into weighted improvement goals:

1. **User feedback** — NPS scores, feature requests, bug reports
   (from ``FeedbackAnalyzer`` / ``FeedbackStore``)
2. **Business metrics** — pipeline completion rates, agent performance
   (from analytics handlers and ``CycleTelemetryCollector``)
3. **Market signals** — trending topics, research papers
   (from ``PulseManager`` / ``PulseAdapter``)
4. **Obsidian-sourced goals** — notes tagged ``#aragora-improve``
   (from ``ObsidianAdapter``)

The aggregator runs periodically (e.g., before each self-improvement cycle)
to produce a prioritized list of ``ImprovementGoal``s that feed into the
``ImprovementQueue`` for ``MetaPlanner`` consumption.

Usage::

    aggregator = SignalAggregator()
    goals = await aggregator.collect_all()
    aggregator.push_to_queue(goals)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal types
# ---------------------------------------------------------------------------


@dataclass
class ExternalSignal:
    """A single signal from an external source."""

    source: str  # "user_feedback", "business_metrics", "market", "obsidian"
    title: str
    description: str
    priority: float  # 0.0-1.0
    category: str  # maps to ImprovementGoal categories
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class AggregatedSignals:
    """Result of collecting signals from all sources."""

    signals: list[ExternalSignal] = field(default_factory=list)
    source_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    collected_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Source weights — tunable per deployment
# ---------------------------------------------------------------------------

DEFAULT_SOURCE_WEIGHTS: dict[str, float] = {
    "user_feedback": 1.5,  # User pain > everything else
    "business_metrics": 1.2,  # Completion rate drops are urgent
    "market": 0.8,  # Trending topics are nice-to-have
    "obsidian": 1.3,  # Explicit user requests are high-signal
}


# ---------------------------------------------------------------------------
# Collector functions (one per source)
# ---------------------------------------------------------------------------


def collect_user_feedback_signals(
    limit: int = 20,
) -> list[ExternalSignal]:
    """Extract improvement signals from user feedback.

    Reads from ``FeedbackStore`` via ``FeedbackAnalyzer``, converting
    unprocessed feedback into ``ExternalSignal``s.
    """
    signals: list[ExternalSignal] = []
    try:
        from aragora.nomic.feedback_analyzer import FeedbackAnalyzer

        analyzer = FeedbackAnalyzer()
        result = analyzer.process_new_feedback()

        # Convert learnings to signals
        for learning in getattr(result, "learnings", [])[:limit]:
            category = _map_feedback_category(getattr(learning, "category", "general"))
            signals.append(
                ExternalSignal(
                    source="user_feedback",
                    title=getattr(learning, "title", "User feedback item"),
                    description=getattr(learning, "description", str(learning)),
                    priority=getattr(learning, "priority", 0.5),
                    category=category,
                    metadata={
                        "feedback_type": getattr(learning, "feedback_type", "general"),
                        "track": getattr(learning, "track", "core"),
                    },
                )
            )
    except ImportError:
        logger.debug("FeedbackAnalyzer not available; skipping user feedback signals")
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("User feedback collection failed: %s", exc)

    return signals


def collect_business_metric_signals(
    completion_rate_threshold: float = 0.7,
    cost_spike_threshold: float = 1.5,
) -> list[ExternalSignal]:
    """Extract improvement signals from business metrics.

    Checks pipeline completion rates, agent timeout rates, and cost
    trends to identify areas needing attention.
    """
    signals: list[ExternalSignal] = []
    try:
        from aragora.nomic.cycle_telemetry import CycleTelemetryCollector

        telemetry = CycleTelemetryCollector()
        records = telemetry.get_recent_cycles(n=50)

        if not records:
            return signals

        # Check success rate
        total = len(records)
        successes = sum(1 for r in records if getattr(r, "success", False))
        success_rate = successes / total if total > 0 else 1.0

        if success_rate < completion_rate_threshold:
            signals.append(
                ExternalSignal(
                    source="business_metrics",
                    title=f"Low pipeline success rate ({success_rate:.0%})",
                    description=(
                        f"Only {successes}/{total} recent improvement cycles succeeded. "
                        f"Threshold is {completion_rate_threshold:.0%}. "
                        "Investigate common failure modes."
                    ),
                    priority=min(1.0, (1.0 - success_rate) * 1.5),
                    category="reliability",
                    metadata={
                        "success_rate": success_rate,
                        "total_cycles": total,
                    },
                )
            )

        # Check cost trends
        costs = [getattr(r, "cost_usd", 0.0) for r in records if getattr(r, "cost_usd", 0.0) > 0]
        if len(costs) >= 10:
            recent_avg = sum(costs[:5]) / 5
            older_avg = sum(costs[5:10]) / 5
            if older_avg > 0 and recent_avg / older_avg > cost_spike_threshold:
                signals.append(
                    ExternalSignal(
                        source="business_metrics",
                        title="Cost spike detected in recent cycles",
                        description=(
                            f"Average cycle cost rose from ${older_avg:.2f} "
                            f"to ${recent_avg:.2f} ({recent_avg / older_avg:.1f}x). "
                            "Consider model downgrades or prompt optimization."
                        ),
                        priority=0.6,
                        category="performance",
                        metadata={
                            "recent_avg_cost": recent_avg,
                            "older_avg_cost": older_avg,
                        },
                    )
                )

        # Check timeout patterns
        timeout_agents: dict[str, int] = {}
        for r in records:
            for agent in getattr(r, "agents_used", []):
                timeout_agents.setdefault(agent, 0)
            if not getattr(r, "success", True):
                for agent in getattr(r, "agents_used", []):
                    timeout_agents[agent] = timeout_agents.get(agent, 0) + 1

        for agent, failures in timeout_agents.items():
            total_uses = sum(1 for r in records if agent in getattr(r, "agents_used", []))
            if total_uses >= 5 and failures / total_uses > 0.4:
                signals.append(
                    ExternalSignal(
                        source="business_metrics",
                        title=f"Agent '{agent}' has high failure rate",
                        description=(
                            f"Agent '{agent}' failed in {failures}/{total_uses} "
                            f"cycles ({failures / total_uses:.0%}). "
                            "Consider fallback configuration or model change."
                        ),
                        priority=0.7,
                        category="reliability",
                        metadata={
                            "agent": agent,
                            "failure_rate": failures / total_uses,
                        },
                    )
                )

    except ImportError:
        logger.debug("CycleTelemetryCollector not available; skipping business metrics")
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Business metric collection failed: %s", exc)

    return signals


async def collect_market_signals(
    platforms: list[str] | None = None,
    limit: int = 10,
) -> list[ExternalSignal]:
    """Extract improvement signals from trending topics.

    Uses ``PulseManager`` to fetch trending topics from configured
    platforms and converts relevant ones into improvement signals.
    """
    signals: list[ExternalSignal] = []
    if platforms is None:
        platforms = ["hackernews", "reddit", "arxiv"]

    try:
        from aragora.pulse.ingestor import PulseManager

        manager = PulseManager()
        topics = await manager.get_trending_topics(
            platforms=platforms,
            limit_per_platform=limit,
        )

        # Filter for AI/ML/dev-tools relevant topics
        relevant_keywords = {
            "ai",
            "llm",
            "agent",
            "debate",
            "reasoning",
            "multi-agent",
            "mcp",
            "tool-use",
            "rag",
            "prompt",
            "evaluation",
            "benchmark",
        }

        for topic in topics:
            topic_text = getattr(topic, "topic", "").lower()
            if any(kw in topic_text for kw in relevant_keywords):
                signals.append(
                    ExternalSignal(
                        source="market",
                        title=f"Trending: {getattr(topic, 'topic', 'unknown')}",
                        description=(
                            f"Trending on {getattr(topic, 'platform', 'unknown')} "
                            f"(volume: {getattr(topic, 'volume', 0)}). "
                            "Consider if Aragora should adapt to this trend."
                        ),
                        priority=min(
                            0.6,
                            0.3 + getattr(topic, "volume", 0) / 10000,
                        ),
                        category="features",
                        metadata={
                            "platform": getattr(topic, "platform", "unknown"),
                            "volume": getattr(topic, "volume", 0),
                            "category": getattr(topic, "category", "tech"),
                        },
                    )
                )

    except ImportError:
        logger.debug("PulseManager not available; skipping market signals")
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Market signal collection failed: %s", exc)

    return signals


def collect_obsidian_signals(
    vault_path: str | None = None,
    tag: str = "#aragora-improve",
) -> list[ExternalSignal]:
    """Extract improvement signals from Obsidian notes.

    Scans an Obsidian vault for notes tagged with ``#aragora-improve``
    and converts them into ``ExternalSignal``s. Each tagged note becomes
    an explicit user request for improvement.
    """
    signals: list[ExternalSignal] = []

    if vault_path is None:
        # Try common vault locations
        from pathlib import Path

        candidates = [
            Path.home() / "Documents" / "Obsidian",
            Path.home() / "obsidian",
            Path.home() / "vault",
        ]
        for candidate in candidates:
            if candidate.exists():
                vault_path = str(candidate)
                break

    if vault_path is None:
        logger.debug("No Obsidian vault found; skipping Obsidian signals")
        return signals

    from pathlib import Path

    vault = Path(vault_path)
    if not vault.is_dir():
        return signals

    tag_pattern = re.compile(re.escape(tag), re.IGNORECASE)

    try:
        for md_file in vault.rglob("*.md"):
            # Skip hidden dirs (e.g., .obsidian/)
            if any(part.startswith(".") for part in md_file.parts):
                continue

            try:
                content = md_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            if not tag_pattern.search(content):
                continue

            # Extract the title (first heading or filename)
            title = md_file.stem
            first_heading = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            if first_heading:
                title = first_heading.group(1).strip()

            # Extract priority from optional tag like #priority-high
            priority = 0.7  # Default: user-tagged = high signal
            if "#priority-critical" in content.lower():
                priority = 1.0
            elif "#priority-high" in content.lower():
                priority = 0.8
            elif "#priority-low" in content.lower():
                priority = 0.4

            # Extract category from optional tags
            category = "general"
            category_map = {
                "#performance": "performance",
                "#reliability": "reliability",
                "#ux": "ux",
                "#security": "security",
                "#testing": "test_coverage",
                "#docs": "documentation",
                "#feature": "features",
            }
            for cat_tag, cat_name in category_map.items():
                if cat_tag in content.lower():
                    category = cat_name
                    break

            # Use first paragraph after the tag as description
            description = _extract_description(content, tag)

            signals.append(
                ExternalSignal(
                    source="obsidian",
                    title=title,
                    description=description or f"Improvement requested via {md_file.name}",
                    priority=priority,
                    category=category,
                    metadata={
                        "file": str(md_file.relative_to(vault)),
                        "vault": vault_path,
                    },
                )
            )
    except OSError as exc:
        logger.warning("Obsidian vault scan failed: %s", exc)

    return signals


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class SignalAggregator:
    """Collects and ranks signals from all external sources.

    Usage::

        aggregator = SignalAggregator()
        result = await aggregator.collect_all()
        aggregator.push_to_queue(result.signals)
    """

    def __init__(
        self,
        source_weights: dict[str, float] | None = None,
        obsidian_vault: str | None = None,
        pulse_platforms: list[str] | None = None,
    ):
        self.weights = source_weights or DEFAULT_SOURCE_WEIGHTS.copy()
        self.obsidian_vault = obsidian_vault
        self.pulse_platforms = pulse_platforms

    async def collect_all(
        self,
        include_user_feedback: bool = True,
        include_business_metrics: bool = True,
        include_market: bool = True,
        include_obsidian: bool = True,
    ) -> AggregatedSignals:
        """Collect signals from all enabled sources.

        Each source runs independently — failures in one source don't
        block others (graceful degradation).
        """
        result = AggregatedSignals()

        if include_user_feedback:
            try:
                fb_signals = collect_user_feedback_signals()
                result.signals.extend(fb_signals)
                result.source_counts["user_feedback"] = len(fb_signals)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                result.errors.append(f"user_feedback: {exc}")

        if include_business_metrics:
            try:
                bm_signals = collect_business_metric_signals()
                result.signals.extend(bm_signals)
                result.source_counts["business_metrics"] = len(bm_signals)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                result.errors.append(f"business_metrics: {exc}")

        if include_market:
            try:
                mk_signals = await collect_market_signals(platforms=self.pulse_platforms)
                result.signals.extend(mk_signals)
                result.source_counts["market"] = len(mk_signals)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                result.errors.append(f"market: {exc}")

        if include_obsidian:
            try:
                ob_signals = collect_obsidian_signals(vault_path=self.obsidian_vault)
                result.signals.extend(ob_signals)
                result.source_counts["obsidian"] = len(ob_signals)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                result.errors.append(f"obsidian: {exc}")

        # Apply source weights and sort by priority
        for signal in result.signals:
            weight = self.weights.get(signal.source, 1.0)
            signal.priority = min(1.0, signal.priority * weight)

        result.signals.sort(key=lambda s: s.priority, reverse=True)

        logger.info(
            "Collected %d signals from %d sources (%s)",
            len(result.signals),
            len(result.source_counts),
            result.source_counts,
        )

        return result

    def push_to_queue(
        self,
        signals: list[ExternalSignal],
        limit: int = 20,
    ) -> int:
        """Push top signals to the ``ImprovementQueue`` as goals.

        Returns the number of goals enqueued.
        """
        count = 0
        try:
            from aragora.nomic.improvement_queue import (
                ImprovementQueue,
                ImprovementSuggestion,
            )

            queue = ImprovementQueue()

            for signal in signals[:limit]:
                suggestion = ImprovementSuggestion(
                    debate_id=f"signal-{signal.source}-{int(signal.created_at)}",
                    task=signal.title,
                    suggestion=signal.description,
                    category=signal.category,
                    confidence=signal.priority,
                )
                queue.enqueue(suggestion)
                count += 1

        except ImportError:
            logger.warning("ImprovementQueue not available; signals collected but not queued")

        logger.info("Pushed %d signals to improvement queue", count)
        return count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _map_feedback_category(category: str) -> str:
    """Map feedback categories to improvement goal categories."""
    mapping = {
        "bug_report": "reliability",
        "feature_request": "features",
        "debate_quality": "accuracy",
        "nps": "ux",
        "performance": "performance",
        "documentation": "documentation",
    }
    return mapping.get(category, "general")


def _extract_description(content: str, tag: str) -> str:
    """Extract the first meaningful paragraph after a tag occurrence."""
    tag_pos = content.lower().find(tag.lower())
    if tag_pos == -1:
        return ""

    # Get text after the tag
    after_tag = content[tag_pos + len(tag) :]

    # Find first non-empty paragraph
    paragraphs = after_tag.split("\n\n")
    for para in paragraphs:
        cleaned = para.strip()
        # Skip empty lines, headings, and other tags
        if cleaned and not cleaned.startswith("#") and not cleaned.startswith("---"):
            # Truncate long descriptions
            if len(cleaned) > 500:
                cleaned = cleaned[:497] + "..."
            return cleaned

    return ""
