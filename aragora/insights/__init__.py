"""
Aragora Insights - Extract and aggregate learnings from debates.

This module provides tools for:
- Extracting structured insights from completed debates
- Identifying winning argument patterns
- Tracking agent performance and specializations
- Aggregating meta-learnings across debate history

Key components:
- InsightExtractor: Extracts insights from DebateResult
- InsightStore: Persists insights to SQLite
- InsightAggregator: Cross-debate pattern analysis
"""

from aragora.insights.extractor import (
    DebateInsights,
    Insight,
    InsightExtractor,
    InsightType,
)
from aragora.insights.flip_detector import (
    AgentConsistencyScore,
    FlipDetector,
    FlipEvent,
    format_consistency_for_ui,
    format_flip_for_ui,
)
from aragora.insights.store import InsightStore

__all__ = [
    "Insight",
    "InsightType",
    "DebateInsights",
    "InsightExtractor",
    "InsightStore",
    "FlipEvent",
    "AgentConsistencyScore",
    "FlipDetector",
    "format_flip_for_ui",
    "format_consistency_for_ui",
]
