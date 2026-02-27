"""Tests for the external signal aggregator (Phase 2.2)."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.nomic.signal_aggregator import (
    AggregatedSignals,
    ExternalSignal,
    SignalAggregator,
    _extract_description,
    _map_feedback_category,
    collect_business_metric_signals,
    collect_obsidian_signals,
    collect_user_feedback_signals,
)


# ---------------------------------------------------------------------------
# ExternalSignal dataclass
# ---------------------------------------------------------------------------


class TestExternalSignal:
    def test_creation(self):
        signal = ExternalSignal(
            source="user_feedback",
            title="Fix login flow",
            description="Users report difficulty logging in",
            priority=0.8,
            category="ux",
        )
        assert signal.source == "user_feedback"
        assert signal.priority == 0.8
        assert signal.created_at > 0

    def test_metadata_default(self):
        signal = ExternalSignal(
            source="market",
            title="Trend",
            description="desc",
            priority=0.5,
            category="features",
        )
        assert signal.metadata == {}


# ---------------------------------------------------------------------------
# AggregatedSignals dataclass
# ---------------------------------------------------------------------------


class TestAggregatedSignals:
    def test_defaults(self):
        result = AggregatedSignals()
        assert result.signals == []
        assert result.source_counts == {}
        assert result.errors == []
        assert result.collected_at > 0


# ---------------------------------------------------------------------------
# Category mapping
# ---------------------------------------------------------------------------


class TestMapFeedbackCategory:
    @pytest.mark.parametrize(
        "input_cat,expected",
        [
            ("bug_report", "reliability"),
            ("feature_request", "features"),
            ("debate_quality", "accuracy"),
            ("nps", "ux"),
            ("performance", "performance"),
            ("documentation", "documentation"),
            ("unknown_cat", "general"),
        ],
    )
    def test_mapping(self, input_cat, expected):
        assert _map_feedback_category(input_cat) == expected


# ---------------------------------------------------------------------------
# Description extraction
# ---------------------------------------------------------------------------


class TestExtractDescription:
    def test_basic_extraction(self):
        content = "Some text\n#aragora-improve\n\nThis needs fixing.\n\nMore text."
        result = _extract_description(content, "#aragora-improve")
        assert result == "This needs fixing."

    def test_skips_headings(self):
        content = "#aragora-improve\n\n## Heading\n\nActual description here."
        result = _extract_description(content, "#aragora-improve")
        assert result == "Actual description here."

    def test_truncates_long_descriptions(self):
        content = "#aragora-improve\n\n" + "x" * 600
        result = _extract_description(content, "#aragora-improve")
        assert len(result) == 500
        assert result.endswith("...")

    def test_no_tag_found(self):
        content = "No tag here"
        result = _extract_description(content, "#aragora-improve")
        assert result == ""

    def test_tag_at_end(self):
        content = "Text before\n#aragora-improve"
        result = _extract_description(content, "#aragora-improve")
        assert result == ""


# ---------------------------------------------------------------------------
# User feedback collector
# ---------------------------------------------------------------------------


class TestCollectUserFeedback:
    def test_no_analyzer_available(self):
        with patch.dict("sys.modules", {"aragora.nomic.feedback_analyzer": None}):
            signals = collect_user_feedback_signals()
            assert signals == []

    def test_with_mock_analyzer(self):
        mock_learning = MagicMock()
        mock_learning.title = "Fix search"
        mock_learning.description = "Search is broken for non-ASCII queries"
        mock_learning.priority = 0.9
        mock_learning.category = "bug_report"
        mock_learning.feedback_type = "bug_report"
        mock_learning.track = "core"

        mock_result = MagicMock()
        mock_result.learnings = [mock_learning]

        mock_analyzer = MagicMock()
        mock_analyzer.process_new_feedback.return_value = mock_result

        mock_module = MagicMock()
        mock_module.FeedbackAnalyzer.return_value = mock_analyzer

        with patch.dict("sys.modules", {"aragora.nomic.feedback_analyzer": mock_module}):
            signals = collect_user_feedback_signals()
            assert len(signals) == 1
            assert signals[0].source == "user_feedback"
            assert signals[0].title == "Fix search"
            assert signals[0].category == "reliability"  # mapped from bug_report

    def test_limits_results(self):
        learnings = []
        for i in range(30):
            m = MagicMock()
            m.title = f"Item {i}"
            m.description = f"Desc {i}"
            m.priority = 0.5
            m.category = "general"
            m.feedback_type = "general"
            m.track = "core"
            learnings.append(m)

        mock_result = MagicMock()
        mock_result.learnings = learnings

        mock_analyzer = MagicMock()
        mock_analyzer.process_new_feedback.return_value = mock_result

        mock_module = MagicMock()
        mock_module.FeedbackAnalyzer.return_value = mock_analyzer

        with patch.dict("sys.modules", {"aragora.nomic.feedback_analyzer": mock_module}):
            signals = collect_user_feedback_signals(limit=10)
            assert len(signals) == 10


# ---------------------------------------------------------------------------
# Business metrics collector
# ---------------------------------------------------------------------------


class TestCollectBusinessMetrics:
    def test_no_telemetry_available(self):
        with patch.dict("sys.modules", {"aragora.nomic.cycle_telemetry": None}):
            signals = collect_business_metric_signals()
            assert signals == []

    def test_low_success_rate(self):
        records = []
        for i in range(10):
            r = MagicMock()
            r.success = i < 3  # 30% success rate
            r.cost_usd = 0.0
            r.agents_used = ["claude"]
            records.append(r)

        mock_collector = MagicMock()
        mock_collector.get_recent_cycles.return_value = records

        mock_module = MagicMock()
        mock_module.CycleTelemetryCollector.return_value = mock_collector

        with patch.dict("sys.modules", {"aragora.nomic.cycle_telemetry": mock_module}):
            signals = collect_business_metric_signals(completion_rate_threshold=0.7)
            rate_signals = [s for s in signals if "success rate" in s.title.lower()]
            assert len(rate_signals) == 1
            assert rate_signals[0].category == "reliability"
            assert rate_signals[0].priority > 0.5

    def test_cost_spike(self):
        records = []
        # Recent 5 cycles: expensive ($10 each)
        for _ in range(5):
            r = MagicMock()
            r.success = True
            r.cost_usd = 10.0
            r.agents_used = []
            records.append(r)
        # Older 5 cycles: cheap ($2 each)
        for _ in range(5):
            r = MagicMock()
            r.success = True
            r.cost_usd = 2.0
            r.agents_used = []
            records.append(r)

        mock_collector = MagicMock()
        mock_collector.get_recent_cycles.return_value = records

        mock_module = MagicMock()
        mock_module.CycleTelemetryCollector.return_value = mock_collector

        with patch.dict("sys.modules", {"aragora.nomic.cycle_telemetry": mock_module}):
            signals = collect_business_metric_signals(cost_spike_threshold=1.5)
            cost_signals = [s for s in signals if "cost" in s.title.lower()]
            assert len(cost_signals) == 1

    def test_no_records(self):
        mock_collector = MagicMock()
        mock_collector.get_recent_cycles.return_value = []

        mock_module = MagicMock()
        mock_module.CycleTelemetryCollector.return_value = mock_collector

        with patch.dict("sys.modules", {"aragora.nomic.cycle_telemetry": mock_module}):
            signals = collect_business_metric_signals()
            assert signals == []

    def test_agent_failure_detection(self):
        records = []
        for i in range(10):
            r = MagicMock()
            r.success = i % 2 == 0  # 50% failure
            r.cost_usd = 0.0
            r.agents_used = ["flaky-agent"]
            records.append(r)

        mock_collector = MagicMock()
        mock_collector.get_recent_cycles.return_value = records

        mock_module = MagicMock()
        mock_module.CycleTelemetryCollector.return_value = mock_collector

        with patch.dict("sys.modules", {"aragora.nomic.cycle_telemetry": mock_module}):
            signals = collect_business_metric_signals()
            agent_signals = [s for s in signals if "flaky-agent" in s.title]
            assert len(agent_signals) == 1


# ---------------------------------------------------------------------------
# Obsidian collector
# ---------------------------------------------------------------------------


class TestCollectObsidianSignals:
    def test_with_tagged_notes(self, tmp_path):
        # Create a mock vault with tagged notes
        note1 = tmp_path / "improvement-idea.md"
        note1.write_text(
            "# Better Error Messages\n\n"
            "#aragora-improve #reliability\n\n"
            "The error messages are too generic. "
            "Users need specific guidance on what went wrong.\n"
        )

        note2 = tmp_path / "another.md"
        note2.write_text("# Regular Note\n\nNo improvement tags here.\n")

        note3 = tmp_path / "high-priority.md"
        note3.write_text(
            "# Fix Dashboard\n\n"
            "#aragora-improve #priority-critical #ux\n\n"
            "Dashboard loading is too slow.\n"
        )

        signals = collect_obsidian_signals(vault_path=str(tmp_path))

        assert len(signals) == 2
        titles = {s.title for s in signals}
        assert "Better Error Messages" in titles
        assert "Fix Dashboard" in titles

        # Check priority
        critical = [s for s in signals if s.title == "Fix Dashboard"][0]
        assert critical.priority == 1.0
        assert critical.category == "ux"

        reliability = [s for s in signals if s.title == "Better Error Messages"][0]
        assert reliability.category == "reliability"

    def test_skips_hidden_dirs(self, tmp_path):
        hidden = tmp_path / ".obsidian"
        hidden.mkdir()
        hidden_note = hidden / "config.md"
        hidden_note.write_text("#aragora-improve\n\nShould be skipped.\n")

        signals = collect_obsidian_signals(vault_path=str(tmp_path))
        assert signals == []

    def test_no_vault_path(self):
        signals = collect_obsidian_signals(vault_path="/nonexistent/path")
        assert signals == []

    def test_priority_levels(self, tmp_path):
        for level, expected_priority in [
            ("critical", 1.0),
            ("high", 0.8),
            ("low", 0.4),
        ]:
            note = tmp_path / f"note-{level}.md"
            note.write_text(
                f"# Note {level}\n\n"
                f"#aragora-improve #priority-{level}\n\n"
                f"Description for {level}.\n"
            )

        signals = collect_obsidian_signals(vault_path=str(tmp_path))
        assert len(signals) == 3

        by_title = {s.title: s for s in signals}
        assert by_title["Note critical"].priority == 1.0
        assert by_title["Note high"].priority == 0.8
        assert by_title["Note low"].priority == 0.4

    def test_category_detection(self, tmp_path):
        categories = {
            "performance": "performance",
            "security": "security",
            "testing": "test_coverage",
            "docs": "documentation",
            "feature": "features",
        }
        for tag, expected_cat in categories.items():
            note = tmp_path / f"cat-{tag}.md"
            note.write_text(f"#aragora-improve #{tag}\n\nTest.\n")

        signals = collect_obsidian_signals(vault_path=str(tmp_path))
        detected = {s.category for s in signals}
        assert detected == set(categories.values())


# ---------------------------------------------------------------------------
# SignalAggregator
# ---------------------------------------------------------------------------


class TestSignalAggregator:
    @pytest.mark.asyncio
    async def test_collect_all_empty(self):
        """All sources disabled produces empty result."""
        aggregator = SignalAggregator()
        result = await aggregator.collect_all(
            include_user_feedback=False,
            include_business_metrics=False,
            include_market=False,
            include_obsidian=False,
        )
        assert result.signals == []
        assert result.source_counts == {}

    @pytest.mark.asyncio
    async def test_collect_all_with_obsidian(self, tmp_path):
        """Obsidian signals flow through the aggregator."""
        note = tmp_path / "improve.md"
        note.write_text("# Speed Up Tests\n\n#aragora-improve\n\nTests are slow.\n")

        aggregator = SignalAggregator(obsidian_vault=str(tmp_path))
        result = await aggregator.collect_all(
            include_user_feedback=False,
            include_business_metrics=False,
            include_market=False,
            include_obsidian=True,
        )

        assert len(result.signals) == 1
        assert result.signals[0].source == "obsidian"
        assert result.source_counts["obsidian"] == 1

    @pytest.mark.asyncio
    async def test_source_weights_applied(self, tmp_path):
        """Source weights adjust signal priorities."""
        note = tmp_path / "test.md"
        note.write_text("#aragora-improve\n\nTest note.\n")

        # High weight for obsidian
        aggregator = SignalAggregator(
            source_weights={"obsidian": 2.0},
            obsidian_vault=str(tmp_path),
        )
        result = await aggregator.collect_all(
            include_user_feedback=False,
            include_business_metrics=False,
            include_market=False,
        )

        assert len(result.signals) == 1
        # Priority should be capped at 1.0
        assert result.signals[0].priority <= 1.0

    @pytest.mark.asyncio
    async def test_signals_sorted_by_priority(self, tmp_path):
        """Signals are returned sorted by priority (highest first)."""
        for name, priority_tag in [
            ("low", "#priority-low"),
            ("high", "#priority-high"),
            ("critical", "#priority-critical"),
        ]:
            note = tmp_path / f"{name}.md"
            note.write_text(f"# {name}\n\n#aragora-improve {priority_tag}\n\nDesc.\n")

        aggregator = SignalAggregator(obsidian_vault=str(tmp_path))
        result = await aggregator.collect_all(
            include_user_feedback=False,
            include_business_metrics=False,
            include_market=False,
        )

        priorities = [s.priority for s in result.signals]
        assert priorities == sorted(priorities, reverse=True)

    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Errors in one source don't block others."""
        aggregator = SignalAggregator()

        # Even with all sources "enabled" but none importable, should not crash
        result = await aggregator.collect_all()
        # Should complete without raising
        assert isinstance(result, AggregatedSignals)

    def test_push_to_queue(self):
        """Signals can be pushed to ImprovementQueue."""
        signals = [
            ExternalSignal(
                source="test",
                title=f"Signal {i}",
                description=f"Desc {i}",
                priority=0.5 + i * 0.1,
                category="general",
            )
            for i in range(5)
        ]

        mock_queue_instance = MagicMock()
        mock_queue_module = MagicMock()
        mock_queue_module.ImprovementQueue.return_value = mock_queue_instance

        with patch.dict(
            "sys.modules",
            {"aragora.nomic.improvement_queue": mock_queue_module},
        ):
            aggregator = SignalAggregator()
            count = aggregator.push_to_queue(signals)
            assert count == 5
            assert mock_queue_instance.enqueue.call_count == 5

    def test_push_to_queue_with_limit(self):
        """Push respects the limit parameter."""
        signals = [
            ExternalSignal(
                source="test",
                title=f"Signal {i}",
                description=f"Desc {i}",
                priority=0.5,
                category="general",
            )
            for i in range(10)
        ]

        mock_queue_instance = MagicMock()
        mock_queue_module = MagicMock()
        mock_queue_module.ImprovementQueue.return_value = mock_queue_instance

        with patch.dict(
            "sys.modules",
            {"aragora.nomic.improvement_queue": mock_queue_module},
        ):
            aggregator = SignalAggregator()
            count = aggregator.push_to_queue(signals, limit=3)
            assert count == 3
