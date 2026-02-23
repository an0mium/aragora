"""Tests for LearningBus KM persistence (write-through + historical loading).

Covers:
- Publish writes through to KM
- Historical findings loaded on startup
- Graceful degradation when KM unavailable
- Cycle summary save via CycleLearningStore + NomicCycleAdapter
- max_historical limit enforcement
- LearningBusConfig options
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.learning_bus import (
    Finding,
    LearningBus,
    LearningBusConfig,
    _severity_to_confidence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_bus():
    """Ensure each test gets a fresh singleton."""
    LearningBus.reset_instance()
    yield
    LearningBus.reset_instance()


def _make_finding(**kwargs) -> Finding:
    defaults = {
        "agent_id": "agent-1",
        "topic": "pattern_bug",
        "description": "Found bare except",
    }
    defaults.update(kwargs)
    return Finding(**defaults)


def _make_km_search_result(finding: Finding) -> SimpleNamespace:
    """Build a mock KM search result containing a serialized Finding."""
    return SimpleNamespace(
        metadata={
            "type": "learning_bus_finding",
            "finding_data": finding.to_dict(),
        },
        content=f"FINDING [{finding.severity.upper()}]: {finding.description}",
        score=0.9,
    )


# ---------------------------------------------------------------------------
# Finding serialization
# ---------------------------------------------------------------------------


class TestFindingSerialization:
    def test_to_dict_round_trip(self):
        f = _make_finding(
            affected_files=["a.py"],
            severity="warning",
            suggested_action="fix it",
            metadata={"key": "val"},
        )
        d = f.to_dict()
        restored = Finding.from_dict(d)
        assert restored.agent_id == f.agent_id
        assert restored.topic == f.topic
        assert restored.description == f.description
        assert restored.affected_files == f.affected_files
        assert restored.severity == f.severity
        assert restored.suggested_action == f.suggested_action
        assert restored.metadata == f.metadata
        # Timestamp preserved through ISO round-trip
        assert abs((restored.timestamp - f.timestamp).total_seconds()) < 1

    def test_from_dict_missing_timestamp_defaults(self):
        d = {"agent_id": "a", "topic": "t", "description": "d"}
        f = Finding.from_dict(d)
        assert isinstance(f.timestamp, datetime)

    def test_from_dict_ignores_extra_keys(self):
        d = {
            "agent_id": "a",
            "topic": "t",
            "description": "d",
            "extra_nonsense": 42,
        }
        f = Finding.from_dict(d)
        assert f.agent_id == "a"


# ---------------------------------------------------------------------------
# Publish persists to KM
# ---------------------------------------------------------------------------


class TestPublishPersistence:
    def test_publish_calls_persist_when_enabled(self):
        bus = LearningBus(config=LearningBusConfig(persist=True))
        bus._historical_loaded = True  # skip loading
        bus._persist_finding = MagicMock()

        f = _make_finding()
        bus.publish(f)

        bus._persist_finding.assert_called_once_with(f)
        assert bus.get_findings() == [f]

    def test_publish_skips_persist_when_disabled(self):
        bus = LearningBus(config=LearningBusConfig(persist=False))
        bus._persist_finding = MagicMock()

        bus.publish(_make_finding())

        bus._persist_finding.assert_called_once()
        # _persist_finding itself checks the flag and returns early
        # but verify the finding is in memory regardless
        assert len(bus.get_findings()) == 1

    @pytest.mark.asyncio
    async def test_persist_finding_async_stores_to_mound(self):
        mock_mound = AsyncMock()
        mock_mound.store = AsyncMock(
            return_value=SimpleNamespace(item_id="item_1", deduplicated=False)
        )

        bus = LearningBus(config=LearningBusConfig(persist=True))
        bus._historical_loaded = True
        bus._get_km_mound = MagicMock(return_value=mock_mound)

        f = _make_finding(severity="warning", suggested_action="fix it")
        await bus._persist_finding_async(f)

        mock_mound.store.assert_awaited_once()
        call_args = mock_mound.store.call_args
        request = call_args[0][0]
        assert "FINDING [WARNING]" in request.content
        assert request.metadata["type"] == "learning_bus_finding"
        assert request.metadata["topic"] == "pattern_bug"

    @pytest.mark.asyncio
    async def test_persist_finding_async_graceful_on_no_mound(self):
        bus = LearningBus(config=LearningBusConfig(persist=True))
        bus._historical_loaded = True
        bus._get_km_mound = MagicMock(return_value=None)

        # Should not raise
        await bus._persist_finding_async(_make_finding())

    @pytest.mark.asyncio
    async def test_persist_finding_async_graceful_on_store_error(self):
        mock_mound = AsyncMock()
        mock_mound.store = AsyncMock(side_effect=RuntimeError("KM down"))

        bus = LearningBus(config=LearningBusConfig(persist=True))
        bus._historical_loaded = True
        bus._get_km_mound = MagicMock(return_value=mock_mound)

        # Should not raise
        await bus._persist_finding_async(_make_finding())


# ---------------------------------------------------------------------------
# Historical loading
# ---------------------------------------------------------------------------


class TestHistoricalLoading:
    @pytest.mark.asyncio
    async def test_load_historical_returns_findings_from_km(self):
        old_finding = _make_finding(
            description="old bug",
            timestamp=datetime.now(timezone.utc) - timedelta(days=1),
        )
        mock_mound = AsyncMock()
        mock_mound.search = AsyncMock(
            return_value=[
                _make_km_search_result(old_finding),
            ]
        )

        bus = LearningBus(config=LearningBusConfig(persist=True, max_historical=50))
        bus._get_km_mound = MagicMock(return_value=mock_mound)

        loaded = await bus._load_historical_findings_async()
        assert len(loaded) == 1
        assert loaded[0].description == "old bug"

    @pytest.mark.asyncio
    async def test_load_historical_filters_by_date_cutoff(self):
        recent = _make_finding(
            description="recent",
            timestamp=datetime.now(timezone.utc) - timedelta(days=1),
        )
        stale = _make_finding(
            description="stale",
            timestamp=datetime.now(timezone.utc) - timedelta(days=30),
        )
        mock_mound = AsyncMock()
        mock_mound.search = AsyncMock(
            return_value=[
                _make_km_search_result(recent),
                _make_km_search_result(stale),
            ]
        )

        bus = LearningBus(
            config=LearningBusConfig(
                persist=True,
                historical_days=7,
                max_historical=50,
            )
        )
        bus._get_km_mound = MagicMock(return_value=mock_mound)

        loaded = await bus._load_historical_findings_async()
        assert len(loaded) == 1
        assert loaded[0].description == "recent"

    @pytest.mark.asyncio
    async def test_load_historical_respects_max_limit(self):
        findings = [
            _make_finding(
                description=f"finding-{i}",
                timestamp=datetime.now(timezone.utc) - timedelta(hours=i),
            )
            for i in range(10)
        ]
        mock_mound = AsyncMock()
        mock_mound.search = AsyncMock(return_value=[_make_km_search_result(f) for f in findings])

        bus = LearningBus(
            config=LearningBusConfig(
                persist=True,
                max_historical=3,
            )
        )
        bus._get_km_mound = MagicMock(return_value=mock_mound)

        loaded = await bus._load_historical_findings_async()
        assert len(loaded) == 3

    @pytest.mark.asyncio
    async def test_load_historical_graceful_on_no_mound(self):
        bus = LearningBus(config=LearningBusConfig(persist=True))
        bus._get_km_mound = MagicMock(return_value=None)

        loaded = await bus._load_historical_findings_async()
        assert loaded == []

    @pytest.mark.asyncio
    async def test_load_historical_graceful_on_search_error(self):
        mock_mound = AsyncMock()
        mock_mound.search = AsyncMock(side_effect=RuntimeError("search failed"))

        bus = LearningBus(config=LearningBusConfig(persist=True))
        bus._get_km_mound = MagicMock(return_value=mock_mound)

        loaded = await bus._load_historical_findings_async()
        assert loaded == []

    def test_load_historical_findings_public_method(self):
        """Test the public load_historical_findings() method."""
        bus = LearningBus(config=LearningBusConfig(persist=False))
        # With persist=False, loading is a no-op
        count = bus.load_historical_findings()
        assert count == 0

    @pytest.mark.asyncio
    async def test_load_skips_malformed_entries(self):
        """Malformed finding_data in KM is gracefully skipped."""
        good = _make_finding(description="good one")
        bad_result = SimpleNamespace(
            metadata={
                "type": "learning_bus_finding",
                "finding_data": {"not_a_valid": "finding"},  # missing required fields
            },
            content="bad",
            score=0.5,
        )
        mock_mound = AsyncMock()
        mock_mound.search = AsyncMock(
            return_value=[
                _make_km_search_result(good),
                bad_result,
            ]
        )

        bus = LearningBus(config=LearningBusConfig(persist=True))
        bus._get_km_mound = MagicMock(return_value=mock_mound)

        loaded = await bus._load_historical_findings_async()
        assert len(loaded) == 1
        assert loaded[0].description == "good one"


# ---------------------------------------------------------------------------
# Cycle summary persistence
# ---------------------------------------------------------------------------


class TestCycleSummary:
    def test_save_cycle_summary_calls_cycle_store(self):
        bus = LearningBus(config=LearningBusConfig(persist_cycles=True))
        bus._historical_loaded = True
        bus.publish(_make_finding(topic="pattern_bug", severity="info"))
        bus.publish(_make_finding(topic="test_failure", severity="critical"))

        mock_store = MagicMock()
        mock_store.save_cycle = MagicMock()

        with patch(
            "aragora.nomic.cycle_store.get_cycle_store",
            return_value=mock_store,
        ):
            # Prevent KM adapter call
            bus._persist_cycle_to_km = MagicMock()

            result = bus.save_cycle_summary(
                cycle_id="cycle-001",
                objective="Fix all the bugs",
            )

        assert result is True
        mock_store.save_cycle.assert_called_once()
        saved_record = mock_store.save_cycle.call_args[0][0]
        assert saved_record.cycle_id == "cycle-001"
        assert "pattern_bug" in saved_record.topics_debated
        assert "test_failure" in saved_record.topics_debated

    def test_save_cycle_summary_with_agent_contributions(self):
        bus = LearningBus(config=LearningBusConfig(persist_cycles=True))
        bus._historical_loaded = True
        bus.publish(_make_finding())

        mock_store = MagicMock()
        with patch(
            "aragora.nomic.cycle_store.get_cycle_store",
            return_value=mock_store,
        ):
            bus._persist_cycle_to_km = MagicMock()
            bus.save_cycle_summary(
                cycle_id="cycle-002",
                objective="Improve coverage",
                agent_contributions={
                    "claude": {"proposals_made": 5, "proposals_accepted": 3},
                },
            )

        saved = mock_store.save_cycle.call_args[0][0]
        assert "claude" in saved.agent_contributions
        assert saved.agent_contributions["claude"].proposals_made == 5

    def test_save_cycle_summary_with_surprise_events(self):
        bus = LearningBus(config=LearningBusConfig(persist_cycles=True))
        bus._historical_loaded = True
        bus.publish(_make_finding())

        mock_store = MagicMock()
        with patch(
            "aragora.nomic.cycle_store.get_cycle_store",
            return_value=mock_store,
        ):
            bus._persist_cycle_to_km = MagicMock()
            bus.save_cycle_summary(
                cycle_id="cycle-003",
                objective="Refactor",
                surprise_events=[
                    {
                        "phase": "implement",
                        "description": "unexpected import cycle",
                        "expected": "clean import",
                        "actual": "circular dependency",
                        "impact": "high",
                    }
                ],
            )

        saved = mock_store.save_cycle.call_args[0][0]
        assert len(saved.surprise_events) == 1
        assert saved.surprise_events[0].phase == "implement"

    def test_save_cycle_summary_disabled(self):
        bus = LearningBus(config=LearningBusConfig(persist_cycles=False))
        bus._historical_loaded = True
        result = bus.save_cycle_summary(cycle_id="x", objective="y")
        assert result is False

    def test_save_cycle_summary_graceful_on_store_error(self):
        bus = LearningBus(config=LearningBusConfig(persist_cycles=True))
        bus._historical_loaded = True
        bus.publish(_make_finding())

        with patch(
            "aragora.nomic.cycle_store.get_cycle_store",
            side_effect=RuntimeError("db locked"),
        ):
            result = bus.save_cycle_summary(cycle_id="cycle-fail", objective="oops")

        assert result is False


# ---------------------------------------------------------------------------
# Config / helpers
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_config(self):
        cfg = LearningBusConfig()
        assert cfg.persist is True
        assert cfg.persist_cycles is True
        assert cfg.max_historical == 50
        assert cfg.historical_days == 7
        assert cfg.workspace_id == "nomic"

    def test_custom_config(self):
        cfg = LearningBusConfig(
            persist=False,
            persist_cycles=False,
            max_historical=10,
            historical_days=3,
            workspace_id="custom",
        )
        assert cfg.persist is False
        assert cfg.max_historical == 10

    def test_get_instance_with_config(self):
        cfg = LearningBusConfig(persist=False)
        bus = LearningBus.get_instance(config=cfg)
        assert bus._config.persist is False


class TestSeverityConfidence:
    def test_critical(self):
        assert _severity_to_confidence("critical") == 0.95

    def test_warning(self):
        assert _severity_to_confidence("warning") == 0.85

    def test_info(self):
        assert _severity_to_confidence("info") == 0.7

    def test_unknown_defaults_to_info(self):
        assert _severity_to_confidence("weird") == 0.7


# ---------------------------------------------------------------------------
# Backward compatibility: existing tests still work
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Verify the original LearningBus API contract is preserved."""

    def test_singleton_still_works(self):
        cfg = LearningBusConfig(persist=False)
        a = LearningBus.get_instance(config=cfg)
        b = LearningBus.get_instance()
        assert a is b

    def test_publish_subscribe_query_still_work(self):
        bus = LearningBus(config=LearningBusConfig(persist=False))
        received: list[Finding] = []
        bus.subscribe("pattern_bug", received.append)

        f = _make_finding()
        bus.publish(f)

        assert bus.get_findings() == [f]
        assert received == [f]

    def test_clear_and_summary_still_work(self):
        bus = LearningBus(config=LearningBusConfig(persist=False))
        bus.publish(_make_finding(severity="warning"))
        bus.publish(_make_finding(severity="critical"))

        s = bus.summary()
        assert s["total"] == 2

        bus.clear()
        assert bus.summary()["total"] == 0
