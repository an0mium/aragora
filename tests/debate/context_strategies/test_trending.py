"""Tests for TrendingStrategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.context_strategies.trending import (
    MAX_TRENDING_CACHE_SIZE,
    TRENDING_TIMEOUT,
    TrendingStrategy,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@dataclass
class _FakeTopic:
    """Minimal stand-in for a trending topic."""

    topic: str
    platform: str
    volume: int
    category: str


def _make_topics(n: int = 3) -> list[_FakeTopic]:
    return [
        _FakeTopic(topic=f"Topic {i}", platform="hackernews", volume=1000 * i, category="tech")
        for i in range(1, n + 1)
    ]


class _FakePulseManager:
    """Minimal stand-in for PulseManager."""

    def __init__(self, topics: list[_FakeTopic] | None = None) -> None:
        self._topics = topics if topics is not None else _make_topics()
        self._ingestors: dict[str, Any] = {}

    def add_ingestor(self, name: str, ingestor: Any) -> None:
        self._ingestors[name] = ingestor

    async def get_trending_topics(self, limit_per_platform: int = 3) -> list[_FakeTopic]:
        return self._topics


# ---------------------------------------------------------------------------
# Attribute tests
# ---------------------------------------------------------------------------


class TestTrendingStrategyAttributes:
    """Test class attributes and init."""

    def test_name(self) -> None:
        s = TrendingStrategy()
        assert s.name == "trending"

    def test_default_timeout(self) -> None:
        s = TrendingStrategy()
        assert s.default_timeout == TRENDING_TIMEOUT

    def test_timeout_value(self) -> None:
        assert TRENDING_TIMEOUT == 5.0

    def test_cache_size_value(self) -> None:
        assert MAX_TRENDING_CACHE_SIZE == 50

    def test_default_init(self) -> None:
        s = TrendingStrategy()
        assert s._prompt_builder is None
        assert s._enabled is True
        assert s._topics_cache == []

    def test_custom_init(self) -> None:
        pb = MagicMock()
        s = TrendingStrategy(prompt_builder=pb, enabled=False)
        assert s._prompt_builder is pb
        assert s._enabled is False


class TestTrendingStrategyHelpers:
    """Test helper methods."""

    def test_set_prompt_builder(self) -> None:
        s = TrendingStrategy()
        pb = MagicMock()
        s.set_prompt_builder(pb)
        assert s._prompt_builder is pb

    def test_get_topics_empty(self) -> None:
        s = TrendingStrategy()
        assert s.get_topics() == []

    def test_get_topics_after_cache(self) -> None:
        s = TrendingStrategy()
        topics = _make_topics(2)
        s._topics_cache = topics
        assert s.get_topics() == topics


# ---------------------------------------------------------------------------
# is_available tests
# ---------------------------------------------------------------------------


class TestTrendingIsAvailable:
    """Test is_available."""

    def test_unavailable_when_disabled(self) -> None:
        s = TrendingStrategy(enabled=False)
        assert s.is_available() is False

    def test_available_when_enabled_and_pulse_exists(self) -> None:
        s = TrendingStrategy(enabled=True)
        # Patch to ensure import succeeds
        pulse_mod = MagicMock()
        pulse_mod.PulseManager = MagicMock
        with patch.dict("sys.modules", {"aragora.pulse.ingestor": pulse_mod}):
            assert s.is_available() is True

    def test_unavailable_when_pulse_missing(self) -> None:
        s = TrendingStrategy(enabled=True)
        with patch(
            "builtins.__import__",
            side_effect=_import_error_for("aragora.pulse.ingestor"),
        ):
            assert s.is_available() is False


# ---------------------------------------------------------------------------
# gather tests
# ---------------------------------------------------------------------------


class TestTrendingStrategyGather:
    """Test gather method."""

    @pytest.mark.asyncio
    async def test_gather_disabled_returns_none(self) -> None:
        s = TrendingStrategy(enabled=False)
        result = await s.gather("task")
        assert result is None

    @pytest.mark.asyncio
    async def test_gather_success(self) -> None:
        s = TrendingStrategy()
        topics = _make_topics(3)
        manager = _FakePulseManager(topics=topics)

        with _patch_pulse_imports(manager):
            result = await s.gather("task")

        assert result is not None
        assert "TRENDING CONTEXT" in result
        assert "Topic 1" in result
        assert "Topic 2" in result
        assert "Topic 3" in result

    @pytest.mark.asyncio
    async def test_gather_formats_topics_correctly(self) -> None:
        s = TrendingStrategy()
        topics = [
            _FakeTopic(topic="AI Ethics", platform="reddit", volume=5000, category="tech"),
        ]
        manager = _FakePulseManager(topics=topics)

        with _patch_pulse_imports(manager):
            result = await s.gather("task")

        assert result is not None
        assert "AI Ethics" in result
        assert "reddit" in result
        assert "5,000" in result
        assert "tech" in result

    @pytest.mark.asyncio
    async def test_gather_limits_to_5_topics(self) -> None:
        s = TrendingStrategy()
        topics = _make_topics(10)
        manager = _FakePulseManager(topics=topics)

        with _patch_pulse_imports(manager):
            result = await s.gather("task")

        assert result is not None
        # Should contain topics 1-5 but not 6-10
        assert "Topic 5" in result
        assert "Topic 6" not in result

    @pytest.mark.asyncio
    async def test_gather_caches_topics(self) -> None:
        s = TrendingStrategy()
        topics = _make_topics(3)
        manager = _FakePulseManager(topics=topics)

        with _patch_pulse_imports(manager):
            await s.gather("task")

        cached = s.get_topics()
        assert len(cached) == 3
        assert cached[0].topic == "Topic 1"

    @pytest.mark.asyncio
    async def test_gather_cache_respects_max_size(self) -> None:
        s = TrendingStrategy()
        topics = _make_topics(100)
        manager = _FakePulseManager(topics=topics)

        with _patch_pulse_imports(manager):
            await s.gather("task")

        assert len(s.get_topics()) == MAX_TRENDING_CACHE_SIZE

    @pytest.mark.asyncio
    async def test_gather_updates_prompt_builder(self) -> None:
        pb = MagicMock()
        s = TrendingStrategy(prompt_builder=pb)
        topics = _make_topics(2)
        manager = _FakePulseManager(topics=topics)

        with _patch_pulse_imports(manager):
            await s.gather("task")

        pb.set_trending_topics.assert_called_once()
        call_args = pb.set_trending_topics.call_args[0][0]
        assert len(call_args) == 2

    @pytest.mark.asyncio
    async def test_gather_empty_topics_returns_none(self) -> None:
        s = TrendingStrategy()
        manager = _FakePulseManager(topics=[])

        with _patch_pulse_imports(manager):
            result = await s.gather("task")

        assert result is None

    @pytest.mark.asyncio
    async def test_gather_handles_import_error(self) -> None:
        s = TrendingStrategy()
        with patch(
            "builtins.__import__",
            side_effect=_import_error_for("aragora.pulse.ingestor"),
        ):
            result = await s.gather("task")
        assert result is None

    @pytest.mark.asyncio
    async def test_gather_handles_connection_error(self) -> None:
        s = TrendingStrategy()
        manager = MagicMock()
        manager.add_ingestor = MagicMock()
        manager.get_trending_topics = AsyncMock(side_effect=ConnectionError("down"))

        with _patch_pulse_imports(manager):
            result = await s.gather("task")
        assert result is None

    @pytest.mark.asyncio
    async def test_gather_handles_value_error(self) -> None:
        s = TrendingStrategy()
        manager = MagicMock()
        manager.add_ingestor = MagicMock()
        manager.get_trending_topics = AsyncMock(side_effect=ValueError("bad"))

        with _patch_pulse_imports(manager):
            result = await s.gather("task")
        assert result is None

    @pytest.mark.asyncio
    async def test_gather_handles_unexpected_error(self) -> None:
        s = TrendingStrategy()
        manager = MagicMock()
        manager.add_ingestor = MagicMock()
        manager.get_trending_topics = AsyncMock(side_effect=Exception("weird"))

        with _patch_pulse_imports(manager):
            result = await s.gather("task")
        assert result is None

    @pytest.mark.asyncio
    async def test_gather_with_timeout_integration(self) -> None:
        s = TrendingStrategy()
        topics = _make_topics(2)
        manager = _FakePulseManager(topics=topics)

        with _patch_pulse_imports(manager):
            result = await s.gather_with_timeout("task", timeout=5.0)
        assert result is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_real_import = __import__


def _import_error_for(module_name: str):
    """Side effect that raises ImportError for a specific module."""

    def _side_effect(name, *args, **kwargs):
        if name == module_name:
            raise ImportError(f"No module named '{module_name}'")
        return _real_import(name, *args, **kwargs)

    return _side_effect


class _patch_pulse_imports:
    """Context manager to patch pulse-related imports."""

    def __init__(self, manager: Any) -> None:
        self._manager = manager
        self._patches: list[Any] = []

    def __enter__(self) -> "_patch_pulse_imports":
        pulse_mod = MagicMock()
        pulse_mod.PulseManager = MagicMock(return_value=self._manager)
        pulse_mod.GoogleTrendsIngestor = MagicMock
        pulse_mod.HackerNewsIngestor = MagicMock
        pulse_mod.RedditIngestor = MagicMock
        pulse_mod.GitHubTrendingIngestor = MagicMock

        p = patch.dict("sys.modules", {"aragora.pulse.ingestor": pulse_mod})
        p.start()
        self._patches.append(p)
        return self

    def __exit__(self, *args: Any) -> None:
        for p in reversed(self._patches):
            p.stop()
