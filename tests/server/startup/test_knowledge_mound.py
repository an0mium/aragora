"""
Tests for aragora.server.startup.knowledge_mound module.

Tests TTS integration, Knowledge Mound configuration,
initialization, and adapter setup.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_mock_backend(value: str) -> MagicMock:
    """Create a mock enum value with .value attribute."""
    mock = MagicMock()
    mock.value = value
    return mock


# =============================================================================
# init_tts_integration Tests
# =============================================================================


class TestInitTTSIntegration:
    """Tests for init_tts_integration function."""

    @pytest.mark.asyncio
    async def test_tts_available(self) -> None:
        """Test TTS integration when voice synthesis is available."""
        mock_event_bus = MagicMock()
        mock_integration = MagicMock()
        mock_integration.is_available = True

        mock_event_module = MagicMock()
        mock_event_module.get_event_bus = MagicMock(return_value=mock_event_bus)

        mock_tts_module = MagicMock()
        mock_tts_module.init_tts_integration = MagicMock(return_value=mock_integration)

        with patch.dict(
            "sys.modules",
            {
                "aragora.debate.event_bus": mock_event_module,
                "aragora.server.stream.tts_integration": mock_tts_module,
            },
        ):
            from aragora.server.startup.knowledge_mound import init_tts_integration

            result = await init_tts_integration()

        assert result is True
        mock_tts_module.init_tts_integration.assert_called_once_with(event_bus=mock_event_bus)

    @pytest.mark.asyncio
    async def test_tts_unavailable(self) -> None:
        """Test TTS integration when voice synthesis is unavailable."""
        mock_event_bus = MagicMock()
        mock_integration = MagicMock()
        mock_integration.is_available = False

        mock_event_module = MagicMock()
        mock_event_module.get_event_bus = MagicMock(return_value=mock_event_bus)

        mock_tts_module = MagicMock()
        mock_tts_module.init_tts_integration = MagicMock(return_value=mock_integration)

        with patch.dict(
            "sys.modules",
            {
                "aragora.debate.event_bus": mock_event_module,
                "aragora.server.stream.tts_integration": mock_tts_module,
            },
        ):
            from aragora.server.startup.knowledge_mound import init_tts_integration

            result = await init_tts_integration()

        assert result is False

    @pytest.mark.asyncio
    async def test_import_error(self) -> None:
        """Test ImportError returns False."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.debate.event_bus": None,
                "aragora.server.stream.tts_integration": None,
            },
        ):
            import importlib
            import aragora.server.startup.knowledge_mound as km_module

            importlib.reload(km_module)
            result = await km_module.init_tts_integration()

        assert result is False

    @pytest.mark.asyncio
    async def test_runtime_error(self) -> None:
        """Test RuntimeError returns False."""
        mock_event_module = MagicMock()
        mock_event_module.get_event_bus = MagicMock(side_effect=RuntimeError("event bus error"))

        with patch.dict(
            "sys.modules",
            {
                "aragora.debate.event_bus": mock_event_module,
                "aragora.server.stream.tts_integration": MagicMock(),
            },
        ):
            from aragora.server.startup.knowledge_mound import init_tts_integration

            result = await init_tts_integration()

        assert result is False


# =============================================================================
# get_km_config_from_env Tests
# =============================================================================


class TestGetKMConfigFromEnv:
    """Tests for get_km_config_from_env function."""

    def test_default_auto_selects_sqlite(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test auto backend defaults to SQLite without postgres URL."""
        monkeypatch.delenv("KM_BACKEND", raising=False)
        monkeypatch.delenv("KM_POSTGRES_URL", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)

        mock_sqlite = _make_mock_backend("sqlite")
        mock_postgres = _make_mock_backend("postgres")

        mock_types = MagicMock()
        mock_types.MoundConfig = MagicMock(return_value=MagicMock())
        mock_types.MoundBackend = MagicMock()
        mock_types.MoundBackend.SQLITE = mock_sqlite
        mock_types.MoundBackend.POSTGRES = mock_postgres

        with patch.dict("sys.modules", {"aragora.knowledge.mound.types": mock_types}):
            from aragora.server.startup.knowledge_mound import get_km_config_from_env

            config = get_km_config_from_env()

        mock_types.MoundConfig.assert_called_once()
        call_kwargs = mock_types.MoundConfig.call_args[1]
        assert call_kwargs["backend"] == mock_sqlite

    def test_auto_selects_postgres_with_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test auto backend selects Postgres when URL provided."""
        monkeypatch.setenv("KM_BACKEND", "auto")
        monkeypatch.setenv("KM_POSTGRES_URL", "postgresql://localhost:5432/test")

        mock_sqlite = _make_mock_backend("sqlite")
        mock_postgres = _make_mock_backend("postgres")

        mock_types = MagicMock()
        mock_types.MoundConfig = MagicMock(return_value=MagicMock())
        mock_types.MoundBackend = MagicMock()
        mock_types.MoundBackend.SQLITE = mock_sqlite
        mock_types.MoundBackend.POSTGRES = mock_postgres

        with patch.dict("sys.modules", {"aragora.knowledge.mound.types": mock_types}):
            from aragora.server.startup.knowledge_mound import get_km_config_from_env

            config = get_km_config_from_env()

        call_kwargs = mock_types.MoundConfig.call_args[1]
        assert call_kwargs["backend"] == mock_postgres

    def test_explicit_postgres_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test explicit postgres backend selection."""
        monkeypatch.setenv("KM_BACKEND", "postgres")
        monkeypatch.setenv("KM_POSTGRES_URL", "postgresql://localhost:5432/test")
        monkeypatch.setenv("KM_POSTGRES_POOL_SIZE", "20")

        mock_sqlite = _make_mock_backend("sqlite")
        mock_postgres = _make_mock_backend("postgres")

        mock_types = MagicMock()
        mock_types.MoundConfig = MagicMock(return_value=MagicMock())
        mock_types.MoundBackend = MagicMock()
        mock_types.MoundBackend.SQLITE = mock_sqlite
        mock_types.MoundBackend.POSTGRES = mock_postgres

        with patch.dict("sys.modules", {"aragora.knowledge.mound.types": mock_types}):
            from aragora.server.startup.knowledge_mound import get_km_config_from_env

            config = get_km_config_from_env()

        call_kwargs = mock_types.MoundConfig.call_args[1]
        assert call_kwargs["backend"] == mock_postgres
        assert call_kwargs["postgres_pool_size"] == 20

    def test_unknown_backend_defaults_sqlite(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test unknown backend falls back to SQLite."""
        monkeypatch.setenv("KM_BACKEND", "unknown_db")

        mock_sqlite = _make_mock_backend("sqlite")
        mock_postgres = _make_mock_backend("postgres")

        mock_types = MagicMock()
        mock_types.MoundConfig = MagicMock(return_value=MagicMock())
        mock_types.MoundBackend = MagicMock()
        mock_types.MoundBackend.SQLITE = mock_sqlite
        mock_types.MoundBackend.POSTGRES = mock_postgres

        with patch.dict("sys.modules", {"aragora.knowledge.mound.types": mock_types}):
            from aragora.server.startup.knowledge_mound import get_km_config_from_env

            config = get_km_config_from_env()

        call_kwargs = mock_types.MoundConfig.call_args[1]
        assert call_kwargs["backend"] == mock_sqlite

    def test_feature_flags_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test feature flags parsed from environment."""
        monkeypatch.setenv("KM_BACKEND", "sqlite")
        monkeypatch.setenv("KM_ENABLE_STALENESS", "false")
        monkeypatch.setenv("KM_ENABLE_CULTURE", "true")
        monkeypatch.setenv("KM_ENABLE_DEDUP", "0")
        monkeypatch.setenv("KM_ENABLE_COST_ADAPTER", "1")

        mock_sqlite = _make_mock_backend("sqlite")

        mock_types = MagicMock()
        mock_types.MoundConfig = MagicMock(return_value=MagicMock())
        mock_types.MoundBackend = MagicMock()
        mock_types.MoundBackend.SQLITE = mock_sqlite

        with patch.dict("sys.modules", {"aragora.knowledge.mound.types": mock_types}):
            from aragora.server.startup.knowledge_mound import get_km_config_from_env

            config = get_km_config_from_env()

        call_kwargs = mock_types.MoundConfig.call_args[1]
        assert call_kwargs["enable_staleness_detection"] is False
        assert call_kwargs["enable_culture_accumulator"] is True
        assert call_kwargs["enable_deduplication"] is False
        assert call_kwargs["enable_cost_adapter"] is True

    def test_confidence_thresholds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test confidence thresholds parsed from environment."""
        monkeypatch.setenv("KM_BACKEND", "sqlite")
        monkeypatch.setenv("KM_EVIDENCE_MIN_RELIABILITY", "0.8")
        monkeypatch.setenv("KM_PULSE_MIN_QUALITY", "0.5")
        monkeypatch.setenv("KM_INSIGHT_MIN_CONFIDENCE", "0.9")
        monkeypatch.setenv("KM_BELIEF_MIN_CONFIDENCE", "0.95")

        mock_sqlite = _make_mock_backend("sqlite")

        mock_types = MagicMock()
        mock_types.MoundConfig = MagicMock(return_value=MagicMock())
        mock_types.MoundBackend = MagicMock()
        mock_types.MoundBackend.SQLITE = mock_sqlite

        with patch.dict("sys.modules", {"aragora.knowledge.mound.types": mock_types}):
            from aragora.server.startup.knowledge_mound import get_km_config_from_env

            config = get_km_config_from_env()

        call_kwargs = mock_types.MoundConfig.call_args[1]
        assert call_kwargs["evidence_min_reliability"] == 0.8
        assert call_kwargs["pulse_min_quality"] == 0.5
        assert call_kwargs["insight_min_confidence"] == 0.9
        assert call_kwargs["belief_min_confidence"] == 0.95

    def test_redis_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Redis configuration from environment."""
        monkeypatch.setenv("KM_BACKEND", "sqlite")
        monkeypatch.setenv("KM_REDIS_URL", "redis://localhost:6379/0")
        monkeypatch.setenv("KM_REDIS_CACHE_TTL", "600")
        monkeypatch.setenv("KM_REDIS_CULTURE_TTL", "7200")

        mock_sqlite = _make_mock_backend("sqlite")

        mock_types = MagicMock()
        mock_types.MoundConfig = MagicMock(return_value=MagicMock())
        mock_types.MoundBackend = MagicMock()
        mock_types.MoundBackend.SQLITE = mock_sqlite

        with patch.dict("sys.modules", {"aragora.knowledge.mound.types": mock_types}):
            from aragora.server.startup.knowledge_mound import get_km_config_from_env

            config = get_km_config_from_env()

        call_kwargs = mock_types.MoundConfig.call_args[1]
        assert call_kwargs["redis_url"] == "redis://localhost:6379/0"
        assert call_kwargs["redis_cache_ttl"] == 600
        assert call_kwargs["redis_culture_ttl"] == 7200

    def test_weaviate_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Weaviate configuration from environment."""
        monkeypatch.setenv("KM_BACKEND", "sqlite")
        monkeypatch.setenv("KM_WEAVIATE_URL", "http://localhost:8080")
        monkeypatch.setenv("KM_WEAVIATE_API_KEY", "test-api-key")
        monkeypatch.setenv("KM_WEAVIATE_COLLECTION", "TestCollection")

        mock_sqlite = _make_mock_backend("sqlite")

        mock_types = MagicMock()
        mock_types.MoundConfig = MagicMock(return_value=MagicMock())
        mock_types.MoundBackend = MagicMock()
        mock_types.MoundBackend.SQLITE = mock_sqlite

        with patch.dict("sys.modules", {"aragora.knowledge.mound.types": mock_types}):
            from aragora.server.startup.knowledge_mound import get_km_config_from_env

            config = get_km_config_from_env()

        call_kwargs = mock_types.MoundConfig.call_args[1]
        assert call_kwargs["weaviate_url"] == "http://localhost:8080"
        assert call_kwargs["weaviate_api_key"] == "test-api-key"
        assert call_kwargs["weaviate_collection"] == "TestCollection"


# =============================================================================
# init_knowledge_mound_from_env Tests
# =============================================================================


class TestInitKnowledgeMoundFromEnv:
    """Tests for init_knowledge_mound_from_env function."""

    @pytest.mark.asyncio
    async def test_successful_initialization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test successful KM initialization from environment."""
        monkeypatch.setenv("KM_BACKEND", "sqlite")

        mock_mound = MagicMock()
        mock_mound.initialize = AsyncMock()

        mock_sqlite = _make_mock_backend("sqlite")
        mock_config = MagicMock()
        mock_config.backend = mock_sqlite

        mock_km_module = MagicMock()
        mock_km_module.set_mound_config = MagicMock()
        mock_km_module.get_knowledge_mound = MagicMock(return_value=mock_mound)

        mock_types = MagicMock()
        mock_types.MoundConfig = MagicMock(return_value=mock_config)
        mock_types.MoundBackend = MagicMock()
        mock_types.MoundBackend.SQLITE = mock_sqlite

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": mock_km_module,
                "aragora.knowledge.mound.types": mock_types,
            },
        ):
            from aragora.server.startup.knowledge_mound import (
                init_knowledge_mound_from_env,
            )

            result = await init_knowledge_mound_from_env()

        assert result is True
        mock_km_module.set_mound_config.assert_called_once()
        mock_mound.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_import_error(self) -> None:
        """Test ImportError returns False."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": None,
                "aragora.knowledge.mound.types": None,
            },
        ):
            import importlib
            import aragora.server.startup.knowledge_mound as km_module

            importlib.reload(km_module)
            result = await km_module.init_knowledge_mound_from_env()

        assert result is False

    @pytest.mark.asyncio
    async def test_connection_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test ConnectionError returns False."""
        monkeypatch.setenv("KM_BACKEND", "postgres")
        monkeypatch.setenv("KM_POSTGRES_URL", "postgresql://localhost:5432/test")

        mock_mound = MagicMock()
        mock_mound.initialize = AsyncMock(side_effect=ConnectionError("database connection failed"))

        mock_postgres = _make_mock_backend("postgres")
        mock_config = MagicMock()
        mock_config.backend = mock_postgres

        mock_km_module = MagicMock()
        mock_km_module.set_mound_config = MagicMock()
        mock_km_module.get_knowledge_mound = MagicMock(return_value=mock_mound)

        mock_types = MagicMock()
        mock_types.MoundConfig = MagicMock(return_value=mock_config)
        mock_types.MoundBackend = MagicMock()
        mock_types.MoundBackend.POSTGRES = mock_postgres

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": mock_km_module,
                "aragora.knowledge.mound.types": mock_types,
            },
        ):
            from aragora.server.startup.knowledge_mound import (
                init_knowledge_mound_from_env,
            )

            result = await init_knowledge_mound_from_env()

        assert result is False


# =============================================================================
# init_km_adapters Tests
# =============================================================================


class TestInitKMAdapters:
    """Tests for init_km_adapters function."""

    @pytest.mark.asyncio
    async def test_successful_initialization(self) -> None:
        """Test successful KM adapters initialization."""
        mock_manager = MagicMock()
        mock_cross_subs = MagicMock()
        mock_cross_subs.get_cross_subscriber_manager = MagicMock(return_value=mock_manager)

        mock_ranking_adapter = MagicMock()
        mock_rlm_adapter = MagicMock()

        mock_adapters = MagicMock()
        mock_adapters.RankingAdapter = MagicMock(return_value=mock_ranking_adapter)
        mock_adapters.RlmAdapter = MagicMock(return_value=mock_rlm_adapter)

        mock_metrics = MagicMock()
        mock_metrics.KMMetrics = MagicMock
        mock_metrics.set_metrics = MagicMock()

        mock_bridge = MagicMock()
        mock_bridge.create_km_bridge = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.events.cross_subscribers": mock_cross_subs,
                "aragora.knowledge.mound.adapters": mock_adapters,
                "aragora.knowledge.mound.metrics": mock_metrics,
                "aragora.knowledge.mound.websocket_bridge": mock_bridge,
            },
        ):
            from aragora.server.startup.knowledge_mound import init_km_adapters

            result = await init_km_adapters()

        assert result is True
        mock_adapters.RankingAdapter.assert_called_once()
        mock_adapters.RlmAdapter.assert_called_once()
        mock_metrics.set_metrics.assert_called_once()
        mock_bridge.create_km_bridge.assert_called_once()

    @pytest.mark.asyncio
    async def test_metrics_import_error_continues(self) -> None:
        """Test that metrics ImportError doesn't stop initialization."""
        mock_manager = MagicMock()
        mock_cross_subs = MagicMock()
        mock_cross_subs.get_cross_subscriber_manager = MagicMock(return_value=mock_manager)

        mock_adapters = MagicMock()
        mock_adapters.RankingAdapter = MagicMock()
        mock_adapters.RlmAdapter = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.events.cross_subscribers": mock_cross_subs,
                "aragora.knowledge.mound.adapters": mock_adapters,
                "aragora.knowledge.mound.metrics": None,  # ImportError
                "aragora.knowledge.mound.websocket_bridge": None,  # ImportError
            },
        ):
            import importlib
            import aragora.server.startup.knowledge_mound as km_module

            importlib.reload(km_module)
            result = await km_module.init_km_adapters()

        assert result is True

    @pytest.mark.asyncio
    async def test_import_error(self) -> None:
        """Test ImportError returns False."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.events.cross_subscribers": None,
                "aragora.knowledge.mound.adapters": None,
            },
        ):
            import importlib
            import aragora.server.startup.knowledge_mound as km_module

            importlib.reload(km_module)
            result = await km_module.init_km_adapters()

        assert result is False

    @pytest.mark.asyncio
    async def test_runtime_error(self) -> None:
        """Test RuntimeError returns False."""
        mock_cross_subs = MagicMock()
        mock_cross_subs.get_cross_subscriber_manager = MagicMock(
            side_effect=RuntimeError("manager error")
        )

        with patch.dict(
            "sys.modules",
            {
                "aragora.events.cross_subscribers": mock_cross_subs,
                "aragora.knowledge.mound.adapters": MagicMock(),
            },
        ):
            from aragora.server.startup.knowledge_mound import init_km_adapters

            result = await init_km_adapters()

        assert result is False

    @pytest.mark.asyncio
    async def test_attribute_error(self) -> None:
        """Test AttributeError returns False."""
        mock_manager = MagicMock()
        mock_cross_subs = MagicMock()
        mock_cross_subs.get_cross_subscriber_manager = MagicMock(return_value=mock_manager)

        mock_adapters = MagicMock()
        mock_adapters.RankingAdapter = MagicMock(side_effect=AttributeError("missing attr"))

        with patch.dict(
            "sys.modules",
            {
                "aragora.events.cross_subscribers": mock_cross_subs,
                "aragora.knowledge.mound.adapters": mock_adapters,
            },
        ):
            from aragora.server.startup.knowledge_mound import init_km_adapters

            result = await init_km_adapters()

        assert result is False
