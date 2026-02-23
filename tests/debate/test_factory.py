"""Tests for aragora.debate.factory module.

Covers ArenaFactory lazy loading, creator methods, the create() method,
and the global convenience functions (get_arena_factory, create_arena).
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_factory_singleton():
    """Reset the module-level _factory singleton between tests.

    Prior tests in a larger suite may leave a stale singleton that
    causes ``get_arena_factory()`` and ``create_arena()`` to return
    unexpected instances.  Also guard against ``importlib.reload()``
    side-effects in ``TestCreateMethod`` corrupting the shared module.
    """
    import aragora.debate.factory as mod

    saved = mod._factory
    mod._factory = None
    yield
    mod._factory = saved


def _fresh_factory():
    """Import ArenaFactory without triggering heavy transitive imports."""
    from aragora.debate.factory import ArenaFactory

    return ArenaFactory()


# =========================================================================
# Lazy-loading getter tests
# =========================================================================


class TestLazyLoadingCaching:
    """Each _get_*_cls() should import once and cache the reference."""

    def test_position_tracker_cls_cached(self):
        factory = _fresh_factory()
        mock_cls = MagicMock()
        with patch.dict(
            "sys.modules",
            {"aragora.agents.truth_grounding": MagicMock(PositionTracker=mock_cls)},
        ):
            first = factory._get_position_tracker_cls()
            second = factory._get_position_tracker_cls()
        assert first is mock_cls
        assert second is mock_cls

    def test_calibration_tracker_cls_cached(self):
        factory = _fresh_factory()
        mock_cls = MagicMock()
        with patch.dict(
            "sys.modules",
            {"aragora.agents.calibration": MagicMock(CalibrationTracker=mock_cls)},
        ):
            first = factory._get_calibration_tracker_cls()
            second = factory._get_calibration_tracker_cls()
        assert first is mock_cls
        assert second is mock_cls

    def test_belief_classes_cached(self):
        factory = _fresh_factory()
        mock_bn = MagicMock()
        mock_ba = MagicMock()
        mod = MagicMock(BeliefNetwork=mock_bn, BeliefPropagationAnalyzer=mock_ba)
        with patch.dict("sys.modules", {"aragora.reasoning.belief": mod}):
            first = factory._get_belief_classes()
            second = factory._get_belief_classes()
        assert first == (mock_bn, mock_ba)
        assert second == (mock_bn, mock_ba)

    def test_citation_extractor_cls_cached(self):
        factory = _fresh_factory()
        mock_cls = MagicMock()
        with patch.dict(
            "sys.modules",
            {"aragora.reasoning.citations": MagicMock(CitationExtractor=mock_cls)},
        ):
            first = factory._get_citation_extractor_cls()
            second = factory._get_citation_extractor_cls()
        assert first is mock_cls
        assert second is mock_cls

    def test_insight_classes_cached(self):
        factory = _fresh_factory()
        mock_ie = MagicMock()
        mock_is = MagicMock()
        mod = MagicMock(InsightExtractor=mock_ie, InsightStore=mock_is)
        with patch.dict("sys.modules", {"aragora.insights": mod}):
            first = factory._get_insight_classes()
            second = factory._get_insight_classes()
        assert first == (mock_ie, mock_is)
        assert second == (mock_ie, mock_is)

    def test_critique_store_cls_cached(self):
        factory = _fresh_factory()
        mock_cls = MagicMock()
        with patch.dict(
            "sys.modules",
            {"aragora.memory.store": MagicMock(CritiqueStore=mock_cls)},
        ):
            first = factory._get_critique_store_cls()
            second = factory._get_critique_store_cls()
        assert first is mock_cls
        assert second is mock_cls

    def test_argument_cartographer_cls_cached(self):
        factory = _fresh_factory()
        mock_cls = MagicMock()
        with patch.dict(
            "sys.modules",
            {"aragora.visualization.mapper": MagicMock(ArgumentCartographer=mock_cls)},
        ):
            first = factory._get_argument_cartographer_cls()
            second = factory._get_argument_cartographer_cls()
        assert first is mock_cls
        assert second is mock_cls


class TestLazyLoadingImportFailure:
    """Import failures should return None gracefully."""

    def test_position_tracker_import_error(self):
        factory = _fresh_factory()
        with patch.dict("sys.modules", {"aragora.agents.truth_grounding": None}):
            result = factory._get_position_tracker_cls()
        assert result is None

    def test_calibration_tracker_import_error(self):
        factory = _fresh_factory()
        with patch.dict("sys.modules", {"aragora.agents.calibration": None}):
            result = factory._get_calibration_tracker_cls()
        assert result is None

    def test_belief_classes_import_error(self):
        factory = _fresh_factory()
        with patch.dict("sys.modules", {"aragora.reasoning.belief": None}):
            result = factory._get_belief_classes()
        assert result == (None, None)

    def test_citation_extractor_import_error(self):
        factory = _fresh_factory()
        with patch.dict("sys.modules", {"aragora.reasoning.citations": None}):
            result = factory._get_citation_extractor_cls()
        assert result is None

    def test_insight_classes_import_error(self):
        factory = _fresh_factory()
        with patch.dict("sys.modules", {"aragora.insights": None}):
            result = factory._get_insight_classes()
        assert result == (None, None)

    def test_critique_store_import_error(self):
        factory = _fresh_factory()
        with patch.dict("sys.modules", {"aragora.memory.store": None}):
            result = factory._get_critique_store_cls()
        assert result is None

    def test_argument_cartographer_import_error(self):
        factory = _fresh_factory()
        with patch.dict("sys.modules", {"aragora.visualization.mapper": None}):
            result = factory._get_argument_cartographer_cls()
        assert result is None


# =========================================================================
# Creator method tests
# =========================================================================


class TestCreatorMethods:
    """Creator methods should instantiate when class is available, else None."""

    def test_create_position_tracker_success(self):
        factory = _fresh_factory()
        mock_cls = MagicMock()
        factory._position_tracker_cls = mock_cls
        result = factory.create_position_tracker(foo="bar")
        mock_cls.assert_called_once_with(foo="bar")
        assert result is mock_cls.return_value

    def test_create_position_tracker_unavailable(self):
        factory = _fresh_factory()
        # _position_tracker_cls stays None; getter also returns None
        with patch.dict("sys.modules", {"aragora.agents.truth_grounding": None}):
            assert factory.create_position_tracker() is None

    def test_create_calibration_tracker_success(self):
        factory = _fresh_factory()
        mock_cls = MagicMock()
        factory._calibration_tracker_cls = mock_cls
        result = factory.create_calibration_tracker(alpha=0.5)
        mock_cls.assert_called_once_with(alpha=0.5)
        assert result is mock_cls.return_value

    def test_create_calibration_tracker_unavailable(self):
        factory = _fresh_factory()
        with patch.dict("sys.modules", {"aragora.agents.calibration": None}):
            assert factory.create_calibration_tracker() is None

    def test_create_belief_network_success(self):
        factory = _fresh_factory()
        mock_bn = MagicMock()
        factory._belief_network_cls = mock_bn
        factory._belief_analyzer_cls = MagicMock()
        result = factory.create_belief_network(depth=3)
        mock_bn.assert_called_once_with(depth=3)
        assert result is mock_bn.return_value

    def test_create_belief_analyzer_success(self):
        factory = _fresh_factory()
        mock_ba = MagicMock()
        factory._belief_network_cls = MagicMock()  # must be set so getter skips import
        factory._belief_analyzer_cls = mock_ba
        result = factory.create_belief_analyzer(iterations=10)
        mock_ba.assert_called_once_with(iterations=10)
        assert result is mock_ba.return_value

    def test_create_belief_network_unavailable(self):
        factory = _fresh_factory()
        with patch.dict("sys.modules", {"aragora.reasoning.belief": None}):
            assert factory.create_belief_network() is None

    def test_create_citation_extractor_success(self):
        factory = _fresh_factory()
        mock_cls = MagicMock()
        factory._citation_extractor_cls = mock_cls
        result = factory.create_citation_extractor(strict=True)
        mock_cls.assert_called_once_with(strict=True)
        assert result is mock_cls.return_value

    def test_create_insight_extractor_success(self):
        factory = _fresh_factory()
        mock_ie = MagicMock()
        factory._insight_extractor_cls = mock_ie
        factory._insight_store_cls = MagicMock()
        result = factory.create_insight_extractor(limit=5)
        mock_ie.assert_called_once_with(limit=5)
        assert result is mock_ie.return_value

    def test_create_insight_store_success(self):
        factory = _fresh_factory()
        mock_is = MagicMock()
        factory._insight_extractor_cls = MagicMock()  # must be set
        factory._insight_store_cls = mock_is
        result = factory.create_insight_store(path="/tmp")
        mock_is.assert_called_once_with(path="/tmp")
        assert result is mock_is.return_value

    def test_create_critique_store_success(self):
        factory = _fresh_factory()
        mock_cls = MagicMock()
        factory._critique_store_cls = mock_cls
        result = factory.create_critique_store(max_size=100)
        mock_cls.assert_called_once_with(max_size=100)
        assert result is mock_cls.return_value

    def test_create_argument_cartographer_success(self):
        factory = _fresh_factory()
        mock_cls = MagicMock()
        factory._argument_cartographer_cls = mock_cls
        result = factory.create_argument_cartographer(layout="radial")
        mock_cls.assert_called_once_with(layout="radial")
        assert result is mock_cls.return_value

    def test_create_argument_cartographer_unavailable(self):
        factory = _fresh_factory()
        with patch.dict("sys.modules", {"aragora.visualization.mapper": None}):
            assert factory.create_argument_cartographer() is None


# =========================================================================
# create() method tests
# =========================================================================


class TestCreateMethod:
    """ArenaFactory.create() wires config objects and delegates to Arena.create()."""

    @pytest.fixture(autouse=True)
    def _patch_arena_imports(self):
        """Patch the heavy imports inside create() so we never load the real Arena.

        Uses sys.modules patching without ``importlib.reload()`` to avoid
        re-executing module-level imports that can fail when prior tests
        in a larger suite have modified transitive dependencies.
        """
        self.mock_arena_cls = MagicMock()
        self.mock_arena_cls.create.return_value = MagicMock(name="arena_instance")

        self.mock_agent_cfg = MagicMock(name="AgentConfig")
        self.mock_memory_cfg = MagicMock(name="MemoryConfig")
        self.mock_streaming_cfg = MagicMock(name="StreamingConfig")
        self.mock_obs_cfg = MagicMock(name="ObservabilityConfig")

        mock_orchestrator = MagicMock(Arena=self.mock_arena_cls)
        mock_arena_config = MagicMock(
            AgentConfig=self.mock_agent_cfg,
            MemoryConfig=self.mock_memory_cfg,
            StreamingConfig=self.mock_streaming_cfg,
            ObservabilityConfig=self.mock_obs_cfg,
        )

        import aragora.debate.factory as factory_mod

        # Save originals from sys.modules so we can restore them
        saved_orchestrator = sys.modules.get("aragora.debate.orchestrator")
        saved_arena_config = sys.modules.get("aragora.debate.arena_config")

        # Inject mocks into sys.modules so the local imports inside
        # create() pick them up without needing a full module reload.
        sys.modules["aragora.debate.orchestrator"] = mock_orchestrator
        sys.modules["aragora.debate.arena_config"] = mock_arena_config

        # Invalidate any cached references in the factory module's import
        # cache so the next ``from aragora.debate.orchestrator import Arena``
        # inside create() re-resolves from sys.modules.
        if hasattr(importlib, "invalidate_caches"):
            importlib.invalidate_caches()

        self.factory_mod = factory_mod
        yield

        # Restore original modules
        if saved_orchestrator is not None:
            sys.modules["aragora.debate.orchestrator"] = saved_orchestrator
        else:
            sys.modules.pop("aragora.debate.orchestrator", None)

        if saved_arena_config is not None:
            sys.modules["aragora.debate.arena_config"] = saved_arena_config
        else:
            sys.modules.pop("aragora.debate.arena_config", None)

    def _make_env_agents(self):
        env = MagicMock(name="environment")
        agents = [MagicMock(name="agent1"), MagicMock(name="agent2")]
        return env, agents

    def test_create_minimal_args(self):
        factory = self.factory_mod.ArenaFactory()
        env, agents = self._make_env_agents()
        result = factory.create(env, agents)
        self.mock_arena_cls.create.assert_called_once()
        call_kw = self.mock_arena_cls.create.call_args[1]
        assert call_kw["environment"] is env
        assert call_kw["agents"] is agents
        assert result is self.mock_arena_cls.create.return_value

    def test_enable_position_tracking_auto_creates(self):
        factory = self.factory_mod.ArenaFactory()
        mock_pt = MagicMock(name="PositionTrackerCls")
        factory._position_tracker_cls = mock_pt
        env, agents = self._make_env_agents()
        factory.create(env, agents, enable_position_tracking=True)
        mock_pt.assert_called_once_with()
        cfg_call = self.mock_agent_cfg.call_args
        assert cfg_call[1]["position_tracker"] is mock_pt.return_value

    def test_enable_calibration_auto_creates(self):
        factory = self.factory_mod.ArenaFactory()
        mock_ct = MagicMock(name="CalibrationTrackerCls")
        factory._calibration_tracker_cls = mock_ct
        env, agents = self._make_env_agents()
        factory.create(env, agents, enable_calibration=True)
        mock_ct.assert_called_once_with()
        cfg_call = self.mock_agent_cfg.call_args
        assert cfg_call[1]["calibration_tracker"] is mock_ct.return_value

    def test_enable_insights_auto_creates_store(self):
        factory = self.factory_mod.ArenaFactory()
        mock_ie = MagicMock(name="InsightExtractorCls")
        mock_is = MagicMock(name="InsightStoreCls")
        factory._insight_extractor_cls = mock_ie
        factory._insight_store_cls = mock_is
        env, agents = self._make_env_agents()
        factory.create(env, agents, enable_insights=True)
        mock_is.assert_called_once_with()
        cfg_call = self.mock_memory_cfg.call_args
        assert cfg_call[1]["insight_store"] is mock_is.return_value

    def test_enable_critique_patterns_auto_creates_memory(self):
        factory = self.factory_mod.ArenaFactory()
        mock_cs = MagicMock(name="CritiqueStoreCls")
        factory._critique_store_cls = mock_cs
        env, agents = self._make_env_agents()
        factory.create(env, agents, enable_critique_patterns=True)
        mock_cs.assert_called_once_with()
        cfg_call = self.mock_memory_cfg.call_args
        assert cfg_call[1]["memory"] is mock_cs.return_value

    def test_explicit_instance_overrides_enable_flag(self):
        factory = self.factory_mod.ArenaFactory()
        mock_pt_cls = MagicMock(name="PositionTrackerCls")
        factory._position_tracker_cls = mock_pt_cls
        env, agents = self._make_env_agents()
        explicit_tracker = MagicMock(name="explicit_tracker")
        factory.create(
            env,
            agents,
            enable_position_tracking=True,
            position_tracker=explicit_tracker,
        )
        # The class constructor should NOT have been called
        mock_pt_cls.assert_not_called()
        cfg_call = self.mock_agent_cfg.call_args
        assert cfg_call[1]["position_tracker"] is explicit_tracker

    def test_config_objects_constructed_correctly(self):
        factory = self.factory_mod.ArenaFactory()
        env, agents = self._make_env_agents()
        spectator = MagicMock(name="spectator")
        factory.create(
            env,
            agents,
            spectator=spectator,
            loop_id="test-loop",
            agent_weights={"a": 1.0},
        )
        # StreamingConfig should have spectator and loop_id
        streaming_call = self.mock_streaming_cfg.call_args
        assert streaming_call[1]["spectator"] is spectator
        assert streaming_call[1]["loop_id"] == "test-loop"
        # AgentConfig should have agent_weights
        agent_call = self.mock_agent_cfg.call_args
        assert agent_call[1]["agent_weights"] == {"a": 1.0}

    def test_protocol_passed_through(self):
        factory = self.factory_mod.ArenaFactory()
        env, agents = self._make_env_agents()
        protocol = MagicMock(name="protocol")
        factory.create(env, agents, protocol=protocol)
        call_kw = self.mock_arena_cls.create.call_args[1]
        assert call_kw["protocol"] is protocol


# =========================================================================
# Global function tests
# =========================================================================


class TestGlobalFunctions:
    """get_arena_factory() returns singleton; create_arena delegates."""

    def test_get_arena_factory_returns_singleton(self):
        import aragora.debate.factory as mod

        # Reset the singleton
        mod._factory = None
        f1 = mod.get_arena_factory()
        f2 = mod.get_arena_factory()
        assert f1 is f2
        # Cleanup
        mod._factory = None

    def test_get_arena_factory_is_arena_factory_instance(self):
        import aragora.debate.factory as mod
        from aragora.debate.factory import ArenaFactory

        mod._factory = None
        f = mod.get_arena_factory()
        assert isinstance(f, ArenaFactory)
        mod._factory = None

    def test_create_arena_delegates_to_factory(self):
        import aragora.debate.factory as mod

        mock_factory = MagicMock()
        mod._factory = mock_factory
        env = MagicMock()
        agents = [MagicMock()]
        result = mod.create_arena(env, agents, enable_insights=True)
        mock_factory.create.assert_called_once_with(env, agents, enable_insights=True)
        assert result is mock_factory.create.return_value
        # Cleanup
        mod._factory = None
