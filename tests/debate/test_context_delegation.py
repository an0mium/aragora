"""Tests for aragora.debate.context_delegation.ContextDelegator."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.debate.context_delegation import ContextDelegator
from aragora.debate.state_cache import DebateStateCache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_gatherer():
    g = MagicMock()
    g.get_continuum_context = MagicMock(return_value=("ctx-text", ["id1"], {"id1": "fast"}))
    g.gather_all = AsyncMock(return_value="research-result")
    g.gather_aragora_context = AsyncMock(return_value="aragora-ctx")
    g.gather_evidence_context = AsyncMock(return_value="evidence-ctx")
    g.gather_trending_context = AsyncMock(return_value="trending-ctx")
    g.refresh_evidence_for_round = AsyncMock(return_value=(3, {"snippets": []}))
    g.evidence_pack = {"pack": True}
    return g


@pytest.fixture()
def mock_memory_manager():
    m = MagicMock()
    m.fetch_historical_context = AsyncMock(return_value="historical-ctx")
    m._format_patterns_for_prompt = MagicMock(return_value="formatted-patterns")
    m.get_successful_patterns = MagicMock(return_value="successful-patterns")
    return m


@pytest.fixture()
def cache():
    return DebateStateCache()


@pytest.fixture()
def mock_grounder():
    g = MagicMock()
    g.set_evidence_pack = MagicMock()
    return g


@pytest.fixture()
def mock_env():
    env = MagicMock()
    env.task = "Design a rate limiter"
    return env


@pytest.fixture()
def mock_auth_context():
    ac = MagicMock()
    ac.workspace_id = "ws-123"
    ac.org_id = "org-456"
    return ac


@pytest.fixture()
def mock_continuum_memory():
    return MagicMock()


@pytest.fixture()
def delegator(mock_gatherer, mock_memory_manager, cache, mock_grounder, mock_env):
    return ContextDelegator(
        context_gatherer=mock_gatherer,
        memory_manager=mock_memory_manager,
        cache=cache,
        evidence_grounder=mock_grounder,
        continuum_memory=MagicMock(),
        env=mock_env,
    )


# ---------------------------------------------------------------------------
# 1. Init and defaults
# ---------------------------------------------------------------------------


class TestInit:
    def test_init_stores_all_params(
        self, mock_gatherer, mock_memory_manager, cache, mock_grounder, mock_env
    ):
        d = ContextDelegator(
            context_gatherer=mock_gatherer,
            memory_manager=mock_memory_manager,
            cache=cache,
            evidence_grounder=mock_grounder,
            env=mock_env,
        )
        assert d.context_gatherer is mock_gatherer
        assert d.memory_manager is mock_memory_manager
        assert d._cache is cache
        assert d.evidence_grounder is mock_grounder
        assert d.env is mock_env

    def test_extract_domain_fn_default_returns_general(self):
        d = ContextDelegator()
        assert d._extract_domain() == "general"

    def test_extract_domain_fn_custom(self):
        d = ContextDelegator(extract_domain_fn=lambda: "finance")
        assert d._extract_domain() == "finance"

    def test_all_params_default_to_none(self):
        d = ContextDelegator()
        assert d.context_gatherer is None
        assert d.memory_manager is None
        assert d._cache is None
        assert d.evidence_grounder is None
        assert d.continuum_memory is None
        assert d.env is None
        assert d._auth_context is None


# ---------------------------------------------------------------------------
# 2-5. _resolve_tenant_id
# ---------------------------------------------------------------------------


class TestResolveTenantId:
    def test_no_auth_context_returns_none(self):
        d = ContextDelegator(auth_context=None)
        assert d._resolve_tenant_id() is None

    def test_enforce_off_returns_none(self, mock_auth_context, monkeypatch):
        monkeypatch.setenv("ARAGORA_MEMORY_TENANT_ENFORCE", "0")
        d = ContextDelegator(auth_context=mock_auth_context)
        assert d._resolve_tenant_id() is None

    def test_returns_workspace_id_when_present(self, mock_auth_context, monkeypatch):
        monkeypatch.setenv("ARAGORA_MEMORY_TENANT_ENFORCE", "1")
        d = ContextDelegator(auth_context=mock_auth_context)
        assert d._resolve_tenant_id() == "ws-123"

    def test_returns_org_id_as_fallback(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_MEMORY_TENANT_ENFORCE", "1")
        ac = MagicMock()
        ac.workspace_id = None
        ac.org_id = "org-789"
        d = ContextDelegator(auth_context=ac)
        assert d._resolve_tenant_id() == "org-789"

    def test_returns_none_when_neither_id_present(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_MEMORY_TENANT_ENFORCE", "1")
        ac = MagicMock()
        ac.workspace_id = None
        ac.org_id = None
        d = ContextDelegator(auth_context=ac)
        assert d._resolve_tenant_id() is None

    def test_default_enforce_is_on(self, mock_auth_context, monkeypatch):
        """When ARAGORA_MEMORY_TENANT_ENFORCE is not set, default is '1' (on)."""
        monkeypatch.delenv("ARAGORA_MEMORY_TENANT_ENFORCE", raising=False)
        d = ContextDelegator(auth_context=mock_auth_context)
        assert d._resolve_tenant_id() == "ws-123"


# ---------------------------------------------------------------------------
# 6-9. get_continuum_context
# ---------------------------------------------------------------------------


class TestGetContinuumContext:
    def test_cache_hit_returns_cached_value(self, mock_gatherer, cache):
        cache.continuum_context = "cached-memories"
        d = ContextDelegator(
            context_gatherer=mock_gatherer,
            cache=cache,
            continuum_memory=MagicMock(),
        )
        assert d.get_continuum_context() == "cached-memories"
        mock_gatherer.get_continuum_context.assert_not_called()

    def test_no_continuum_memory_returns_empty(self, mock_gatherer, cache):
        d = ContextDelegator(
            context_gatherer=mock_gatherer,
            cache=cache,
            continuum_memory=None,
        )
        assert d.get_continuum_context() == ""
        mock_gatherer.get_continuum_context.assert_not_called()

    def test_calls_gatherer_with_correct_params_and_tracks(
        self, mock_gatherer, cache, mock_env, mock_auth_context, monkeypatch
    ):
        monkeypatch.setenv("ARAGORA_MEMORY_TENANT_ENFORCE", "1")
        cm = MagicMock()
        d = ContextDelegator(
            context_gatherer=mock_gatherer,
            cache=cache,
            continuum_memory=cm,
            env=mock_env,
            auth_context=mock_auth_context,
            extract_domain_fn=lambda: "tech",
        )
        with patch(
            "aragora.debate.context_delegation.build_access_envelope",
            create=True,
        ):
            result = d.get_continuum_context()

        assert result == "ctx-text"
        mock_gatherer.get_continuum_context.assert_called_once_with(
            continuum_memory=cm,
            domain="tech",
            task="Design a rate limiter",
            tenant_id="ws-123",
            auth_context=mock_auth_context,
        )
        assert cache.continuum_context == "ctx-text"
        assert cache.continuum_retrieved_ids == ["id1"]
        assert cache.continuum_retrieved_tiers == {"id1": "fast"}

    def test_build_access_envelope_import_error_handled(
        self, mock_gatherer, cache, mock_env, monkeypatch
    ):
        monkeypatch.delenv("ARAGORA_MEMORY_TENANT_ENFORCE", raising=False)
        cm = MagicMock()
        d = ContextDelegator(
            context_gatherer=mock_gatherer,
            cache=cache,
            continuum_memory=cm,
            env=mock_env,
        )
        # The import may fail naturally or succeed; either way the method
        # should not raise. We patch builtins.__import__ to force ImportError
        # for the specific module.
        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def fake_import(name, *args, **kwargs):
            if name == "aragora.memory.access":
                raise ImportError("no access module")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = d.get_continuum_context()

        assert result == "ctx-text"

    def test_no_cache_still_returns_context(self, mock_gatherer, mock_env):
        cm = MagicMock()
        d = ContextDelegator(
            context_gatherer=mock_gatherer,
            cache=None,
            continuum_memory=cm,
            env=mock_env,
        )
        result = d.get_continuum_context()
        assert result == "ctx-text"

    def test_no_env_uses_empty_task(self, mock_gatherer, cache):
        cm = MagicMock()
        d = ContextDelegator(
            context_gatherer=mock_gatherer,
            cache=cache,
            continuum_memory=cm,
            env=None,
        )
        d.get_continuum_context()
        call_kwargs = mock_gatherer.get_continuum_context.call_args.kwargs
        assert call_kwargs["task"] == ""


# ---------------------------------------------------------------------------
# 10-11. perform_research
# ---------------------------------------------------------------------------


class TestPerformResearch:
    @pytest.mark.asyncio
    async def test_calls_gather_all_updates_cache_and_grounder(
        self, mock_gatherer, cache, mock_grounder
    ):
        d = ContextDelegator(
            context_gatherer=mock_gatherer,
            cache=cache,
            evidence_grounder=mock_grounder,
        )
        result = await d.perform_research("test task")
        assert result == "research-result"
        mock_gatherer.gather_all.assert_awaited_once_with("test task")
        assert cache.evidence_pack == {"pack": True}
        mock_grounder.set_evidence_pack.assert_called_once_with({"pack": True})

    @pytest.mark.asyncio
    async def test_no_cache_no_grounder_doesnt_error(self, mock_gatherer):
        d = ContextDelegator(
            context_gatherer=mock_gatherer,
            cache=None,
            evidence_grounder=None,
        )
        result = await d.perform_research("test task")
        assert result == "research-result"


# ---------------------------------------------------------------------------
# 12. gather_aragora_context
# ---------------------------------------------------------------------------


class TestGatherAragoraContext:
    @pytest.mark.asyncio
    async def test_delegates_to_gatherer(self, mock_gatherer):
        d = ContextDelegator(context_gatherer=mock_gatherer)
        result = await d.gather_aragora_context("some task")
        assert result == "aragora-ctx"
        mock_gatherer.gather_aragora_context.assert_awaited_once_with("some task")


# ---------------------------------------------------------------------------
# 13. gather_evidence_context
# ---------------------------------------------------------------------------


class TestGatherEvidenceContext:
    @pytest.mark.asyncio
    async def test_calls_gatherer_updates_cache_and_grounder(
        self, mock_gatherer, cache, mock_grounder
    ):
        d = ContextDelegator(
            context_gatherer=mock_gatherer,
            cache=cache,
            evidence_grounder=mock_grounder,
        )
        result = await d.gather_evidence_context("evidence task")
        assert result == "evidence-ctx"
        mock_gatherer.gather_evidence_context.assert_awaited_once_with("evidence task")
        assert cache.evidence_pack == {"pack": True}
        mock_grounder.set_evidence_pack.assert_called_once_with({"pack": True})

    @pytest.mark.asyncio
    async def test_no_cache_no_grounder(self, mock_gatherer):
        d = ContextDelegator(
            context_gatherer=mock_gatherer,
            cache=None,
            evidence_grounder=None,
        )
        result = await d.gather_evidence_context("task")
        assert result == "evidence-ctx"


# ---------------------------------------------------------------------------
# 14. gather_trending_context
# ---------------------------------------------------------------------------


class TestGatherTrendingContext:
    @pytest.mark.asyncio
    async def test_delegates_to_gatherer(self, mock_gatherer):
        d = ContextDelegator(context_gatherer=mock_gatherer)
        result = await d.gather_trending_context()
        assert result == "trending-ctx"
        mock_gatherer.gather_trending_context.assert_awaited_once()


# ---------------------------------------------------------------------------
# 15-16. refresh_evidence_for_round
# ---------------------------------------------------------------------------


class TestRefreshEvidenceForRound:
    @pytest.mark.asyncio
    async def test_with_updated_pack_updates_all_three(self, mock_gatherer, cache, mock_grounder):
        updated_pack = {"new": "evidence"}
        mock_gatherer.refresh_evidence_for_round = AsyncMock(return_value=(5, updated_pack))
        prompt_builder = MagicMock()
        d = ContextDelegator(
            context_gatherer=mock_gatherer,
            cache=cache,
            evidence_grounder=mock_grounder,
        )
        count = await d.refresh_evidence_for_round(
            combined_text="some text",
            evidence_collector=MagicMock(),
            task="task",
            evidence_store_callback=MagicMock(),
            prompt_builder=prompt_builder,
        )
        assert count == 5
        assert cache.evidence_pack == updated_pack
        mock_grounder.set_evidence_pack.assert_called_once_with(updated_pack)
        prompt_builder.set_evidence_pack.assert_called_once_with(updated_pack)

    @pytest.mark.asyncio
    async def test_no_updated_pack_doesnt_update(self, mock_gatherer, cache, mock_grounder):
        mock_gatherer.refresh_evidence_for_round = AsyncMock(return_value=(0, None))
        prompt_builder = MagicMock()
        d = ContextDelegator(
            context_gatherer=mock_gatherer,
            cache=cache,
            evidence_grounder=mock_grounder,
        )
        count = await d.refresh_evidence_for_round(
            combined_text="text",
            evidence_collector=MagicMock(),
            task="task",
            evidence_store_callback=MagicMock(),
            prompt_builder=prompt_builder,
        )
        assert count == 0
        assert cache.evidence_pack is None  # unchanged from default
        mock_grounder.set_evidence_pack.assert_not_called()
        prompt_builder.set_evidence_pack.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_cache_no_grounder_no_builder_with_updated_pack(self, mock_gatherer):
        updated_pack = {"data": True}
        mock_gatherer.refresh_evidence_for_round = AsyncMock(return_value=(2, updated_pack))
        d = ContextDelegator(
            context_gatherer=mock_gatherer,
            cache=None,
            evidence_grounder=None,
        )
        count = await d.refresh_evidence_for_round(
            combined_text="text",
            evidence_collector=MagicMock(),
            task="task",
            evidence_store_callback=MagicMock(),
            prompt_builder=None,
        )
        assert count == 2


# ---------------------------------------------------------------------------
# 17-18. fetch_historical_context
# ---------------------------------------------------------------------------


class TestFetchHistoricalContext:
    @pytest.mark.asyncio
    async def test_no_memory_manager_returns_empty(self):
        d = ContextDelegator(memory_manager=None)
        result = await d.fetch_historical_context("task")
        assert result == ""

    @pytest.mark.asyncio
    async def test_delegates_to_manager(self, mock_memory_manager):
        d = ContextDelegator(memory_manager=mock_memory_manager)
        result = await d.fetch_historical_context("task", limit=5)
        assert result == "historical-ctx"
        mock_memory_manager.fetch_historical_context.assert_awaited_once_with("task", 5)

    @pytest.mark.asyncio
    async def test_default_limit_is_three(self, mock_memory_manager):
        d = ContextDelegator(memory_manager=mock_memory_manager)
        await d.fetch_historical_context("task")
        mock_memory_manager.fetch_historical_context.assert_awaited_once_with("task", 3)


# ---------------------------------------------------------------------------
# 19. format_patterns_for_prompt
# ---------------------------------------------------------------------------


class TestFormatPatternsForPrompt:
    def test_no_manager_returns_empty(self):
        d = ContextDelegator(memory_manager=None)
        assert d.format_patterns_for_prompt([{"p": 1}]) == ""

    def test_delegates_to_manager(self, mock_memory_manager):
        patterns = [{"strategy": "divide", "success_rate": 0.9}]
        d = ContextDelegator(memory_manager=mock_memory_manager)
        result = d.format_patterns_for_prompt(patterns)
        assert result == "formatted-patterns"
        mock_memory_manager._format_patterns_for_prompt.assert_called_once_with(patterns)


# ---------------------------------------------------------------------------
# 20. get_successful_patterns
# ---------------------------------------------------------------------------


class TestGetSuccessfulPatterns:
    def test_no_manager_returns_empty(self):
        d = ContextDelegator(memory_manager=None)
        assert d.get_successful_patterns() == ""

    def test_delegates_to_manager(self, mock_memory_manager):
        d = ContextDelegator(memory_manager=mock_memory_manager)
        result = d.get_successful_patterns(limit=10)
        assert result == "successful-patterns"
        mock_memory_manager.get_successful_patterns.assert_called_once_with(10)

    def test_default_limit_is_five(self, mock_memory_manager):
        d = ContextDelegator(memory_manager=mock_memory_manager)
        d.get_successful_patterns()
        mock_memory_manager.get_successful_patterns.assert_called_once_with(5)
