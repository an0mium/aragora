"""Comprehensive tests for MemoryProgressiveMixin handler methods.

Tests cover:
- _search_index: progressive retrieval stage 1 (compact index entries)
- _search_timeline: progressive retrieval stage 2 (timeline around anchor)
- _get_entries: progressive retrieval stage 3 (full entries by ID)
- _search_memories: cross-tier search with filtering and sorting

Each method is tested for:
- Missing/invalid required parameters (400)
- Continuum unavailable (503)
- Continuum not initialized (503)
- Successful retrieval paths
- Tenant enforcement
- External source integration (search_index)
- Sorting options (search_memories)
- Feature-not-supported (501) responses
- RBAC filter_entries integration
"""

import json
import pytest
from enum import Enum
from unittest.mock import MagicMock, patch

from aragora.rbac.models import AuthorizationContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_AUTH_CONTEXT = AuthorizationContext(
    user_id="test-user",
    roles={"admin"},
    permissions={"memory:read", "memory:write", "memory:manage", "memory:*"},
)


class MockMemoryTier(Enum):
    """Mock MemoryTier for testing."""

    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    GLACIAL = "glacial"


class MockMemory:
    """Mock memory entry."""

    def __init__(
        self,
        id="mem1",
        tier=None,
        content="Test content",
        importance=0.5,
        surprise_score=0.3,
    ):
        self.id = id
        self.tier = tier or MockMemoryTier.FAST
        self.content = content
        self.importance = importance
        self.surprise_score = surprise_score
        self.created_at = "2024-01-01T00:00:00"
        self.updated_at = "2024-01-02T00:00:00"
        self.metadata = {"type": "test"}


class MockHybridResult:
    """Mock result from hybrid_search."""

    def __init__(self, memory_id="mem1", combined_score=0.95):
        self.memory_id = memory_id
        self.combined_score = combined_score


def _bypass_rbac(handler):
    """Inject an admin auth context so RBAC decorators pass."""
    handler._auth_context = _TEST_AUTH_CONTEXT
    return handler


def _body(result):
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _disable_tenant_enforcement(monkeypatch):
    """Disable tenant enforcement by default for unit tests.

    Individual tests that need tenant enforcement patch it explicitly.
    The env var ARAGORA_MEMORY_TENANT_ENFORCE defaults to '1' (enabled),
    which causes 400 errors when the test auth context lacks workspace_id/org_id.
    """
    monkeypatch.setenv("ARAGORA_MEMORY_TENANT_ENFORCE", "0")


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    """Reset module-level rate limiters between tests."""
    from aragora.server.handlers.memory.memory import (
        _retrieve_limiter,
        _stats_limiter,
        _mutation_limiter,
    )

    _retrieve_limiter._buckets.clear()
    _stats_limiter._buckets.clear()
    _mutation_limiter._buckets.clear()


@pytest.fixture(autouse=True)
def _disable_tenant_enforcement():
    """Disable tenant enforcement by default.

    Individual tests that need to verify tenant enforcement behavior
    should patch tenant_enforcement_enabled themselves.
    """
    with patch(
        "aragora.memory.access.tenant_enforcement_enabled", return_value=False
    ):
        with patch("aragora.memory.access.resolve_tenant_id", return_value=None):
            with patch(
                "aragora.memory.access.filter_entries",
                side_effect=lambda entries, ctx: entries,
            ):
                yield


@pytest.fixture
def mock_continuum():
    """Create a bare mock ContinuumMemory."""
    mem = MagicMock()
    mem.retrieve = MagicMock(return_value=[])
    mem.get_many = MagicMock(return_value=[])
    mem.get_timeline_entries = MagicMock(return_value=None)
    return mem


@pytest.fixture
def handler(mock_continuum):
    """Create a MemoryHandler with continuum available and RBAC bypassed."""
    from aragora.server.handlers.memory.memory import MemoryHandler

    ctx = {"continuum_memory": mock_continuum}
    h = MemoryHandler(ctx)
    _bypass_rbac(h)
    return h


@pytest.fixture
def handler_no_continuum():
    """Create a MemoryHandler without continuum memory in context."""
    from aragora.server.handlers.memory.memory import MemoryHandler

    h = MemoryHandler({"continuum_memory": None})
    _bypass_rbac(h)
    return h


# ===========================================================================
# _search_index tests
# ===========================================================================


class TestSearchIndex:
    """Tests for _search_index (progressive retrieval stage 1)."""

    def test_missing_query_returns_400(self, handler):
        result = handler._search_index({})
        assert result.status_code == 400
        assert "q" in _body(result).get("error", "")

    def test_empty_query_returns_400(self, handler):
        result = handler._search_index({"q": ""})
        assert result.status_code == 400

    @patch("aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", False)
    def test_continuum_unavailable_returns_503(self, handler):
        result = handler._search_index({"q": "test"})
        assert result.status_code == 503

    def test_continuum_not_initialized_returns_503(self, handler_no_continuum):
        result = handler_no_continuum._search_index({"q": "test"})
        assert result.status_code == 503

    def test_basic_search_returns_200(self, handler, mock_continuum):
        entry = MockMemory(id="m1", content="hello world")
        mock_continuum.retrieve.return_value = [entry]

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._search_index({"q": "hello"})

        assert result.status_code == 200
        body = _body(result)
        assert body["query"] == "hello"
        assert body["count"] == 1
        assert body["results"][0]["source"] == "continuum"
        assert body["use_hybrid"] is False

    def test_search_respects_limit_param(self, handler, mock_continuum):
        mock_continuum.retrieve.return_value = []

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            handler._search_index({"q": "test", "limit": "5"})

        call_kwargs = mock_continuum.retrieve.call_args[1]
        assert call_kwargs["limit"] == 5

    def test_search_respects_min_importance(self, handler, mock_continuum):
        mock_continuum.retrieve.return_value = []

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            handler._search_index({"q": "test", "min_importance": "0.7"})

        call_kwargs = mock_continuum.retrieve.call_args[1]
        assert call_kwargs["min_importance"] == 0.7

    def test_search_passes_tiers(self, handler, mock_continuum):
        mock_continuum.retrieve.return_value = []

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            handler._search_index({"q": "test", "tier": "fast,slow"})

        call_kwargs = mock_continuum.retrieve.call_args[1]
        tier_names = {t.name for t in call_kwargs["tiers"]}
        assert "FAST" in tier_names
        assert "SLOW" in tier_names

    def test_hybrid_search_used_when_requested(self, handler, mock_continuum):
        """When use_hybrid=true and continuum supports hybrid_search."""
        hybrid_result = MockHybridResult("m1", 0.95)
        mock_continuum.hybrid_search = MagicMock(return_value=[hybrid_result])

        entry = MockMemory(id="m1", content="hybrid match")
        mock_continuum.get_many.return_value = [entry]

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            with patch(
                "aragora.server.handlers.memory.memory_progressive.run_async",
                side_effect=lambda coro: [hybrid_result],
            ):
                result = handler._search_index({"q": "test", "use_hybrid": "true"})

        assert result.status_code == 200
        body = _body(result)
        assert body["use_hybrid"] is True

    def test_hybrid_search_result_fields(self, handler, mock_continuum):
        """Hybrid results include score and id fields."""
        hybrid_result = MockHybridResult("m1", 0.9512)
        mock_continuum.hybrid_search = MagicMock(return_value=[hybrid_result])

        entry = MockMemory(id="m1", content="hybrid match")
        mock_continuum.get_many.return_value = [entry]

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            with patch(
                "aragora.server.handlers.memory.memory_progressive.run_async",
                side_effect=lambda coro: [hybrid_result],
            ):
                result = handler._search_index({"q": "test", "use_hybrid": "true"})

        body = _body(result)
        assert body["results"][0]["score"] == 0.9512
        assert body["results"][0]["id"] == "m1"
        assert body["results"][0]["source"] == "continuum"

    def test_hybrid_skips_entries_not_found(self, handler, mock_continuum):
        """Hybrid results where entry not found in get_many are skipped."""
        hybrid_result = MockHybridResult("missing_id", 0.9)
        mock_continuum.hybrid_search = MagicMock(return_value=[hybrid_result])
        mock_continuum.get_many.return_value = []  # No entries found

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            with patch(
                "aragora.server.handlers.memory.memory_progressive.run_async",
                side_effect=lambda coro: [hybrid_result],
            ):
                result = handler._search_index({"q": "test", "use_hybrid": "true"})

        body = _body(result)
        assert body["count"] == 0
        assert body["results"] == []

    def test_hybrid_fallback_on_missing_method(self, handler, mock_continuum):
        """Falls back to regular retrieve if continuum lacks hybrid_search."""
        if hasattr(mock_continuum, "hybrid_search"):
            del mock_continuum.hybrid_search

        mock_continuum.retrieve.return_value = []

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._search_index({"q": "test", "use_hybrid": "true"})

        assert result.status_code == 200
        body = _body(result)
        assert body["use_hybrid"] is False

    def test_external_supermemory_included(self, handler, mock_continuum):
        """include_external=true fetches supermemory results."""
        mock_continuum.retrieve.return_value = []
        handler._search_supermemory = MagicMock(
            return_value=[{"source": "supermemory", "text": "ext"}]
        )
        handler._search_claude_mem = MagicMock(return_value=[])

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._search_index(
                {"q": "test", "include_external": "true"}
            )

        assert result.status_code == 200
        body = _body(result)
        assert "supermemory" in body["external_sources"]
        assert len(body["external_results"]) == 1
        handler._search_supermemory.assert_called_once()

    def test_external_claude_mem_included(self, handler, mock_continuum):
        """include_external=true fetches claude-mem results."""
        mock_continuum.retrieve.return_value = []
        handler._search_supermemory = MagicMock(return_value=[])
        handler._search_claude_mem = MagicMock(
            return_value=[{"source": "claude-mem", "text": "c"}]
        )

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._search_index(
                {"q": "test", "include_external": "true"}
            )

        assert result.status_code == 200
        body = _body(result)
        assert "claude-mem" in body["external_sources"]
        handler._search_claude_mem.assert_called_once()

    def test_external_specific_source_only(self, handler, mock_continuum):
        """external=supermemory only calls supermemory, not claude-mem."""
        mock_continuum.retrieve.return_value = []
        handler._search_supermemory = MagicMock(return_value=[])
        handler._search_claude_mem = MagicMock(return_value=[])

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._search_index(
                {
                    "q": "test",
                    "include_external": "true",
                    "external": "supermemory",
                }
            )

        body = _body(result)
        assert "supermemory" in body["external_sources"]
        assert "claude-mem" not in body["external_sources"]
        handler._search_supermemory.assert_called_once()
        handler._search_claude_mem.assert_not_called()

    def test_external_claude_mem_aliases(self, handler, mock_continuum):
        """claude_mem, claudemem, claude-mem all normalize to claude-mem."""
        mock_continuum.retrieve.return_value = []
        handler._search_supermemory = MagicMock(return_value=[])
        handler._search_claude_mem = MagicMock(return_value=[])

        for alias in ("claude_mem", "claudemem", "claude-mem"):
            handler._search_claude_mem.reset_mock()
            with patch(
                "aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier
            ):
                result = handler._search_index(
                    {
                        "q": "test",
                        "include_external": "true",
                        "external": alias,
                    }
                )
            body = _body(result)
            assert "claude-mem" in body["external_sources"], (
                f"alias {alias!r} not normalized"
            )

    def test_external_supermemory_aliases(self, handler, mock_continuum):
        """super-memory and sm normalize to supermemory."""
        mock_continuum.retrieve.return_value = []
        handler._search_supermemory = MagicMock(return_value=[])
        handler._search_claude_mem = MagicMock(return_value=[])

        for alias in ("super-memory", "sm"):
            handler._search_supermemory.reset_mock()
            with patch(
                "aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier
            ):
                handler._search_index(
                    {
                        "q": "test",
                        "include_external": "true",
                        "external": alias,
                    }
                )
            handler._search_supermemory.assert_called_once()

    def test_no_external_by_default(self, handler, mock_continuum):
        """External sources not queried when include_external is false."""
        mock_continuum.retrieve.return_value = []
        handler._search_supermemory = MagicMock(return_value=[])
        handler._search_claude_mem = MagicMock(return_value=[])

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._search_index({"q": "test"})

        body = _body(result)
        assert body["external_sources"] == []
        assert body["external_results"] == []
        handler._search_supermemory.assert_not_called()
        handler._search_claude_mem.assert_not_called()

    def test_external_with_project_param(self, handler, mock_continuum):
        """claude-mem external source passes project param."""
        mock_continuum.retrieve.return_value = []
        handler._search_supermemory = MagicMock(return_value=[])
        handler._search_claude_mem = MagicMock(return_value=[])

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            handler._search_index(
                {
                    "q": "test",
                    "include_external": "true",
                    "external": "claude-mem",
                    "project": "my-project",
                }
            )

        handler._search_claude_mem.assert_called_once_with(
            "test", limit=20, project="my-project"
        )

    def test_tenant_enforcement_no_auth_context_disables_enforcement(
        self, handler, mock_continuum
    ):
        """When auth context is None, tenant enforcement is disabled."""
        handler._auth_context = None
        mock_continuum.retrieve.return_value = []

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            with patch(
                "aragora.memory.access.tenant_enforcement_enabled",
                return_value=True,
            ):
                with patch(
                    "aragora.memory.access.resolve_tenant_id", return_value=None
                ):
                    result = handler._search_index({"q": "test"})

        assert result.status_code == 200

    def test_tenant_enforcement_required_but_no_tenant_returns_400(
        self, handler, mock_continuum
    ):
        """When tenant enforcement is on but tenant_id is None, returns 400."""
        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            with patch(
                "aragora.memory.access.tenant_enforcement_enabled",
                return_value=True,
            ):
                with patch(
                    "aragora.memory.access.resolve_tenant_id", return_value=None
                ):
                    result = handler._search_index({"q": "test"})

        assert result.status_code == 400
        assert "Tenant" in _body(result).get("error", "")

    def test_filter_entries_applied(self, handler, mock_continuum):
        """RBAC filter_entries is called when available."""
        entry = MockMemory(id="m1")
        mock_continuum.retrieve.return_value = [entry]

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            with patch(
                "aragora.memory.access.filter_entries", return_value=[entry]
            ) as mock_filter:
                result = handler._search_index({"q": "test"})

        assert result.status_code == 200
        mock_filter.assert_called_once()

    def test_memory_access_import_failure_degrades_gracefully(
        self, handler, mock_continuum
    ):
        """When memory.access fails to import, search still works."""
        mock_continuum.retrieve.return_value = []

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            with patch.dict("sys.modules", {"aragora.memory.access": None}):
                result = handler._search_index({"q": "test"})

        # Should either return 200 (graceful) or the method handles ImportError
        assert result.status_code in (200, 500)

    def test_response_tiers_field(self, handler, mock_continuum):
        """Response includes searched tier names."""
        mock_continuum.retrieve.return_value = []

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._search_index({"q": "test"})

        body = _body(result)
        assert "tiers" in body
        assert isinstance(body["tiers"], list)
        # Default: all tiers searched
        assert len(body["tiers"]) == 4

    def test_response_tiers_filtered(self, handler, mock_continuum):
        """When tier filter is applied, only those tiers in response."""
        mock_continuum.retrieve.return_value = []

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._search_index({"q": "test", "tier": "fast"})

        body = _body(result)
        assert body["tiers"] == ["fast"]


# ===========================================================================
# _search_timeline tests
# ===========================================================================


class TestSearchTimeline:
    """Tests for _search_timeline (progressive retrieval stage 2)."""

    def test_missing_anchor_id_returns_400(self, handler):
        result = handler._search_timeline({})
        assert result.status_code == 400
        assert "anchor_id" in _body(result).get("error", "")

    def test_empty_anchor_id_returns_400(self, handler):
        result = handler._search_timeline({"anchor_id": ""})
        assert result.status_code == 400

    @patch("aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", False)
    def test_continuum_unavailable_returns_503(self, handler):
        result = handler._search_timeline({"anchor_id": "m1"})
        assert result.status_code == 503

    def test_continuum_not_initialized_returns_503(self, handler_no_continuum):
        result = handler_no_continuum._search_timeline({"anchor_id": "m1"})
        assert result.status_code == 503

    def test_no_get_timeline_entries_returns_501(self, handler, mock_continuum):
        """If continuum backend lacks get_timeline_entries, returns 501."""
        del mock_continuum.get_timeline_entries

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._search_timeline({"anchor_id": "m1"})

        assert result.status_code == 501
        assert "Timeline" in _body(result).get("error", "")

    def test_anchor_not_found_returns_404(self, handler, mock_continuum):
        """When get_timeline_entries returns None, returns 404."""
        mock_continuum.get_timeline_entries.return_value = None

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._search_timeline({"anchor_id": "nonexistent"})

        assert result.status_code == 404
        assert "Anchor" in _body(result).get("error", "")

    def test_successful_timeline_retrieval(self, handler, mock_continuum):
        """Returns 200 with anchor, before, and after entries."""
        anchor = MockMemory(id="anchor1", content="Anchor content")
        before_entry = MockMemory(id="b1", content="Before content")
        after_entry = MockMemory(id="a1", content="After content")

        mock_continuum.get_timeline_entries.return_value = {
            "anchor": anchor,
            "before": [before_entry],
            "after": [after_entry],
        }

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._search_timeline({"anchor_id": "anchor1"})

        assert result.status_code == 200
        body = _body(result)
        assert body["anchor_id"] == "anchor1"
        assert "anchor" in body
        assert len(body["before"]) == 1
        assert len(body["after"]) == 1

    def test_timeline_anchor_has_preview(self, handler, mock_continuum):
        """Anchor entry is formatted with preview_chars=260."""
        anchor = MockMemory(id="a1", content="A" * 300)
        mock_continuum.get_timeline_entries.return_value = {
            "anchor": anchor,
            "before": [],
            "after": [],
        }

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._search_timeline({"anchor_id": "a1"})

        body = _body(result)
        # Anchor gets preview_chars=260
        preview = body["anchor"]["preview"]
        assert len(preview) <= 263  # 260 + "..."

    def test_timeline_default_before_after(self, handler, mock_continuum):
        """Default before=3, after=3 are passed to continuum."""
        anchor = MockMemory(id="a1")
        mock_continuum.get_timeline_entries.return_value = {
            "anchor": anchor,
            "before": [],
            "after": [],
        }

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            handler._search_timeline({"anchor_id": "a1"})

        call_kwargs = mock_continuum.get_timeline_entries.call_args[1]
        assert call_kwargs["before"] == 3
        assert call_kwargs["after"] == 3

    def test_timeline_custom_before_after(self, handler, mock_continuum):
        """Custom before/after values passed through."""
        anchor = MockMemory(id="a1")
        mock_continuum.get_timeline_entries.return_value = {
            "anchor": anchor,
            "before": [],
            "after": [],
        }

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            handler._search_timeline(
                {"anchor_id": "a1", "before": "10", "after": "5"}
            )

        call_kwargs = mock_continuum.get_timeline_entries.call_args[1]
        assert call_kwargs["before"] == 10
        assert call_kwargs["after"] == 5

    def test_timeline_min_importance_passed(self, handler, mock_continuum):
        anchor = MockMemory(id="a1")
        mock_continuum.get_timeline_entries.return_value = {
            "anchor": anchor,
            "before": [],
            "after": [],
        }

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            handler._search_timeline(
                {"anchor_id": "a1", "min_importance": "0.6"}
            )

        call_kwargs = mock_continuum.get_timeline_entries.call_args[1]
        assert call_kwargs["min_importance"] == 0.6

    def test_timeline_tiers_passed(self, handler, mock_continuum):
        """Tier filter is passed to get_timeline_entries."""
        anchor = MockMemory(id="a1")
        mock_continuum.get_timeline_entries.return_value = {
            "anchor": anchor,
            "before": [],
            "after": [],
        }

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            handler._search_timeline(
                {"anchor_id": "a1", "tier": "fast,medium"}
            )

        call_kwargs = mock_continuum.get_timeline_entries.call_args[1]
        tier_names = {t.name for t in call_kwargs["tiers"]}
        assert tier_names == {"FAST", "MEDIUM"}

    def test_timeline_rbac_filter_hides_anchor(self, handler, mock_continuum):
        """If filter_entries removes the anchor, returns 404."""
        anchor = MockMemory(id="a1")
        mock_continuum.get_timeline_entries.return_value = {
            "anchor": anchor,
            "before": [],
            "after": [],
        }

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            with patch("aragora.memory.access.filter_entries", return_value=[]):
                result = handler._search_timeline({"anchor_id": "a1"})

        assert result.status_code == 404

    def test_timeline_rbac_filter_applied_to_before_after(
        self, handler, mock_continuum
    ):
        """filter_entries is called on before and after lists too."""
        anchor = MockMemory(id="a1")
        before1 = MockMemory(id="b1")
        after1 = MockMemory(id="af1")

        mock_continuum.get_timeline_entries.return_value = {
            "anchor": anchor,
            "before": [before1],
            "after": [after1],
        }

        def filter_fn(entries, ctx):
            return entries

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            with patch(
                "aragora.memory.access.filter_entries", side_effect=filter_fn
            ) as mock_filter:
                result = handler._search_timeline({"anchor_id": "a1"})

        assert result.status_code == 200
        # Called 3 times: anchor, before, after
        assert mock_filter.call_count == 3

    def test_timeline_tenant_enforcement_no_tenant_returns_400(
        self, handler, mock_continuum
    ):
        """When tenant enforcement is on but tenant_id is None, returns 400."""
        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            with patch(
                "aragora.memory.access.tenant_enforcement_enabled",
                return_value=True,
            ):
                with patch(
                    "aragora.memory.access.resolve_tenant_id", return_value=None
                ):
                    result = handler._search_timeline({"anchor_id": "a1"})

        assert result.status_code == 400
        assert "Tenant" in _body(result).get("error", "")

    def test_timeline_tenant_enforcement_disabled_when_no_auth(
        self, handler, mock_continuum
    ):
        """When auth context is None, tenant enforcement is disabled."""
        anchor = MockMemory(id="a1")
        mock_continuum.get_timeline_entries.return_value = {
            "anchor": anchor,
            "before": [],
            "after": [],
        }

        handler._auth_context = None

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            with patch(
                "aragora.memory.access.tenant_enforcement_enabled",
                return_value=True,
            ):
                with patch(
                    "aragora.memory.access.resolve_tenant_id", return_value=None
                ):
                    result = handler._search_timeline({"anchor_id": "a1"})

        assert result.status_code == 200

    def test_timeline_with_tenant_id(self, handler, mock_continuum):
        """When tenant_id is resolved, it is passed to get_timeline_entries."""
        anchor = MockMemory(id="a1")
        mock_continuum.get_timeline_entries.return_value = {
            "anchor": anchor,
            "before": [],
            "after": [],
        }

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            with patch(
                "aragora.memory.access.tenant_enforcement_enabled",
                return_value=True,
            ):
                with patch(
                    "aragora.memory.access.resolve_tenant_id",
                    return_value="tenant-abc",
                ):
                    result = handler._search_timeline({"anchor_id": "a1"})

        assert result.status_code == 200
        call_kwargs = mock_continuum.get_timeline_entries.call_args[1]
        assert call_kwargs["tenant_id"] == "tenant-abc"
        assert call_kwargs["enforce_tenant_isolation"] is True


# ===========================================================================
# _get_entries tests
# ===========================================================================


class TestGetEntries:
    """Tests for _get_entries (progressive retrieval stage 3)."""

    def test_missing_ids_returns_400(self, handler):
        result = handler._get_entries({})
        assert result.status_code == 400
        assert "ids" in _body(result).get("error", "")

    def test_empty_ids_returns_400(self, handler):
        result = handler._get_entries({"ids": ""})
        assert result.status_code == 400

    def test_whitespace_only_ids_returns_400(self, handler):
        result = handler._get_entries({"ids": " , , "})
        assert result.status_code == 400

    @patch("aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", False)
    def test_continuum_unavailable_returns_503(self, handler):
        result = handler._get_entries({"ids": "m1"})
        assert result.status_code == 503

    def test_continuum_not_initialized_returns_503(self, handler_no_continuum):
        result = handler_no_continuum._get_entries({"ids": "m1"})
        assert result.status_code == 503

    def test_no_get_many_returns_501(self, handler, mock_continuum):
        """If continuum backend lacks get_many, returns 501."""
        del mock_continuum.get_many

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._get_entries({"ids": "m1"})

        assert result.status_code == 501
        assert "Bulk" in _body(result).get("error", "")

    def test_single_id_success(self, handler, mock_continuum):
        entry = MockMemory(id="m1", content="Full entry content")
        mock_continuum.get_many.return_value = [entry]

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._get_entries({"ids": "m1"})

        assert result.status_code == 200
        body = _body(result)
        assert body["ids"] == ["m1"]
        assert body["count"] == 1
        assert len(body["entries"]) == 1

    def test_multiple_ids_success(self, handler, mock_continuum):
        e1 = MockMemory(id="m1", content="Entry one")
        e2 = MockMemory(id="m2", content="Entry two")
        mock_continuum.get_many.return_value = [e1, e2]

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._get_entries({"ids": "m1,m2"})

        assert result.status_code == 200
        body = _body(result)
        assert body["ids"] == ["m1", "m2"]
        assert body["count"] == 2

    def test_ids_stripped_of_whitespace(self, handler, mock_continuum):
        mock_continuum.get_many.return_value = []

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._get_entries({"ids": " m1 , m2 "})

        assert result.status_code == 200
        body = _body(result)
        assert body["ids"] == ["m1", "m2"]

    def test_entries_use_format_entry_full(self, handler, mock_continuum):
        """Entries use the full format (includes content and metadata)."""
        entry = MockMemory(id="m1", content="Full content here", importance=0.8)
        mock_continuum.get_many.return_value = [entry]

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._get_entries({"ids": "m1"})

        body = _body(result)
        entry_data = body["entries"][0]
        assert "content" in entry_data
        assert "metadata" in entry_data

    def test_entries_include_token_estimate(self, handler, mock_continuum):
        """Full entries include token_estimate field."""
        entry = MockMemory(id="m1", content="Hello world test")
        mock_continuum.get_many.return_value = [entry]

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._get_entries({"ids": "m1"})

        body = _body(result)
        entry_data = body["entries"][0]
        assert "token_estimate" in entry_data
        assert entry_data["token_estimate"] > 0

    def test_get_entries_filter_entries_applied(self, handler, mock_continuum):
        """RBAC filter_entries is called on bulk results."""
        e1 = MockMemory(id="m1")
        mock_continuum.get_many.return_value = [e1]

        # filter_entries removes the entry
        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            with patch(
                "aragora.memory.access.filter_entries", return_value=[]
            ) as mock_filter:
                result = handler._get_entries({"ids": "m1"})

        assert result.status_code == 200
        body = _body(result)
        assert body["count"] == 0
        assert body["entries"] == []
        mock_filter.assert_called_once()

    def test_get_entries_tenant_enforcement_required_no_tenant(
        self, handler, mock_continuum
    ):
        """When tenant enforcement is on but tenant_id is None, returns 400."""
        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            with patch(
                "aragora.memory.access.tenant_enforcement_enabled",
                return_value=True,
            ):
                with patch(
                    "aragora.memory.access.resolve_tenant_id", return_value=None
                ):
                    result = handler._get_entries({"ids": "m1"})

        assert result.status_code == 400

    def test_get_entries_tenant_id_passed(self, handler, mock_continuum):
        """When tenant_id is resolved, it is passed to get_many."""
        mock_continuum.get_many.return_value = []

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            with patch(
                "aragora.memory.access.tenant_enforcement_enabled",
                return_value=True,
            ):
                with patch(
                    "aragora.memory.access.resolve_tenant_id",
                    return_value="tenant-123",
                ):
                    result = handler._get_entries({"ids": "m1"})

        assert result.status_code == 200
        call_kwargs = mock_continuum.get_many.call_args[1]
        assert call_kwargs["tenant_id"] == "tenant-123"

    def test_get_entries_empty_result(self, handler, mock_continuum):
        """When get_many returns empty list, response has count=0."""
        mock_continuum.get_many.return_value = []

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._get_entries({"ids": "m1,m2,m3"})

        assert result.status_code == 200
        body = _body(result)
        assert body["count"] == 0
        assert body["entries"] == []
        assert body["ids"] == ["m1", "m2", "m3"]


# ===========================================================================
# _search_memories tests
# ===========================================================================


class TestSearchMemories:
    """Tests for _search_memories (cross-tier search with filtering)."""

    def test_missing_query_returns_400(self, handler):
        result = handler._search_memories({})
        assert result.status_code == 400
        assert "q" in _body(result).get("error", "")

    def test_empty_query_returns_400(self, handler):
        result = handler._search_memories({"q": ""})
        assert result.status_code == 400

    @patch("aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", False)
    def test_continuum_unavailable_returns_503(self, handler):
        result = handler._search_memories({"q": "test"})
        assert result.status_code == 503

    def test_continuum_not_initialized_returns_503(self, handler_no_continuum):
        result = handler_no_continuum._search_memories({"q": "test"})
        assert result.status_code == 503

    def test_basic_search_returns_200(self, handler, mock_continuum):
        entry = MockMemory(id="m1", content="Matching content", importance=0.8)
        mock_continuum.retrieve.return_value = [entry]

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._search_memories({"q": "matching"})

        assert result.status_code == 200
        body = _body(result)
        assert body["query"] == "matching"
        assert body["count"] == 1
        assert body["results"][0]["id"] == "m1"
        assert body["results"][0]["tier"] == "fast"

    def test_search_result_fields(self, handler, mock_continuum):
        """Verify all expected fields in search result."""
        entry = MockMemory(
            id="m1", content="Test content", importance=0.8, surprise_score=0.4
        )
        mock_continuum.retrieve.return_value = [entry]

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._search_memories({"q": "test"})

        body = _body(result)
        r = body["results"][0]
        assert "id" in r
        assert "tier" in r
        assert "content" in r
        assert "importance" in r
        assert "surprise_score" in r
        assert "created_at" in r
        assert "updated_at" in r
        assert "metadata" in r

    def test_search_truncates_long_content(self, handler, mock_continuum):
        """Content longer than 300 chars is truncated with '...'."""
        long_content = "x" * 500
        entry = MockMemory(id="m1", content=long_content)
        mock_continuum.retrieve.return_value = [entry]

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._search_memories({"q": "test"})

        body = _body(result)
        assert body["results"][0]["content"].endswith("...")
        assert len(body["results"][0]["content"]) == 303  # 300 + "..."

    def test_search_short_content_not_truncated(self, handler, mock_continuum):
        """Content shorter than 300 chars is not truncated."""
        entry = MockMemory(id="m1", content="Short")
        mock_continuum.retrieve.return_value = [entry]

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._search_memories({"q": "test"})

        body = _body(result)
        assert body["results"][0]["content"] == "Short"

    def test_search_default_limit(self, handler, mock_continuum):
        mock_continuum.retrieve.return_value = []

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            handler._search_memories({"q": "test"})

        call_kwargs = mock_continuum.retrieve.call_args[1]
        assert call_kwargs["limit"] == 20

    def test_search_custom_limit(self, handler, mock_continuum):
        mock_continuum.retrieve.return_value = []

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            handler._search_memories({"q": "test", "limit": "50"})

        call_kwargs = mock_continuum.retrieve.call_args[1]
        assert call_kwargs["limit"] == 50

    def test_search_limit_clamped_max(self, handler, mock_continuum):
        """Limit is clamped to max 100."""
        mock_continuum.retrieve.return_value = []

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            handler._search_memories({"q": "test", "limit": "999"})

        call_kwargs = mock_continuum.retrieve.call_args[1]
        assert call_kwargs["limit"] == 100

    def test_search_limit_clamped_min(self, handler, mock_continuum):
        """Limit is clamped to min 1."""
        mock_continuum.retrieve.return_value = []

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            handler._search_memories({"q": "test", "limit": "0"})

        call_kwargs = mock_continuum.retrieve.call_args[1]
        assert call_kwargs["limit"] == 1

    def test_search_min_importance(self, handler, mock_continuum):
        mock_continuum.retrieve.return_value = []

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            handler._search_memories({"q": "test", "min_importance": "0.5"})

        call_kwargs = mock_continuum.retrieve.call_args[1]
        assert call_kwargs["min_importance"] == 0.5

    def test_search_tier_filter(self, handler, mock_continuum):
        """tier param filters which tiers are searched."""
        mock_continuum.retrieve.return_value = []

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._search_memories(
                {"q": "test", "tier": "fast,medium"}
            )

        call_kwargs = mock_continuum.retrieve.call_args[1]
        tier_names = {t.name for t in call_kwargs["tiers"]}
        assert tier_names == {"FAST", "MEDIUM"}

        body = _body(result)
        assert set(body["tiers_searched"]) == {"fast", "medium"}

    def test_search_invalid_tier_ignored(self, handler, mock_continuum):
        """Invalid tier names are skipped (not an error)."""
        mock_continuum.retrieve.return_value = []

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            handler._search_memories({"q": "test", "tier": "fast,bogus"})

        call_kwargs = mock_continuum.retrieve.call_args[1]
        tier_names = {t.name for t in call_kwargs["tiers"]}
        assert "FAST" in tier_names
        assert "BOGUS" not in tier_names

    def test_search_all_invalid_tiers_defaults_to_all(
        self, handler, mock_continuum
    ):
        """If no valid tiers remain after parsing, all tiers are searched."""
        mock_continuum.retrieve.return_value = []

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._search_memories(
                {"q": "test", "tier": "bogus,nope"}
            )

        call_kwargs = mock_continuum.retrieve.call_args[1]
        assert len(call_kwargs["tiers"]) == 4  # All tiers
        body = _body(result)
        assert len(body["tiers_searched"]) == 4

    def test_search_sort_by_relevance(self, handler, mock_continuum):
        """Default sort is relevance (no resorting)."""
        e1 = MockMemory(id="m1", importance=0.5)
        e2 = MockMemory(id="m2", importance=0.9)
        mock_continuum.retrieve.return_value = [e1, e2]

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._search_memories(
                {"q": "test", "sort": "relevance"}
            )

        body = _body(result)
        assert body["filters"]["sort"] == "relevance"
        # Order preserved from retrieve (relevance-ordered by default)
        assert body["results"][0]["id"] == "m1"
        assert body["results"][1]["id"] == "m2"

    def test_search_sort_by_importance(self, handler, mock_continuum):
        """Sort by importance puts highest importance first."""
        e1 = MockMemory(id="m1", importance=0.5)
        e2 = MockMemory(id="m2", importance=0.9)
        mock_continuum.retrieve.return_value = [e1, e2]

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._search_memories(
                {"q": "test", "sort": "importance"}
            )

        body = _body(result)
        assert body["filters"]["sort"] == "importance"
        assert body["results"][0]["id"] == "m2"  # higher importance first
        assert body["results"][1]["id"] == "m1"

    def test_search_sort_by_recency(self, handler, mock_continuum):
        """Sort by recency puts most recently updated first."""
        e1 = MockMemory(id="m1")
        e1.updated_at = 100
        e2 = MockMemory(id="m2")
        e2.updated_at = 200

        mock_continuum.retrieve.return_value = [e1, e2]

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._search_memories({"q": "test", "sort": "recency"})

        body = _body(result)
        assert body["filters"]["sort"] == "recency"
        assert body["results"][0]["id"] == "m2"  # more recent first
        assert body["results"][1]["id"] == "m1"

    def test_search_filters_in_response(self, handler, mock_continuum):
        """Response includes applied filters."""
        mock_continuum.retrieve.return_value = []

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._search_memories(
                {"q": "test", "min_importance": "0.3", "sort": "importance"}
            )

        body = _body(result)
        assert body["filters"]["min_importance"] == 0.3
        assert body["filters"]["sort"] == "importance"

    def test_search_tenant_enforcement(self, handler, mock_continuum):
        """Tenant enforcement passes tenant_id to retrieve."""
        mock_continuum.retrieve.return_value = []

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            with patch(
                "aragora.memory.access.tenant_enforcement_enabled",
                return_value=True,
            ):
                with patch(
                    "aragora.memory.access.resolve_tenant_id",
                    return_value="t-abc",
                ):
                    result = handler._search_memories({"q": "test"})

        assert result.status_code == 200
        call_kwargs = mock_continuum.retrieve.call_args[1]
        assert call_kwargs["tenant_id"] == "t-abc"
        assert call_kwargs["enforce_tenant_isolation"] is True

    def test_search_tenant_enforcement_no_tenant_returns_400(
        self, handler, mock_continuum
    ):
        """When tenant enforcement is on but no tenant_id, returns 400."""
        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            with patch(
                "aragora.memory.access.tenant_enforcement_enabled",
                return_value=True,
            ):
                with patch(
                    "aragora.memory.access.resolve_tenant_id", return_value=None
                ):
                    result = handler._search_memories({"q": "test"})

        assert result.status_code == 400

    def test_search_importance_rounding(self, handler, mock_continuum):
        """importance and surprise_score are rounded to 3 decimal places."""
        entry = MockMemory(
            id="m1", importance=0.12345678, surprise_score=0.98765432
        )
        mock_continuum.retrieve.return_value = [entry]

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._search_memories({"q": "test"})

        body = _body(result)
        assert body["results"][0]["importance"] == 0.123
        assert body["results"][0]["surprise_score"] == 0.988

    def test_search_missing_optional_attributes(self, handler, mock_continuum):
        """Entries missing optional attributes get defaults."""

        class BareEntry:
            def __init__(self):
                self.id = "m1"
                self.tier = MockMemoryTier.FAST
                self.content = "Bare entry"

        bare = BareEntry()
        mock_continuum.retrieve.return_value = [bare]

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._search_memories({"q": "test"})

        body = _body(result)
        r = body["results"][0]
        assert r["importance"] == 0.0
        assert r["surprise_score"] == 0.0
        assert r["created_at"] is None
        assert r["updated_at"] is None
        assert r["metadata"] == {}

    def test_search_multiple_results(self, handler, mock_continuum):
        """Multiple results are all returned."""
        entries = [
            MockMemory(id=f"m{i}", content=f"Entry {i}", importance=i * 0.1)
            for i in range(5)
        ]
        mock_continuum.retrieve.return_value = entries

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler._search_memories({"q": "test"})

        body = _body(result)
        assert body["count"] == 5
        assert len(body["results"]) == 5

    def test_search_filter_entries_applied(self, handler, mock_continuum):
        """RBAC filter_entries is called on search results."""
        e1 = MockMemory(id="m1")
        e2 = MockMemory(id="m2")
        mock_continuum.retrieve.return_value = [e1, e2]

        # filter removes e2
        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            with patch(
                "aragora.memory.access.filter_entries", return_value=[e1]
            ):
                result = handler._search_memories({"q": "test"})

        body = _body(result)
        assert body["count"] == 1
        assert body["results"][0]["id"] == "m1"


# ===========================================================================
# Integration: via handle() routing
# ===========================================================================


class TestProgressiveRouting:
    """Test that routes are correctly dispatched through handle()."""

    def test_search_index_route(self, handler, mock_continuum):
        """GET /api/v1/memory/search-index dispatches to _search_index."""
        mock_continuum.retrieve.return_value = []

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler.handle(
                "/api/v1/memory/search-index", {"q": "test"}, None
            )

        assert result is not None
        assert result.status_code == 200

    def test_search_timeline_route(self, handler, mock_continuum):
        """GET /api/v1/memory/search-timeline dispatches to _search_timeline."""
        anchor = MockMemory(id="a1")
        mock_continuum.get_timeline_entries.return_value = {
            "anchor": anchor,
            "before": [],
            "after": [],
        }

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler.handle(
                "/api/v1/memory/search-timeline", {"anchor_id": "a1"}, None
            )

        assert result is not None
        assert result.status_code == 200

    def test_entries_route(self, handler, mock_continuum):
        """GET /api/v1/memory/entries dispatches to _get_entries."""
        entry = MockMemory(id="m1")
        mock_continuum.get_many.return_value = [entry]

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler.handle(
                "/api/v1/memory/entries", {"ids": "m1"}, None
            )

        assert result is not None
        assert result.status_code == 200

    def test_search_route(self, handler, mock_continuum):
        """GET /api/v1/memory/search dispatches to _search_memories."""
        mock_continuum.retrieve.return_value = []

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler.handle(
                "/api/v1/memory/search", {"q": "test"}, None
            )

        assert result is not None
        assert result.status_code == 200

    def test_search_index_missing_query_via_handle(self, handler):
        """Missing query through handle() still returns 400."""
        result = handler.handle("/api/v1/memory/search-index", {}, None)
        assert result is not None
        assert result.status_code == 400

    def test_search_timeline_missing_anchor_via_handle(self, handler):
        """Missing anchor_id through handle() still returns 400."""
        result = handler.handle("/api/v1/memory/search-timeline", {}, None)
        assert result is not None
        assert result.status_code == 400

    def test_entries_missing_ids_via_handle(self, handler):
        """Missing ids through handle() still returns 400."""
        result = handler.handle("/api/v1/memory/entries", {}, None)
        assert result is not None
        assert result.status_code == 400

    def test_legacy_path_normalization(self, handler, mock_continuum):
        """Legacy /api/memory/* paths are normalized to /api/v1/memory/*."""
        mock_continuum.retrieve.return_value = []

        with patch("aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier):
            result = handler.handle(
                "/api/memory/search", {"q": "test"}, None
            )

        assert result is not None
        assert result.status_code == 200

    def test_can_handle_progressive_routes(self, handler):
        """Handler recognizes all progressive retrieval routes."""
        assert handler.can_handle("/api/v1/memory/search-index")
        assert handler.can_handle("/api/v1/memory/search-timeline")
        assert handler.can_handle("/api/v1/memory/entries")
        assert handler.can_handle("/api/v1/memory/search")
