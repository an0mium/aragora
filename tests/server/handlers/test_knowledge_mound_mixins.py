"""Tests for Knowledge Mound operation mixins routing.

Tests validate that all 20 Knowledge Mound mixin endpoints are correctly
routed in the handler. This ensures the can_handle() method recognizes
all Phase A2 Knowledge Mound features:
- Extraction operations (extract from debate, promote claims)
- Curation operations (policy, quality scores)
- Pruning operations (get prunable, execute, auto-prune)
- Contradiction detection (detect, resolve, stats)
- Governance (roles, permissions, audit)
- Analytics (coverage, usage, quality trends)
- Deduplication (clusters, merge)
- Staleness (get stale items, revalidate)
- Confidence decay (apply decay)
- Export (D3, GraphML)
- Sync (continuum, consensus, facts)

Note: Full implementation tests require integration with actual KnowledgeMound
backend. These routing tests validate the handler recognizes all endpoints.
"""

import pytest

from aragora.server.handlers.knowledge_base.mound.handler import KnowledgeMoundHandler


@pytest.fixture
def mound_handler():
    """Create a knowledge mound handler with mocked dependencies."""
    ctx = {"storage": None, "elo_system": None, "nomic_dir": None}
    handler = KnowledgeMoundHandler(ctx)
    return handler


# =============================================================================
# Extraction Mixin Tests
# =============================================================================


class TestExtractionCanHandle:
    """Test can_handle for extraction endpoints."""

    def test_can_handle_extract_debate(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/extraction/debate")

    def test_can_handle_promote(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/extraction/promote")

    def test_can_handle_extraction_stats(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/extraction/stats")


# =============================================================================
# Curation Mixin Tests
# =============================================================================


class TestCurationCanHandle:
    """Test can_handle for curation endpoints."""

    def test_can_handle_policy(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/curation/policy")

    def test_can_handle_status(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/curation/status")

    def test_can_handle_run(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/curation/run")


# =============================================================================
# Pruning Mixin Tests
# =============================================================================


class TestPruningCanHandle:
    """Test can_handle for pruning endpoints."""

    def test_can_handle_items(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/pruning/items")

    def test_can_handle_execute(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/pruning/execute")

    def test_can_handle_auto(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/pruning/auto")

    def test_can_handle_restore(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/pruning/restore")

    def test_can_handle_decay(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/pruning/decay")


# =============================================================================
# Contradiction Mixin Tests
# =============================================================================


class TestContradictionCanHandle:
    """Test can_handle for contradiction endpoints."""

    def test_can_handle_detect(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/contradictions/detect")

    def test_can_handle_list(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/contradictions")

    def test_can_handle_resolve(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/contradictions/contra-1/resolve")

    def test_can_handle_stats(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/contradictions/stats")


# =============================================================================
# Governance Mixin Tests
# =============================================================================


class TestGovernanceCanHandle:
    """Test can_handle for governance endpoints."""

    def test_can_handle_roles(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/governance/roles")

    def test_can_handle_permissions(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/governance/permissions/user-123")

    def test_can_handle_audit(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/governance/audit")


# =============================================================================
# Analytics Mixin Tests
# =============================================================================


class TestAnalyticsCanHandle:
    """Test can_handle for analytics endpoints."""

    def test_can_handle_coverage(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/analytics/coverage")

    def test_can_handle_usage(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/analytics/usage")

    def test_can_handle_quality_trend(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/analytics/quality/trend")


# =============================================================================
# Deduplication Mixin Tests
# =============================================================================


class TestDedupCanHandle:
    """Test can_handle for deduplication endpoints."""

    def test_can_handle_clusters(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/dedup/clusters")

    def test_can_handle_merge(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/dedup/merge")

    def test_can_handle_auto_merge(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/dedup/auto-merge")


# =============================================================================
# Staleness Mixin Tests
# =============================================================================


class TestStalenessCanHandle:
    """Test can_handle for staleness endpoints."""

    def test_can_handle_stale(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/stale")

    def test_can_handle_revalidate(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/revalidate/node-123")

    def test_can_handle_schedule(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/schedule-revalidation")


# =============================================================================
# Export Mixin Tests
# =============================================================================


class TestExportCanHandle:
    """Test can_handle for export endpoints."""

    def test_can_handle_d3(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/export/d3")

    def test_can_handle_graphml(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/export/graphml")


# =============================================================================
# Sync Mixin Tests
# =============================================================================


class TestSyncCanHandle:
    """Test can_handle for sync endpoints."""

    def test_can_handle_sync_continuum(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/sync/continuum")

    def test_can_handle_sync_consensus(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/sync/consensus")

    def test_can_handle_sync_facts(self, mound_handler):
        assert mound_handler.can_handle("/api/v1/knowledge/mound/sync/facts")
