"""Tests for Knowledge Mound prometheus metrics."""

import pytest

# Import from main prometheus module to avoid circular import
from aragora.server.prometheus import (
    record_knowledge_access_grant,
    record_knowledge_federation_sync,
    record_knowledge_global_fact,
    record_knowledge_global_query,
    record_knowledge_share,
    record_knowledge_visibility_change,
    set_knowledge_federation_regions,
    set_knowledge_shared_items,
    timed_knowledge_federation_sync,
)


class TestVisibilityMetrics:
    """Tests for visibility change metrics."""

    def test_record_visibility_change(self):
        """Should record visibility level changes."""
        # Should not raise
        record_knowledge_visibility_change(
            from_level="workspace",
            to_level="public",
            workspace_id="ws-123",
        )

    def test_record_visibility_change_all_levels(self):
        """Should handle all visibility levels."""
        levels = ["private", "workspace", "organization", "public", "system"]
        for from_level in levels:
            for to_level in levels:
                if from_level != to_level:
                    record_knowledge_visibility_change(
                        from_level=from_level,
                        to_level=to_level,
                        workspace_id="test-ws",
                    )


class TestAccessGrantMetrics:
    """Tests for access grant metrics."""

    def test_record_grant_action(self):
        """Should record grant actions."""
        record_knowledge_access_grant(
            action="grant",
            grantee_type="user",
            workspace_id="ws-123",
        )

    def test_record_revoke_action(self):
        """Should record revoke actions."""
        record_knowledge_access_grant(
            action="revoke",
            grantee_type="workspace",
            workspace_id="ws-456",
        )

    def test_all_grantee_types(self):
        """Should handle all grantee types."""
        grantee_types = ["user", "role", "workspace", "organization"]
        for grantee_type in grantee_types:
            record_knowledge_access_grant(
                action="grant",
                grantee_type=grantee_type,
                workspace_id="test-ws",
            )


class TestSharingMetrics:
    """Tests for sharing metrics."""

    def test_record_share_action(self):
        """Should record share actions."""
        record_knowledge_share(action="share", target_type="workspace")

    def test_record_accept_action(self):
        """Should record accept actions."""
        record_knowledge_share(action="accept", target_type="user")

    def test_record_decline_action(self):
        """Should record decline actions."""
        record_knowledge_share(action="decline", target_type="workspace")

    def test_record_revoke_share(self):
        """Should record revoke share actions."""
        record_knowledge_share(action="revoke", target_type="user")

    def test_set_shared_items_count(self):
        """Should set shared items count."""
        set_knowledge_shared_items(workspace_id="ws-123", count=5)
        set_knowledge_shared_items(workspace_id="ws-123", count=0)


class TestGlobalKnowledgeMetrics:
    """Tests for global knowledge metrics."""

    def test_record_stored_fact(self):
        """Should record stored facts."""
        record_knowledge_global_fact(action="stored")

    def test_record_promoted_fact(self):
        """Should record promoted facts."""
        record_knowledge_global_fact(action="promoted")

    def test_record_queried_fact(self):
        """Should record queried facts."""
        record_knowledge_global_fact(action="queried")

    def test_record_query_with_results(self):
        """Should record queries with results."""
        record_knowledge_global_query(has_results=True)

    def test_record_query_without_results(self):
        """Should record queries without results."""
        record_knowledge_global_query(has_results=False)


class TestFederationMetrics:
    """Tests for federation metrics."""

    def test_record_push_sync(self):
        """Should record push sync operations."""
        record_knowledge_federation_sync(
            region_id="region-1",
            direction="push",
            status="success",
            nodes_synced=42,
            duration_seconds=1.5,
        )

    def test_record_pull_sync(self):
        """Should record pull sync operations."""
        record_knowledge_federation_sync(
            region_id="region-2",
            direction="pull",
            status="success",
            nodes_synced=10,
        )

    def test_record_failed_sync(self):
        """Should record failed sync operations."""
        record_knowledge_federation_sync(
            region_id="region-1",
            direction="push",
            status="failed",
            duration_seconds=0.5,
        )

    def test_set_federation_regions(self):
        """Should set federation region counts."""
        set_knowledge_federation_regions(
            enabled=3,
            disabled=1,
            healthy=2,
            unhealthy=1,
        )

    def test_set_federation_regions_defaults(self):
        """Should accept default values."""
        set_knowledge_federation_regions()


class TestTimedFederationSync:
    """Tests for timed federation sync context manager."""

    def test_successful_sync(self):
        """Should track successful sync with timing."""
        with timed_knowledge_federation_sync("region-1", "push") as ctx:
            ctx["nodes_synced"] = 25
            ctx["status"] = "success"
        # Should not raise

    def test_failed_sync_raises(self):
        """Should track failed sync and re-raise exception."""
        with pytest.raises(ValueError):
            with timed_knowledge_federation_sync("region-2", "pull") as ctx:
                ctx["nodes_synced"] = 0
                raise ValueError("Sync failed")

    def test_default_status(self):
        """Should default to success status."""
        with timed_knowledge_federation_sync("region-1", "push") as ctx:
            ctx["nodes_synced"] = 10
        # ctx["status"] defaults to "success"

    def test_context_values(self):
        """Should provide context dict to populate."""
        with timed_knowledge_federation_sync("region-1", "push") as ctx:
            assert "status" in ctx
            assert "nodes_synced" in ctx
            ctx["nodes_synced"] = 100


class TestMetricLabels:
    """Tests for metric label handling."""

    def test_special_characters_in_workspace_id(self):
        """Should handle special characters in workspace IDs."""
        record_knowledge_visibility_change(
            from_level="private",
            to_level="public",
            workspace_id="ws/special-chars_123",
        )

    def test_empty_strings(self):
        """Should handle empty strings."""
        record_knowledge_share(action="", target_type="")

    def test_unicode_values(self):
        """Should handle unicode values."""
        record_knowledge_visibility_change(
            from_level="private",
            to_level="public",
            workspace_id="ws-unicode-\u00e9\u00e8\u00ea",
        )
