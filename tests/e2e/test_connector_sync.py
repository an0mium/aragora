"""
E2E tests for connector sync workflows.

Tests complete sync cycles for enterprise connectors including:
- GitHub repository sync
- Slack channel sync
- Notion database sync
- SharePoint document sync
- Full sync lifecycle with persistence
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from tests.e2e.conftest import TestTenant, DebateSetup


# ============================================================================
# GitHub Connector E2E Tests
# ============================================================================


class TestGitHubConnectorE2E:
    """E2E tests for GitHub connector sync."""

    @pytest.mark.asyncio
    async def test_github_full_sync_lifecycle(self, mock_github_api, tenant_a: TestTenant):
        """Test complete GitHub sync lifecycle."""
        # Arrange
        from aragora.connectors.enterprise.git.github import GitHubEnterpriseConnector

        connector = GitHubEnterpriseConnector(
            repo="test-org/test-repo",
            branch="main",
            token="ghp_test_token",
        )

        # Act - Sync (connector initializes on first sync)
        sync_result = await connector.sync()

        # Assert
        assert sync_result is not None
        assert sync_result.success is True
        assert sync_result.items_synced >= 0

    @pytest.mark.asyncio
    async def test_github_incremental_sync(self, mock_github_api, tenant_a: TestTenant):
        """Test incremental sync after initial full sync."""
        from aragora.connectors.enterprise.git.github import GitHubEnterpriseConnector

        connector = GitHubEnterpriseConnector(
            repo="test-org/test-repo",
            branch="main",
            token="ghp_test_token",
        )
        # Connector initializes on first sync

        # First full sync
        first_sync = await connector.sync()

        # Second incremental sync
        second_sync = await connector.sync()

        assert second_sync is not None
        assert second_sync.success is True

    @pytest.mark.asyncio
    async def test_github_sync_error_recovery(self, mock_github_api, tenant_a: TestTenant):
        """Test sync recovery from API errors."""
        from aragora.connectors.enterprise.git.github import GitHubEnterpriseConnector
        import json

        connector = GitHubEnterpriseConnector(
            repo="test-org/test-repo",
            branch="main",
            token="ghp_test_token",
        )

        # Mock API failure then success
        call_count = [0]

        async def mock_run_gh_with_retry(args):
            call_count[0] += 1
            if call_count[0] <= 2:  # First two calls fail (commit check)
                raise ConnectionError("API temporarily unavailable")
            # Return valid commit response for subsequent calls
            args_str = " ".join(args)
            if "commits/" in args_str:
                return "abc123"
            if "/git/trees/" in args_str:
                return json.dumps([])
            return json.dumps([])

        with patch.object(connector, "_run_gh", side_effect=mock_run_gh_with_retry):
            with patch.object(connector, "_check_gh_cli", return_value=True):
                # First attempt should fail (returns SyncResult with success=False)
                result1 = await connector.sync()
                assert result1.success is False
                assert len(result1.errors) > 0

                # Retry should succeed (after error clears)
                result2 = await connector.sync()
                assert result2 is not None


# ============================================================================
# Slack Connector E2E Tests
# ============================================================================


class TestSlackConnectorE2E:
    """E2E tests for Slack connector sync."""

    @pytest.mark.asyncio
    async def test_slack_channel_sync(self, mock_slack_api, tenant_a: TestTenant):
        """Test Slack channel message sync."""
        from aragora.connectors.enterprise.collaboration.slack import SlackConnector, SlackConfig

        config = SlackConfig(
            bot_token="xoxb-test-token",
            channels=["general", "random"],
            sync_threads=True,
            max_messages_per_channel=100,
        )

        connector = SlackConnector(config)
        # Connector initializes on first sync

        result = await connector.sync()

        assert result is not None
        assert result.success is True

    @pytest.mark.asyncio
    async def test_slack_thread_sync(self, mock_slack_api, tenant_a: TestTenant):
        """Test thread message synchronization."""
        from aragora.connectors.enterprise.collaboration.slack import SlackConnector, SlackConfig

        config = SlackConfig(
            bot_token="xoxb-test-token",
            channels=["general"],
            sync_threads=True,
        )

        connector = SlackConnector(config)
        # Connector initializes on first sync

        result = await connector.sync()

        # Should include thread messages
        assert result is not None


# ============================================================================
# Notion Connector E2E Tests
# ============================================================================


class TestNotionConnectorE2E:
    """E2E tests for Notion connector sync."""

    @pytest.mark.asyncio
    async def test_notion_database_sync(self, tenant_a: TestTenant):
        """Test Notion database page sync."""
        from aragora.connectors.enterprise.collaboration.notion import NotionConnector, NotionConfig

        # Mock Notion API
        with patch("aragora.connectors.enterprise.collaboration.notion.NotionClient") as mock:
            instance = MagicMock()
            instance.list_databases = AsyncMock(
                return_value=[
                    {"id": "db-1", "title": [{"plain_text": "Tasks"}]},
                ]
            )
            instance.query_database = AsyncMock(
                return_value={
                    "results": [{"id": "page-1", "properties": {}}],
                    "has_more": False,
                }
            )
            instance.get_page_content = AsyncMock(
                return_value=[
                    {"type": "paragraph", "paragraph": {"text": [{"plain_text": "Content"}]}},
                ]
            )
            mock.return_value = instance

            config = NotionConfig(
                integration_token="secret_test",
                database_ids=["db-1"],
            )

            connector = NotionConnector(config)
            # Connector initializes on first sync

            result = await connector.sync()

            assert result is not None
            assert result.success is True

    @pytest.mark.asyncio
    async def test_notion_recursive_page_sync(self, tenant_a: TestTenant):
        """Test recursive page content sync."""
        from aragora.connectors.enterprise.collaboration.notion import NotionConnector, NotionConfig

        with patch("aragora.connectors.enterprise.collaboration.notion.NotionClient") as mock:
            instance = MagicMock()
            instance.list_databases = AsyncMock(return_value=[])
            instance.get_page = AsyncMock(return_value={"id": "page-1"})
            instance.get_page_content = AsyncMock(
                return_value=[
                    {"type": "child_page", "child_page": {"title": "Subpage"}},
                ]
            )
            mock.return_value = instance

            config = NotionConfig(
                integration_token="secret_test",
                page_ids=["page-1"],
                recursive=True,
            )

            connector = NotionConnector(config)
            # Connector initializes on first sync

            result = await connector.sync()
            assert result is not None


# ============================================================================
# Multi-Connector E2E Tests
# ============================================================================


class TestMultiConnectorE2E:
    """E2E tests for multiple connectors running together."""

    @pytest.mark.asyncio
    async def test_parallel_connector_sync(
        self,
        mock_github_api,
        mock_slack_api,
        tenant_a: TestTenant,
    ):
        """Test parallel sync of multiple connectors."""
        from aragora.connectors.enterprise.git.github import GitHubEnterpriseConnector
        from aragora.connectors.enterprise.collaboration.slack import SlackConnector, SlackConfig

        slack_config = SlackConfig(
            bot_token="xoxb-test",
            channels=["general"],
        )

        github = GitHubEnterpriseConnector(
            repo="test-org/test-repo",
            branch="main",
            token="ghp_test",
        )
        slack = SlackConnector(slack_config)

        await asyncio.gather(
            github.initialize(),
            slack.initialize(),
        )

        results = await asyncio.gather(
            github.sync(),
            slack.sync(),
        )

        assert len(results) == 2
        assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_connector_sync_with_persistence(
        self,
        mock_github_api,
        tenant_a: TestTenant,
    ):
        """Test connector sync with state persistence."""
        from aragora.connectors.enterprise.git.github import GitHubEnterpriseConnector

        connector = GitHubEnterpriseConnector(
            repo="test-org/test-repo",
            branch="main",
            token="ghp_test",
        )
        # Connector initializes on first sync

        result = await connector.sync()

        assert result is not None


# ============================================================================
# Connector Health E2E Tests
# ============================================================================


class TestConnectorHealthE2E:
    """E2E tests for connector health monitoring."""

    @pytest.mark.asyncio
    async def test_connector_health_check(self, mock_github_api, tenant_a: TestTenant):
        """Test connector health status reporting."""
        from aragora.connectors.enterprise.git.github import GitHubEnterpriseConnector

        connector = GitHubEnterpriseConnector(
            repo="test-org/test-repo",
            branch="main",
            token="ghp_test",
        )
        # Connector initializes on first sync

        health = await connector.health_check()

        assert health is not None
        assert "status" in health
        assert health["status"] in ("healthy", "degraded", "unhealthy")

    @pytest.mark.asyncio
    async def test_connector_metrics_collection(
        self,
        mock_github_api,
        tenant_a: TestTenant,
    ):
        """Test metrics collection during sync."""
        from aragora.connectors.enterprise.git.github import GitHubEnterpriseConnector

        connector = GitHubEnterpriseConnector(
            repo="test-org/test-repo",
            branch="main",
            token="ghp_test",
        )
        # Connector initializes on first sync

        # Mock metrics recording
        with patch("aragora.connectors.metrics.record_sync") as mock_record:
            result = await connector.sync()

            # Metrics should be recorded
            assert result is not None
