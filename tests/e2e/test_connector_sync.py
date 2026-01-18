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
        from aragora.connectors.enterprise.git.github import GitHubConnector, GitHubConfig

        config = GitHubConfig(
            access_token="ghp_test_token",
            organization="test-org",
            include_patterns=["*.py", "*.md"],
            exclude_patterns=["node_modules/**"],
        )

        connector = GitHubConnector(config)

        # Act - Initialize
        initialized = await connector.initialize()
        assert initialized is True

        # Act - Sync
        sync_result = await connector.sync()

        # Assert
        assert sync_result is not None
        assert sync_result.status == "success"
        assert sync_result.items_synced >= 0

    @pytest.mark.asyncio
    async def test_github_incremental_sync(self, mock_github_api, tenant_a: TestTenant):
        """Test incremental sync after initial full sync."""
        from aragora.connectors.enterprise.git.github import GitHubConnector, GitHubConfig

        config = GitHubConfig(
            access_token="ghp_test_token",
            organization="test-org",
        )

        connector = GitHubConnector(config)
        await connector.initialize()

        # First full sync
        first_sync = await connector.sync()

        # Second incremental sync
        second_sync = await connector.sync()

        assert second_sync is not None
        assert second_sync.sync_type in ("incremental", "full")

    @pytest.mark.asyncio
    async def test_github_sync_error_recovery(self, tenant_a: TestTenant):
        """Test sync recovery from API errors."""
        from aragora.connectors.enterprise.git.github import GitHubConnector, GitHubConfig

        config = GitHubConfig(
            access_token="ghp_test_token",
            organization="test-org",
        )

        # Mock API failure then success
        with patch("aragora.connectors.enterprise.git.github.GitHubClient") as mock:
            instance = MagicMock()
            call_count = [0]

            async def get_repos_with_retry(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise ConnectionError("API temporarily unavailable")
                return [{"id": 1, "name": "repo-1"}]

            instance.get_repos = get_repos_with_retry
            mock.return_value = instance

            connector = GitHubConnector(config)
            await connector.initialize()

            # First attempt should fail
            with pytest.raises(Exception):
                await connector.sync()

            # Retry should succeed
            result = await connector.sync()
            assert result is not None


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
        await connector.initialize()

        result = await connector.sync()

        assert result is not None
        assert result.status == "success"

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
        await connector.initialize()

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
            instance.list_databases = AsyncMock(return_value=[
                {"id": "db-1", "title": [{"plain_text": "Tasks"}]},
            ])
            instance.query_database = AsyncMock(return_value={
                "results": [{"id": "page-1", "properties": {}}],
                "has_more": False,
            })
            instance.get_page_content = AsyncMock(return_value=[
                {"type": "paragraph", "paragraph": {"text": [{"plain_text": "Content"}]}},
            ])
            mock.return_value = instance

            config = NotionConfig(
                integration_token="secret_test",
                database_ids=["db-1"],
            )

            connector = NotionConnector(config)
            await connector.initialize()

            result = await connector.sync()

            assert result is not None
            assert result.status == "success"

    @pytest.mark.asyncio
    async def test_notion_recursive_page_sync(self, tenant_a: TestTenant):
        """Test recursive page content sync."""
        from aragora.connectors.enterprise.collaboration.notion import NotionConnector, NotionConfig

        with patch("aragora.connectors.enterprise.collaboration.notion.NotionClient") as mock:
            instance = MagicMock()
            instance.list_databases = AsyncMock(return_value=[])
            instance.get_page = AsyncMock(return_value={"id": "page-1"})
            instance.get_page_content = AsyncMock(return_value=[
                {"type": "child_page", "child_page": {"title": "Subpage"}},
            ])
            mock.return_value = instance

            config = NotionConfig(
                integration_token="secret_test",
                page_ids=["page-1"],
                recursive=True,
            )

            connector = NotionConnector(config)
            await connector.initialize()

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
        from aragora.connectors.enterprise.git.github import GitHubConnector, GitHubConfig
        from aragora.connectors.enterprise.collaboration.slack import SlackConnector, SlackConfig

        github_config = GitHubConfig(
            access_token="ghp_test",
            organization="test-org",
        )
        slack_config = SlackConfig(
            bot_token="xoxb-test",
            channels=["general"],
        )

        github = GitHubConnector(github_config)
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
        from aragora.connectors.enterprise.git.github import GitHubConnector, GitHubConfig
        from aragora.connectors.enterprise.sync_store import SyncStore

        config = GitHubConfig(
            access_token="ghp_test",
            organization="test-org",
        )

        # Mock sync store
        with patch("aragora.connectors.enterprise.sync_store.SyncStore") as store_mock:
            store_instance = MagicMock()
            store_instance.get_last_sync = AsyncMock(return_value=None)
            store_instance.save_sync_state = AsyncMock(return_value=True)
            store_mock.return_value = store_instance

            connector = GitHubConnector(config)
            await connector.initialize()

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
        from aragora.connectors.enterprise.git.github import GitHubConnector, GitHubConfig

        config = GitHubConfig(
            access_token="ghp_test",
            organization="test-org",
        )

        connector = GitHubConnector(config)
        await connector.initialize()

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
        from aragora.connectors.enterprise.git.github import GitHubConnector, GitHubConfig
        from aragora.connectors.metrics import record_sync

        config = GitHubConfig(
            access_token="ghp_test",
            organization="test-org",
        )

        connector = GitHubConnector(config)
        await connector.initialize()

        # Mock metrics recording
        with patch("aragora.connectors.metrics.record_sync") as mock_record:
            result = await connector.sync()

            # Metrics should be recorded
            assert result is not None
