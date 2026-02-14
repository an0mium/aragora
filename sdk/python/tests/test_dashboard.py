"""Tests for Dashboard namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestDashboardOverview:
    """Tests for dashboard overview methods."""

    def test_get_overview(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"inbox": {"total_unread": 5}}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.dashboard.get_overview()
            mock_request.assert_called_once_with("GET", "/api/v1/dashboard", params=None)
            assert result["inbox"]["total_unread"] == 5
            client.close()

    def test_get_overview_with_refresh(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"inbox": {"total_unread": 3}}
            client = AragoraClient(base_url="https://api.aragora.ai")
            client.dashboard.get_overview(refresh=True)
            mock_request.assert_called_once_with(
                "GET", "/api/v1/dashboard", params={"refresh": True}
            )
            client.close()

    def test_get_stats(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"charts": []}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.dashboard.get_stats(period="month")
            mock_request.assert_called_once_with(
                "GET", "/api/v1/dashboard/stats", params={"period": "month"}
            )
            assert result == {"charts": []}
            client.close()

    def test_get_inbox_summary(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"total": 42, "urgent": 3}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.dashboard.get_inbox_summary()
            mock_request.assert_called_once_with("GET", "/api/v1/dashboard/inbox-summary")
            assert result["total"] == 42
            client.close()

    def test_get_stat_cards(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"cards": [{"label": "Debates", "value": 150}]}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.dashboard.get_stat_cards()
            mock_request.assert_called_once_with("GET", "/api/v1/dashboard/stat-cards")
            assert result["cards"][0]["label"] == "Debates"
            client.close()

class TestDashboardActivity:
    """Tests for activity and quick actions."""

    def test_get_activity(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"items": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.dashboard.get_activity()
            mock_request.assert_called_once_with("GET", "/api/v1/dashboard/activity", params=None)
            assert result["total"] == 0
            client.close()

    def test_get_activity_with_filters(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"items": []}
            client = AragoraClient(base_url="https://api.aragora.ai")
            client.dashboard.get_activity(limit=10, offset=5, activity_type="debate")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/dashboard/activity",
                params={"limit": 10, "offset": 5, "type": "debate"},
            )
            client.close()

    def test_get_recent_activity_convenience(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"items": [{"id": "a1"}]}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.dashboard.get_recent_activity(limit=5)
            mock_request.assert_called_once_with(
                "GET", "/api/v1/dashboard/activity", params={"limit": 5}
            )
            assert len(result["items"]) == 1
            client.close()

    def test_get_quick_actions(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"actions": [{"id": "archive_old"}]}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.dashboard.get_quick_actions()
            mock_request.assert_called_once_with("GET", "/api/v1/dashboard/quick-actions")
            assert result["actions"][0]["id"] == "archive_old"
            client.close()

class TestDashboardDebates:
    """Tests for debate listing methods."""

    def test_list_debates(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"debates": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.dashboard.list_debates()
            mock_request.assert_called_once_with("GET", "/api/v1/dashboard/debates", params=None)
            assert result["total"] == 0
            client.close()

    def test_list_debates_with_filters(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"debates": [{"id": "d1"}]}
            client = AragoraClient(base_url="https://api.aragora.ai")
            client.dashboard.list_debates(
                status="active",
                start_date="2025-01-01",
                end_date="2025-01-31",
                limit=10,
                offset=0,
            )
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/dashboard/debates",
                params={
                    "status": "active",
                    "start_date": "2025-01-01",
                    "end_date": "2025-01-31",
                    "limit": 10,
                    "offset": 0,
                },
            )
            client.close()

class TestDashboardTeamPerformance:
    """Tests for team performance methods."""

    def test_get_team_performance(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"teams": []}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.dashboard.get_team_performance()
            mock_request.assert_called_once_with(
                "GET", "/api/v1/dashboard/team-performance", params=None
            )
            assert result["teams"] == []
            client.close()

    def test_get_team_performance_with_filters(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"teams": [{"id": "t1"}]}
            client = AragoraClient(base_url="https://api.aragora.ai")
            client.dashboard.get_team_performance(
                sort_by="win_rate", sort_order="desc", min_debates=5, limit=10
            )
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/dashboard/team-performance",
                params={
                    "sort_by": "win_rate",
                    "sort_order": "desc",
                    "min_debates": 5,
                    "limit": 10,
                },
            )
            client.close()

class TestDashboardEmailAnalytics:
    """Tests for email analytics methods."""

    def test_get_top_senders(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"senders": []}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.dashboard.get_top_senders()
            mock_request.assert_called_once_with(
                "GET", "/api/v1/dashboard/top-senders", params=None
            )
            assert result["senders"] == []
            client.close()

    def test_get_top_senders_with_filters(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"senders": [{"email": "a@b.com"}]}
            client = AragoraClient(base_url="https://api.aragora.ai")
            client.dashboard.get_top_senders(
                domain="example.com", min_messages=10, sort_by="count", limit=5
            )
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/dashboard/top-senders",
                params={
                    "domain": "example.com",
                    "min_messages": 10,
                    "sort_by": "count",
                    "limit": 5,
                },
            )
            client.close()

    def test_get_labels(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"labels": [{"name": "inbox", "count": 42}]}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.dashboard.get_labels()
            mock_request.assert_called_once_with("GET", "/api/v1/dashboard/labels")
            assert result["labels"][0]["name"] == "inbox"
            client.close()

class TestDashboardUrgentItems:
    """Tests for urgent items and pending actions."""

    def test_get_urgent_items(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"items": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.dashboard.get_urgent_items()
            mock_request.assert_called_once_with("GET", "/api/v1/dashboard/urgent", params=None)
            assert result["total"] == 0
            client.close()

    def test_get_urgent_items_with_filters(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"items": [{"id": "u1"}]}
            client = AragoraClient(base_url="https://api.aragora.ai")
            client.dashboard.get_urgent_items(
                action_type="review",
                min_importance=8,
                include_deadline_passed=True,
                limit=5,
            )
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/dashboard/urgent",
                params={
                    "action_type": "review",
                    "min_importance": 8,
                    "include_deadline_passed": True,
                    "limit": 5,
                },
            )
            client.close()

    def test_get_pending_actions(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"actions": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.dashboard.get_pending_actions()
            mock_request.assert_called_once_with(
                "GET", "/api/v1/dashboard/pending-actions", params=None
            )
            assert result["total"] == 0
            client.close()

    def test_get_pending_actions_with_pagination(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"actions": [{"id": "a1"}]}
            client = AragoraClient(base_url="https://api.aragora.ai")
            client.dashboard.get_pending_actions(limit=10, offset=20)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/dashboard/pending-actions",
                params={"limit": 10, "offset": 20},
            )
            client.close()

class TestDashboardSearchExport:
    """Tests for search and export methods."""

    def test_search(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"results": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.dashboard.search("rate limiter")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/dashboard/search",
                params={"query": "rate limiter"},
            )
            assert result["total"] == 0
            client.close()

    def test_search_with_filters(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"results": [{"id": "r1"}]}
            client = AragoraClient(base_url="https://api.aragora.ai")
            client.dashboard.search("rate limiter", types=["debate", "action"], limit=5)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/dashboard/search",
                params={"query": "rate limiter", "types": "debate,action", "limit": 5},
            )
            client.close()

    def test_export_data(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"url": "https://example.com/export.csv"}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.dashboard.export_data("csv")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/dashboard/export",
                json={"format": "csv"},
            )
            assert "url" in result
            client.close()

    def test_export_data_with_options(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"url": "https://example.com/export.json"}
            client = AragoraClient(base_url="https://api.aragora.ai")
            client.dashboard.export_data(
                "json",
                include=["debates", "actions"],
                start_date="2025-01-01",
                end_date="2025-01-31",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/dashboard/export",
                json={
                    "format": "json",
                    "include": ["debates", "actions"],
                    "start_date": "2025-01-01",
                    "end_date": "2025-01-31",
                },
            )
            client.close()

class TestAsyncDashboard:
    """Tests for async dashboard methods."""

    @pytest.mark.asyncio
    async def test_get_overview(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"inbox": {"total_unread": 5}}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.dashboard.get_overview()
            mock_request.assert_called_once_with("GET", "/api/v1/dashboard", params=None)
            assert result["inbox"]["total_unread"] == 5
            await client.close()

    @pytest.mark.asyncio
    async def test_list_debates(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"debates": []}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.dashboard.list_debates(status="active")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/dashboard/debates",
                params={"status": "active"},
            )
            assert result["debates"] == []
            await client.close()

    @pytest.mark.asyncio
    async def test_get_stat_cards(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"cards": []}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.dashboard.get_stat_cards()
            mock_request.assert_called_once_with("GET", "/api/v1/dashboard/stat-cards")
            assert result["cards"] == []
            await client.close()

    @pytest.mark.asyncio
    async def test_get_team_performance(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"teams": []}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.dashboard.get_team_performance(sort_by="win_rate")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/dashboard/team-performance",
                params={"sort_by": "win_rate"},
            )
            assert result["teams"] == []
            await client.close()

    @pytest.mark.asyncio
    async def test_get_urgent_items(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"items": []}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.dashboard.get_urgent_items(min_importance=5)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/dashboard/urgent",
                params={"min_importance": 5},
            )
            assert result["items"] == []
            await client.close()

    @pytest.mark.asyncio
    async def test_search(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"results": []}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.dashboard.search("test query")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/dashboard/search",
                params={"query": "test query"},
            )
            assert result["results"] == []
            await client.close()

    @pytest.mark.asyncio
    async def test_export_data(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"url": "https://example.com/export.pdf"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.dashboard.export_data("pdf")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/dashboard/export",
                json={"format": "pdf"},
            )
            assert "url" in result
            await client.close()

    @pytest.mark.asyncio
    async def test_get_recent_activity(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"items": []}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.dashboard.get_recent_activity()
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/dashboard/activity",
                params={"limit": 20},
            )
            assert result["items"] == []
            await client.close()
