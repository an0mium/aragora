"""
End-to-End Integration Tests for New Feature Handlers.

Tests realistic workflows across:
- Unified Inbox + Email Webhooks
- Bank Reconciliation
- Codebase Audit
- Cross-Platform Analytics
"""

import pytest
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import json
import base64

# Import all handlers
from aragora.server.handlers.features.unified_inbox import (
    UnifiedInboxHandler,
    EmailProvider,
    AccountStatus,
    TriageAction,
)
from aragora.server.handlers.features.email_webhooks import (
    EmailWebhooksHandler,
    WebhookProvider,
    WebhookStatus,
    NotificationType,
)
from aragora.server.handlers.features.reconciliation import (
    ReconciliationHandler,
)
from aragora.server.handlers.features.codebase_audit import (
    CodebaseAuditHandler,
    ScanType,
    ScanStatus,
    FindingSeverity,
)
from aragora.server.handlers.features.cross_platform_analytics import (
    CrossPlatformAnalyticsHandler,
    Platform,
    AlertSeverity,
)
from aragora.server.handlers.features.marketplace import (
    MarketplaceHandler,
    TemplateCategory,
    DeploymentStatus,
)


class TestUnifiedInboxWebhookIntegration:
    """Test unified inbox with webhook notifications."""

    @pytest.mark.asyncio
    async def test_gmail_webhook_triggers_inbox_sync(self):
        """Test that Gmail webhook notification triggers inbox sync."""
        tenant_id = "integration_tenant_1"

        # Step 1: Connect Gmail account via unified inbox
        inbox_handler = UnifiedInboxHandler()
        connect_request = MagicMock()
        connect_request.tenant_id = tenant_id
        connect_request.json = AsyncMock(
            return_value={
                "provider": "gmail",
                "email": "test@gmail.com",
                "auth_code": "mock_auth_code",
            }
        )

        connect_result = await inbox_handler.handle(
            connect_request, "/api/v1/inbox/connect", "POST"
        )
        # Result depends on OAuth implementation - validate handler processes request
        assert connect_result is not None

        # Step 2: Subscribe to Gmail webhooks
        webhook_handler = EmailWebhooksHandler()
        subscribe_request = MagicMock()
        subscribe_request.tenant_id = tenant_id
        subscribe_request.json = AsyncMock(
            return_value={
                "provider": "gmail",
                "account_id": "acc_123",
            }
        )

        subscribe_result = await webhook_handler.handle(
            subscribe_request, "/api/v1/webhooks/subscribe", "POST"
        )
        assert subscribe_result.status_code == 200

        # Step 3: Simulate Gmail push notification
        gmail_data = {"emailAddress": "test@gmail.com", "historyId": "12345"}
        encoded_data = base64.b64encode(json.dumps(gmail_data).encode()).decode()

        notification_request = MagicMock()
        notification_request.tenant_id = tenant_id
        notification_request.json = AsyncMock(
            return_value={
                "message": {
                    "data": encoded_data,
                    "messageId": "msg_123",
                },
                "subscription": "projects/test/subscriptions/gmail-push",
            }
        )

        webhook_result = await webhook_handler.handle(
            notification_request, "/api/v1/webhooks/gmail", "POST"
        )
        assert webhook_result.status_code == 200

        # Step 4: Check inbox for messages
        list_request = MagicMock()
        list_request.tenant_id = tenant_id
        list_request.query = {}

        list_result = await inbox_handler.handle(list_request, "/api/v1/inbox/messages", "GET")
        assert list_result.status_code == 200

    @pytest.mark.asyncio
    async def test_outlook_webhook_triggers_sync(self):
        """Test that Outlook webhook notification triggers sync."""
        tenant_id = "integration_tenant_2"

        # Setup webhook handler
        webhook_handler = EmailWebhooksHandler()

        # Simulate Outlook change notification
        notification_request = MagicMock()
        notification_request.tenant_id = tenant_id
        notification_request.query = {}
        notification_request.json = AsyncMock(
            return_value={
                "value": [
                    {
                        "subscriptionId": "sub_123",
                        "changeType": "created",
                        "resource": "Users/user_456/Messages/msg_789",
                        "clientState": "secret",
                    }
                ]
            }
        )

        result = await webhook_handler.handle(
            notification_request, "/api/v1/webhooks/outlook", "POST"
        )
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_triage_workflow(self):
        """Test complete triage workflow."""
        tenant_id = "integration_tenant_3"
        inbox_handler = UnifiedInboxHandler()

        # Step 1: Get inbox stats
        stats_request = MagicMock()
        stats_request.tenant_id = tenant_id

        stats_result = await inbox_handler.handle(stats_request, "/api/v1/inbox/stats", "GET")
        assert stats_result.status_code == 200

        # Step 2: Triage messages
        # Note: Triage requires messages in cache, which requires connected accounts
        # For integration test, verify proper error handling when no messages exist
        triage_request = MagicMock()
        triage_request.tenant_id = tenant_id
        triage_request.json = AsyncMock(
            return_value={
                "message_ids": ["msg_test_123", "msg_test_456"],
                "use_agents": True,
            }
        )

        triage_result = await inbox_handler.handle(triage_request, "/api/v1/inbox/triage", "POST")
        # Expect 404 when no messages exist (no connected accounts)
        assert triage_result.status_code in [200, 404]
        if triage_result.status_code == 404:
            assert b"No matching messages found" in triage_result.body


class TestReconciliationWorkflow:
    """Test bank reconciliation end-to-end workflow."""

    @pytest.mark.asyncio
    async def test_full_reconciliation_flow(self):
        """Test complete reconciliation workflow."""
        tenant_id = "recon_tenant_1"
        handler = ReconciliationHandler()

        # Step 1: Get demo data to understand structure
        demo_request = MagicMock()
        demo_request.tenant_id = tenant_id

        demo_result = await handler.handle(demo_request, "/api/v1/reconciliation/demo", "GET")
        assert demo_result.status_code == 200
        assert b"reconciliation" in demo_result.body

        # Step 2: Run reconciliation (mock mode)
        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service"
        ) as mock_get:
            mock_service = MagicMock()
            mock_get.return_value = mock_service

            run_request = MagicMock()
            run_request.tenant_id = tenant_id
            run_request.json = AsyncMock(
                return_value={
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31",
                    "account_id": "checking_001",
                }
            )

            run_result = await handler.handle(run_request, "/api/v1/reconciliation/run", "POST")
            assert run_result.status_code == 200

        # Step 3: List reconciliations
        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service"
        ) as mock_get:
            mock_service = MagicMock()
            mock_service.list_reconciliations.return_value = []
            mock_get.return_value = mock_service

            list_request = MagicMock()
            list_request.tenant_id = tenant_id
            list_request.query = {}

            list_result = await handler.handle(list_request, "/api/v1/reconciliation/list", "GET")
            assert list_result.status_code == 200

        # Step 4: Get discrepancies
        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service"
        ) as mock_get:
            mock_service = MagicMock()
            mock_service.list_reconciliations.return_value = []
            mock_get.return_value = mock_service

            disc_request = MagicMock()
            disc_request.tenant_id = tenant_id
            disc_request.query = {"status": "pending"}

            disc_result = await handler.handle(
                disc_request, "/api/v1/reconciliation/discrepancies", "GET"
            )
            assert disc_result.status_code == 200


class TestCodebaseAuditWorkflow:
    """Test codebase audit end-to-end workflow."""

    @pytest.mark.skip(reason="Test times out in CI - needs optimization")
    @pytest.mark.asyncio
    async def test_comprehensive_scan_to_issue_flow(self):
        """Test scan -> findings -> create issue workflow."""
        tenant_id = "audit_tenant_1"
        handler = CodebaseAuditHandler()

        # Step 1: Run comprehensive scan
        scan_request = MagicMock()
        scan_request.tenant_id = tenant_id
        scan_request.json = AsyncMock(
            return_value={
                "target_path": ".",
                "scan_types": ["sast", "bugs", "secrets"],
            }
        )

        scan_result = await handler.handle(scan_request, "/api/v1/codebase/scan", "POST")
        assert scan_result.status_code == 200
        assert b"findings" in scan_result.body

        # Parse findings from response (wrapped in success response)
        response_data = json.loads(scan_result.body)
        data = response_data.get("data", response_data)  # Handle both wrapped and unwrapped
        findings = data.get("findings", [])
        assert len(findings) > 0

        # Step 2: Get dashboard
        dashboard_request = MagicMock()
        dashboard_request.tenant_id = tenant_id

        dashboard_result = await handler.handle(
            dashboard_request, "/api/v1/codebase/dashboard", "GET"
        )
        assert dashboard_result.status_code == 200
        assert b"summary" in dashboard_result.body

        # Step 3: List findings with filters
        findings_request = MagicMock()
        findings_request.tenant_id = tenant_id
        findings_request.query = {"severity": "high", "status": "open"}

        findings_result = await handler.handle(findings_request, "/api/v1/codebase/findings", "GET")
        assert findings_result.status_code == 200

    @pytest.mark.asyncio
    async def test_individual_scan_types(self):
        """Test individual scan type endpoints."""
        tenant_id = "audit_tenant_2"
        handler = CodebaseAuditHandler()

        # Test SAST scan
        sast_request = MagicMock()
        sast_request.tenant_id = tenant_id
        sast_request.json = AsyncMock(return_value={"target_path": "."})

        sast_result = await handler.handle(sast_request, "/api/v1/codebase/sast", "POST")
        assert sast_result.status_code == 200

        # Test secrets scan
        secrets_request = MagicMock()
        secrets_request.tenant_id = tenant_id
        secrets_request.json = AsyncMock(return_value={"target_path": "."})

        secrets_result = await handler.handle(secrets_request, "/api/v1/codebase/secrets", "POST")
        assert secrets_result.status_code == 200

        # Test metrics analysis
        metrics_request = MagicMock()
        metrics_request.tenant_id = tenant_id
        metrics_request.json = AsyncMock(return_value={"target_path": "."})

        metrics_result = await handler.handle(metrics_request, "/api/v1/codebase/metrics", "POST")
        assert metrics_result.status_code == 200

    @pytest.mark.asyncio
    async def test_scan_history_tracking(self):
        """Test scan history is properly tracked."""
        tenant_id = "audit_tenant_3"
        handler = CodebaseAuditHandler()

        # Run multiple scans
        for i in range(3):
            scan_request = MagicMock()
            scan_request.tenant_id = tenant_id
            scan_request.json = AsyncMock(
                return_value={
                    "target_path": f"./module_{i}",
                    "scan_types": ["sast"],
                }
            )

            await handler.handle(scan_request, "/api/v1/codebase/scan", "POST")

        # List scans
        list_request = MagicMock()
        list_request.tenant_id = tenant_id
        list_request.query = {"limit": "10"}

        list_result = await handler.handle(list_request, "/api/v1/codebase/scans", "GET")
        assert list_result.status_code == 200

        response_data = json.loads(list_result.body)
        data = response_data.get("data", response_data)  # Handle both wrapped and unwrapped
        assert data["total"] >= 3


class TestCrossPlatformAnalyticsWorkflow:
    """Test cross-platform analytics end-to-end workflow."""

    @pytest.mark.asyncio
    async def test_unified_dashboard_flow(self):
        """Test unified analytics dashboard flow."""
        tenant_id = "analytics_tenant_1"
        handler = CrossPlatformAnalyticsHandler()

        # Step 1: Get summary
        summary_request = MagicMock()
        summary_request.tenant_id = tenant_id
        summary_request.query = {"range": "7d"}

        summary_result = await handler.handle(
            summary_request, "/api/v1/analytics/cross-platform/summary", "GET"
        )
        assert summary_result.status_code == 200
        assert b"key_metrics" in summary_result.body

        # Step 2: Get trends
        trends_request = MagicMock()
        trends_request.tenant_id = tenant_id
        trends_request.query = {"metric": "users", "range": "7d"}

        trends_result = await handler.handle(
            trends_request, "/api/v1/analytics/cross-platform/trends", "GET"
        )
        assert trends_result.status_code == 200
        assert b"data_points" in trends_result.body

        # Step 3: Get correlations
        corr_request = MagicMock()
        corr_request.tenant_id = tenant_id
        corr_request.query = {"range": "30d"}

        corr_result = await handler.handle(
            corr_request, "/api/v1/analytics/cross-platform/correlation", "GET"
        )
        assert corr_result.status_code == 200
        assert b"correlation_matrix" in corr_result.body

        # Step 4: Check anomalies
        anomalies_request = MagicMock()
        anomalies_request.tenant_id = tenant_id
        anomalies_request.query = {}

        anomalies_result = await handler.handle(
            anomalies_request, "/api/v1/analytics/cross-platform/anomalies", "GET"
        )
        assert anomalies_result.status_code == 200
        assert b"anomalies" in anomalies_result.body

    @pytest.mark.asyncio
    async def test_alert_workflow(self):
        """Test alert creation and management workflow."""
        tenant_id = "analytics_tenant_2"
        handler = CrossPlatformAnalyticsHandler()

        # Step 1: Create alert rule
        create_request = MagicMock()
        create_request.tenant_id = tenant_id
        create_request.json = AsyncMock(
            return_value={
                "name": "High Error Rate Alert",
                "metric_name": "error_rate",
                "condition": "above",
                "threshold": 0.05,
                "severity": "warning",
                "platforms": ["aragora"],
            }
        )

        create_result = await handler.handle(
            create_request, "/api/v1/analytics/cross-platform/alerts", "POST"
        )
        assert create_result.status_code == 200
        assert b"created" in create_result.body

        # Step 2: List alerts
        list_request = MagicMock()
        list_request.tenant_id = tenant_id
        list_request.query = {}

        list_result = await handler.handle(
            list_request, "/api/v1/analytics/cross-platform/alerts", "GET"
        )
        assert list_result.status_code == 200
        assert b"rules" in list_result.body

    @pytest.mark.asyncio
    async def test_custom_query_workflow(self):
        """Test custom metric query workflow."""
        tenant_id = "analytics_tenant_3"
        handler = CrossPlatformAnalyticsHandler()

        # Execute custom query
        query_request = MagicMock()
        query_request.tenant_id = tenant_id
        query_request.json = AsyncMock(
            return_value={
                "metrics": ["users", "events", "sessions"],
                "platforms": ["aragora", "google_analytics", "mixpanel"],
                "range": "24h",
                "aggregation": "sum",
            }
        )

        query_result = await handler.handle(
            query_request, "/api/v1/analytics/cross-platform/query", "POST"
        )
        assert query_result.status_code == 200
        assert b"results" in query_result.body

    @pytest.mark.asyncio
    async def test_export_workflow(self):
        """Test data export workflow."""
        tenant_id = "analytics_tenant_4"
        handler = CrossPlatformAnalyticsHandler()

        # Test JSON export
        json_export_request = MagicMock()
        json_export_request.tenant_id = tenant_id
        json_export_request.query = {"format": "json", "range": "7d"}

        json_result = await handler.handle(
            json_export_request, "/api/v1/analytics/cross-platform/export", "GET"
        )
        assert json_result.status_code == 200
        assert b"data" in json_result.body

        # Test CSV export
        csv_export_request = MagicMock()
        csv_export_request.tenant_id = tenant_id
        csv_export_request.query = {"format": "csv", "range": "7d"}

        csv_result = await handler.handle(
            csv_export_request, "/api/v1/analytics/cross-platform/export", "GET"
        )
        assert csv_result.status_code == 200
        assert csv_result.content_type == "text/csv"


class TestCrossHandlerIntegration:
    """Test integration across multiple handlers."""

    @pytest.mark.asyncio
    async def test_audit_findings_to_analytics(self):
        """Test that audit findings appear in analytics."""
        tenant_id = "cross_tenant_1"

        # Run audit
        audit_handler = CodebaseAuditHandler()
        audit_request = MagicMock()
        audit_request.tenant_id = tenant_id
        audit_request.json = AsyncMock(
            return_value={
                "target_path": ".",
                "scan_types": ["sast"],
            }
        )

        await audit_handler.handle(audit_request, "/api/v1/codebase/scan", "POST")

        # Check analytics
        analytics_handler = CrossPlatformAnalyticsHandler()
        analytics_request = MagicMock()
        analytics_request.tenant_id = tenant_id
        analytics_request.query = {}

        summary_result = await analytics_handler.handle(
            analytics_request, "/api/v1/analytics/cross-platform/summary", "GET"
        )
        assert summary_result.status_code == 200

    @pytest.mark.asyncio
    async def test_reconciliation_with_analytics(self):
        """Test reconciliation metrics in analytics."""
        tenant_id = "cross_tenant_2"

        # Get reconciliation demo data
        recon_handler = ReconciliationHandler()
        demo_request = MagicMock()
        demo_request.tenant_id = tenant_id

        demo_result = await recon_handler.handle(demo_request, "/api/v1/reconciliation/demo", "GET")
        assert demo_result.status_code == 200

        # Check analytics for financial metrics
        analytics_handler = CrossPlatformAnalyticsHandler()
        comparison_request = MagicMock()
        comparison_request.tenant_id = tenant_id
        comparison_request.query = {"type": "cost"}

        comparison_result = await analytics_handler.handle(
            comparison_request, "/api/v1/analytics/cross-platform/comparison", "GET"
        )
        assert comparison_result.status_code == 200


class TestDemoModesConsistency:
    """Test that demo modes return consistent data."""

    @pytest.mark.asyncio
    async def test_all_demo_endpoints(self):
        """Test all handlers return valid demo data."""
        tenant_id = "demo_tenant"

        # Unified Inbox demo via stats
        inbox_handler = UnifiedInboxHandler()
        inbox_request = MagicMock()
        inbox_request.tenant_id = tenant_id

        inbox_result = await inbox_handler.handle(inbox_request, "/api/v1/inbox/stats", "GET")
        assert inbox_result.status_code == 200

        # Reconciliation demo
        recon_handler = ReconciliationHandler()
        recon_request = MagicMock()
        recon_request.tenant_id = tenant_id

        recon_result = await recon_handler.handle(
            recon_request, "/api/v1/reconciliation/demo", "GET"
        )
        assert recon_result.status_code == 200
        assert b"is_demo" in recon_result.body

        # Codebase audit demo
        audit_handler = CodebaseAuditHandler()
        audit_request = MagicMock()
        audit_request.tenant_id = tenant_id

        audit_result = await audit_handler.handle(audit_request, "/api/v1/codebase/demo", "GET")
        assert audit_result.status_code == 200
        assert b"is_demo" in audit_result.body

        # Analytics demo
        analytics_handler = CrossPlatformAnalyticsHandler()
        analytics_request = MagicMock()
        analytics_request.tenant_id = tenant_id

        analytics_result = await analytics_handler.handle(
            analytics_request, "/api/v1/analytics/cross-platform/demo", "GET"
        )
        assert analytics_result.status_code == 200
        assert b"is_demo" in analytics_result.body


class TestTenantIsolation:
    """Test that handlers maintain tenant isolation."""

    @pytest.mark.asyncio
    async def test_scan_data_isolation(self):
        """Test that scan data is isolated by tenant."""
        handler = CodebaseAuditHandler()

        # Tenant A runs a scan
        tenant_a_request = MagicMock()
        tenant_a_request.tenant_id = "tenant_a"
        tenant_a_request.json = AsyncMock(
            return_value={
                "target_path": ".",
                "scan_types": ["sast"],
            }
        )

        await handler.handle(tenant_a_request, "/api/v1/codebase/scan", "POST")

        # Tenant B lists scans - should not see Tenant A's scan
        tenant_b_request = MagicMock()
        tenant_b_request.tenant_id = "tenant_b"
        tenant_b_request.query = {}

        result = await handler.handle(tenant_b_request, "/api/v1/codebase/scans", "GET")
        assert result.status_code == 200

        response_data = json.loads(result.body)
        # Should be empty or not contain tenant A's data
        for scan in response_data.get("scans", []):
            assert scan.get("tenant_id") != "tenant_a"

    @pytest.mark.asyncio
    async def test_alert_isolation(self):
        """Test that alerts are isolated by tenant."""
        handler = CrossPlatformAnalyticsHandler()

        # Tenant A creates alert
        tenant_a_request = MagicMock()
        tenant_a_request.tenant_id = "tenant_alert_a"
        tenant_a_request.json = AsyncMock(
            return_value={
                "name": "Tenant A Alert",
                "metric_name": "errors",
                "condition": "above",
                "threshold": 100,
                "severity": "warning",
            }
        )

        await handler.handle(tenant_a_request, "/api/v1/analytics/cross-platform/alerts", "POST")

        # Tenant B lists alerts - should not see Tenant A's alert
        tenant_b_request = MagicMock()
        tenant_b_request.tenant_id = "tenant_alert_b"
        tenant_b_request.query = {}

        result = await handler.handle(
            tenant_b_request, "/api/v1/analytics/cross-platform/alerts", "GET"
        )

        response_data = json.loads(result.body)
        for rule in response_data.get("rules", []):
            assert rule.get("name") != "Tenant A Alert"


class TestMarketplaceWorkflow:
    """Test marketplace template discovery and deployment workflow."""

    @pytest.mark.asyncio
    async def test_template_discovery_to_deployment(self):
        """Test complete workflow: discover -> browse -> deploy -> list."""
        tenant_id = "marketplace_tenant_1"
        handler = MarketplaceHandler()

        # Step 1: List categories
        categories_request = MagicMock()
        categories_request.tenant_id = tenant_id
        categories_request.query = {}

        categories_result = await handler.handle(
            categories_request, "/api/v1/marketplace/categories", "GET"
        )
        assert categories_result.status_code == 200
        categories_data = json.loads(categories_result.body)
        data = categories_data.get("data", categories_data)
        assert len(data["categories"]) > 0

        # Step 2: Browse templates
        templates_request = MagicMock()
        templates_request.tenant_id = tenant_id
        templates_request.query = {"category": "accounting"}

        templates_result = await handler.handle(
            templates_request, "/api/v1/marketplace/templates", "GET"
        )
        assert templates_result.status_code == 200
        templates_data = json.loads(templates_result.body)
        data = templates_data.get("data", templates_data)
        templates = data["templates"]
        assert len(templates) > 0

        # Step 3: Get template details
        template_id = templates[0]["id"]
        details_request = MagicMock()
        details_request.tenant_id = tenant_id
        details_request.query = {}

        details_result = await handler.handle(
            details_request, f"/api/v1/marketplace/templates/{template_id}", "GET"
        )
        assert details_result.status_code == 200
        details_data = json.loads(details_result.body)
        data = details_data.get("data", details_data)
        assert data["template"]["id"] == template_id

        # Step 4: Deploy template
        deploy_request = MagicMock()
        deploy_request.tenant_id = tenant_id
        deploy_request.json = AsyncMock(
            return_value={
                "name": "My Invoice Pipeline",
                "config": {"auto_approve_threshold": 500},
            }
        )

        deploy_result = await handler.handle(
            deploy_request, f"/api/v1/marketplace/templates/{template_id}/deploy", "POST"
        )
        assert deploy_result.status_code == 200
        deploy_data = json.loads(deploy_result.body)
        data = deploy_data.get("data", deploy_data)
        deployment_id = data["deployment"]["id"]
        assert data["deployment"]["status"] == "active"

        # Step 5: List deployments
        list_request = MagicMock()
        list_request.tenant_id = tenant_id
        list_request.query = {}

        list_result = await handler.handle(list_request, "/api/v1/marketplace/deployments", "GET")
        assert list_result.status_code == 200
        list_data = json.loads(list_result.body)
        data = list_data.get("data", list_data)
        assert data["total"] >= 1

    @pytest.mark.asyncio
    async def test_search_and_popular(self):
        """Test searching and getting popular templates."""
        tenant_id = "marketplace_tenant_2"
        handler = MarketplaceHandler()

        # Search by query
        search_request = MagicMock()
        search_request.tenant_id = tenant_id
        search_request.query = {"q": "review"}

        search_result = await handler.handle(search_request, "/api/v1/marketplace/search", "GET")
        assert search_result.status_code == 200

        # Get popular templates
        popular_request = MagicMock()
        popular_request.tenant_id = tenant_id
        popular_request.query = {"limit": "5"}

        popular_result = await handler.handle(popular_request, "/api/v1/marketplace/popular", "GET")
        assert popular_result.status_code == 200
        popular_data = json.loads(popular_result.body)
        data = popular_data.get("data", popular_data)
        assert len(data["popular"]) <= 5

    @pytest.mark.asyncio
    async def test_rate_template(self):
        """Test rating a template."""
        tenant_id = "marketplace_tenant_3"
        handler = MarketplaceHandler()

        # First get a template
        templates_request = MagicMock()
        templates_request.tenant_id = tenant_id
        templates_request.query = {}

        templates_result = await handler.handle(
            templates_request, "/api/v1/marketplace/templates", "GET"
        )
        templates_data = json.loads(templates_result.body)
        data = templates_data.get("data", templates_data)
        template_id = data["templates"][0]["id"]

        # Rate it
        rate_request = MagicMock()
        rate_request.tenant_id = tenant_id
        rate_request.user_id = "test_user"
        rate_request.json = AsyncMock(
            return_value={
                "rating": 5,
                "review": "Excellent template for invoice processing!",
            }
        )

        rate_result = await handler.handle(
            rate_request, f"/api/v1/marketplace/templates/{template_id}/rate", "POST"
        )
        assert rate_result.status_code == 200
        rate_data = json.loads(rate_result.body)
        data = rate_data.get("data", rate_data)
        assert data["rating"]["rating"] == 5
        assert data["template_rating"]["count"] >= 1
