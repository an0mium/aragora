"""Tests for trial endpoints in billing handler."""

from __future__ import annotations

import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from aragora.billing.models import Organization, SubscriptionTier
from aragora.billing.trial_manager import TrialManager, TrialStatus
from aragora.server.handlers.billing.core import BillingHandler


@dataclass
class MockUser:
    """Mock user for testing."""

    user_id: str
    role: str = "owner"


@dataclass
class MockDBUser:
    """Mock database user for testing."""

    id: str
    email: str
    org_id: str | None


class TestTrialStatus:
    """Tests for GET /api/v1/billing/trial endpoint."""

    def test_get_trial_status_active(self):
        """Test getting trial status for active trial."""
        # Create org with active trial
        org = Organization(
            id="org-123",
            name="Test Org",
            slug="test-org",
            tier=SubscriptionTier.FREE,
        )
        org.start_trial(duration_days=7, debates_limit=10)

        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = MockDBUser(
            id="user-123", email="test@example.com", org_id="org-123"
        )
        mock_user_store.get_organization_by_id.return_value = org

        handler = BillingHandler(ctx={"user_store": mock_user_store})
        mock_handler = MagicMock()
        user = MockUser(user_id="user-123")

        # Bypass decorators with __wrapped__.__wrapped__
        fn = handler._get_trial_status.__wrapped__.__wrapped__
        result = fn(handler, mock_handler, user=user)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "trial" in data
        assert data["trial"]["is_active"] is True
        assert data["trial"]["is_expired"] is False
        assert data["trial"]["days_remaining"] >= 6  # At least 6 days
        assert data["trial"]["debates_remaining"] == 10
        assert "upgrade_options" in data["trial"]
        assert len(data["trial"]["upgrade_options"]) == 3

    def test_get_trial_status_expired(self):
        """Test getting trial status for expired trial."""
        # Create org with expired trial
        org = Organization(
            id="org-123",
            name="Test Org",
            slug="test-org",
            tier=SubscriptionTier.FREE,
        )
        org.trial_started_at = datetime.now(timezone.utc) - timedelta(days=10)
        org.trial_expires_at = datetime.now(timezone.utc) - timedelta(days=3)
        org.trial_debates_limit = 10

        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = MockDBUser(
            id="user-123", email="test@example.com", org_id="org-123"
        )
        mock_user_store.get_organization_by_id.return_value = org

        handler = BillingHandler(ctx={"user_store": mock_user_store})
        mock_handler = MagicMock()
        user = MockUser(user_id="user-123")

        fn = handler._get_trial_status.__wrapped__.__wrapped__
        result = fn(handler, mock_handler, user=user)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["trial"]["is_active"] is False
        assert data["trial"]["is_expired"] is True
        assert data["trial"]["days_remaining"] == 0
        assert "upgrade_options" in data["trial"]

    def test_get_trial_status_no_trial(self):
        """Test getting trial status when no trial exists."""
        org = Organization(
            id="org-123",
            name="Test Org",
            slug="test-org",
            tier=SubscriptionTier.FREE,
        )

        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = MockDBUser(
            id="user-123", email="test@example.com", org_id="org-123"
        )
        mock_user_store.get_organization_by_id.return_value = org

        handler = BillingHandler(ctx={"user_store": mock_user_store})
        mock_handler = MagicMock()
        user = MockUser(user_id="user-123")

        fn = handler._get_trial_status.__wrapped__.__wrapped__
        result = fn(handler, mock_handler, user=user)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["trial"]["is_active"] is False
        assert data["trial"]["is_expired"] is False
        assert "upgrade_options" not in data["trial"]

    def test_get_trial_status_warning_expiring_soon(self):
        """Test warning message when trial is expiring soon."""
        # Create org with trial expiring in 2 days
        org = Organization(
            id="org-123",
            name="Test Org",
            slug="test-org",
            tier=SubscriptionTier.FREE,
        )
        org.trial_started_at = datetime.now(timezone.utc) - timedelta(days=5)
        org.trial_expires_at = datetime.now(timezone.utc) + timedelta(days=2)
        org.trial_debates_limit = 10

        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = MockDBUser(
            id="user-123", email="test@example.com", org_id="org-123"
        )
        mock_user_store.get_organization_by_id.return_value = org

        handler = BillingHandler(ctx={"user_store": mock_user_store})
        mock_handler = MagicMock()
        user = MockUser(user_id="user-123")

        fn = handler._get_trial_status.__wrapped__.__wrapped__
        result = fn(handler, mock_handler, user=user)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "warning" in data["trial"]
        assert "expires" in data["trial"]["warning"]

    def test_get_trial_status_user_not_found(self):
        """Test error when user not found."""
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = None

        handler = BillingHandler(ctx={"user_store": mock_user_store})
        mock_handler = MagicMock()
        user = MockUser(user_id="user-123")

        fn = handler._get_trial_status.__wrapped__.__wrapped__
        result = fn(handler, mock_handler, user=user)

        assert result.status_code == 404

    def test_get_trial_status_no_org(self):
        """Test error when organization not found."""
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = MockDBUser(
            id="user-123", email="test@example.com", org_id="org-123"
        )
        mock_user_store.get_organization_by_id.return_value = None

        handler = BillingHandler(ctx={"user_store": mock_user_store})
        mock_handler = MagicMock()
        user = MockUser(user_id="user-123")

        fn = handler._get_trial_status.__wrapped__.__wrapped__
        result = fn(handler, mock_handler, user=user)

        assert result.status_code == 404


class TestStartTrial:
    """Tests for POST /api/v1/billing/trial/start endpoint."""

    def test_start_trial_success(self):
        """Test successfully starting a trial."""
        org = Organization(
            id="org-123",
            name="Test Org",
            slug="test-org",
            tier=SubscriptionTier.FREE,
        )

        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = MockDBUser(
            id="user-123", email="test@example.com", org_id="org-123"
        )
        mock_user_store.get_organization_by_id.return_value = org

        handler = BillingHandler(ctx={"user_store": mock_user_store})
        mock_handler = MagicMock()
        user = MockUser(user_id="user-123")

        # _start_trial has @handle_errors and @log_request decorators
        fn = handler._start_trial.__wrapped__.__wrapped__
        result = fn(handler, mock_handler, user=user)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "trial" in data
        assert data["trial"]["is_active"] is True
        assert "message" in data
        # days_remaining can be 6 or 7 depending on timing
        assert "days" in data["message"]
        assert "10 debates" in data["message"]

    def test_start_trial_already_active(self):
        """Test starting trial when already active returns current status."""
        org = Organization(
            id="org-123",
            name="Test Org",
            slug="test-org",
            tier=SubscriptionTier.FREE,
        )
        org.start_trial(duration_days=7, debates_limit=10)

        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = MockDBUser(
            id="user-123", email="test@example.com", org_id="org-123"
        )
        mock_user_store.get_organization_by_id.return_value = org

        handler = BillingHandler(ctx={"user_store": mock_user_store})
        mock_handler = MagicMock()
        user = MockUser(user_id="user-123")

        fn = handler._start_trial.__wrapped__.__wrapped__
        result = fn(handler, mock_handler, user=user)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["message"] == "Trial already active"

    def test_start_trial_paid_subscription(self):
        """Test starting trial with paid subscription returns error."""
        org = Organization(
            id="org-123",
            name="Test Org",
            slug="test-org",
            tier=SubscriptionTier.STARTER,  # Paid tier
        )

        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = MockDBUser(
            id="user-123", email="test@example.com", org_id="org-123"
        )
        mock_user_store.get_organization_by_id.return_value = org

        handler = BillingHandler(ctx={"user_store": mock_user_store})
        mock_handler = MagicMock()
        user = MockUser(user_id="user-123")

        fn = handler._start_trial.__wrapped__.__wrapped__
        result = fn(handler, mock_handler, user=user)

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "paid subscription" in data["error"]

    def test_start_trial_no_user(self):
        """Test starting trial without authentication."""
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = None

        handler = BillingHandler(ctx={"user_store": mock_user_store})
        mock_handler = MagicMock()
        handler.read_json_body = MagicMock(return_value=None)

        fn = handler._start_trial.__wrapped__.__wrapped__
        result = fn(handler, mock_handler, user=None)

        assert result.status_code == 401

    def test_start_trial_signup_flow(self):
        """Test starting trial from signup flow with user_id in body."""
        org = Organization(
            id="org-123",
            name="Test Org",
            slug="test-org",
            tier=SubscriptionTier.FREE,
        )

        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = MockDBUser(
            id="user-123", email="test@example.com", org_id="org-123"
        )
        mock_user_store.get_organization_by_id.return_value = org

        handler = BillingHandler(ctx={"user_store": mock_user_store})
        mock_handler = MagicMock()
        handler.read_json_body = MagicMock(return_value={"user_id": "user-123"})

        fn = handler._start_trial.__wrapped__.__wrapped__
        result = fn(handler, mock_handler, user=None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["trial"]["is_active"] is True


class TestTrialManager:
    """Tests for TrialManager utility class."""

    def test_trial_manager_start_trial(self):
        """Test TrialManager.start_trial."""
        org = Organization(
            id="org-123",
            name="Test Org",
            slug="test-org",
            tier=SubscriptionTier.FREE,
        )

        trial_mgr = TrialManager()
        status = trial_mgr.start_trial(org)

        assert status.is_active is True
        # days_remaining can be 6 or 7 depending on timing (uses calendar days)
        assert status.days_remaining >= 6
        assert status.days_remaining <= 7
        assert status.debates_remaining == 10
        assert org.trial_started_at is not None
        assert org.trial_expires_at is not None

    def test_trial_manager_can_start_debate_in_trial(self):
        """Test TrialManager.can_start_debate during trial."""
        org = Organization(
            id="org-123",
            name="Test Org",
            slug="test-org",
            tier=SubscriptionTier.FREE,
        )
        org.start_trial(duration_days=7, debates_limit=10)

        trial_mgr = TrialManager()
        allowed, reason = trial_mgr.can_start_debate(org)

        assert allowed is True
        assert reason == ""

    def test_trial_manager_can_start_debate_at_limit(self):
        """Test TrialManager.can_start_debate when at trial limit."""
        org = Organization(
            id="org-123",
            name="Test Org",
            slug="test-org",
            tier=SubscriptionTier.FREE,
        )
        org.start_trial(duration_days=7, debates_limit=10)
        org.debates_used_this_month = 10  # At limit

        trial_mgr = TrialManager()
        allowed, reason = trial_mgr.can_start_debate(org)

        assert allowed is False
        assert "limit reached" in reason

    def test_trial_manager_can_start_debate_expired(self):
        """Test TrialManager.can_start_debate when trial expired."""
        org = Organization(
            id="org-123",
            name="Test Org",
            slug="test-org",
            tier=SubscriptionTier.FREE,
        )
        org.trial_started_at = datetime.now(timezone.utc) - timedelta(days=10)
        org.trial_expires_at = datetime.now(timezone.utc) - timedelta(days=3)
        org.trial_debates_limit = 10

        trial_mgr = TrialManager()
        allowed, reason = trial_mgr.can_start_debate(org)

        assert allowed is False
        assert "expired" in reason.lower()

    def test_trial_manager_record_debate(self):
        """Test TrialManager.record_debate."""
        org = Organization(
            id="org-123",
            name="Test Org",
            slug="test-org",
            tier=SubscriptionTier.FREE,
        )
        org.start_trial(duration_days=7, debates_limit=10)

        trial_mgr = TrialManager()
        success = trial_mgr.record_debate(org)

        assert success is True
        assert org.debates_used_this_month == 1

    def test_trial_manager_convert_trial(self):
        """Test TrialManager.convert_trial."""
        org = Organization(
            id="org-123",
            name="Test Org",
            slug="test-org",
            tier=SubscriptionTier.FREE,
        )
        org.start_trial(duration_days=7, debates_limit=10)

        trial_mgr = TrialManager()
        success = trial_mgr.convert_trial(org, SubscriptionTier.STARTER)

        assert success is True
        assert org.trial_converted is True
        assert org.tier == SubscriptionTier.STARTER

    def test_trial_manager_extend_trial(self):
        """Test TrialManager.extend_trial."""
        org = Organization(
            id="org-123",
            name="Test Org",
            slug="test-org",
            tier=SubscriptionTier.FREE,
        )
        org.start_trial(duration_days=7, debates_limit=10)
        original_expires = org.trial_expires_at

        trial_mgr = TrialManager()
        status = trial_mgr.extend_trial(org, additional_days=7)

        assert status.days_remaining > 7  # More than original
        assert org.trial_expires_at > original_expires

    def test_trial_status_to_dict(self):
        """Test TrialStatus.to_dict serialization."""
        status = TrialStatus(
            is_active=True,
            is_expired=False,
            days_remaining=5,
            debates_remaining=8,
            debates_used=2,
            debates_limit=10,
            started_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
            expires_at=datetime(2026, 2, 8, tzinfo=timezone.utc),
            converted=False,
        )

        result = status.to_dict()

        assert result["is_active"] is True
        assert result["days_remaining"] == 5
        assert result["debates_remaining"] == 8
        assert result["started_at"] == "2026-02-01T00:00:00+00:00"
        assert result["expires_at"] == "2026-02-08T00:00:00+00:00"
