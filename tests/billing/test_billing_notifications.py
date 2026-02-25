"""
Comprehensive tests for aragora.billing.notifications module.

Tests cover:
- BillingNotifier class with all notification methods
- NotificationResult dataclass
- SMTP email delivery
- Webhook notifications
- Payment failure notifications with urgency levels
- Trial expiration warnings
- Subscription cancellation notifications
- Downgrade notifications
- Budget alert notifications at all levels
- Forecast overage alerts
- Credit expiration notifications
- Fallback behavior (email -> webhook -> log)
- Edge cases: empty emails, zero budgets, special characters
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from unittest.mock import MagicMock, patch, Mock, call
from urllib.error import URLError

import pytest

from aragora.billing.notifications import (
    BillingNotifier,
    NotificationResult,
    get_billing_notifier,
    SMTP_HOST,
    SMTP_PORT,
    SMTP_USER,
    SMTP_PASSWORD,
    SMTP_FROM,
    NOTIFICATION_WEBHOOK,
)
from aragora.billing.models import SubscriptionTier


# =============================================================================
# NotificationResult Tests
# =============================================================================


class TestNotificationResult:
    """Tests for NotificationResult dataclass."""

    def test_success_result(self):
        """Test successful notification result."""
        result = NotificationResult(success=True, method="email")
        assert result.success is True
        assert result.method == "email"
        assert result.error is None

    def test_failure_result(self):
        """Test failed notification result."""
        result = NotificationResult(
            success=False,
            method="webhook",
            error="Connection refused",
        )
        assert result.success is False
        assert result.method == "webhook"
        assert result.error == "Connection refused"

    def test_log_method_result(self):
        """Test log fallback result."""
        result = NotificationResult(success=True, method="log")
        assert result.method == "log"

    def test_error_none_by_default(self):
        """Test error is None by default."""
        result = NotificationResult(success=True, method="email")
        assert result.error is None

    def test_method_values(self):
        """Test various method values are accepted."""
        for method in ["email", "webhook", "log"]:
            result = NotificationResult(success=True, method=method)
            assert result.method == method


# =============================================================================
# BillingNotifier Initialization Tests
# =============================================================================


class TestBillingNotifierInit:
    """Tests for BillingNotifier initialization."""

    def test_default_initialization(self):
        """Test initialization with default values."""
        notifier = BillingNotifier()
        # Uses environment variables or module defaults
        assert notifier.smtp_host == SMTP_HOST
        assert notifier.smtp_port == SMTP_PORT
        assert notifier.smtp_from == SMTP_FROM

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        notifier = BillingNotifier(
            smtp_host="mail.example.com",
            smtp_port=465,
            smtp_user="user@example.com",
            smtp_password="secret",
            smtp_from="billing@example.com",
            webhook_url="https://hooks.example.com/billing",
        )
        assert notifier.smtp_host == "mail.example.com"
        assert notifier.smtp_port == 465
        assert notifier.smtp_user == "user@example.com"
        assert notifier.smtp_password == "secret"
        assert notifier.smtp_from == "billing@example.com"
        assert notifier.webhook_url == "https://hooks.example.com/billing"

    def test_is_smtp_configured(self):
        """Test SMTP configuration detection."""
        # Not configured
        notifier = BillingNotifier()
        assert notifier._is_smtp_configured() is False

        # Partially configured
        notifier = BillingNotifier(smtp_host="mail.example.com")
        assert notifier._is_smtp_configured() is False

        # Fully configured
        notifier = BillingNotifier(
            smtp_host="mail.example.com",
            smtp_user="user@example.com",
            smtp_password="secret",
        )
        assert notifier._is_smtp_configured() is True

    def test_is_smtp_configured_missing_user(self):
        """Test SMTP not configured when missing user."""
        notifier = BillingNotifier(
            smtp_host="mail.example.com",
            smtp_password="secret",
        )
        assert notifier._is_smtp_configured() is False

    def test_is_smtp_configured_missing_password(self):
        """Test SMTP not configured when missing password."""
        notifier = BillingNotifier(
            smtp_host="mail.example.com",
            smtp_user="user@example.com",
        )
        assert notifier._is_smtp_configured() is False


# =============================================================================
# Email Sending Tests
# =============================================================================


class TestEmailSending:
    """Tests for email sending functionality."""

    def test_send_email_not_configured(self):
        """Test email sending when SMTP not configured."""
        notifier = BillingNotifier()
        result = notifier._send_email(
            to_email="user@example.com",
            subject="Test",
            html_body="<p>Test</p>",
        )
        assert result.success is False
        assert result.method == "email"
        assert "not configured" in result.error.lower()

    @patch("smtplib.SMTP")
    def test_send_email_success(self, mock_smtp_class):
        """Test successful email sending."""
        mock_smtp = MagicMock()
        mock_smtp_class.return_value.__enter__ = Mock(return_value=mock_smtp)
        mock_smtp_class.return_value.__exit__ = Mock(return_value=False)

        notifier = BillingNotifier(
            smtp_host="mail.example.com",
            smtp_user="user@example.com",
            smtp_password="secret",
        )

        result = notifier._send_email(
            to_email="recipient@example.com",
            subject="Test Subject",
            html_body="<p>HTML body</p>",
            text_body="Text body",
        )

        assert result.success is True
        assert result.method == "email"
        mock_smtp.starttls.assert_called_once()
        mock_smtp.login.assert_called_once()
        mock_smtp.sendmail.assert_called_once()

    @patch("smtplib.SMTP")
    def test_send_email_failure(self, mock_smtp_class):
        """Test email sending failure."""
        mock_smtp_class.side_effect = OSError("SMTP connection failed")

        notifier = BillingNotifier(
            smtp_host="mail.example.com",
            smtp_user="user@example.com",
            smtp_password="secret",
        )

        result = notifier._send_email(
            to_email="recipient@example.com",
            subject="Test",
            html_body="<p>Test</p>",
        )

        assert result.success is False
        assert result.method == "email"
        assert result.error  # Sanitized error message present

    @patch("smtplib.SMTP")
    def test_send_email_html_only(self, mock_smtp_class):
        """Test email with HTML body only (no text_body)."""
        mock_smtp = MagicMock()
        mock_smtp_class.return_value.__enter__ = Mock(return_value=mock_smtp)
        mock_smtp_class.return_value.__exit__ = Mock(return_value=False)

        notifier = BillingNotifier(
            smtp_host="mail.example.com",
            smtp_user="user@example.com",
            smtp_password="secret",
        )

        result = notifier._send_email(
            to_email="recipient@example.com",
            subject="Test",
            html_body="<p>HTML only</p>",
        )

        assert result.success is True


# =============================================================================
# Webhook Sending Tests
# =============================================================================


class TestWebhookSending:
    """Tests for webhook notification sending."""

    def test_send_webhook_not_configured(self):
        """Test webhook sending when not configured."""
        notifier = BillingNotifier()
        result = notifier._send_webhook({"event": "test"})

        assert result.success is False
        assert result.method == "webhook"
        assert "not configured" in result.error.lower()

    @patch("aragora.billing.notifications.urlopen")
    def test_send_webhook_success(self, mock_urlopen):
        """Test successful webhook sending."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"{}"
        mock_urlopen.return_value.__enter__ = Mock(return_value=mock_response)
        mock_urlopen.return_value.__exit__ = Mock(return_value=False)

        notifier = BillingNotifier(webhook_url="https://hooks.example.com/billing")

        result = notifier._send_webhook({"event": "test", "data": "value"})

        assert result.success is True
        assert result.method == "webhook"
        mock_urlopen.assert_called_once()

    @patch("aragora.billing.notifications.urlopen")
    def test_send_webhook_failure(self, mock_urlopen):
        """Test webhook sending failure."""
        mock_urlopen.side_effect = URLError("Connection refused")

        notifier = BillingNotifier(webhook_url="https://hooks.example.com/billing")

        result = notifier._send_webhook({"event": "test"})

        assert result.success is False
        assert result.method == "webhook"
        assert result.error  # Sanitized error message present


# =============================================================================
# Payment Failed Notification Tests
# =============================================================================


class TestPaymentFailedNotification:
    """Tests for payment failure notifications."""

    def test_notify_payment_failed_first_attempt(self):
        """Test payment failed notification for first attempt falls back to log."""
        notifier = BillingNotifier()

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(
                success=False, method="email", error="no smtp"
            )
            with patch.object(notifier, "_send_webhook") as mock_webhook:
                mock_webhook.return_value = NotificationResult(
                    success=False, method="webhook", error="no webhook"
                )

                result = notifier.notify_payment_failed(
                    org_id="org-123",
                    org_name="Test Org",
                    email="admin@example.com",
                    attempt_count=1,
                )

                # Falls back to log
                assert result.success is True
                assert result.method == "log"

    def test_notify_payment_failed_second_attempt(self):
        """Test payment failed notification for second attempt uses IMPORTANT urgency."""
        notifier = BillingNotifier()

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            result = notifier.notify_payment_failed(
                org_id="org-123",
                org_name="Test Org",
                email="admin@example.com",
                attempt_count=2,
            )

            assert result.success is True
            assert result.method == "email"
            # Verify email was called; _send_email(email, subject, html_body, text_body)
            call_args = mock_email.call_args
            subject = call_args[0][1]  # positional arg: subject
            assert "Payment Failed" in subject

    def test_notify_payment_failed_final_attempt(self):
        """Test payment failed notification for final attempt (3rd) triggers URGENT."""
        notifier = BillingNotifier()

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            result = notifier.notify_payment_failed(
                org_id="org-123",
                org_name="Test Org",
                email="admin@example.com",
                attempt_count=3,
            )

            assert result.success is True
            # The subject stays the same but HTML body contains URGENT
            call_args = mock_email.call_args
            html_body = call_args[0][2]  # positional arg: html_body
            assert "URGENT" in html_body

    def test_notify_payment_failed_with_invoice_url(self):
        """Test payment failed notification includes invoice URL in body."""
        notifier = BillingNotifier()

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            notifier.notify_payment_failed(
                org_id="org-123",
                org_name="Test Org",
                email="admin@example.com",
                attempt_count=1,
                invoice_url="https://pay.stripe.com/invoice/123",
            )

            call_args = mock_email.call_args
            html_body = call_args[0][2]  # positional arg: html_body
            assert "pay.stripe.com" in html_body

    def test_notify_payment_failed_webhook_fallback(self):
        """Test payment failed falls back to webhook when email fails."""
        notifier = BillingNotifier(webhook_url="https://hooks.example.com")

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(
                success=False, method="email", error="failed"
            )
            with patch.object(notifier, "_send_webhook") as mock_webhook:
                mock_webhook.return_value = NotificationResult(success=True, method="webhook")

                result = notifier.notify_payment_failed(
                    org_id="org-123",
                    org_name="Test Org",
                    email="admin@example.com",
                )

                assert result.success is True
                assert result.method == "webhook"

    def test_notify_payment_failed_webhook_payload(self):
        """Test payment failed webhook includes correct payload."""
        notifier = BillingNotifier(webhook_url="https://hooks.example.com")

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(
                success=False, method="email", error="failed"
            )
            with patch.object(notifier, "_send_webhook") as mock_webhook:
                mock_webhook.return_value = NotificationResult(success=True, method="webhook")

                notifier.notify_payment_failed(
                    org_id="org-123",
                    org_name="Test Org",
                    email="admin@example.com",
                    attempt_count=2,
                )

                call_args = mock_webhook.call_args
                payload = call_args[0][0]
                assert payload["event"] == "payment_failed"
                assert payload["org_id"] == "org-123"
                assert payload["org_name"] == "Test Org"
                assert payload["attempt_count"] == 2
                assert payload["urgency"] == "IMPORTANT"

    def test_notify_payment_failed_urgency_levels(self):
        """Test urgency levels for different attempt counts."""
        notifier = BillingNotifier(webhook_url="https://hooks.example.com")

        for attempt, expected_urgency in [
            (1, "NOTICE"),
            (2, "IMPORTANT"),
            (3, "URGENT"),
            (4, "URGENT"),
        ]:
            with patch.object(notifier, "_send_email") as mock_email:
                mock_email.return_value = NotificationResult(
                    success=False, method="email", error="failed"
                )
                with patch.object(notifier, "_send_webhook") as mock_webhook:
                    mock_webhook.return_value = NotificationResult(success=True, method="webhook")

                    notifier.notify_payment_failed(
                        org_id="org-123",
                        org_name="Test Org",
                        email="admin@example.com",
                        attempt_count=attempt,
                    )

                    payload = mock_webhook.call_args[0][0]
                    assert payload["urgency"] == expected_urgency, (
                        f"attempt={attempt}: expected {expected_urgency}, got {payload['urgency']}"
                    )


# =============================================================================
# Trial Ending Notification Tests
# =============================================================================


class TestTrialEndingNotification:
    """Tests for trial ending notifications."""

    def test_notify_trial_ending_7_days(self):
        """Test trial ending notification with 7 days remaining (INFO)."""
        notifier = BillingNotifier()
        trial_end = datetime.now(timezone.utc) + timedelta(days=7)

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            result = notifier.notify_trial_ending(
                org_id="org-123",
                org_name="Test Org",
                email="admin@example.com",
                days_remaining=7,
                trial_end=trial_end,
            )

            assert result.success is True
            call_args = mock_email.call_args
            subject = call_args[0][1]
            assert "7 Days Left" in subject

    def test_notify_trial_ending_3_days(self):
        """Test trial ending notification with 3 days remaining (REMINDER)."""
        notifier = BillingNotifier()
        trial_end = datetime.now(timezone.utc) + timedelta(days=3)

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            result = notifier.notify_trial_ending(
                org_id="org-123",
                org_name="Test Org",
                email="admin@example.com",
                days_remaining=3,
                trial_end=trial_end,
            )

            assert result.success is True
            call_args = mock_email.call_args
            subject = call_args[0][1]
            assert "3 Days" in subject

    def test_notify_trial_ending_1_day(self):
        """Test trial ending notification with 1 day remaining (URGENT)."""
        notifier = BillingNotifier()
        trial_end = datetime.now(timezone.utc) + timedelta(days=1)

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            result = notifier.notify_trial_ending(
                org_id="org-123",
                org_name="Test Org",
                email="admin@example.com",
                days_remaining=1,
                trial_end=trial_end,
            )

            assert result.success is True
            call_args = mock_email.call_args
            subject = call_args[0][1]
            assert "Tomorrow" in subject

    def test_notify_trial_ending_0_days(self):
        """Test trial ending notification with 0 days remaining is URGENT."""
        notifier = BillingNotifier()
        trial_end = datetime.now(timezone.utc)

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            result = notifier.notify_trial_ending(
                org_id="org-123",
                org_name="Test Org",
                email="admin@example.com",
                days_remaining=0,
                trial_end=trial_end,
            )

            assert result.success is True
            call_args = mock_email.call_args
            subject = call_args[0][1]
            assert "Tomorrow" in subject

    def test_notify_trial_ending_log_fallback(self):
        """Test trial ending falls back to logging when email and webhook fail."""
        notifier = BillingNotifier()
        trial_end = datetime.now(timezone.utc) + timedelta(days=5)

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(
                success=False, method="email", error="failed"
            )
            with patch.object(notifier, "_send_webhook") as mock_webhook:
                mock_webhook.return_value = NotificationResult(
                    success=False, method="webhook", error="failed"
                )

                result = notifier.notify_trial_ending(
                    org_id="org-123",
                    org_name="Test Org",
                    email="admin@example.com",
                    days_remaining=5,
                    trial_end=trial_end,
                )

                assert result.success is True
                assert result.method == "log"

    def test_notify_trial_ending_webhook_payload(self):
        """Test trial ending webhook payload has correct fields."""
        notifier = BillingNotifier(webhook_url="https://hooks.example.com")
        trial_end = datetime.now(timezone.utc) + timedelta(days=2)

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(
                success=False, method="email", error="failed"
            )
            with patch.object(notifier, "_send_webhook") as mock_webhook:
                mock_webhook.return_value = NotificationResult(success=True, method="webhook")

                notifier.notify_trial_ending(
                    org_id="org-123",
                    org_name="Test Org",
                    email="admin@example.com",
                    days_remaining=2,
                    trial_end=trial_end,
                )

                payload = mock_webhook.call_args[0][0]
                assert payload["event"] == "trial_ending"
                assert payload["org_id"] == "org-123"
                assert payload["days_remaining"] == 2
                assert payload["urgency"] == "REMINDER"

    def test_notify_trial_ending_html_includes_date(self):
        """Test trial ending HTML body includes formatted trial end date."""
        notifier = BillingNotifier()
        trial_end = datetime(2025, 3, 15, tzinfo=timezone.utc)

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            notifier.notify_trial_ending(
                org_id="org-123",
                org_name="Test Org",
                email="admin@example.com",
                days_remaining=5,
                trial_end=trial_end,
            )

            call_args = mock_email.call_args
            html_body = call_args[0][2]
            assert "March 15, 2025" in html_body


# =============================================================================
# Subscription Canceled Notification Tests
# =============================================================================


class TestSubscriptionCanceledNotification:
    """Tests for subscription cancellation notifications."""

    def test_notify_subscription_canceled(self):
        """Test subscription canceled notification includes correct subject."""
        notifier = BillingNotifier()

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            result = notifier.notify_subscription_canceled(
                org_id="org-123",
                org_name="Test Org",
                email="admin@example.com",
            )

            assert result.success is True
            call_args = mock_email.call_args
            subject = call_args[0][1]
            assert "Canceled" in subject

    def test_notify_subscription_canceled_with_reason(self):
        """Test subscription canceled notification with reason."""
        notifier = BillingNotifier()

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            result = notifier.notify_subscription_canceled(
                org_id="org-123",
                org_name="Test Org",
                email="admin@example.com",
                reason="Too expensive",
            )

            assert result.success is True

    def test_notify_subscription_canceled_webhook(self):
        """Test subscription canceled notification via webhook."""
        notifier = BillingNotifier(webhook_url="https://hooks.example.com")

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(
                success=False, method="email", error="failed"
            )
            with patch.object(notifier, "_send_webhook") as mock_webhook:
                mock_webhook.return_value = NotificationResult(success=True, method="webhook")

                result = notifier.notify_subscription_canceled(
                    org_id="org-123",
                    org_name="Test Org",
                    email="admin@example.com",
                    reason="budget_constraints",
                )

                assert result.success is True
                assert result.method == "webhook"
                call_args = mock_webhook.call_args
                payload = call_args[0][0]
                assert payload["event"] == "subscription_canceled"
                assert payload["reason"] == "budget_constraints"

    def test_notify_subscription_canceled_html_contains_reactivation_link(self):
        """Test cancellation email contains reactivation link."""
        notifier = BillingNotifier()

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            notifier.notify_subscription_canceled(
                org_id="org-123",
                org_name="Test Org",
                email="admin@example.com",
            )

            call_args = mock_email.call_args
            html_body = call_args[0][2]
            assert "REACTIVATE" in html_body
            assert "aragora.ai/billing" in html_body


# =============================================================================
# Downgraded Notification Tests
# =============================================================================


class TestDowngradedNotification:
    """Tests for subscription downgrade notifications."""

    def test_notify_downgraded(self):
        """Test subscription downgraded notification."""
        notifier = BillingNotifier()

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            result = notifier.notify_downgraded(
                org_id="org-123",
                org_name="Test Org",
                email="admin@example.com",
                previous_tier=SubscriptionTier.PROFESSIONAL,
            )

            assert result.success is True
            call_args = mock_email.call_args
            subject = call_args[0][1]
            html_body = call_args[0][2]
            assert "Downgraded" in subject
            assert "PROFESSIONAL" in html_body

    def test_notify_downgraded_with_invoice_url(self):
        """Test subscription downgraded notification with invoice URL."""
        notifier = BillingNotifier()

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            result = notifier.notify_downgraded(
                org_id="org-123",
                org_name="Test Org",
                email="admin@example.com",
                previous_tier=SubscriptionTier.ENTERPRISE,
                invoice_url="https://pay.stripe.com/invoice/456",
            )

            assert result.success is True
            call_args = mock_email.call_args
            html_body = call_args[0][2]
            assert "pay.stripe.com" in html_body

    def test_notify_downgraded_without_invoice_url(self):
        """Test downgrade notification without invoice URL uses default billing link."""
        notifier = BillingNotifier()

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            notifier.notify_downgraded(
                org_id="org-123",
                org_name="Test Org",
                email="admin@example.com",
                previous_tier=SubscriptionTier.PROFESSIONAL,
            )

            call_args = mock_email.call_args
            html_body = call_args[0][2]
            assert "aragora.ai/billing" in html_body

    def test_notify_downgraded_webhook_payload(self):
        """Test downgrade notification webhook payload."""
        notifier = BillingNotifier(webhook_url="https://hooks.example.com")

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(
                success=False, method="email", error="failed"
            )
            with patch.object(notifier, "_send_webhook") as mock_webhook:
                mock_webhook.return_value = NotificationResult(success=True, method="webhook")

                notifier.notify_downgraded(
                    org_id="org-123",
                    org_name="Test Org",
                    email="admin@example.com",
                    previous_tier=SubscriptionTier.PROFESSIONAL,
                )

                call_args = mock_webhook.call_args
                payload = call_args[0][0]
                assert payload["event"] == "subscription_downgraded"
                assert payload["previous_tier"] == "professional"
                assert payload["new_tier"] == "free"

    def test_notify_downgraded_enterprise(self):
        """Test downgrade notification for Enterprise tier."""
        notifier = BillingNotifier()

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            notifier.notify_downgraded(
                org_id="org-123",
                org_name="Test Org",
                email="admin@example.com",
                previous_tier=SubscriptionTier.ENTERPRISE,
            )

            call_args = mock_email.call_args
            html_body = call_args[0][2]
            assert "ENTERPRISE" in html_body


# =============================================================================
# Budget Alert Notification Tests
# =============================================================================


class TestBudgetAlertNotification:
    """Tests for budget alert notifications."""

    def test_notify_budget_alert_info(self):
        """Test budget alert notification at info level."""
        notifier = BillingNotifier()

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            result = notifier.notify_budget_alert(
                tenant_id="tenant-123",
                email="admin@example.com",
                alert_level="info",
                current_spend="$50.00",
                budget_limit="$100.00",
                percent_used=50.0,
            )

            assert result.success is True
            call_args = mock_email.call_args
            subject = call_args[0][1]
            assert "50%" in subject
            assert "Budget Update" in subject

    def test_notify_budget_alert_warning(self):
        """Test budget alert notification at warning level."""
        notifier = BillingNotifier()

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            result = notifier.notify_budget_alert(
                tenant_id="tenant-123",
                email="admin@example.com",
                alert_level="warning",
                current_spend="$80.00",
                budget_limit="$100.00",
                percent_used=80.0,
                org_name="Test Org",
            )

            assert result.success is True
            call_args = mock_email.call_args
            subject = call_args[0][1]
            assert "Warning" in subject

    def test_notify_budget_alert_critical(self):
        """Test budget alert notification at critical level."""
        notifier = BillingNotifier()

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            result = notifier.notify_budget_alert(
                tenant_id="tenant-123",
                email="admin@example.com",
                alert_level="critical",
                current_spend="$95.00",
                budget_limit="$100.00",
                percent_used=95.0,
            )

            assert result.success is True
            call_args = mock_email.call_args
            subject = call_args[0][1]
            assert "Critical" in subject

    def test_notify_budget_alert_exceeded(self):
        """Test budget alert notification when exceeded."""
        notifier = BillingNotifier()

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            result = notifier.notify_budget_alert(
                tenant_id="tenant-123",
                email="admin@example.com",
                alert_level="exceeded",
                current_spend="$120.00",
                budget_limit="$100.00",
                percent_used=120.0,
            )

            assert result.success is True
            call_args = mock_email.call_args
            subject = call_args[0][1]
            assert "Exceeded" in subject

    def test_notify_budget_alert_unknown_level(self):
        """Test budget alert with unknown alert level (defaults to warning config)."""
        notifier = BillingNotifier()

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            result = notifier.notify_budget_alert(
                tenant_id="tenant-123",
                email="admin@example.com",
                alert_level="unknown",
                current_spend="$75.00",
                budget_limit="$100.00",
                percent_used=75.0,
            )

            assert result.success is True

    def test_notify_budget_alert_uses_org_name(self):
        """Test budget alert uses org_name when provided."""
        notifier = BillingNotifier()

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            notifier.notify_budget_alert(
                tenant_id="tenant-123",
                email="admin@example.com",
                alert_level="info",
                current_spend="$50.00",
                budget_limit="$100.00",
                percent_used=50.0,
                org_name="Acme Corp",
            )

            call_args = mock_email.call_args
            html_body = call_args[0][2]
            assert "Acme Corp" in html_body

    def test_notify_budget_alert_uses_tenant_id_as_fallback(self):
        """Test budget alert uses tenant_id when org_name not provided."""
        notifier = BillingNotifier()

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            notifier.notify_budget_alert(
                tenant_id="tenant-abc",
                email="admin@example.com",
                alert_level="info",
                current_spend="$50.00",
                budget_limit="$100.00",
                percent_used=50.0,
            )

            call_args = mock_email.call_args
            html_body = call_args[0][2]
            assert "tenant-abc" in html_body

    def test_notify_budget_alert_webhook_payload(self):
        """Test budget alert webhook payload contains all fields."""
        notifier = BillingNotifier(webhook_url="https://hooks.example.com")

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(
                success=False, method="email", error="failed"
            )
            with patch.object(notifier, "_send_webhook") as mock_webhook:
                mock_webhook.return_value = NotificationResult(success=True, method="webhook")

                notifier.notify_budget_alert(
                    tenant_id="tenant-123",
                    email="admin@example.com",
                    alert_level="critical",
                    current_spend="$95.00",
                    budget_limit="$100.00",
                    percent_used=95.0,
                    org_name="Test Org",
                )

                payload = mock_webhook.call_args[0][0]
                assert payload["event"] == "budget_alert"
                assert payload["tenant_id"] == "tenant-123"
                assert payload["alert_level"] == "critical"
                assert payload["percent_used"] == 95.0


# =============================================================================
# Forecast Overage Notification Tests
# =============================================================================


class TestForecastOverageNotification:
    """Tests for forecast overage notifications."""

    def test_notify_forecast_overage(self):
        """Test forecast overage notification."""
        notifier = BillingNotifier()
        projected_date = datetime.now(timezone.utc) + timedelta(days=10)

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            result = notifier.notify_forecast_overage(
                org_id="org-123",
                email="admin@example.com",
                org_name="Test Org",
                budget_name="Monthly API Budget",
                current_spent=500.00,
                budget_limit=1000.00,
                projected_date=projected_date,
                projected_amount=1200.00,
            )

            assert result.success is True
            call_args = mock_email.call_args
            subject = call_args[0][1]
            assert "Forecast" in subject
            assert "Monthly API Budget" in subject

    def test_notify_forecast_overage_calculations(self):
        """Test forecast overage notification includes correct calculations."""
        notifier = BillingNotifier()
        projected_date = datetime.now(timezone.utc) + timedelta(days=5)

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            notifier.notify_forecast_overage(
                org_id="org-123",
                email="admin@example.com",
                org_name="Test Org",
                budget_name="Test Budget",
                current_spent=800.00,
                budget_limit=1000.00,
                projected_date=projected_date,
                projected_amount=1500.00,  # $500 overage
            )

            call_args = mock_email.call_args
            html_body = call_args[0][2]
            assert "$1,500" in html_body or "1,500" in html_body  # Projected amount
            assert "$500" in html_body or "500" in html_body  # Overage amount

    def test_notify_forecast_overage_webhook(self):
        """Test forecast overage webhook payload."""
        notifier = BillingNotifier(webhook_url="https://hooks.example.com")
        projected_date = datetime.now(timezone.utc) + timedelta(days=7)

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(
                success=False, method="email", error="failed"
            )
            with patch.object(notifier, "_send_webhook") as mock_webhook:
                mock_webhook.return_value = NotificationResult(success=True, method="webhook")

                notifier.notify_forecast_overage(
                    org_id="org-123",
                    email="admin@example.com",
                    org_name="Test Org",
                    budget_name="API Calls",
                    current_spent=600.00,
                    budget_limit=800.00,
                    projected_date=projected_date,
                    projected_amount=1000.00,
                )

                call_args = mock_webhook.call_args
                payload = call_args[0][0]
                assert payload["event"] == "forecast_overage"
                assert payload["current_spent"] == 600.00
                assert payload["budget_limit"] == 800.00
                assert payload["projected_amount"] == 1000.00
                assert payload["overage_amount"] == 200.00

    def test_notify_forecast_overage_days_calculation(self):
        """Test forecast overage correctly calculates days until exceed."""
        notifier = BillingNotifier()
        projected_date = datetime.now(timezone.utc) + timedelta(days=15)

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            notifier.notify_forecast_overage(
                org_id="org-123",
                email="admin@example.com",
                org_name="Test Org",
                budget_name="Monthly Budget",
                current_spent=500.00,
                budget_limit=1000.00,
                projected_date=projected_date,
                projected_amount=1200.00,
            )

            call_args = mock_email.call_args
            subject = call_args[0][1]
            # Should contain the number of days
            assert "15 days" in subject or "14 days" in subject  # Approximate due to timing


# =============================================================================
# Credit Expiring Notification Tests
# =============================================================================


class TestCreditExpiringNotification:
    """Tests for credit expiration notifications."""

    def test_notify_credit_expiring(self):
        """Test credit expiring notification."""
        notifier = BillingNotifier()
        expiration_date = datetime.now(timezone.utc) + timedelta(days=14)

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            result = notifier.notify_credit_expiring(
                org_id="org-123",
                email="admin@example.com",
                org_name="Test Org",
                expiring_amount_cents=5000,  # $50.00
                expiration_date=expiration_date,
                days_until=14,
            )

            assert result.success is True
            call_args = mock_email.call_args
            subject = call_args[0][1]
            assert "$50.00" in subject
            assert "14 days" in subject

    def test_notify_credit_expiring_large_amount(self):
        """Test credit expiring notification with large amount."""
        notifier = BillingNotifier()
        expiration_date = datetime.now(timezone.utc) + timedelta(days=7)

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            notifier.notify_credit_expiring(
                org_id="org-123",
                email="admin@example.com",
                org_name="Test Org",
                expiring_amount_cents=100000,  # $1,000.00
                expiration_date=expiration_date,
                days_until=7,
            )

            call_args = mock_email.call_args
            subject = call_args[0][1]
            assert "$1,000.00" in subject or "$1000.00" in subject

    def test_notify_credit_expiring_webhook(self):
        """Test credit expiring webhook payload."""
        notifier = BillingNotifier(webhook_url="https://hooks.example.com")
        expiration_date = datetime.now(timezone.utc) + timedelta(days=3)

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(
                success=False, method="email", error="failed"
            )
            with patch.object(notifier, "_send_webhook") as mock_webhook:
                mock_webhook.return_value = NotificationResult(success=True, method="webhook")

                notifier.notify_credit_expiring(
                    org_id="org-123",
                    email="admin@example.com",
                    org_name="Test Org",
                    expiring_amount_cents=2500,  # $25.00
                    expiration_date=expiration_date,
                    days_until=3,
                )

                call_args = mock_webhook.call_args
                payload = call_args[0][0]
                assert payload["event"] == "credit_expiring"
                assert payload["expiring_amount_cents"] == 2500
                assert payload["expiring_amount_usd"] == 25.00
                assert payload["days_until"] == 3

    def test_notify_credit_expiring_cents_to_dollars_conversion(self):
        """Test correct conversion from cents to dollars."""
        notifier = BillingNotifier()
        expiration_date = datetime.now(timezone.utc) + timedelta(days=5)

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            notifier.notify_credit_expiring(
                org_id="org-123",
                email="admin@example.com",
                org_name="Test Org",
                expiring_amount_cents=199,  # $1.99
                expiration_date=expiration_date,
                days_until=5,
            )

            call_args = mock_email.call_args
            subject = call_args[0][1]
            assert "$1.99" in subject

    def test_notify_credit_expiring_zero_cents(self):
        """Test credit expiring with zero cents."""
        notifier = BillingNotifier()
        expiration_date = datetime.now(timezone.utc) + timedelta(days=5)

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            result = notifier.notify_credit_expiring(
                org_id="org-123",
                email="admin@example.com",
                org_name="Test Org",
                expiring_amount_cents=0,
                expiration_date=expiration_date,
                days_until=5,
            )

            assert result.success is True
            call_args = mock_email.call_args
            subject = call_args[0][1]
            assert "$0.00" in subject


# =============================================================================
# Global Notifier Tests
# =============================================================================


class TestGlobalNotifier:
    """Tests for global notifier singleton."""

    def test_get_billing_notifier(self):
        """Test getting global billing notifier."""
        notifier = get_billing_notifier()
        assert isinstance(notifier, BillingNotifier)

    def test_get_billing_notifier_singleton(self):
        """Test global notifier is singleton."""
        notifier1 = get_billing_notifier()
        notifier2 = get_billing_notifier()
        assert notifier1 is notifier2


# =============================================================================
# Fallback Chain Tests
# =============================================================================


class TestFallbackChain:
    """Tests for notification fallback chain (email -> webhook -> log)."""

    def test_email_success_no_fallback(self):
        """Test successful email doesn't trigger fallbacks."""
        notifier = BillingNotifier(webhook_url="https://hooks.example.com")

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")
            with patch.object(notifier, "_send_webhook") as mock_webhook:
                result = notifier.notify_payment_failed(
                    org_id="org-123",
                    org_name="Test Org",
                    email="admin@example.com",
                )

                assert result.method == "email"
                mock_webhook.assert_not_called()

    def test_email_fails_webhook_succeeds(self):
        """Test webhook fallback when email fails."""
        notifier = BillingNotifier(webhook_url="https://hooks.example.com")

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(
                success=False, method="email", error="failed"
            )
            with patch.object(notifier, "_send_webhook") as mock_webhook:
                mock_webhook.return_value = NotificationResult(success=True, method="webhook")

                result = notifier.notify_payment_failed(
                    org_id="org-123",
                    org_name="Test Org",
                    email="admin@example.com",
                )

                assert result.method == "webhook"

    def test_all_fail_fallback_to_log(self):
        """Test log fallback when all methods fail."""
        notifier = BillingNotifier(webhook_url="https://hooks.example.com")

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(
                success=False, method="email", error="failed"
            )
            with patch.object(notifier, "_send_webhook") as mock_webhook:
                mock_webhook.return_value = NotificationResult(
                    success=False, method="webhook", error="failed"
                )

                result = notifier.notify_payment_failed(
                    org_id="org-123",
                    org_name="Test Org",
                    email="admin@example.com",
                )

                assert result.success is True
                assert result.method == "log"

    def test_fallback_chain_for_all_notification_types(self):
        """Test fallback chain works for all notification types."""
        notifier = BillingNotifier()
        trial_end = datetime.now(timezone.utc) + timedelta(days=5)
        projected_date = datetime.now(timezone.utc) + timedelta(days=10)
        expiration_date = datetime.now(timezone.utc) + timedelta(days=7)

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(
                success=False, method="email", error="failed"
            )
            with patch.object(notifier, "_send_webhook") as mock_webhook:
                mock_webhook.return_value = NotificationResult(
                    success=False, method="webhook", error="failed"
                )

                # All should fall back to log
                r1 = notifier.notify_trial_ending("org-1", "Org", "e@e.com", 5, trial_end)
                assert r1.method == "log"

                r2 = notifier.notify_subscription_canceled("org-1", "Org", "e@e.com")
                assert r2.method == "log"

                r3 = notifier.notify_downgraded(
                    "org-1", "Org", "e@e.com", SubscriptionTier.PROFESSIONAL
                )
                assert r3.method == "log"

                r4 = notifier.notify_budget_alert("t-1", "e@e.com", "info", "$50", "$100", 50.0)
                assert r4.method == "log"

                r5 = notifier.notify_forecast_overage(
                    "org-1",
                    "e@e.com",
                    "Org",
                    "Budget",
                    500.0,
                    1000.0,
                    projected_date,
                    1200.0,
                )
                assert r5.method == "log"

                r6 = notifier.notify_credit_expiring(
                    "org-1", "e@e.com", "Org", 5000, expiration_date, 7
                )
                assert r6.method == "log"


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_email(self):
        """Test notification with empty email."""
        notifier = BillingNotifier()

        # Should still fall back to log
        result = notifier.notify_payment_failed(
            org_id="org-123",
            org_name="Test Org",
            email="",
        )
        assert result.success is True
        assert result.method == "log"

    def test_zero_percent_budget(self):
        """Test budget alert with 0% usage."""
        notifier = BillingNotifier()

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            result = notifier.notify_budget_alert(
                tenant_id="tenant-123",
                email="admin@example.com",
                alert_level="info",
                current_spend="$0.00",
                budget_limit="$100.00",
                percent_used=0.0,
            )

            assert result.success is True

    def test_negative_days_remaining(self):
        """Test trial notification with negative days (already expired)."""
        notifier = BillingNotifier()
        trial_end = datetime.now(timezone.utc) - timedelta(days=1)

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            result = notifier.notify_trial_ending(
                org_id="org-123",
                org_name="Test Org",
                email="admin@example.com",
                days_remaining=-1,
                trial_end=trial_end,
            )

            assert result.success is True

    def test_very_large_budget(self):
        """Test budget alert with very large amounts."""
        notifier = BillingNotifier()

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            result = notifier.notify_budget_alert(
                tenant_id="tenant-123",
                email="admin@example.com",
                alert_level="warning",
                current_spend="$999,999.99",
                budget_limit="$1,000,000.00",
                percent_used=99.99999,
            )

            assert result.success is True

    def test_special_characters_in_org_name(self):
        """Test notification with special characters in org name."""
        notifier = BillingNotifier()

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            result = notifier.notify_payment_failed(
                org_id="org-123",
                org_name="Test <script>alert('XSS')</script> Org",
                email="admin@example.com",
            )

            assert result.success is True

    def test_unicode_in_org_name(self):
        """Test notification with unicode characters in org name."""
        notifier = BillingNotifier()

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            result = notifier.notify_payment_failed(
                org_id="org-123",
                org_name="Test Org GmbH",
                email="admin@example.com",
            )

            assert result.success is True

    def test_very_high_attempt_count(self):
        """Test payment failed with very high attempt count."""
        notifier = BillingNotifier()

        with patch.object(notifier, "_send_email") as mock_email:
            mock_email.return_value = NotificationResult(success=True, method="email")

            result = notifier.notify_payment_failed(
                org_id="org-123",
                org_name="Test Org",
                email="admin@example.com",
                attempt_count=100,
            )

            assert result.success is True
            # Should still be URGENT (>= 3)
            html_body = mock_email.call_args[0][2]
            assert "URGENT" in html_body
