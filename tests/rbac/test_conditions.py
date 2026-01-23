"""Tests for ABAC condition evaluation."""

from datetime import datetime, timezone

import pytest

from aragora.rbac.conditions import (
    ConditionEvaluator,
    ConditionResult,
    IPCondition,
    ResourceOwnerCondition,
    ResourceStatusCondition,
    TagCondition,
    TimeCondition,
    get_condition_evaluator,
)


class TestTimeCondition:
    """Test TimeCondition evaluation."""

    def test_business_hours_allowed(self):
        """Test access during business hours."""
        condition = TimeCondition(
            allowed_hours=(9, 17),
            allowed_days={0, 1, 2, 3, 4},  # Mon-Fri
        )

        # Tuesday at 10:00
        context = {
            "current_time": datetime(2024, 1, 9, 10, 0, tzinfo=timezone.utc),  # Tuesday
        }

        result = condition.evaluate(True, context)
        assert result.satisfied is True

    def test_business_hours_denied_weekend(self):
        """Test access denied on weekend."""
        condition = TimeCondition(
            allowed_hours=(9, 17),
            allowed_days={0, 1, 2, 3, 4},  # Mon-Fri
        )

        # Saturday at 10:00
        context = {
            "current_time": datetime(2024, 1, 13, 10, 0, tzinfo=timezone.utc),  # Saturday
        }

        result = condition.evaluate(True, context)
        assert result.satisfied is False
        assert "Sat" in result.reason

    def test_business_hours_denied_after_hours(self):
        """Test access denied after hours."""
        condition = TimeCondition(
            allowed_hours=(9, 17),
        )

        # Tuesday at 20:00
        context = {
            "current_time": datetime(2024, 1, 9, 20, 0, tzinfo=timezone.utc),
        }

        result = condition.evaluate(True, context)
        assert result.satisfied is False
        assert "20:00" in result.reason

    def test_date_range(self):
        """Test date range restrictions."""
        condition = TimeCondition(
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
        )

        # Within range
        context = {
            "current_time": datetime(2024, 6, 15, tzinfo=timezone.utc),
        }
        result = condition.evaluate(True, context)
        assert result.satisfied is True

        # Before range
        context = {
            "current_time": datetime(2023, 12, 15, tzinfo=timezone.utc),
        }
        result = condition.evaluate(True, context)
        assert result.satisfied is False

        # After range
        context = {
            "current_time": datetime(2025, 1, 15, tzinfo=timezone.utc),
        }
        result = condition.evaluate(True, context)
        assert result.satisfied is False


class TestIPCondition:
    """Test IPCondition evaluation."""

    def test_single_ip_allowed(self):
        """Test single IP in allowlist."""
        condition = IPCondition()
        context = {"ip_address": "10.0.0.1"}

        result = condition.evaluate(["10.0.0.1"], context)
        assert result.satisfied is True

    def test_single_ip_denied(self):
        """Test IP not in allowlist."""
        condition = IPCondition()
        context = {"ip_address": "10.0.0.1"}

        result = condition.evaluate(["10.0.0.2"], context)
        assert result.satisfied is False

    def test_cidr_range_allowed(self):
        """Test IP in CIDR range."""
        condition = IPCondition()
        context = {"ip_address": "10.0.1.50"}

        result = condition.evaluate(["10.0.0.0/8"], context)
        assert result.satisfied is True

    def test_cidr_range_denied(self):
        """Test IP not in CIDR range."""
        condition = IPCondition()
        context = {"ip_address": "192.168.1.1"}

        result = condition.evaluate(["10.0.0.0/8"], context)
        assert result.satisfied is False

    def test_multiple_ranges(self):
        """Test multiple IP ranges."""
        condition = IPCondition()

        result = condition.evaluate(
            ["10.0.0.0/8", "192.168.0.0/16"],
            {"ip_address": "192.168.1.1"},
        )
        assert result.satisfied is True

        result = condition.evaluate(
            ["10.0.0.0/8", "192.168.0.0/16"],
            {"ip_address": "172.16.1.1"},
        )
        assert result.satisfied is False

    def test_blocklist(self):
        """Test IP blocklist."""
        condition = IPCondition()
        context = {"ip_address": "10.0.0.1"}

        result = condition.evaluate(
            {"blocklist": ["10.0.0.1"]},
            context,
        )
        assert result.satisfied is False

    def test_require_private(self):
        """Test private IP requirement."""
        condition = IPCondition()

        # Private IP
        result = condition.evaluate(
            {"require_private": True},
            {"ip_address": "10.0.0.1"},
        )
        assert result.satisfied is True

        # Public IP
        result = condition.evaluate(
            {"require_private": True},
            {"ip_address": "8.8.8.8"},
        )
        assert result.satisfied is False

    def test_missing_ip(self):
        """Test missing IP in context."""
        condition = IPCondition()
        result = condition.evaluate(["10.0.0.1"], {})
        assert result.satisfied is False

    def test_invalid_ip(self):
        """Test invalid IP address."""
        condition = IPCondition()
        result = condition.evaluate(["10.0.0.1"], {"ip_address": "invalid"})
        assert result.satisfied is False


class TestResourceOwnerCondition:
    """Test ResourceOwnerCondition evaluation."""

    def test_direct_owner(self):
        """Test direct resource ownership."""
        condition = ResourceOwnerCondition()
        context = {
            "actor_id": "user-123",
            "resource_owner": "user-123",
        }

        result = condition.evaluate(True, context)
        assert result.satisfied is True

    def test_not_owner(self):
        """Test non-owner access."""
        condition = ResourceOwnerCondition()
        context = {
            "actor_id": "user-123",
            "resource_owner": "user-456",
        }

        result = condition.evaluate(True, context)
        assert result.satisfied is False

    def test_group_owner(self):
        """Test group ownership."""
        condition = ResourceOwnerCondition()
        context = {
            "actor_id": "user-123",
            "resource_owner": "user-456",
            "resource_owner_group": ["user-123", "user-789"],
        }

        result = condition.evaluate(True, context)
        assert result.satisfied is True


class TestResourceStatusCondition:
    """Test ResourceStatusCondition evaluation."""

    def test_single_status_allowed(self):
        """Test single allowed status."""
        condition = ResourceStatusCondition()
        context = {"resource_status": "active"}

        result = condition.evaluate("active", context)
        assert result.satisfied is True

    def test_single_status_denied(self):
        """Test status not allowed."""
        condition = ResourceStatusCondition()
        context = {"resource_status": "deleted"}

        result = condition.evaluate("active", context)
        assert result.satisfied is False

    def test_multiple_statuses(self):
        """Test multiple allowed statuses."""
        condition = ResourceStatusCondition()
        context = {"resource_status": "pending"}

        result = condition.evaluate(["active", "pending"], context)
        assert result.satisfied is True


class TestTagCondition:
    """Test TagCondition evaluation."""

    def test_single_required_tag(self):
        """Test single required tag."""
        condition = TagCondition()
        context = {"resource_tags": ["public", "reviewed"]}

        result = condition.evaluate("public", context)
        assert result.satisfied is True

    def test_missing_required_tag(self):
        """Test missing required tag."""
        condition = TagCondition()
        context = {"resource_tags": ["public"]}

        result = condition.evaluate("reviewed", context)
        assert result.satisfied is False

    def test_all_tags_required(self):
        """Test all tags required."""
        condition = TagCondition()
        context = {"resource_tags": ["public", "reviewed", "approved"]}

        # All present
        result = condition.evaluate(["public", "reviewed"], context)
        assert result.satisfied is True

        # One missing
        context["resource_tags"] = ["public"]
        result = condition.evaluate(["public", "reviewed"], context)
        assert result.satisfied is False

    def test_any_tag_required(self):
        """Test any tag required."""
        condition = TagCondition()
        context = {"resource_tags": ["public"]}

        result = condition.evaluate({"any": ["public", "reviewed"]}, context)
        assert result.satisfied is True

        context["resource_tags"] = ["private"]
        result = condition.evaluate({"any": ["public", "reviewed"]}, context)
        assert result.satisfied is False

    def test_forbidden_tags(self):
        """Test forbidden tags."""
        condition = TagCondition()
        context = {"resource_tags": ["public", "sensitive"]}

        result = condition.evaluate({"none": ["classified"]}, context)
        assert result.satisfied is True

        context["resource_tags"] = ["public", "classified"]
        result = condition.evaluate({"none": ["classified"]}, context)
        assert result.satisfied is False


class TestConditionEvaluator:
    """Test ConditionEvaluator integration."""

    def test_evaluate_multiple_conditions(self):
        """Test evaluating multiple conditions."""
        evaluator = ConditionEvaluator()

        conditions = {
            "ip_address": ["10.0.0.0/8"],
            "resource_status": "active",
        }

        context = {
            "ip_address": "10.0.0.1",
            "resource_status": "active",
        }

        satisfied, results = evaluator.evaluate(conditions, context)
        assert satisfied is True
        assert len(results) == 2

    def test_evaluate_failing_condition(self):
        """Test with one failing condition."""
        evaluator = ConditionEvaluator()

        conditions = {
            "ip_address": ["10.0.0.0/8"],
            "resource_status": "active",
        }

        context = {
            "ip_address": "10.0.0.1",
            "resource_status": "deleted",
        }

        satisfied, results = evaluator.evaluate(conditions, context)
        assert satisfied is False

    def test_custom_condition(self):
        """Test custom condition registration."""
        evaluator = ConditionEvaluator()

        # Register custom condition
        evaluator.register_custom(
            "is_premium",
            lambda expected, ctx: ctx.get("user_tier") == "premium",
        )

        # Test custom condition
        satisfied, _ = evaluator.evaluate(
            {"is_premium": True},
            {"user_tier": "premium"},
        )
        assert satisfied is True

        satisfied, _ = evaluator.evaluate(
            {"is_premium": True},
            {"user_tier": "free"},
        )
        assert satisfied is False

    def test_equality_fallback(self):
        """Test fallback to equality for unknown conditions."""
        evaluator = ConditionEvaluator()

        satisfied, _ = evaluator.evaluate(
            {"custom_field": "expected_value"},
            {"custom_field": "expected_value"},
        )
        assert satisfied is True

        satisfied, _ = evaluator.evaluate(
            {"custom_field": "expected_value"},
            {"custom_field": "other_value"},
        )
        assert satisfied is False


class TestConditionEvaluatorSingleton:
    """Test singleton access."""

    def test_get_condition_evaluator(self):
        """Test getting singleton instance."""
        evaluator = get_condition_evaluator()
        assert evaluator is not None

        # Same instance returned
        evaluator2 = get_condition_evaluator()
        assert evaluator is evaluator2
