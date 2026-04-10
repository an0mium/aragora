"""Tests for swarm delivery policy helpers."""

from __future__ import annotations

import pytest

from aragora.swarm.delivery_policy import apply_delivery_policy, sensitive_scope_categories


@pytest.mark.parametrize(
    ("path", "category"),
    [
        ("aragora/server/auth/session.py", "auth_rbac"),
        ("aragora/billing/invoice_store.py", "billing_payments"),
        ("aragora/secrets/token_store.py", "secrets_credentials"),
        (".github/workflows/release.yml", "deployment_infra"),
        ("aragora/security/scanner.py", "security_enforcement"),
        ("migrations/20260410_drop_old_table.py", "destructive_migrations"),
    ],
)
def test_sensitive_scope_categories_matches_each_category(path: str, category: str) -> None:
    """Known sensitive markers map to their policy categories."""
    assert sensitive_scope_categories([path]) == [category]


def test_sensitive_scope_categories_ignores_empty_paths() -> None:
    """Blank and missing file scopes do not create policy categories."""
    assert sensitive_scope_categories(None) == []
    assert sensitive_scope_categories(["", "   "]) == []


def test_sensitive_scope_categories_deduplicates_preserving_order() -> None:
    """Repeated matches keep the first category occurrence only."""
    categories = sensitive_scope_categories(
        [
            "aragora/server/auth/session.py",
            "aragora/server/auth/rbac.py",
            "aragora/security/scanner.py",
            "aragora/server/auth/oauth.py",
        ]
    )

    assert categories == ["auth_rbac", "security_enforcement"]


def test_sensitive_scope_categories_normalizes_case_and_whitespace() -> None:
    """Path matching is case-insensitive and trims surrounding whitespace."""
    assert sensitive_scope_categories(["  ARAGORA/BILLING/Invoices.py  "]) == ["billing_payments"]


def test_apply_delivery_policy_preserves_safe_low_risk_request() -> None:
    """Safe scopes keep the requested low-risk automation settings."""
    policy = apply_delivery_policy(
        file_scope=["aragora/live/src/components/StatusBadge.tsx"],
        requested_risk="low",
        requested_merge_class="low_risk",
        requested_autonomy_mode="fire_and_forget",
    )

    assert policy == {
        "requested_risk": "low",
        "requested_merge_class": "low_risk",
        "requested_autonomy_mode": "fire_and_forget",
        "effective_risk": "low",
        "effective_merge_class": "low_risk",
        "effective_autonomy_mode": "fire_and_forget",
        "sensitive_scope_categories": [],
        "downgraded": False,
        "policy_reasons": [],
    }


def test_apply_delivery_policy_normalizes_invalid_requests() -> None:
    """Unknown request values fall back to conservative defaults."""
    policy = apply_delivery_policy(
        file_scope=[],
        requested_risk="surprising",
        requested_merge_class="auto",
        requested_autonomy_mode="full_auto",
    )

    assert policy["requested_risk"] == "medium"
    assert policy["requested_merge_class"] == "manual"
    assert policy["requested_autonomy_mode"] == "checkpoint"
    assert policy["effective_risk"] == "medium"
    assert policy["effective_merge_class"] == "manual"
    assert policy["effective_autonomy_mode"] == "checkpoint"
    assert policy["downgraded"] is False
    assert policy["policy_reasons"] == []


def test_apply_delivery_policy_downgrades_sensitive_scope() -> None:
    """Sensitive paths force high risk, manual merge, and checkpoint autonomy."""
    policy = apply_delivery_policy(
        file_scope=["aragora/server/auth/session.py"],
        requested_risk="low",
        requested_merge_class="low_risk",
        requested_autonomy_mode="fire_and_forget",
    )

    assert policy["effective_risk"] == "high"
    assert policy["effective_merge_class"] == "manual"
    assert policy["effective_autonomy_mode"] == "checkpoint"
    assert policy["sensitive_scope_categories"] == ["auth_rbac"]
    assert policy["downgraded"] is True
    assert policy["policy_reasons"] == ["sensitive_scope:auth_rbac"]


def test_apply_delivery_policy_preserves_requested_high_risk() -> None:
    """Sensitive scopes do not lower an already-high requested risk."""
    policy = apply_delivery_policy(
        file_scope=["aragora/security/csrf.py"],
        requested_risk="high",
        requested_merge_class="low_risk",
        requested_autonomy_mode="adaptive",
    )

    assert policy["requested_risk"] == "high"
    assert policy["effective_risk"] == "high"
    assert policy["effective_merge_class"] == "manual"
    assert policy["effective_autonomy_mode"] == "checkpoint"


def test_apply_delivery_policy_reports_each_sensitive_category_once() -> None:
    """Overlapping sensitive scopes emit one reason per matched category."""
    policy = apply_delivery_policy(
        file_scope=[
            "aragora/server/auth/session.py",
            "aragora/server/auth/rbac.py",
            ".github/workflows/deploy.yml",
            "migrations/20260410_drop_old_table.py",
        ],
        requested_risk="medium",
        requested_merge_class="low_risk",
        requested_autonomy_mode="adaptive",
    )

    assert policy["sensitive_scope_categories"] == [
        "auth_rbac",
        "deployment_infra",
        "destructive_migrations",
    ]
    assert policy["policy_reasons"] == [
        "sensitive_scope:auth_rbac",
        "sensitive_scope:deployment_infra",
        "sensitive_scope:destructive_migrations",
    ]


def test_apply_delivery_policy_handles_non_string_scope_values() -> None:
    """Non-string scope values are coerced through the normal path matcher."""
    policy = apply_delivery_policy(
        file_scope=[None, 123, "API_KEY_ROTATION.md"],  # type: ignore[list-item]
        requested_risk="low",
        requested_merge_class="low_risk",
        requested_autonomy_mode="fire_and_forget",
    )

    assert policy["sensitive_scope_categories"] == ["secrets_credentials"]
    assert policy["policy_reasons"] == ["sensitive_scope:secrets_credentials"]
