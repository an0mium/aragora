"""Tests for tier-based feature gating."""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from unittest.mock import patch, MagicMock

from aragora.billing.tier_gating import (
    TIER_ORDER,
    TIER_DISPLAY_NAMES,
    FEATURE_TIER_MAP,
    tier_sufficient,
    TierInsufficientError,
    require_tier,
    DebateRateLimiter,
    get_debate_rate_limiter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeAuthContext:
    """Minimal AuthorizationContext stand-in for tests."""
    user_id: str = "user-1"
    org_id: str | None = "org-1"
    roles: set[str] = field(default_factory=lambda: {"member"})
    subscription_tier: str | None = None


# ---------------------------------------------------------------------------
# tier_sufficient
# ---------------------------------------------------------------------------

class TestTierSufficient:
    def test_same_tier_is_sufficient(self):
        assert tier_sufficient("professional", "professional") is True

    def test_higher_tier_is_sufficient(self):
        assert tier_sufficient("enterprise", "professional") is True

    def test_lower_tier_is_insufficient(self):
        assert tier_sufficient("free", "professional") is False

    def test_free_is_sufficient_for_free(self):
        assert tier_sufficient("free", "free") is True

    def test_unknown_tier_treated_as_zero(self):
        assert tier_sufficient("unknown", "free") is True
        assert tier_sufficient("unknown", "starter") is False


# ---------------------------------------------------------------------------
# TIER_ORDER and FEATURE_TIER_MAP
# ---------------------------------------------------------------------------

class TestTierConstants:
    def test_tier_order_has_all_tiers(self):
        assert "free" in TIER_ORDER
        assert "starter" in TIER_ORDER
        assert "professional" in TIER_ORDER
        assert "enterprise" in TIER_ORDER
        assert "enterprise_plus" in TIER_ORDER

    def test_tier_order_is_ascending(self):
        tiers = list(TIER_ORDER.keys())
        for i in range(len(tiers) - 1):
            assert TIER_ORDER[tiers[i]] < TIER_ORDER[tiers[i + 1]]

    def test_feature_tier_map_has_entries(self):
        assert "knowledge_mound" in FEATURE_TIER_MAP
        assert "slack_integration" in FEATURE_TIER_MAP
        assert "sso" in FEATURE_TIER_MAP

    def test_display_names_match_tiers(self):
        for tier in TIER_ORDER:
            assert tier in TIER_DISPLAY_NAMES


# ---------------------------------------------------------------------------
# TierInsufficientError
# ---------------------------------------------------------------------------

class TestTierInsufficientError:
    def test_error_message_without_feature(self):
        err = TierInsufficientError("professional", "free")
        assert "Professional" in str(err)

    def test_error_message_with_feature(self):
        err = TierInsufficientError("enterprise", "free", feature="SSO")
        assert "SSO" in str(err)
        assert "Enterprise" in str(err)

    def test_to_response_structure(self):
        err = TierInsufficientError("professional", "free", feature="Knowledge Mound")
        resp = err.to_response()
        assert resp["code"] == "tier_insufficient"
        assert resp["required_tier"] == "professional"
        assert resp["current_tier"] == "free"
        assert resp["upgrade_url"] == "/pricing"
        assert "upgrade_prompt" in resp
        assert "Professional" in resp["upgrade_prompt"]


# ---------------------------------------------------------------------------
# @require_tier decorator
# ---------------------------------------------------------------------------

class TestRequireTierDecorator:
    def test_sync_function_allowed(self):
        """Sync function proceeds when tier is sufficient."""
        @require_tier("professional")
        def my_func(context):
            return "ok"

        ctx = FakeAuthContext(subscription_tier="professional")
        with patch(
            "aragora.billing.tier_gating._resolve_org_tier",
            return_value="professional",
        ):
            assert my_func(context=ctx) == "ok"

    def test_sync_function_blocked(self):
        """Sync function raises TierInsufficientError when tier is too low."""
        @require_tier("professional", feature_name="Knowledge Mound")
        def my_func(context):
            return "ok"

        ctx = FakeAuthContext(subscription_tier="free")
        with patch(
            "aragora.billing.tier_gating._resolve_org_tier",
            return_value="free",
        ):
            with pytest.raises(TierInsufficientError) as exc_info:
                my_func(context=ctx)
            assert exc_info.value.required_tier == "professional"
            assert exc_info.value.current_tier == "free"

    @pytest.mark.asyncio
    async def test_async_function_allowed(self):
        """Async function proceeds when tier is sufficient."""
        @require_tier("enterprise")
        async def my_func(context):
            return "enterprise_result"

        ctx = FakeAuthContext(subscription_tier="enterprise")
        with patch(
            "aragora.billing.tier_gating._resolve_org_tier",
            return_value="enterprise",
        ):
            result = await my_func(context=ctx)
            assert result == "enterprise_result"

    @pytest.mark.asyncio
    async def test_async_function_blocked(self):
        """Async function raises TierInsufficientError when tier is too low."""
        @require_tier("enterprise", feature_name="SSO")
        async def my_func(context):
            return "ok"

        ctx = FakeAuthContext(subscription_tier="starter")
        with patch(
            "aragora.billing.tier_gating._resolve_org_tier",
            return_value="starter",
        ):
            with pytest.raises(TierInsufficientError):
                await my_func(context=ctx)

    def test_graceful_degradation_no_context(self):
        """If no context is found, access is allowed (graceful degradation)."""
        @require_tier("professional")
        def my_func(data):
            return "ok"

        # Passing a plain dict (no user_id/org_id attrs)
        assert my_func(data={"key": "val"}) == "ok"

    def test_graceful_degradation_unresolvable_tier(self):
        """If org tier cannot be resolved, access is allowed."""
        @require_tier("professional")
        def my_func(context):
            return "ok"

        ctx = FakeAuthContext(org_id="org-1")
        with patch(
            "aragora.billing.tier_gating._resolve_org_tier",
            return_value=None,
        ):
            assert my_func(context=ctx) == "ok"

    def test_invalid_tier_name_raises_valueerror(self):
        """Passing an unknown tier to the decorator raises ValueError."""
        with pytest.raises(ValueError, match="Unknown tier"):
            @require_tier("platinum")
            def my_func(context):
                return "ok"

    def test_context_extracted_from_positional_arg(self):
        """Context can be found in positional args by duck typing."""
        @require_tier("professional")
        def my_func(ctx):
            return "ok"

        ctx = FakeAuthContext(subscription_tier="enterprise")
        with patch(
            "aragora.billing.tier_gating._resolve_org_tier",
            return_value="enterprise",
        ):
            assert my_func(ctx) == "ok"

    def test_context_extracted_from_handler_auth_context(self):
        """Context can be found via handler._auth_context attribute."""
        @require_tier("professional")
        def my_func(handler):
            return "ok"

        ctx = FakeAuthContext(subscription_tier="professional")
        handler = MagicMock()
        handler._auth_context = ctx
        # Remove user_id/org_id from handler mock to avoid duck-type match
        del handler.user_id
        del handler.org_id
        with patch(
            "aragora.billing.tier_gating._resolve_org_tier",
            return_value="professional",
        ):
            assert my_func(handler) == "ok"


# ---------------------------------------------------------------------------
# DebateRateLimiter
# ---------------------------------------------------------------------------

class TestDebateRateLimiter:
    def test_allows_within_limit(self):
        limiter = DebateRateLimiter()
        with patch(
            "aragora.billing.tier_gating.TIER_LIMITS",
            create=True,
        ):
            # Use the actual TIER_LIMITS
            result = limiter.check_and_increment("org-1", "free")
            assert result is None  # Allowed

    def test_blocks_at_limit(self):
        limiter = DebateRateLimiter()
        # Fill up to the free tier limit (10 debates)
        for _ in range(10):
            limiter.check_and_increment("org-1", "free")
        # 11th should be blocked
        result = limiter.check_and_increment("org-1", "free")
        assert result is not None
        assert result["code"] == "debate_limit_exceeded"
        assert result["upgrade_url"] == "/pricing"

    def test_different_orgs_independent(self):
        limiter = DebateRateLimiter()
        for _ in range(10):
            limiter.check_and_increment("org-1", "free")
        # org-2 should still be allowed
        result = limiter.check_and_increment("org-2", "free")
        assert result is None

    def test_reset_clears_count(self):
        limiter = DebateRateLimiter()
        for _ in range(10):
            limiter.check_and_increment("org-1", "free")
        limiter.reset("org-1")
        result = limiter.check_and_increment("org-1", "free")
        assert result is None

    def test_get_usage(self):
        limiter = DebateRateLimiter()
        assert limiter.get_usage("org-1") == 0
        limiter.check_and_increment("org-1", "free")
        assert limiter.get_usage("org-1") == 1

    def test_invalid_tier_allows_gracefully(self):
        limiter = DebateRateLimiter()
        result = limiter.check_and_increment("org-1", "nonexistent")
        assert result is None  # Graceful degradation


class TestGetDebateRateLimiter:
    def test_returns_singleton(self):
        # Reset the global
        import aragora.billing.tier_gating as mod
        mod._debate_limiter = None
        limiter1 = get_debate_rate_limiter()
        limiter2 = get_debate_rate_limiter()
        assert limiter1 is limiter2
        # Cleanup
        mod._debate_limiter = None
