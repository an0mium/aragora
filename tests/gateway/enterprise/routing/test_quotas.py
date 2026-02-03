"""Tests for gateway enterprise routing quotas."""

from __future__ import annotations

import pytest

from aragora.gateway.enterprise.routing.quotas import (
    QuotaStatus,
    QuotaTracker,
    TenantQuotas,
)


# ---------------------------------------------------------------------------
# TenantQuotas dataclass
# ---------------------------------------------------------------------------


class TestTenantQuotas:
    """Tests for TenantQuotas dataclass."""

    def test_defaults(self) -> None:
        q = TenantQuotas()
        assert q.requests_per_minute == 60
        assert q.requests_per_hour == 1000
        assert q.requests_per_day == 10000
        assert q.concurrent_requests == 10
        assert q.bandwidth_bytes_per_minute == 10 * 1024 * 1024
        assert q.warn_threshold == 0.8

    def test_custom_values(self) -> None:
        q = TenantQuotas(requests_per_minute=100, warn_threshold=0.9)
        assert q.requests_per_minute == 100
        assert q.warn_threshold == 0.9


# ---------------------------------------------------------------------------
# QuotaStatus dataclass
# ---------------------------------------------------------------------------


class TestQuotaStatus:
    """Tests for QuotaStatus dataclass."""

    def test_to_dict(self) -> None:
        from datetime import datetime

        reset = datetime(2026, 1, 1, 0, 0, 0)
        s = QuotaStatus(
            tenant_id="t1",
            used=50,
            remaining=10,
            limit=60,
            reset_time=reset,
            quota_type="requests_per_minute",
            is_exceeded=False,
            is_warning=True,
            percentage_used=83.33,
        )
        d = s.to_dict()
        assert d["tenant_id"] == "t1"
        assert d["used"] == 50
        assert d["remaining"] == 10
        assert d["limit"] == 60
        assert d["reset_time"] == reset.isoformat()
        assert d["quota_type"] == "requests_per_minute"
        assert d["is_exceeded"] is False
        assert d["is_warning"] is True
        assert d["percentage_used"] == 83.33

    def test_defaults(self) -> None:
        from datetime import datetime

        s = QuotaStatus(
            tenant_id="t1",
            used=0,
            remaining=60,
            limit=60,
            reset_time=datetime.utcnow(),
        )
        assert s.quota_type == "requests_per_minute"
        assert s.is_exceeded is False
        assert s.is_warning is False
        assert s.percentage_used == 0.0


# ---------------------------------------------------------------------------
# QuotaTracker
# ---------------------------------------------------------------------------


class TestQuotaTracker:
    """Tests for QuotaTracker."""

    @pytest.mark.asyncio
    async def test_first_request_allowed(self) -> None:
        tracker = QuotaTracker()
        quotas = TenantQuotas(requests_per_minute=10)
        allowed, status = await tracker.check_and_consume("t1", quotas)
        assert allowed is True
        assert status is None

    @pytest.mark.asyncio
    async def test_minute_rate_limit(self) -> None:
        tracker = QuotaTracker()
        quotas = TenantQuotas(requests_per_minute=3, requests_per_hour=1000, requests_per_day=10000)

        for _ in range(3):
            allowed, _ = await tracker.check_and_consume("t1", quotas)
            assert allowed is True

        # 4th request should be denied
        allowed, status = await tracker.check_and_consume("t1", quotas)
        assert allowed is False
        assert status is not None
        assert status.quota_type == "requests_per_minute"
        assert status.is_exceeded is True

    @pytest.mark.asyncio
    async def test_concurrent_limit(self) -> None:
        tracker = QuotaTracker()
        quotas = TenantQuotas(concurrent_requests=2)

        # Consume 2 concurrent slots
        await tracker.check_and_consume("t1", quotas)
        await tracker.check_and_consume("t1", quotas)

        # 3rd should fail on concurrent
        allowed, status = await tracker.check_and_consume("t1", quotas)
        assert allowed is False
        assert status is not None
        assert status.quota_type == "concurrent_requests"

    @pytest.mark.asyncio
    async def test_release_concurrent(self) -> None:
        tracker = QuotaTracker()
        quotas = TenantQuotas(concurrent_requests=1)

        await tracker.check_and_consume("t1", quotas)
        # Now at limit
        allowed, _ = await tracker.check_and_consume("t1", quotas)
        assert allowed is False

        # Release one slot
        await tracker.release_concurrent("t1")

        # Should succeed now (but per-minute/hour/day may also constrain)
        # With defaults they should be fine
        allowed, _ = await tracker.check_and_consume("t1", quotas)
        assert allowed is True

    @pytest.mark.asyncio
    async def test_release_concurrent_below_zero(self) -> None:
        tracker = QuotaTracker()
        # Releasing without any consumed should not go negative
        await tracker.release_concurrent("t1")
        assert tracker._concurrent["t1"] == 0

    @pytest.mark.asyncio
    async def test_bandwidth_limit(self) -> None:
        tracker = QuotaTracker()
        quotas = TenantQuotas(bandwidth_bytes_per_minute=100)

        allowed, _ = await tracker.check_and_consume("t1", quotas, bytes_size=80)
        assert allowed is True

        # This would exceed 100 bytes
        allowed, status = await tracker.check_and_consume("t1", quotas, bytes_size=30)
        assert allowed is False
        assert status is not None
        assert status.quota_type == "bandwidth_bytes_per_minute"

    @pytest.mark.asyncio
    async def test_hour_rate_limit(self) -> None:
        tracker = QuotaTracker()
        quotas = TenantQuotas(
            requests_per_minute=1000,
            requests_per_hour=3,
            requests_per_day=10000,
        )

        for _ in range(3):
            allowed, _ = await tracker.check_and_consume("t1", quotas)
            assert allowed is True

        allowed, status = await tracker.check_and_consume("t1", quotas)
        assert allowed is False
        assert status is not None
        assert status.quota_type == "requests_per_hour"

    @pytest.mark.asyncio
    async def test_day_rate_limit(self) -> None:
        tracker = QuotaTracker()
        quotas = TenantQuotas(
            requests_per_minute=1000,
            requests_per_hour=1000,
            requests_per_day=2,
        )

        for _ in range(2):
            allowed, _ = await tracker.check_and_consume("t1", quotas)
            assert allowed is True

        allowed, status = await tracker.check_and_consume("t1", quotas)
        assert allowed is False
        assert status is not None
        assert status.quota_type == "requests_per_day"

    @pytest.mark.asyncio
    async def test_get_status(self) -> None:
        tracker = QuotaTracker()
        quotas = TenantQuotas(requests_per_minute=10)

        await tracker.check_and_consume("t1", quotas)
        await tracker.check_and_consume("t1", quotas)

        statuses = await tracker.get_status("t1", quotas)
        assert "requests_per_minute" in statuses
        assert "requests_per_hour" in statuses
        assert "requests_per_day" in statuses
        assert "concurrent_requests" in statuses

        minute_status = statuses["requests_per_minute"]
        assert minute_status.used == 2
        assert minute_status.remaining == 8
        assert minute_status.limit == 10

    @pytest.mark.asyncio
    async def test_get_status_warning_threshold(self) -> None:
        tracker = QuotaTracker()
        quotas = TenantQuotas(requests_per_minute=5, warn_threshold=0.5)

        # Consume 3 of 5 (60% > 50% threshold)
        for _ in range(3):
            await tracker.check_and_consume("t1", quotas)

        statuses = await tracker.get_status("t1", quotas)
        assert statuses["requests_per_minute"].is_warning is True

    @pytest.mark.asyncio
    async def test_reset(self) -> None:
        tracker = QuotaTracker()
        quotas = TenantQuotas(requests_per_minute=2)

        await tracker.check_and_consume("t1", quotas)
        await tracker.check_and_consume("t1", quotas)

        # At limit
        allowed, _ = await tracker.check_and_consume("t1", quotas)
        assert allowed is False

        await tracker.reset("t1")

        # Should succeed again
        allowed, _ = await tracker.check_and_consume("t1", quotas)
        assert allowed is True

    @pytest.mark.asyncio
    async def test_reset_nonexistent_tenant(self) -> None:
        tracker = QuotaTracker()
        # Should not raise
        await tracker.reset("nonexistent")

    @pytest.mark.asyncio
    async def test_independent_tenants(self) -> None:
        tracker = QuotaTracker()
        quotas = TenantQuotas(requests_per_minute=2)

        await tracker.check_and_consume("t1", quotas)
        await tracker.check_and_consume("t1", quotas)

        # t1 exhausted but t2 should be fine
        allowed, _ = await tracker.check_and_consume("t2", quotas)
        assert allowed is True
