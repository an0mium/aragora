"""Tests for billing-owned notification sink registration."""

from unittest.mock import AsyncMock

import pytest

from aragora.billing.notification_sink import (
    clear_billing_notification_sink,
    notify_budget_runway,
    notify_cost_anomaly,
    register_billing_notification_sink,
)


@pytest.fixture(autouse=True)
def reset_notification_sink():
    clear_billing_notification_sink()
    yield
    clear_billing_notification_sink()


@pytest.mark.asyncio
async def test_registered_sink_receives_budget_runway_notification():
    sink = AsyncMock()
    register_billing_notification_sink(sink)

    delivered = await notify_budget_runway(
        title="Budget Alert: WARNING",
        message="Budget running low",
        severity="warning",
        workspace_id="ws-1",
    )

    assert delivered is True
    sink.notify_budget_runway.assert_awaited_once_with(
        title="Budget Alert: WARNING",
        message="Budget running low",
        severity="warning",
        workspace_id="ws-1",
    )


@pytest.mark.asyncio
async def test_registered_sink_receives_cost_anomaly_notification():
    sink = AsyncMock()
    register_billing_notification_sink(sink)

    delivered = await notify_cost_anomaly(
        anomaly_type="spike",
        severity="critical",
        amount=12.0,
        expected=3.0,
        workspace_id="ws-1",
        details="unexpected growth",
    )

    assert delivered is True
    sink.notify_cost_anomaly.assert_awaited_once_with(
        anomaly_type="spike",
        severity="critical",
        amount=12.0,
        expected=3.0,
        workspace_id="ws-1",
        details="unexpected growth",
    )


@pytest.mark.asyncio
async def test_missing_sink_is_best_effort():
    assert (
        await notify_budget_runway(
            title="Budget Alert",
            message="warning",
            severity="warning",
            workspace_id="ws-1",
        )
        is False
    )
