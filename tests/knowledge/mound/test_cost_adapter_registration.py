"""Tests for knowledge-side billing cost adapter composition."""

from unittest.mock import MagicMock, patch

from aragora.knowledge.mound.cost_adapter_registration import (
    register_billing_cost_adapter,
)


def test_register_billing_cost_adapter_attaches_adapter_to_tracker():
    tracker = MagicMock()
    adapter = MagicMock()

    with (
        patch("aragora.billing.cost_tracker.get_cost_tracker", return_value=tracker),
        patch(
            "aragora.knowledge.mound.adapters.cost_adapter.CostAdapter",
            return_value=adapter,
        ) as adapter_type,
    ):
        register_billing_cost_adapter()

    adapter_type.assert_called_once_with(enable_dual_write=True)
    tracker.set_km_adapter.assert_called_once_with(adapter)
