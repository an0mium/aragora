"""Tests for all 4 feature-domain circuit breaker modules.

Covers:
- aragora.server.handlers.features.marketplace.circuit_breaker
- aragora.server.handlers.features.crm.circuit_breaker
- aragora.server.handlers.features.devops.circuit_breaker
- aragora.server.handlers.features.ecommerce.circuit_breaker

Each module wraps ``SimpleCircuitBreaker`` with domain-specific globals,
accessor helpers, status helpers, and reset functions.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Imports: Marketplace
# ---------------------------------------------------------------------------
from aragora.server.handlers.features.marketplace.circuit_breaker import (
    MarketplaceCircuitBreaker,
    _get_circuit_breaker as marketplace_get_cb,
    _get_marketplace_circuit_breaker,
    _reset_circuit_breaker as marketplace_reset,
    get_marketplace_circuit_breaker_status,
)

# ---------------------------------------------------------------------------
# Imports: CRM
# ---------------------------------------------------------------------------
from aragora.server.handlers.features.crm.circuit_breaker import (
    CRMCircuitBreaker,
    get_crm_circuit_breaker,
    reset_crm_circuit_breaker,
)

# ---------------------------------------------------------------------------
# Imports: DevOps
# ---------------------------------------------------------------------------
from aragora.server.handlers.features.devops.circuit_breaker import (
    DevOpsCircuitBreaker,
    get_devops_circuit_breaker,
    get_devops_circuit_breaker_status,
)

# ---------------------------------------------------------------------------
# Imports: Ecommerce
# ---------------------------------------------------------------------------
from aragora.server.handlers.features.ecommerce.circuit_breaker import (
    EcommerceCircuitBreaker,
    get_ecommerce_circuit_breaker,
    reset_ecommerce_circuit_breaker,
)

from aragora.resilience.simple_circuit_breaker import SimpleCircuitBreaker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_all_circuit_breakers():
    """Reset every domain circuit breaker before and after each test."""
    # Marketplace uses lazy init with a module-global; reset destroys the instance
    marketplace_reset()
    # CRM and Ecommerce use eager singletons; reset restores to closed state
    reset_crm_circuit_breaker()
    reset_ecommerce_circuit_breaker()
    # DevOps uses lazy init; reset by patching the module-global to None
    import aragora.server.handlers.features.devops.circuit_breaker as devops_mod

    devops_mod._devops_circuit_breaker = None
    yield
    marketplace_reset()
    reset_crm_circuit_breaker()
    reset_ecommerce_circuit_breaker()
    devops_mod._devops_circuit_breaker = None


# =========================================================================
# Marketplace Circuit Breaker
# =========================================================================


class TestMarketplaceAlias:
    """MarketplaceCircuitBreaker is a backwards-compat alias."""

    def test_alias_is_simple_circuit_breaker(self):
        assert MarketplaceCircuitBreaker is SimpleCircuitBreaker

    def test_can_instantiate_via_alias(self):
        cb = MarketplaceCircuitBreaker("test-alias")
        assert cb.name == "test-alias"
        assert cb.state == SimpleCircuitBreaker.CLOSED


class TestMarketplaceGetCircuitBreaker:
    """Tests for _get_circuit_breaker / _get_marketplace_circuit_breaker."""

    def test_lazy_creation(self):
        """First call creates the instance."""
        cb = marketplace_get_cb()
        assert isinstance(cb, SimpleCircuitBreaker)
        assert cb.name == "Marketplace"

    def test_singleton_behaviour(self):
        """Subsequent calls return the same instance."""
        cb1 = marketplace_get_cb()
        cb2 = marketplace_get_cb()
        assert cb1 is cb2

    def test_public_accessor_delegates(self):
        """_get_marketplace_circuit_breaker returns the same object."""
        cb = _get_marketplace_circuit_breaker()
        assert cb is marketplace_get_cb()

    def test_half_open_max_calls_is_three(self):
        cb = marketplace_get_cb()
        assert cb.half_open_max_calls == 3

    def test_reset_forces_recreation(self):
        """After reset, a new instance is created."""
        cb_before = marketplace_get_cb()
        marketplace_reset()
        cb_after = marketplace_get_cb()
        assert cb_before is not cb_after


class TestMarketplaceStatus:
    """Tests for get_marketplace_circuit_breaker_status."""

    def test_status_returns_dict(self):
        status = get_marketplace_circuit_breaker_status()
        assert isinstance(status, dict)

    def test_status_keys(self):
        status = get_marketplace_circuit_breaker_status()
        expected_keys = {
            "state",
            "failure_count",
            "success_count",
            "failure_threshold",
            "cooldown_seconds",
            "last_failure_time",
        }
        assert expected_keys == set(status.keys())

    def test_initial_state_is_closed(self):
        status = get_marketplace_circuit_breaker_status()
        assert status["state"] == "closed"
        assert status["failure_count"] == 0
        assert status["success_count"] == 0
        assert status["last_failure_time"] is None

    def test_status_reflects_failures(self):
        cb = marketplace_get_cb()
        cb.record_failure()
        cb.record_failure()
        status = get_marketplace_circuit_breaker_status()
        assert status["failure_count"] == 2
        assert status["last_failure_time"] is not None


# =========================================================================
# CRM Circuit Breaker
# =========================================================================


class TestCRMAlias:
    """CRMCircuitBreaker is a backwards-compat alias."""

    def test_alias_is_simple_circuit_breaker(self):
        assert CRMCircuitBreaker is SimpleCircuitBreaker


class TestCRMGetCircuitBreaker:
    """Tests for get_crm_circuit_breaker."""

    def test_returns_simple_circuit_breaker(self):
        cb = get_crm_circuit_breaker()
        assert isinstance(cb, SimpleCircuitBreaker)

    def test_name_is_crm(self):
        cb = get_crm_circuit_breaker()
        assert cb.name == "CRM"

    def test_half_open_max_calls_is_two(self):
        cb = get_crm_circuit_breaker()
        assert cb.half_open_max_calls == 2

    def test_returns_same_instance(self):
        assert get_crm_circuit_breaker() is get_crm_circuit_breaker()


class TestCRMReset:
    """Tests for reset_crm_circuit_breaker."""

    def test_reset_clears_failures(self):
        cb = get_crm_circuit_breaker()
        for _ in range(3):
            cb.record_failure()
        assert cb.get_status()["failure_count"] == 3
        reset_crm_circuit_breaker()
        assert cb.get_status()["failure_count"] == 0

    def test_reset_returns_to_closed(self):
        cb = get_crm_circuit_breaker()
        # Trip the breaker open
        for _ in range(cb.failure_threshold):
            cb.record_failure()
        assert cb.state == SimpleCircuitBreaker.OPEN
        reset_crm_circuit_breaker()
        assert cb.state == SimpleCircuitBreaker.CLOSED

    def test_reset_preserves_same_instance(self):
        """Reset calls .reset() on the existing instance, not a new one."""
        cb_before = get_crm_circuit_breaker()
        reset_crm_circuit_breaker()
        cb_after = get_crm_circuit_breaker()
        assert cb_before is cb_after


# =========================================================================
# DevOps Circuit Breaker
# =========================================================================


class TestDevOpsAlias:
    """DevOpsCircuitBreaker is a backwards-compat alias."""

    def test_alias_is_simple_circuit_breaker(self):
        assert DevOpsCircuitBreaker is SimpleCircuitBreaker


class TestDevOpsGetCircuitBreaker:
    """Tests for get_devops_circuit_breaker."""

    def test_lazy_creation(self):
        cb = get_devops_circuit_breaker()
        assert isinstance(cb, SimpleCircuitBreaker)

    def test_name_is_devops(self):
        cb = get_devops_circuit_breaker()
        assert cb.name == "DevOps"

    def test_half_open_max_calls_is_three(self):
        cb = get_devops_circuit_breaker()
        assert cb.half_open_max_calls == 3

    def test_singleton_behaviour(self):
        cb1 = get_devops_circuit_breaker()
        cb2 = get_devops_circuit_breaker()
        assert cb1 is cb2


class TestDevOpsStatus:
    """Tests for get_devops_circuit_breaker_status."""

    def test_returns_dict(self):
        status = get_devops_circuit_breaker_status()
        assert isinstance(status, dict)

    def test_initial_state_closed(self):
        status = get_devops_circuit_breaker_status()
        assert status["state"] == "closed"

    def test_status_reflects_open_after_threshold(self):
        cb = get_devops_circuit_breaker()
        for _ in range(cb.failure_threshold):
            cb.record_failure()
        status = get_devops_circuit_breaker_status()
        assert status["state"] == "open"


# =========================================================================
# Ecommerce Circuit Breaker
# =========================================================================


class TestEcommerceAlias:
    """EcommerceCircuitBreaker is a backwards-compat alias."""

    def test_alias_is_simple_circuit_breaker(self):
        assert EcommerceCircuitBreaker is SimpleCircuitBreaker


class TestEcommerceGetCircuitBreaker:
    """Tests for get_ecommerce_circuit_breaker."""

    def test_returns_simple_circuit_breaker(self):
        cb = get_ecommerce_circuit_breaker()
        assert isinstance(cb, SimpleCircuitBreaker)

    def test_name_is_ecommerce(self):
        cb = get_ecommerce_circuit_breaker()
        assert cb.name == "Ecommerce"

    def test_half_open_max_calls_is_two(self):
        cb = get_ecommerce_circuit_breaker()
        assert cb.half_open_max_calls == 2

    def test_returns_same_instance(self):
        assert get_ecommerce_circuit_breaker() is get_ecommerce_circuit_breaker()


class TestEcommerceReset:
    """Tests for reset_ecommerce_circuit_breaker."""

    def test_reset_clears_failures(self):
        cb = get_ecommerce_circuit_breaker()
        for _ in range(3):
            cb.record_failure()
        reset_ecommerce_circuit_breaker()
        assert cb.get_status()["failure_count"] == 0

    def test_reset_returns_to_closed(self):
        cb = get_ecommerce_circuit_breaker()
        for _ in range(cb.failure_threshold):
            cb.record_failure()
        assert cb.state == SimpleCircuitBreaker.OPEN
        reset_ecommerce_circuit_breaker()
        assert cb.state == SimpleCircuitBreaker.CLOSED

    def test_reset_preserves_instance(self):
        cb_before = get_ecommerce_circuit_breaker()
        reset_ecommerce_circuit_breaker()
        cb_after = get_ecommerce_circuit_breaker()
        assert cb_before is cb_after


# =========================================================================
# Cross-Domain Behaviour Tests
# =========================================================================


class TestCrossDomainIsolation:
    """Circuit breakers for different domains are independent."""

    def test_marketplace_and_crm_are_separate(self):
        mp = marketplace_get_cb()
        crm = get_crm_circuit_breaker()
        assert mp is not crm
        assert mp.name != crm.name

    def test_devops_and_ecommerce_are_separate(self):
        do = get_devops_circuit_breaker()
        ec = get_ecommerce_circuit_breaker()
        assert do is not ec
        assert do.name != ec.name

    def test_failure_in_one_does_not_affect_another(self):
        crm = get_crm_circuit_breaker()
        ec = get_ecommerce_circuit_breaker()
        # Trip CRM open
        for _ in range(crm.failure_threshold):
            crm.record_failure()
        assert crm.state == SimpleCircuitBreaker.OPEN
        # Ecommerce should still be closed
        assert ec.state == SimpleCircuitBreaker.CLOSED


class TestExportedAll:
    """CRM module defines __all__."""

    def test_crm_all_exports(self):
        from aragora.server.handlers.features.crm import circuit_breaker as crm_mod

        assert "CRMCircuitBreaker" in crm_mod.__all__
        assert "get_crm_circuit_breaker" in crm_mod.__all__
        assert "reset_crm_circuit_breaker" in crm_mod.__all__
