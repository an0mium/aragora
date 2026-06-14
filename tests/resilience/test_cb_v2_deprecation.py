"""Tests for the circuit_breaker_v2 deprecation shim.

circuit_breaker_v2 was consolidated into simple_circuit_breaker (deferred
batch 5.2). The v2 module must remain importable as a shim that:

1. Emits a DeprecationWarning on import.
2. Re-exports every public name identity-equal to its canonical target in
   aragora.resilience.simple_circuit_breaker (same objects, shared registry
   state -- not copies).
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest

SHIM_MODULE = "aragora.resilience.circuit_breaker_v2"
CANONICAL_MODULE = "aragora.resilience.simple_circuit_breaker"

# Public API the shim must preserve (original circuit_breaker_v2.__all__).
PUBLIC_NAMES = [
    "CircuitState",
    "CircuitBreakerOpenError",
    "CircuitBreakerConfig",
    "CircuitBreakerStats",
    "BaseCircuitBreaker",
    "with_circuit_breaker",
    "with_circuit_breaker_sync",
    "get_circuit_breaker",
    "reset_all_circuit_breakers",
    "get_all_circuit_breakers",
]

# Private registry internals that existing tests/fixtures reach into.
PRIVATE_NAMES = [
    "_circuit_breakers",
    "_circuit_breakers_lock",
]


def _fresh_import_shim():
    """Import the shim module fresh so import-time warnings re-fire."""
    sys.modules.pop(SHIM_MODULE, None)
    return importlib.import_module(SHIM_MODULE)


class TestDeprecationWarning:
    def test_import_emits_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _fresh_import_shim()

        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deprecations, "importing circuit_breaker_v2 must emit DeprecationWarning"
        message = str(deprecations[0].message)
        assert "circuit_breaker_v2" in message
        assert "simple_circuit_breaker" in message

    def test_cached_reimport_does_not_rewarn(self):
        """A second (cached) import must not re-emit the warning."""
        _fresh_import_shim()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.import_module(SHIM_MODULE)

        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert not deprecations


class TestReExportIdentity:
    @pytest.fixture()
    def modules(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            shim = _fresh_import_shim()
        canonical = importlib.import_module(CANONICAL_MODULE)
        return shim, canonical

    @pytest.mark.parametrize("name", PUBLIC_NAMES)
    def test_public_name_identity(self, modules, name):
        shim, canonical = modules
        assert getattr(shim, name) is getattr(canonical, name)

    @pytest.mark.parametrize("name", PRIVATE_NAMES)
    def test_private_registry_identity(self, modules, name):
        shim, canonical = modules
        assert getattr(shim, name) is getattr(canonical, name)

    def test_all_matches_public_names(self, modules):
        shim, _ = modules
        assert sorted(shim.__all__) == sorted(PUBLIC_NAMES)

    def test_registry_state_is_shared(self, modules):
        """Breakers created via the shim are visible through the canonical API."""
        shim, canonical = modules
        canonical.reset_all_circuit_breakers()
        name = "test_cb_v2_deprecation_shared"
        cb = shim.get_circuit_breaker(name)
        try:
            assert canonical.get_all_circuit_breakers().get(name) is cb
        finally:
            with canonical._circuit_breakers_lock:
                canonical._circuit_breakers.pop(name, None)


class TestPackageLevelExports:
    def test_resilience_package_does_not_warn(self):
        """aragora.resilience must import canonical names, not the shim."""
        import aragora.resilience as resilience
        from aragora.resilience import simple_circuit_breaker

        assert resilience.BaseCircuitBreaker is simple_circuit_breaker.BaseCircuitBreaker
        assert resilience.CircuitBreakerConfig is simple_circuit_breaker.CircuitBreakerConfig
        assert resilience.get_v2_circuit_breaker is simple_circuit_breaker.get_circuit_breaker
