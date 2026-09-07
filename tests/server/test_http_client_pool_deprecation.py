"""Tests for the aragora.server.http_client_pool deprecation shim.

The shared HTTP client connection pool moved DOWN to
aragora.observability.http_client_pool during the P4a layering work so that
infrastructure-layer modules (aragora.observability, aragora.billing,
aragora.queue) can reach it without importing aragora.server. The old module
must remain importable as a shim that:

1. Emits a DeprecationWarning on import naming the old and new paths.
2. Resolves every public name identity-equal to its canonical target in
   aragora.observability.http_client_pool (same objects, not copies), so that
   monkeypatches against the old dotted path still hit the moved module.
"""

from __future__ import annotations

import importlib
import sys
import warnings

SHIM_MODULE = "aragora.server.http_client_pool"
CANONICAL_MODULE = "aragora.observability.http_client_pool"

# Names that were importable from the old aragora.server.http_client_pool location.
PUBLIC_NAMES = [
    "HTTPClientPool",
    "HTTPPoolConfig",
    "HTTPPoolMetrics",
    "ProviderMetrics",
    "PROVIDER_CONFIGS",
    "get_http_pool",
    "close_http_pool",
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
        assert deprecations, (
            "importing aragora.server.http_client_pool must emit DeprecationWarning"
        )
        messages = " ".join(str(w.message) for w in deprecations)
        assert "aragora.server.http_client_pool" in messages
        assert "aragora.observability.http_client_pool" in messages


class TestReExports:
    def test_public_names_identity_equal(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            shim = _fresh_import_shim()
            canonical = importlib.import_module(CANONICAL_MODULE)

        for name in PUBLIC_NAMES:
            assert getattr(shim, name) is getattr(canonical, name), (
                f"{name} must be identity-equal between shim and canonical module"
            )

    def test_canonical_module_lives_under_observability(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            canonical = importlib.import_module(CANONICAL_MODULE)

        assert canonical.__name__ == CANONICAL_MODULE
        # get_http_pool returns the process-wide singleton instance.
        pool = canonical.get_http_pool()
        assert isinstance(pool, canonical.HTTPClientPool)
