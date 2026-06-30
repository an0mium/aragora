"""Tests for the aragora.server.redis_config deprecation shim.

The Redis connection-pool surface moved down to aragora.utils.redis_config
during the P4a layering work so foundation modules (aragora.utils.redis_cache)
can reach it without importing aragora.server. The old module must remain
importable as a shim that:

1. Emits a DeprecationWarning on import naming the old and new paths.
2. Re-exports every public name identity-equal to its canonical target in
   aragora.utils.redis_config (same objects, shared module-level pool/availability
   state -- not copies).
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest

SHIM_MODULE = "aragora.server.redis_config"
CANONICAL_MODULE = "aragora.utils.redis_config"

# Public API the shim must preserve (aragora.utils.redis_config.__all__).
PUBLIC_NAMES = [
    "get_redis_url",
    "get_redis_pool",
    "get_redis_client",
    "get_async_redis_client",
    "is_redis_available",
    "close_redis_pool",
    "reset_redis_state",
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
        assert deprecations, "importing aragora.server.redis_config must emit DeprecationWarning"
        message = str(deprecations[0].message)
        assert "aragora.server.redis_config" in message
        assert "aragora.utils.redis_config" in message

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

    def test_all_matches_public_names(self, modules):
        shim, _ = modules
        assert sorted(shim.__all__) == sorted(PUBLIC_NAMES)

    def test_state_reset_is_shared(self, modules):
        """reset_redis_state via the shim must clear the canonical module's state."""
        shim, canonical = modules
        canonical._redis_available = True
        canonical._redis_pool = object()
        shim.reset_redis_state()
        assert canonical._redis_available is None
        assert canonical._redis_pool is None
