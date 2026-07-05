"""Tests for the aragora.server.storage deprecation shim.

The debate-storage surface moved down to aragora.storage.debate_storage during
the P4a layering work so infrastructure-layer modules (aragora.storage.adapters)
can reach it without importing aragora.server. The old module must remain
importable as a shim that:

1. Emits a DeprecationWarning on import naming the old and new paths.
2. Re-exports every public name identity-equal to its canonical target in
   aragora.storage.debate_storage (same objects, not copies).
"""

from __future__ import annotations

import importlib
import sys
import warnings

SHIM_MODULE = "aragora.server.storage"
CANONICAL_MODULE = "aragora.storage.debate_storage"

# Names that were importable from the old aragora.server.storage location.
PUBLIC_NAMES = [
    "DebateStorage",
    "DebateMetadata",
    "get_debates_db",
    "DB_TIMEOUT",
    "_escape_like_pattern",
    "_validate_sql_identifier",
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
        assert deprecations, "importing aragora.server.storage must emit DeprecationWarning"
        messages = " ".join(str(w.message) for w in deprecations)
        assert "aragora.server.storage" in messages
        assert "aragora.storage.debate_storage" in messages


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

    def test_adapter_import_does_not_route_through_shim(self):
        # Force a cold import so the shim's import-time DeprecationWarning would
        # re-fire if aragora.storage.adapters (or its import chain) still routed
        # through aragora.server.storage instead of aragora.storage.debate_storage.
        # A clean run proves the adapter reaches the storage surface via the
        # canonical module, which is the behavior VAL-P4A-002 requires.
        cold = ("aragora.storage.adapters", CANONICAL_MODULE, SHIM_MODULE)
        saved = {name: sys.modules[name] for name in cold if name in sys.modules}
        try:
            for name in cold:
                sys.modules.pop(name, None)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                importlib.import_module("aragora.storage.adapters")
            offenders = [
                str(w.message)
                for w in caught
                if issubclass(w.category, DeprecationWarning) and SHIM_MODULE in str(w.message)
            ]
            assert not offenders, (
                f"importing aragora.storage.adapters must not route through the "
                f"deprecated {SHIM_MODULE} shim; saw: {offenders}"
            )
        finally:
            sys.modules.update(saved)
