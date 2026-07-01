"""Two-sided deprecation-shim behavior for the consolidated caching substrate.

Covers VAL-P4A-008: ``aragora.cache``, ``aragora.performance.adaptive_cache``,
and ``aragora.utils.redis_cache`` must (a) import cleanly and re-export their
public surface from ``aragora.caching``, and (b) emit a ``DeprecationWarning``
naming the old path so importing under ``-W error::DeprecationWarning`` fails.
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest

# (module path, token that must appear in its DeprecationWarning message)
SHIMS = [
    ("aragora.cache", "aragora.cache"),
    ("aragora.performance.adaptive_cache", "adaptive_cache"),
    ("aragora.utils.redis_cache", "redis_cache"),
]
MODS = [mod for mod, _ in SHIMS]


@pytest.fixture(autouse=True)
def _restore_cache_modules():
    saved = {name: sys.modules.get(name) for name in MODS}
    yield
    for name, module in saved.items():
        if module is not None:
            sys.modules[name] = module
        else:
            sys.modules.pop(name, None)


@pytest.mark.parametrize(("modname", "token"), SHIMS)
def test_plain_import_warns_with_old_path_token(modname, token):
    sys.modules.pop(modname, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module(modname)
    deprecations = [
        str(w.message)
        for w in caught
        if issubclass(w.category, DeprecationWarning) and token in str(w.message)
    ]
    assert deprecations, (
        f"{modname} did not emit a DeprecationWarning containing {token!r}; "
        f"saw {[str(w.message) for w in caught]}"
    )


@pytest.mark.parametrize("modname", MODS)
def test_import_fails_under_deprecation_as_error(modname):
    sys.modules.pop(modname, None)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(DeprecationWarning):
            importlib.import_module(modname)


def test_back_compat_symbols_resolve_to_new_home():
    from aragora.cache import TTLCache, get_cache  # noqa: F401
    from aragora.performance.adaptive_cache import AdaptiveTTLCache

    import aragora.caching as caching

    assert TTLCache is caching.TTLCache
    assert AdaptiveTTLCache is caching.AdaptiveTTLCache


def test_redis_back_compat_symbols_resolve_to_new_home():
    from aragora.caching.redis import HybridTTLCache as NewHybrid
    from aragora.caching.redis import RedisTTLCache as NewRedis
    from aragora.utils.redis_cache import HybridTTLCache, RedisTTLCache

    assert RedisTTLCache is NewRedis
    assert HybridTTLCache is NewHybrid


def test_distinct_cachestats_apis_preserved():
    # The three CacheStats classes are intentionally distinct (decorator stats,
    # registry/backend stats, adaptive-cache stats) and must not collapse.
    from aragora.caching import CacheStats as DecoratorStats
    from aragora.caching.adaptive import CacheStats as AdaptiveStats
    from aragora.caching.registry import CacheStats as RegistryStats

    assert DecoratorStats is not RegistryStats
    assert DecoratorStats is not AdaptiveStats
    assert RegistryStats is not AdaptiveStats
