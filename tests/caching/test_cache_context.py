"""
Tests for the CacheContext context manager.

Covers:
- Default enabled state
- Disabling caching within context
- Enabling caching within context
- Nested context managers with state restoration
- Context manager returns self
- is_enabled class method
- Thread safety of CacheContext state changes
- State restoration after exceptions
"""

from __future__ import annotations

import threading

import pytest

from aragora.caching.decorators import CacheContext


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_cache_context():
    """Ensure CacheContext is enabled before and after each test."""
    CacheContext._enabled = True
    yield
    CacheContext._enabled = True


# ===========================================================================
# Test: Default State
# ===========================================================================


class TestDefaultState:
    """Tests for CacheContext default state."""

    def test_default_is_enabled(self):
        """CacheContext is enabled by default."""
        assert CacheContext.is_enabled() is True

    def test_is_enabled_returns_bool(self):
        """is_enabled returns a boolean value."""
        result = CacheContext.is_enabled()
        assert isinstance(result, bool)


# ===========================================================================
# Test: Disabling Cache
# ===========================================================================


class TestDisableCache:
    """Tests for disabling caching via CacheContext."""

    def test_disable_within_context(self):
        """Caching is disabled inside the context block."""
        with CacheContext(enabled=False):
            assert CacheContext.is_enabled() is False

    def test_reenabled_after_context(self):
        """Caching is re-enabled after exiting the context block."""
        with CacheContext(enabled=False):
            pass
        assert CacheContext.is_enabled() is True

    def test_disable_does_not_affect_before(self):
        """State before entering context is not affected."""
        assert CacheContext.is_enabled() is True
        ctx = CacheContext(enabled=False)
        # Not entered yet
        assert CacheContext.is_enabled() is True


# ===========================================================================
# Test: Enabling Cache
# ===========================================================================


class TestEnableCache:
    """Tests for enabling caching via CacheContext."""

    def test_enable_within_disabled_context(self):
        """Caching can be re-enabled inside a disabled context."""
        CacheContext._enabled = False

        with CacheContext(enabled=True):
            assert CacheContext.is_enabled() is True

    def test_enable_context_restores_disabled(self):
        """After an enable context exits, the previous disabled state is restored."""
        CacheContext._enabled = False

        with CacheContext(enabled=True):
            pass

        assert CacheContext.is_enabled() is False

    def test_redundant_enable_no_harm(self):
        """Enabling when already enabled has no harmful effect."""
        assert CacheContext.is_enabled() is True

        with CacheContext(enabled=True):
            assert CacheContext.is_enabled() is True

        assert CacheContext.is_enabled() is True


# ===========================================================================
# Test: Nested Contexts
# ===========================================================================


class TestNestedContexts:
    """Tests for nested CacheContext usage."""

    def test_nested_disable_enable(self):
        """Nested disable then enable restores correctly."""
        assert CacheContext.is_enabled() is True

        with CacheContext(enabled=False):
            assert CacheContext.is_enabled() is False

            with CacheContext(enabled=True):
                assert CacheContext.is_enabled() is True

            assert CacheContext.is_enabled() is False

        assert CacheContext.is_enabled() is True

    def test_nested_disable_disable(self):
        """Nested disable within disable restores correctly."""
        assert CacheContext.is_enabled() is True

        with CacheContext(enabled=False):
            assert CacheContext.is_enabled() is False

            with CacheContext(enabled=False):
                assert CacheContext.is_enabled() is False

            assert CacheContext.is_enabled() is False

        assert CacheContext.is_enabled() is True

    def test_triple_nesting(self):
        """Three levels of nesting restore correctly."""
        with CacheContext(enabled=False):
            with CacheContext(enabled=True):
                with CacheContext(enabled=False):
                    assert CacheContext.is_enabled() is False
                assert CacheContext.is_enabled() is True
            assert CacheContext.is_enabled() is False
        assert CacheContext.is_enabled() is True


# ===========================================================================
# Test: Context Manager Protocol
# ===========================================================================


class TestContextManagerProtocol:
    """Tests for context manager protocol compliance."""

    def test_enter_returns_self(self):
        """__enter__ returns the CacheContext instance itself."""
        ctx = CacheContext(enabled=False)
        with ctx as returned:
            assert returned is ctx

    def test_exit_called(self):
        """__exit__ is called when leaving the context."""
        ctx = CacheContext(enabled=False)
        ctx.__enter__()
        assert CacheContext.is_enabled() is False
        ctx.__exit__(None, None, None)
        assert CacheContext.is_enabled() is True


# ===========================================================================
# Test: Exception Safety
# ===========================================================================


class TestExceptionSafety:
    """Tests for state restoration when exceptions occur."""

    def test_state_restored_after_exception(self):
        """State is restored even if an exception occurs inside the context."""
        assert CacheContext.is_enabled() is True

        with pytest.raises(ValueError):
            with CacheContext(enabled=False):
                assert CacheContext.is_enabled() is False
                raise ValueError("test error")

        assert CacheContext.is_enabled() is True

    def test_nested_exception_restores_outer(self):
        """Exception in inner context restores outer context state."""
        with CacheContext(enabled=False):
            with pytest.raises(RuntimeError):
                with CacheContext(enabled=True):
                    raise RuntimeError("inner error")
            # Should be back to False (outer context)
            assert CacheContext.is_enabled() is False

        assert CacheContext.is_enabled() is True


# ===========================================================================
# Test: Thread Safety
# ===========================================================================


class TestCacheContextThreadSafety:
    """Tests for thread safety of CacheContext state changes."""

    def test_concurrent_context_usage(self):
        """Multiple threads using CacheContext do not raise errors."""
        errors = []

        def toggle_cache(enable: bool):
            try:
                for _ in range(50):
                    with CacheContext(enabled=enable):
                        CacheContext.is_enabled()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=toggle_cache, args=(True,)),
            threading.Thread(target=toggle_cache, args=(False,)),
            threading.Thread(target=toggle_cache, args=(True,)),
            threading.Thread(target=toggle_cache, args=(False,)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
