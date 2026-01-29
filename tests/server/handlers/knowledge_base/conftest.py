"""
Pytest configuration for knowledge_base handler tests.

Clears the global cache before each test to prevent state pollution.
"""

import pytest


@pytest.fixture(autouse=True)
def clear_global_cache():
    """Clear the global handler cache before each test.

    This prevents state pollution from other tests that may have
    populated the ttl_cache with stale data.
    """
    try:
        from aragora.server.handlers.admin.cache import clear_cache
        clear_cache()
    except ImportError:
        pass  # Cache module not available

    yield  # Run the test

    # Clear after test too for good measure
    try:
        from aragora.server.handlers.admin.cache import clear_cache
        clear_cache()
    except ImportError:
        pass
