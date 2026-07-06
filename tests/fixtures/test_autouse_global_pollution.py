"""Regression tests for handler-suite process-global pollution repair."""

from __future__ import annotations

import sys
from threading import Lock
from unittest.mock import MagicMock, patch

from tests.fixtures.autouse import _repair_global_mock_pollution


def test_global_guard_repairs_handler_cache_pollution():
    import aragora.server.handlers as handlers_pkg
    from aragora.server.handlers.selection import SelectionHandler

    handlers_pkg._handler_cache.clear()
    handlers_pkg._all_handlers_cache = None

    mock_handler = MagicMock()
    with patch.object(handlers_pkg, "_lazy_import", return_value=mock_handler):
        polluted_handlers = handlers_pkg._get_all_handlers()

    assert polluted_handlers
    assert all(handler is mock_handler for handler in polluted_handlers)
    assert SelectionHandler not in handlers_pkg.ALL_HANDLERS

    _repair_global_mock_pollution(sys)

    assert SelectionHandler in handlers_pkg.ALL_HANDLERS
    assert not any(isinstance(handler, MagicMock) for handler in handlers_pkg.ALL_HANDLERS)


def test_global_guard_repairs_social_oauth_alias_pollution():
    import aragora.server.handlers.social as social_pkg
    import aragora.server.handlers.social.social_media as social_media

    original_states = social_pkg._oauth_states
    original_states.clear()
    social_media._oauth_states = {}
    social_media._oauth_states_lock = Lock()

    assert social_pkg._oauth_states is original_states
    assert social_pkg._oauth_states is not social_media._oauth_states

    social_pkg._store_oauth_state("polluted-state")

    assert "polluted-state" not in social_pkg._oauth_states
    assert "polluted-state" in social_media._oauth_states

    _repair_global_mock_pollution(sys)

    assert social_pkg._oauth_states is social_media._oauth_states
    assert social_pkg._oauth_states_lock is social_media._oauth_states_lock

    social_pkg._store_oauth_state("guard-state")

    with social_pkg._oauth_states_lock:
        assert "guard-state" in social_pkg._oauth_states
        social_pkg._oauth_states.clear()
