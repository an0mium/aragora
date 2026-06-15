"""
Tests for WebSocket authentication tracking in ServerBase.

Tests the new auth state tracking methods added for consistent
authentication enforcement across WebSocket connections.
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from aragora.server.stream.server_base import (
    ServerBase,
    WS_TOKEN_REVALIDATION_INTERVAL,
)


class TestWsAuthStateTracking:
    """Tests for WebSocket auth state management."""

    def test_set_and_get_ws_auth_state(self):
        """Test setting and getting WebSocket auth state."""
        server = ServerBase()
        ws_id = 12345

        server.set_ws_auth_state(
            ws_id=ws_id,
            authenticated=True,
            token="test_token",
            ip_address="192.168.1.1",
        )

        state = server.get_ws_auth_state(ws_id)
        assert state is not None
        assert state["authenticated"] is True
        assert state["token"] == "test_token"
        assert state["ip_address"] == "192.168.1.1"
        assert "last_validated" in state
        assert "created_at" in state

    def test_is_ws_authenticated_true(self):
        """Test is_ws_authenticated returns True for authenticated connection."""
        server = ServerBase()
        ws_id = 12345

        server.set_ws_auth_state(ws_id=ws_id, authenticated=True)
        assert server.is_ws_authenticated(ws_id) is True

    def test_is_ws_authenticated_false(self):
        """Test is_ws_authenticated returns False for unauthenticated connection."""
        server = ServerBase()
        ws_id = 12345

        server.set_ws_auth_state(ws_id=ws_id, authenticated=False)
        assert server.is_ws_authenticated(ws_id) is False

    def test_is_ws_authenticated_missing(self):
        """Test is_ws_authenticated returns False for unknown connection."""
        server = ServerBase()
        assert server.is_ws_authenticated(99999) is False

    def test_remove_ws_auth_state(self):
        """Test removing WebSocket auth state."""
        server = ServerBase()
        ws_id = 12345

        server.set_ws_auth_state(ws_id=ws_id, authenticated=True)
        removed = server.remove_ws_auth_state(ws_id)

        assert removed is not None
        assert removed["authenticated"] is True
        assert server.get_ws_auth_state(ws_id) is None

    def test_remove_ws_auth_state_missing(self):
        """Test removing non-existent auth state returns None."""
        server = ServerBase()
        assert server.remove_ws_auth_state(99999) is None


class TestWsTokenRevalidation:
    """Tests for WebSocket token revalidation timing."""

    def test_should_revalidate_true_after_interval(self):
        """Test token needs revalidation after interval passes."""
        server = ServerBase()
        ws_id = 12345

        # Set auth state with old last_validated time
        server.set_ws_auth_state(ws_id=ws_id, authenticated=True, token="token")

        # Manually set last_validated to the past
        with server._ws_auth_lock:
            server._ws_auth_states[ws_id]["last_validated"] = (
                time.time() - WS_TOKEN_REVALIDATION_INTERVAL - 1
            )

        assert server.should_revalidate_ws_token(ws_id) is True

    def test_should_revalidate_false_within_interval(self):
        """Test token doesn't need revalidation within interval."""
        server = ServerBase()
        ws_id = 12345

        server.set_ws_auth_state(ws_id=ws_id, authenticated=True, token="token")
        # Fresh state should not need revalidation
        assert server.should_revalidate_ws_token(ws_id) is False

    def test_should_revalidate_false_for_unauthenticated(self):
        """Test unauthenticated connections don't need revalidation."""
        server = ServerBase()
        ws_id = 12345

        server.set_ws_auth_state(ws_id=ws_id, authenticated=False, token="token")
        assert server.should_revalidate_ws_token(ws_id) is False

    def test_should_revalidate_false_for_missing(self):
        """Test missing connections don't need revalidation."""
        server = ServerBase()
        assert server.should_revalidate_ws_token(99999) is False

    def test_mark_ws_token_validated(self):
        """Test marking a token as validated updates timestamp."""
        server = ServerBase()
        ws_id = 12345

        server.set_ws_auth_state(ws_id=ws_id, authenticated=True, token="token")

        # Set old validation time
        with server._ws_auth_lock:
            old_time = time.time() - 1000
            server._ws_auth_states[ws_id]["last_validated"] = old_time

        server.mark_ws_token_validated(ws_id)

        state = server.get_ws_auth_state(ws_id)
        assert state["last_validated"] > old_time

    def test_mark_ws_token_validated_missing_noop(self):
        """Test marking missing connection is a no-op."""
        server = ServerBase()
        # Should not raise
        server.mark_ws_token_validated(99999)


class TestWsAuthRevocation:
    """Tests for WebSocket auth revocation."""

    def test_revoke_ws_auth(self):
        """Test revoking WebSocket authentication."""
        server = ServerBase()
        ws_id = 12345

        server.set_ws_auth_state(ws_id=ws_id, authenticated=True, token="token")
        result = server.revoke_ws_auth(ws_id, "Token expired")

        assert result is True
        assert server.is_ws_authenticated(ws_id) is False

        state = server.get_ws_auth_state(ws_id)
        assert state["revoked_at"] is not None
        assert state["revoke_reason"] == "Token expired"

    def test_revoke_ws_auth_missing(self):
        """Test revoking missing connection returns False."""
        server = ServerBase()
        assert server.revoke_ws_auth(99999) is False

    def test_revoke_ws_auth_already_revoked(self):
        """Test revoking already-revoked connection."""
        server = ServerBase()
        ws_id = 12345

        server.set_ws_auth_state(ws_id=ws_id, authenticated=False)
        result = server.revoke_ws_auth(ws_id, "Again")

        assert result is True


class TestWsTokenRetrieval:
    """Tests for retrieving stored WebSocket tokens."""

    def test_get_ws_token(self):
        """Test retrieving stored token."""
        server = ServerBase()
        ws_id = 12345

        server.set_ws_auth_state(ws_id=ws_id, authenticated=True, token="my_token")
        assert server.get_ws_token(ws_id) == "my_token"

    def test_get_ws_token_missing(self):
        """Test getting token for missing connection."""
        server = ServerBase()
        assert server.get_ws_token(99999) is None

    def test_get_ws_token_none_stored(self):
        """Test getting token when None was stored."""
        server = ServerBase()
        ws_id = 12345

        server.set_ws_auth_state(ws_id=ws_id, authenticated=True, token=None)
        assert server.get_ws_token(ws_id) is None


class TestWsAuthCleanup:
    """Tests for WebSocket auth state cleanup."""

    def test_cleanup_ws_auth_states_removes_orphaned(self):
        """Test cleanup removes auth states for disconnected clients."""
        server = ServerBase()

        # Add auth states for connections that are no longer in clients
        server.set_ws_auth_state(ws_id=111, authenticated=True)
        server.set_ws_auth_state(ws_id=222, authenticated=True)

        # Add a mock client that's still connected
        mock_ws = MagicMock()
        mock_ws.__hash__ = lambda self: id(mock_ws)
        server.clients.add(mock_ws)
        ws_id_connected = id(mock_ws)
        server.set_ws_auth_state(ws_id=ws_id_connected, authenticated=True)

        # Cleanup should remove 111 and 222 but keep the connected one
        removed = server.cleanup_ws_auth_states()

        assert removed == 2
        assert server.get_ws_auth_state(111) is None
        assert server.get_ws_auth_state(222) is None
        assert server.get_ws_auth_state(ws_id_connected) is not None

    def test_cleanup_all_includes_auth_states(self):
        """Test cleanup_all includes auth state cleanup."""
        server = ServerBase()

        server.set_ws_auth_state(ws_id=111, authenticated=True)

        results = server.cleanup_all()
        assert "auth_states" in results
        assert results["auth_states"] == 1


class TestWsAuthStats:
    """Tests for WebSocket auth statistics."""

    def test_get_stats_includes_auth_info(self):
        """Test get_stats includes auth state counts."""
        server = ServerBase()

        server.set_ws_auth_state(ws_id=111, authenticated=True)
        server.set_ws_auth_state(ws_id=222, authenticated=False)
        server.set_ws_auth_state(ws_id=333, authenticated=True)

        stats = server.get_stats()

        assert stats["auth_states"] == 3
        assert stats["authenticated_clients"] == 2


class TestWsAuthThreadSafety:
    """Tests for thread-safety of auth tracking."""

    def test_concurrent_auth_state_updates(self):
        """Test concurrent auth state updates are thread-safe."""
        import threading

        server = ServerBase()
        results = []

        def set_and_check(ws_id):
            server.set_ws_auth_state(ws_id=ws_id, authenticated=True, token=f"token_{ws_id}")
            state = server.get_ws_auth_state(ws_id)
            results.append(state is not None and state["token"] == f"token_{ws_id}")

        threads = [threading.Thread(target=set_and_check, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(results)
        assert len(server._ws_auth_states) == 100

    def test_concurrent_revalidation_check(self):
        """Test concurrent revalidation checks are thread-safe."""
        import threading

        server = ServerBase()
        ws_id = 12345
        server.set_ws_auth_state(ws_id=ws_id, authenticated=True, token="token")

        results = []

        def check_revalidation():
            result = server.should_revalidate_ws_token(ws_id)
            results.append(result)

        threads = [threading.Thread(target=check_revalidation) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should get False (fresh state)
        assert all(r is False for r in results)


class TestWsTokenRevalidationInterval:
    """Tests for the revalidation interval constant."""

    def test_revalidation_interval_is_5_minutes(self):
        """Test revalidation interval is 5 minutes (300 seconds)."""
        assert WS_TOKEN_REVALIDATION_INTERVAL == 300.0

    def test_revalidation_boundary(self):
        """Test exact boundary of revalidation interval."""
        server = ServerBase()
        ws_id = 12345

        server.set_ws_auth_state(ws_id=ws_id, authenticated=True, token="token")

        # Set to well within the interval (1 second buffer to avoid timing issues)
        with server._ws_auth_lock:
            server._ws_auth_states[ws_id]["last_validated"] = (
                time.time() - WS_TOKEN_REVALIDATION_INTERVAL + 1.0
            )

        # Within the interval, should not need revalidation
        assert server.should_revalidate_ws_token(ws_id) is False

        # Clearly past the interval (1 second past)
        with server._ws_auth_lock:
            server._ws_auth_states[ws_id]["last_validated"] = (
                time.time() - WS_TOKEN_REVALIDATION_INTERVAL - 1.0
            )

        assert server.should_revalidate_ws_token(ws_id) is True
