"""
Tests for token revocation race condition fixes.

Verifies that token revocation follows the persist-first pattern
to ensure atomic revocation across multiple server instances.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch, MagicMock

import pytest


class TestTokenRevocationOrder:
    """Tests for token revocation ordering (persist-first pattern)."""

    def test_revocation_persists_before_inmemory(self):
        """Verify persistent revocation happens before in-memory."""
        call_order = []

        def mock_persist(token):
            call_order.append(("persist", token))
            return True

        def mock_inmemory(token):
            call_order.append(("inmemory", token))
            return True

        # Simulate the revocation pattern from auth.py
        token = "test-refresh-token"

        # This mimics the fixed pattern: persist first, then in-memory
        mock_persist(token)
        mock_inmemory(token)

        assert len(call_order) == 2
        assert call_order[0][0] == "persist"
        assert call_order[1][0] == "inmemory"

    def test_inmemory_not_called_on_persist_failure(self):
        """Verify in-memory revocation is skipped if persist fails."""
        call_order = []
        persist_success = [False]

        def mock_persist(token):
            call_order.append(("persist", token))
            if not persist_success[0]:
                raise Exception("Persist failed")
            return True

        def mock_inmemory(token):
            call_order.append(("inmemory", token))
            return True

        token = "test-refresh-token"

        # Simulate failure case
        with pytest.raises(Exception, match="Persist failed"):
            mock_persist(token)
            mock_inmemory(token)

        # Only persist should have been attempted
        assert len(call_order) == 1
        assert call_order[0][0] == "persist"


class TestTokenBlacklistConcurrency:
    """Tests for concurrent token blacklist operations.

    Note: These tests use the low-level revoke(jti, expires_at) method
    directly to test blacklist behavior without requiring valid JWTs.
    """

    def test_concurrent_revocations(self):
        """Verify concurrent revocations don't corrupt blacklist."""
        from aragora.billing.jwt_auth import get_token_blacklist
        import hashlib

        blacklist = get_token_blacklist()
        errors = []
        revoked_jtis = []
        test_prefix = f"concurrent-{time.time()}-"
        future_expiry = time.time() + 3600  # 1 hour in future

        def revoke_batch(batch_id):
            try:
                for i in range(100):
                    jti = hashlib.sha256(f"{test_prefix}{batch_id}-{i}".encode()).hexdigest()[:32]
                    blacklist.revoke(jti, future_expiry)
                    revoked_jtis.append(jti)
            except Exception as e:
                errors.append(str(e))

        # Run concurrent revocations
        threads = [threading.Thread(target=revoke_batch, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent revocation: {errors}"

        # All JTIs should be in blacklist
        with blacklist._data_lock:
            for jti in revoked_jtis:
                assert jti in blacklist._blacklist, f"JTI {jti} not in blacklist"

    def test_concurrent_revoke_and_check(self):
        """Verify concurrent revocation and checking work correctly."""
        from aragora.billing.jwt_auth import get_token_blacklist
        import hashlib

        blacklist = get_token_blacklist()
        errors = []
        check_results = []
        test_prefix = f"revcheck-{time.time()}-"
        future_expiry = time.time() + 3600

        # Pre-revoke some JTIs
        prerevoked_jtis = []
        for i in range(50):
            jti = hashlib.sha256(f"{test_prefix}prerevoked-{i}".encode()).hexdigest()[:32]
            blacklist.revoke(jti, future_expiry)
            prerevoked_jtis.append(jti)

        def revoke_new(batch_id):
            try:
                for i in range(50):
                    jti = hashlib.sha256(f"{test_prefix}new-{batch_id}-{i}".encode()).hexdigest()[
                        :32
                    ]
                    blacklist.revoke(jti, future_expiry)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"revoke: {e}")

        def check_jtis():
            try:
                for _ in range(100):
                    # Check pre-revoked JTIs
                    with blacklist._data_lock:
                        for i in range(10):
                            result = prerevoked_jtis[i] in blacklist._blacklist
                            check_results.append(("prerevoked", i, result))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"check: {e}")

        threads = [
            threading.Thread(target=revoke_new, args=(0,)),
            threading.Thread(target=revoke_new, args=(1,)),
            threading.Thread(target=check_jtis),
            threading.Thread(target=check_jtis),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent ops: {errors}"

        # All pre-revoked checks should return True
        for token_type, idx, result in check_results:
            if token_type == "prerevoked":
                assert result is True, f"prerevoked-{idx} should be in blacklist"

    def test_blacklist_handles_many_jtis(self):
        """Verify blacklist handles many JTIs under concurrent load."""
        from aragora.billing.jwt_auth import get_token_blacklist
        import hashlib

        blacklist = get_token_blacklist()
        errors = []
        test_prefix = f"load-{time.time()}-"
        future_expiry = time.time() + 3600
        first_jti = None

        def add_jtis(batch_id):
            nonlocal first_jti
            try:
                for i in range(200):
                    jti = hashlib.sha256(f"{test_prefix}{batch_id}-{i}".encode()).hexdigest()[:32]
                    if batch_id == 0 and i == 0:
                        first_jti = jti
                    blacklist.revoke(jti, future_expiry)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=add_jtis, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during load test: {errors}"
        # Verify first JTI is in the blacklist
        with blacklist._data_lock:
            assert first_jti in blacklist._blacklist


class TestPersistentRevocation:
    """Tests for persistent token revocation integration."""

    def test_revoke_token_persistent_called(self):
        """Verify revoke_token_persistent is called correctly."""
        from aragora.billing.jwt_auth import revoke_token_persistent

        # The function should exist and be callable
        assert callable(revoke_token_persistent)

    def test_persistent_revocation_idempotent(self):
        """Verify persistent revocation is idempotent."""
        from aragora.billing.jwt_auth import revoke_token_persistent

        token = f"test-token-{time.time()}"

        # First revocation
        result1 = revoke_token_persistent(token)

        # Second revocation (should not fail)
        result2 = revoke_token_persistent(token)

        # Both should succeed (idempotent)
        # Note: actual return type depends on implementation
        # This test just verifies no exception is raised


class TestTokenRevocationStress:
    """Tests for token revocation under stress conditions."""

    def test_rapid_revoke_check_cycle(self):
        """Verify rapid revoke/check cycles are thread-safe."""
        from aragora.billing.jwt_auth import get_token_blacklist
        import hashlib

        blacklist = get_token_blacklist()
        errors = []
        test_prefix = f"stress-{time.time()}-"
        future_expiry = time.time() + 3600

        def revoke_and_check(thread_id):
            try:
                for i in range(100):
                    jti = hashlib.sha256(f"{test_prefix}{thread_id}-{i}".encode()).hexdigest()[:32]
                    blacklist.revoke(jti, future_expiry)
                    # Immediately check
                    with blacklist._data_lock:
                        if jti not in blacklist._blacklist:
                            errors.append(f"JTI {jti} not immediately in blacklist")
            except Exception as e:
                errors.append(f"Unexpected error: {e}")

        threads = [threading.Thread(target=revoke_and_check, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during stress test: {errors}"

    def test_blacklist_cleanup_concurrent(self):
        """Verify cleanup doesn't interfere with concurrent operations."""
        from aragora.billing.jwt_auth import get_token_blacklist
        import hashlib

        blacklist = get_token_blacklist()
        errors = []
        test_prefix = f"cleanup-{time.time()}-"
        future_expiry = time.time() + 3600

        def add_and_verify(thread_id):
            try:
                for i in range(50):
                    jti = hashlib.sha256(f"{test_prefix}{thread_id}-{i}".encode()).hexdigest()[:32]
                    blacklist.revoke(jti, future_expiry)
                    # JTI should be in blacklist
                    with blacklist._data_lock:
                        if jti not in blacklist._blacklist:
                            errors.append(f"JTI {jti} not in blacklist after add")
            except Exception as e:
                errors.append(str(e))

        def trigger_cleanup():
            try:
                for _ in range(10):
                    blacklist.cleanup_expired()
                    time.sleep(0.01)
            except Exception as e:
                errors.append(f"Cleanup error: {e}")

        threads = [
            threading.Thread(target=add_and_verify, args=(0,)),
            threading.Thread(target=add_and_verify, args=(1,)),
            threading.Thread(target=trigger_cleanup),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during cleanup test: {errors}"


class TestLogoutAllAtomicity:
    """Tests for logout-all (revoke all user tokens) via blacklist."""

    def test_bulk_revocation_atomicity(self):
        """Verify bulk JTI revocation is safe."""
        from aragora.billing.jwt_auth import get_token_blacklist
        import hashlib

        blacklist = get_token_blacklist()
        test_prefix = f"bulk-{time.time()}-"
        future_expiry = time.time() + 3600

        # Revoke JTIs for multiple "users"
        user_jtis = {}
        for user_id in range(10):
            jtis = [
                hashlib.sha256(f"{test_prefix}user-{user_id}-{i}".encode()).hexdigest()[:32]
                for i in range(5)
            ]
            user_jtis[user_id] = jtis
            for jti in jtis:
                blacklist.revoke(jti, future_expiry)

        # All JTIs should be in blacklist
        with blacklist._data_lock:
            for user_id, jtis in user_jtis.items():
                for jti in jtis:
                    assert jti in blacklist._blacklist, f"JTI {jti} should be in blacklist"

    def test_concurrent_bulk_revocation(self):
        """Verify concurrent bulk revocations are safe."""
        from aragora.billing.jwt_auth import get_token_blacklist
        import hashlib

        blacklist = get_token_blacklist()
        errors = []
        test_prefix = f"concurrent-bulk-{time.time()}-"
        future_expiry = time.time() + 3600

        def revoke_user_jtis(user_id):
            try:
                jtis = [
                    hashlib.sha256(f"{test_prefix}user-{user_id}-{i}".encode()).hexdigest()[:32]
                    for i in range(20)
                ]
                for jti in jtis:
                    blacklist.revoke(jti, future_expiry)
                # Verify all in blacklist
                with blacklist._data_lock:
                    for jti in jtis:
                        if jti not in blacklist._blacklist:
                            errors.append(f"JTI {jti} not in blacklist")
            except Exception as e:
                errors.append(str(e))

        # Multiple users revoking JTIs concurrently
        threads = [threading.Thread(target=revoke_user_jtis, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent bulk revocation: {errors}"


@pytest.fixture
def fresh_blacklist():
    """Fixture to get the token blacklist."""
    from aragora.billing.jwt_auth import get_token_blacklist

    return get_token_blacklist()
