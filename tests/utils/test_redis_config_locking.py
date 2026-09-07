"""Lifecycle lock symmetry for the shared Redis connection-pool globals.

``get_redis_pool()`` guards first-use init with ``_redis_lock``;
``close_redis_pool()`` and ``reset_redis_state()`` must take the same lock so
teardown/reset cannot race a concurrent init on the shared module globals.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import aragora.utils.redis_config as rc


def test_close_and_reset_take_the_redis_lock(monkeypatch):
    real = threading.Lock()
    acquisitions = {"count": 0}

    class TrackingLock:
        def __enter__(self):
            acquisitions["count"] += 1
            real.acquire()
            return self

        def __exit__(self, *exc):
            real.release()
            return False

    monkeypatch.setattr(rc, "_redis_lock", TrackingLock())
    rc.close_redis_pool()
    rc.reset_redis_state()

    assert acquisitions["count"] >= 2


def test_concurrent_get_close_reset_no_deadlock_or_corruption(monkeypatch):
    # No configured Redis -> get_redis_pool() resolves fast and deterministically.
    monkeypatch.delenv("ARAGORA_REDIS_URL", raising=False)
    rc.reset_redis_state()

    def worker():
        for _ in range(50):
            rc.get_redis_pool()
            rc.close_redis_pool()
            rc.reset_redis_state()
        return True

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker) for _ in range(8)]
        results = [future.result(timeout=30) for future in futures]

    assert all(results)

    rc.reset_redis_state()
    assert rc._redis_pool is None
    assert rc._redis_available is None
