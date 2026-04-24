"""Tests for Knowledge Mound connection/narrowing invariants.

Verifies that the internal ``_require_meta_store`` (core.py) and
``_require_client`` (redis_cache.py) helpers:

* Return the underlying handle after initialization / connect so the
  narrowed-path works unchanged from pre-refactor behavior.
* Raise a clear RuntimeError before initialization / connect with a
  "call initialize()/connect() first" hint.
* Do not change the public lifecycle of ``KnowledgeMound`` or ``RedisCache``.

The ``_unsubscribe`` truthy-check regression is also covered: the attribute
must be safe to inspect when never assigned, and ``unsubscribe_from_invalidation_bus``
must be a no-op in that case.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.knowledge.mound.core import KnowledgeMoundCore
from aragora.knowledge.mound.redis_cache import RedisCache
from aragora.knowledge.mound.types import MoundConfig


# ===========================================================================
# KnowledgeMoundCore._require_meta_store
# ===========================================================================


class TestRequireMetaStore:
    """Invariants for the core meta-store narrowing helper."""

    def test_require_meta_store_raises_before_initialize(self) -> None:
        core = KnowledgeMoundCore(
            config=MoundConfig(
                enable_staleness_detection=False,
                enable_culture_accumulator=False,
            )
        )

        with pytest.raises(RuntimeError) as excinfo:
            core._require_meta_store()

        assert "initialize()" in str(excinfo.value)

    def test_require_meta_store_returns_store_when_set(self) -> None:
        core = KnowledgeMoundCore(
            config=MoundConfig(
                enable_staleness_detection=False,
                enable_culture_accumulator=False,
            )
        )
        sentinel = MagicMock(name="meta_store")
        core._meta_store = sentinel

        assert core._require_meta_store() is sentinel

    @pytest.mark.asyncio
    async def test_adapter_methods_use_required_store(self) -> None:
        """Adapter methods should reach through _require_meta_store without
        tripping on the pre-init None -> returns_None invariant."""
        core = KnowledgeMoundCore(
            config=MoundConfig(
                enable_staleness_detection=False,
                enable_culture_accumulator=False,
            )
        )
        fake_store = MagicMock(name="meta_store")
        fake_store.save_node_async = AsyncMock()
        core._meta_store = fake_store

        await core._save_node({"id": "n1", "content": "x"})

        fake_store.save_node_async.assert_awaited_once_with({"id": "n1", "content": "x"})

    @pytest.mark.asyncio
    async def test_adapter_methods_raise_before_init(self) -> None:
        """Adapter methods that touch the meta store should raise with a clear
        message before initialize() sets it."""
        core = KnowledgeMoundCore(
            config=MoundConfig(
                enable_staleness_detection=False,
                enable_culture_accumulator=False,
            )
        )

        with pytest.raises(RuntimeError) as excinfo:
            await core._save_node({"id": "n1", "content": "x"})

        assert "initialize()" in str(excinfo.value)


# ===========================================================================
# RedisCache._require_client and _unsubscribe guarding
# ===========================================================================


class TestRequireClient:
    """Invariants for the redis client narrowing helper."""

    def test_require_client_raises_before_connect(self) -> None:
        cache = RedisCache(url="redis://localhost:6379")

        with pytest.raises(RuntimeError) as excinfo:
            cache._require_client()

        assert "connect()" in str(excinfo.value)

    def test_require_client_raises_when_connected_but_no_client(self) -> None:
        """Defensive: if the flag is set but the handle is None we still refuse."""
        cache = RedisCache(url="redis://localhost:6379")
        cache._connected = True
        cache._client = None

        with pytest.raises(RuntimeError):
            cache._require_client()

    def test_require_client_returns_client_when_connected(self) -> None:
        cache = RedisCache(url="redis://localhost:6379")
        sentinel = MagicMock(name="redis_client")
        cache._client = sentinel
        cache._connected = True

        assert cache._require_client() is sentinel

    @pytest.mark.asyncio
    async def test_get_node_raises_before_connect(self) -> None:
        cache = RedisCache(url="redis://localhost:6379")

        with pytest.raises(RuntimeError):
            await cache.get_node("n-1")

    @pytest.mark.asyncio
    async def test_get_node_uses_required_client(self) -> None:
        cache = RedisCache(url="redis://localhost:6379")
        cache._client = AsyncMock()
        cache._client.get = AsyncMock(return_value=None)
        cache._connected = True

        result = await cache.get_node("n-1")

        assert result is None
        cache._client.get.assert_awaited_once()


class TestUnsubscribeGuard:
    """The _unsubscribe hook is Callable | None; never assume it's set."""

    def test_unsubscribe_default_is_none(self) -> None:
        cache = RedisCache(url="redis://localhost:6379")
        assert cache._unsubscribe is None

    def test_unsubscribe_from_invalidation_bus_is_noop_when_never_subscribed(
        self,
    ) -> None:
        cache = RedisCache(url="redis://localhost:6379")
        # Must not raise and must leave _unsubscribe as None.
        cache.unsubscribe_from_invalidation_bus()
        assert cache._unsubscribe is None

    def test_unsubscribe_calls_handle_and_clears_it(self) -> None:
        cache = RedisCache(url="redis://localhost:6379")
        handle = MagicMock(return_value=None)
        cache._unsubscribe = handle

        cache.unsubscribe_from_invalidation_bus()

        handle.assert_called_once_with()
        assert cache._unsubscribe is None
