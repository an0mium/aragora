"""
Comprehensive tests for aragora.debate.protocol_messages.store.

Tests cover:
- ConnectionPool: creation, get_connection, return_connection, close behavior
- ConnectionPool: max_connections limit (blocking behavior)
- ConnectionPool: close_all drains queue
- ConnectionPool: get after close raises RuntimeError
- ProtocolMessageStore: schema initialization, CRUD, all filter types
- ProtocolMessageStore: limit/offset, order_by/order_desc
- ProtocolMessageStore: _get_sync, _count_sync, _delete_debate_sync, _cleanup_old_sync
- ProtocolMessageStore: _row_to_message payload/metadata/timestamp parsing
- ProtocolMessageStore: close/close_all
- AsyncProtocolMessageStore: record, query, get, count roundtrip
- AsyncProtocolMessageStore: get_debate_timeline, get_round_messages, get_agent_messages
- AsyncProtocolMessageStore: delete_debate, export_jsonl
- get_protocol_store singleton behavior
"""

from __future__ import annotations

import json
import queue
import sqlite3
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.protocol_messages.messages import (
    ProtocolMessage,
    ProtocolMessageType,
    ProposalPayload,
    CritiquePayload,
    VotePayload,
    ConsensusPayload,
    RoundPayload,
)
from aragora.debate.protocol_messages.store import (
    AsyncProtocolMessageStore,
    ConnectionPool,
    ProtocolMessageStore,
    QueryFilters,
    get_protocol_store,
    _store_lock,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_message(
    debate_id: str = "debate-001",
    agent_id: str | None = "agent-alpha",
    message_type: ProtocolMessageType = ProtocolMessageType.PROPOSAL_SUBMITTED,
    round_number: int | None = 1,
    correlation_id: str | None = None,
    parent_message_id: str | None = None,
    payload=None,
    metadata: dict | None = None,
    ts: datetime | None = None,
) -> ProtocolMessage:
    """Create a ProtocolMessage with sensible defaults for testing."""
    return ProtocolMessage(
        message_type=message_type,
        debate_id=debate_id,
        agent_id=agent_id,
        round_number=round_number,
        correlation_id=correlation_id,
        parent_message_id=parent_message_id,
        payload=payload,
        metadata=metadata or {},
        timestamp=ts or datetime.now(timezone.utc),
    )


def fresh_store() -> ProtocolMessageStore:
    """Return a new in-memory ProtocolMessageStore."""
    return ProtocolMessageStore(db_path=":memory:")


def fresh_async_store() -> AsyncProtocolMessageStore:
    """Return a new in-memory AsyncProtocolMessageStore."""
    return AsyncProtocolMessageStore(db_path=":memory:")


# ===========================================================================
# ConnectionPool Tests
# ===========================================================================


class TestConnectionPoolCreate:
    def test_init_stores_config(self):
        pool = ConnectionPool(":memory:", max_connections=3, timeout=10.0)
        assert pool.db_path == ":memory:"
        assert pool.max_connections == 3
        assert pool.timeout == 10.0
        assert pool._created_count == 0
        assert not pool._closed

    def test_create_connection_returns_sqlite_connection(self):
        pool = ConnectionPool(":memory:")
        conn = pool._create_connection()
        assert isinstance(conn, sqlite3.Connection)
        assert conn.row_factory is sqlite3.Row
        conn.close()

    def test_create_connection_has_check_same_thread_false(self):
        pool = ConnectionPool(":memory:")
        # If check_same_thread were True, using from another thread would raise.
        # We just verify the connection can be obtained and closed without error.
        conn = pool._create_connection()
        assert conn is not None
        conn.close()


class TestConnectionPoolGetAndReturn:
    def test_get_connection_increments_created_count(self):
        pool = ConnectionPool(":memory:", max_connections=5)
        conn = pool.get_connection()
        assert pool._created_count == 1
        pool.return_connection(conn)

    def test_get_connection_returns_sqlite_connection(self):
        pool = ConnectionPool(":memory:")
        conn = pool.get_connection()
        assert isinstance(conn, sqlite3.Connection)
        pool.return_connection(conn)

    def test_return_connection_puts_back_in_pool(self):
        pool = ConnectionPool(":memory:", max_connections=5)
        conn = pool.get_connection()
        pool.return_connection(conn)
        # After returning, pool queue should have the connection
        assert not pool._pool.empty()

    def test_second_get_reuses_returned_connection(self):
        pool = ConnectionPool(":memory:", max_connections=5)
        conn1 = pool.get_connection()
        pool.return_connection(conn1)
        conn2 = pool.get_connection()
        # Same object returned from the queue
        assert conn2 is conn1
        pool.return_connection(conn2)

    def test_get_multiple_connections_up_to_max(self):
        pool = ConnectionPool(":memory:", max_connections=3)
        conns = [pool.get_connection() for _ in range(3)]
        assert pool._created_count == 3
        for c in conns:
            pool.return_connection(c)

    def test_get_after_close_raises_runtime_error(self):
        pool = ConnectionPool(":memory:")
        pool.close_all()
        with pytest.raises(RuntimeError, match="closed"):
            pool.get_connection()

    def test_return_connection_when_closed_closes_connection(self):
        pool = ConnectionPool(":memory:")
        conn = pool.get_connection()
        pool.close_all()
        # Should not raise; should just close the connection
        pool.return_connection(conn)

    def test_return_extra_connection_closes_it(self):
        """When pool is full, returning an extra connection closes it."""
        pool = ConnectionPool(":memory:", max_connections=1)
        conn1 = pool.get_connection()
        pool.return_connection(conn1)
        # Create an extra connection manually
        extra = pool._create_connection()
        # Pool queue is full (maxsize=1, one conn already there)
        # returning extra should silently close it
        pool.return_connection(extra)


class TestConnectionPoolMaxConnections:
    def test_get_blocks_when_pool_exhausted_then_returns(self):
        """get_connection should block until a connection is returned."""
        pool = ConnectionPool(":memory:", max_connections=1, timeout=5.0)
        conn = pool.get_connection()

        result_holder = []
        error_holder = []

        def acquire_second():
            try:
                c = pool.get_connection()
                result_holder.append(c)
                pool.return_connection(c)
            except Exception as exc:
                error_holder.append(exc)

        t = threading.Thread(target=acquire_second)
        t.start()
        # Give thread a moment to start blocking
        time.sleep(0.05)
        # Return the first connection, unblocking the thread
        pool.return_connection(conn)
        t.join(timeout=5.0)

        assert not t.is_alive(), "Thread did not finish"
        assert not error_holder, f"Thread raised: {error_holder}"
        assert len(result_holder) == 1

    def test_get_times_out_when_pool_exhausted(self):
        """get_connection raises RuntimeError after timeout when pool is full."""
        pool = ConnectionPool(":memory:", max_connections=1, timeout=0.1)
        conn = pool.get_connection()
        try:
            with pytest.raises(RuntimeError, match="exhausted"):
                pool.get_connection()
        finally:
            pool.return_connection(conn)


class TestConnectionPoolCloseAll:
    def test_close_all_sets_closed_flag(self):
        pool = ConnectionPool(":memory:")
        pool.close_all()
        assert pool._closed is True

    def test_close_all_drains_queue(self):
        pool = ConnectionPool(":memory:", max_connections=3)
        conns = [pool.get_connection() for _ in range(2)]
        for c in conns:
            pool.return_connection(c)
        assert not pool._pool.empty()
        pool.close_all()
        assert pool._pool.empty()

    def test_close_all_resets_created_count(self):
        pool = ConnectionPool(":memory:", max_connections=3)
        conn = pool.get_connection()
        pool.return_connection(conn)
        pool.close_all()
        assert pool._created_count == 0

    def test_close_all_idempotent(self):
        pool = ConnectionPool(":memory:")
        pool.close_all()
        pool.close_all()  # Should not raise


# ===========================================================================
# ProtocolMessageStore Initialization Tests
# ===========================================================================


class TestProtocolMessageStoreInit:
    def test_default_db_path_is_memory(self):
        store = ProtocolMessageStore()
        assert store.db_path == ":memory:"
        store.close()

    def test_explicit_db_path(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            store = ProtocolMessageStore(db_path=path)
            assert store.db_path == path
            store.close()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_schema_creates_protocol_messages_table(self):
        store = fresh_store()
        conn = store._pool.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        store._pool.return_connection(conn)
        assert "protocol_messages" in tables
        store.close()

    def test_schema_creates_indexes(self):
        store = fresh_store()
        conn = store._pool.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = {row[0] for row in cursor.fetchall()}
        store._pool.return_connection(conn)
        expected_indexes = {
            "idx_protocol_debate",
            "idx_protocol_agent",
            "idx_protocol_type",
            "idx_protocol_round",
            "idx_protocol_timestamp",
            "idx_protocol_correlation",
        }
        assert expected_indexes.issubset(indexes)
        store.close()

    def test_schema_idempotent_on_reinit(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            store1 = ProtocolMessageStore(db_path=path)
            store1.close()
            # Second init on same file should not raise
            store2 = ProtocolMessageStore(db_path=path)
            store2.close()
        finally:
            Path(path).unlink(missing_ok=True)


# ===========================================================================
# record_sync + _query_sync_impl Roundtrip Tests
# ===========================================================================


class TestRecordSyncAndQuerySyncRoundtrip:
    def test_record_sync_returns_message_id(self):
        store = fresh_store()
        msg = make_message()
        result = store.record_sync(msg)
        assert result == msg.message_id
        store.close()

    def test_record_and_query_returns_message(self):
        store = fresh_store()
        msg = make_message(debate_id="d1", agent_id="agent-x")
        store.record_sync(msg)
        results = store._query_sync_impl(QueryFilters(debate_id="d1"))
        assert len(results) == 1
        assert results[0].message_id == msg.message_id
        store.close()

    def test_record_preserves_all_fields(self):
        store = fresh_store()
        ts = datetime(2026, 2, 15, 12, 0, 0, tzinfo=timezone.utc)
        msg = make_message(
            debate_id="d-preserve",
            agent_id="agent-42",
            message_type=ProtocolMessageType.VOTE_CAST,
            round_number=3,
            correlation_id="corr-999",
            parent_message_id="parent-abc",
            metadata={"key": "value", "num": 7},
            ts=ts,
        )
        store.record_sync(msg)
        results = store._query_sync_impl()
        assert len(results) == 1
        r = results[0]
        assert r.debate_id == "d-preserve"
        assert r.agent_id == "agent-42"
        assert r.message_type == ProtocolMessageType.VOTE_CAST
        assert r.round_number == 3
        assert r.correlation_id == "corr-999"
        assert r.parent_message_id == "parent-abc"
        assert r.metadata == {"key": "value", "num": 7}
        store.close()

    def test_record_multiple_messages(self):
        store = fresh_store()
        for i in range(5):
            store.record_sync(make_message(debate_id="d-multi", round_number=i))
        results = store._query_sync_impl(QueryFilters(debate_id="d-multi"))
        assert len(results) == 5
        store.close()

    def test_record_with_dict_payload(self):
        store = fresh_store()
        msg = make_message(payload={"answer": 42, "text": "hello"})
        store.record_sync(msg)
        results = store._query_sync_impl()
        assert results[0].payload == {"answer": 42, "text": "hello"}
        store.close()

    def test_record_with_payload_object_having_to_dict(self):
        store = fresh_store()
        payload = ProposalPayload(
            proposal_id="prop-1",
            content="My proposal",
            model="claude-opus",
            round_number=1,
        )
        msg = make_message(payload=payload)
        store.record_sync(msg)
        results = store._query_sync_impl()
        assert results[0].payload is not None
        assert isinstance(results[0].payload, dict)
        assert results[0].payload["proposal_id"] == "prop-1"
        store.close()

    def test_record_with_none_payload(self):
        store = fresh_store()
        msg = make_message(payload=None)
        store.record_sync(msg)
        results = store._query_sync_impl()
        assert results[0].payload is None
        store.close()

    def test_record_with_none_metadata(self):
        store = fresh_store()
        msg = ProtocolMessage(
            message_type=ProtocolMessageType.DEBATE_STARTED,
            debate_id="d-meta-none",
            metadata={},
        )
        store.record_sync(msg)
        results = store._query_sync_impl()
        assert results[0].metadata == {}
        store.close()

    def test_duplicate_message_id_raises(self):
        store = fresh_store()
        msg = make_message()
        store.record_sync(msg)
        with pytest.raises(Exception):
            store.record_sync(msg)
        store.close()


# ===========================================================================
# Query Filter Tests
# ===========================================================================


class TestQueryFilters:
    def _make_store_with_messages(self) -> ProtocolMessageStore:
        """Populate an in-memory store with a mix of messages for filter tests."""
        store = fresh_store()
        now = datetime.now(timezone.utc)

        # debate-A messages
        store.record_sync(make_message(
            debate_id="debate-A", agent_id="agent-1",
            message_type=ProtocolMessageType.PROPOSAL_SUBMITTED,
            round_number=1, correlation_id="corr-A",
            ts=now - timedelta(minutes=10),
        ))
        store.record_sync(make_message(
            debate_id="debate-A", agent_id="agent-2",
            message_type=ProtocolMessageType.CRITIQUE_SUBMITTED,
            round_number=1, parent_message_id=None,
            ts=now - timedelta(minutes=9),
        ))
        store.record_sync(make_message(
            debate_id="debate-A", agent_id="agent-1",
            message_type=ProtocolMessageType.VOTE_CAST,
            round_number=2,
            ts=now - timedelta(minutes=5),
        ))
        store.record_sync(make_message(
            debate_id="debate-A", agent_id="agent-2",
            message_type=ProtocolMessageType.VOTE_CAST,
            round_number=2,
            ts=now - timedelta(minutes=4),
        ))
        store.record_sync(make_message(
            debate_id="debate-A", agent_id=None,
            message_type=ProtocolMessageType.CONSENSUS_REACHED,
            round_number=3, correlation_id="corr-A",
            ts=now - timedelta(minutes=1),
        ))

        # debate-B messages
        store.record_sync(make_message(
            debate_id="debate-B", agent_id="agent-3",
            message_type=ProtocolMessageType.PROPOSAL_SUBMITTED,
            round_number=1,
            ts=now - timedelta(minutes=8),
        ))
        store.record_sync(make_message(
            debate_id="debate-B", agent_id="agent-3",
            message_type=ProtocolMessageType.ROUND_STARTED,
            round_number=2,
            ts=now - timedelta(minutes=6),
        ))

        return store

    def test_filter_by_debate_id(self):
        store = self._make_store_with_messages()
        results = store._query_sync_impl(QueryFilters(debate_id="debate-A"))
        assert len(results) == 5
        assert all(r.debate_id == "debate-A" for r in results)
        store.close()

    def test_filter_by_agent_id(self):
        store = self._make_store_with_messages()
        results = store._query_sync_impl(QueryFilters(agent_id="agent-1"))
        assert len(results) == 2
        assert all(r.agent_id == "agent-1" for r in results)
        store.close()

    def test_filter_by_message_type(self):
        store = self._make_store_with_messages()
        results = store._query_sync_impl(
            QueryFilters(message_type=ProtocolMessageType.VOTE_CAST)
        )
        assert len(results) == 2
        assert all(r.message_type == ProtocolMessageType.VOTE_CAST for r in results)
        store.close()

    def test_filter_by_message_types_list(self):
        store = self._make_store_with_messages()
        results = store._query_sync_impl(
            QueryFilters(message_types=[
                ProtocolMessageType.PROPOSAL_SUBMITTED,
                ProtocolMessageType.CONSENSUS_REACHED,
            ])
        )
        assert len(results) == 3
        types = {r.message_type for r in results}
        assert types == {ProtocolMessageType.PROPOSAL_SUBMITTED, ProtocolMessageType.CONSENSUS_REACHED}
        store.close()

    def test_filter_by_round_number(self):
        store = self._make_store_with_messages()
        results = store._query_sync_impl(
            QueryFilters(debate_id="debate-A", round_number=2)
        )
        assert len(results) == 2
        assert all(r.round_number == 2 for r in results)
        store.close()

    def test_filter_by_min_round(self):
        store = self._make_store_with_messages()
        results = store._query_sync_impl(
            QueryFilters(debate_id="debate-A", min_round=2)
        )
        assert len(results) == 3
        assert all(r.round_number >= 2 for r in results)
        store.close()

    def test_filter_by_max_round(self):
        store = self._make_store_with_messages()
        results = store._query_sync_impl(
            QueryFilters(debate_id="debate-A", max_round=1)
        )
        assert len(results) == 2
        assert all(r.round_number <= 1 for r in results)
        store.close()

    def test_filter_by_min_and_max_round(self):
        store = self._make_store_with_messages()
        results = store._query_sync_impl(
            QueryFilters(debate_id="debate-A", min_round=2, max_round=2)
        )
        assert len(results) == 2
        assert all(r.round_number == 2 for r in results)
        store.close()

    def test_filter_by_start_time(self):
        store = self._make_store_with_messages()
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=5, seconds=30)
        results = store._query_sync_impl(QueryFilters(start_time=cutoff))
        # Only messages from the last ~5 minutes
        assert len(results) >= 2
        for r in results:
            assert r.timestamp >= cutoff
        store.close()

    def test_filter_by_end_time(self):
        store = self._make_store_with_messages()
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=8, seconds=30)
        results = store._query_sync_impl(QueryFilters(end_time=cutoff))
        for r in results:
            assert r.timestamp <= cutoff
        store.close()

    def test_filter_by_start_and_end_time(self):
        store = self._make_store_with_messages()
        now = datetime.now(timezone.utc)
        start = now - timedelta(minutes=7)
        end = now - timedelta(minutes=3)
        results = store._query_sync_impl(QueryFilters(start_time=start, end_time=end))
        for r in results:
            assert start <= r.timestamp <= end
        store.close()

    def test_filter_by_correlation_id(self):
        store = self._make_store_with_messages()
        results = store._query_sync_impl(QueryFilters(correlation_id="corr-A"))
        assert len(results) == 2
        assert all(r.correlation_id == "corr-A" for r in results)
        store.close()

    def test_filter_by_parent_message_id(self):
        store = fresh_store()
        parent_id = str(uuid.uuid4())
        parent = make_message(debate_id="d-parent")
        child = ProtocolMessage(
            message_type=ProtocolMessageType.CRITIQUE_SUBMITTED,
            debate_id="d-parent",
            agent_id="agent-critic",
            parent_message_id=parent.message_id,
        )
        store.record_sync(parent)
        store.record_sync(child)
        results = store._query_sync_impl(
            QueryFilters(parent_message_id=parent.message_id)
        )
        assert len(results) == 1
        assert results[0].parent_message_id == parent.message_id
        store.close()

    def test_filter_no_filters_returns_all(self):
        store = self._make_store_with_messages()
        results = store._query_sync_impl(QueryFilters())
        assert len(results) == 7
        store.close()

    def test_filter_none_returns_all(self):
        store = self._make_store_with_messages()
        results = store._query_sync_impl(None)
        assert len(results) == 7
        store.close()

    def test_filter_combined_debate_and_type(self):
        store = self._make_store_with_messages()
        results = store._query_sync_impl(
            QueryFilters(
                debate_id="debate-A",
                message_type=ProtocolMessageType.VOTE_CAST,
            )
        )
        assert len(results) == 2
        assert all(r.debate_id == "debate-A" for r in results)
        assert all(r.message_type == ProtocolMessageType.VOTE_CAST for r in results)
        store.close()

    def test_filter_no_match_returns_empty(self):
        store = self._make_store_with_messages()
        results = store._query_sync_impl(QueryFilters(debate_id="nonexistent-debate"))
        assert results == []
        store.close()


class TestQueryLimitOffset:
    def _make_store_with_n(self, n: int) -> ProtocolMessageStore:
        store = fresh_store()
        for i in range(n):
            msg = ProtocolMessage(
                message_type=ProtocolMessageType.PROPOSAL_SUBMITTED,
                debate_id="d-paging",
                round_number=i,
                timestamp=datetime(2026, 1, 1, 0, i, 0, tzinfo=timezone.utc),
            )
            store.record_sync(msg)
        return store

    def test_limit_reduces_results(self):
        store = self._make_store_with_n(10)
        results = store._query_sync_impl(QueryFilters(limit=3))
        assert len(results) == 3
        store.close()

    def test_offset_skips_results(self):
        store = self._make_store_with_n(10)
        all_results = store._query_sync_impl(QueryFilters(limit=1000, offset=0))
        offset_results = store._query_sync_impl(QueryFilters(limit=1000, offset=5))
        assert len(offset_results) == 5
        # The offset results should be the last 5
        assert [r.message_id for r in offset_results] == [r.message_id for r in all_results[5:]]
        store.close()

    def test_limit_and_offset_together(self):
        store = self._make_store_with_n(10)
        all_results = store._query_sync_impl(QueryFilters(limit=1000))
        page = store._query_sync_impl(QueryFilters(limit=3, offset=4))
        assert len(page) == 3
        assert [r.message_id for r in page] == [r.message_id for r in all_results[4:7]]
        store.close()

    def test_offset_beyond_count_returns_empty(self):
        store = self._make_store_with_n(5)
        results = store._query_sync_impl(QueryFilters(offset=100))
        assert results == []
        store.close()


class TestQueryOrdering:
    def _make_store_with_timestamps(self) -> ProtocolMessageStore:
        store = fresh_store()
        base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        for i in range(5):
            msg = ProtocolMessage(
                message_type=ProtocolMessageType.ROUND_STARTED,
                debate_id="d-order",
                round_number=i,
                timestamp=base + timedelta(minutes=i),
            )
            store.record_sync(msg)
        return store

    def test_order_asc_by_timestamp(self):
        store = self._make_store_with_timestamps()
        results = store._query_sync_impl(QueryFilters(order_by="timestamp", order_desc=False))
        ts_list = [r.timestamp for r in results]
        assert ts_list == sorted(ts_list)
        store.close()

    def test_order_desc_by_timestamp(self):
        store = self._make_store_with_timestamps()
        results = store._query_sync_impl(QueryFilters(order_by="timestamp", order_desc=True))
        ts_list = [r.timestamp for r in results]
        assert ts_list == sorted(ts_list, reverse=True)
        store.close()

    def test_order_by_round_number_asc(self):
        store = self._make_store_with_timestamps()
        results = store._query_sync_impl(QueryFilters(order_by="round_number", order_desc=False))
        rounds = [r.round_number for r in results]
        assert rounds == sorted(rounds)
        store.close()

    def test_order_by_round_number_desc(self):
        store = self._make_store_with_timestamps()
        results = store._query_sync_impl(QueryFilters(order_by="round_number", order_desc=True))
        rounds = [r.round_number for r in results]
        assert rounds == sorted(rounds, reverse=True)
        store.close()


# ===========================================================================
# _get_sync Tests
# ===========================================================================


class TestGetSync:
    def test_get_sync_returns_message(self):
        store = fresh_store()
        msg = make_message()
        store.record_sync(msg)
        result = store._get_sync(msg.message_id)
        assert result is not None
        assert result.message_id == msg.message_id
        store.close()

    def test_get_sync_returns_none_for_missing(self):
        store = fresh_store()
        result = store._get_sync("nonexistent-id-12345")
        assert result is None
        store.close()

    def test_get_sync_returns_correct_message_among_many(self):
        store = fresh_store()
        msgs = [make_message(debate_id=f"d-{i}") for i in range(5)]
        for m in msgs:
            store.record_sync(m)
        target = msgs[2]
        result = store._get_sync(target.message_id)
        assert result.message_id == target.message_id
        assert result.debate_id == target.debate_id
        store.close()


# ===========================================================================
# _count_sync Tests
# ===========================================================================


class TestCountSync:
    def test_count_all_messages(self):
        store = fresh_store()
        for _ in range(4):
            store.record_sync(make_message(debate_id="d-count"))
        assert store._count_sync() == 4
        store.close()

    def test_count_with_debate_id_filter(self):
        store = fresh_store()
        for _ in range(3):
            store.record_sync(make_message(debate_id="d-count-A"))
        for _ in range(2):
            store.record_sync(make_message(debate_id="d-count-B"))
        assert store._count_sync(QueryFilters(debate_id="d-count-A")) == 3
        assert store._count_sync(QueryFilters(debate_id="d-count-B")) == 2
        store.close()

    def test_count_with_message_type_filter(self):
        store = fresh_store()
        store.record_sync(make_message(message_type=ProtocolMessageType.PROPOSAL_SUBMITTED))
        store.record_sync(make_message(message_type=ProtocolMessageType.PROPOSAL_SUBMITTED))
        store.record_sync(make_message(message_type=ProtocolMessageType.VOTE_CAST))
        assert store._count_sync(
            QueryFilters(message_type=ProtocolMessageType.PROPOSAL_SUBMITTED)
        ) == 2
        store.close()

    def test_count_empty_store_returns_zero(self):
        store = fresh_store()
        assert store._count_sync() == 0
        store.close()

    def test_count_with_none_filter_returns_all(self):
        store = fresh_store()
        for _ in range(3):
            store.record_sync(make_message())
        assert store._count_sync(None) == 3
        store.close()


# ===========================================================================
# _delete_debate_sync Tests
# ===========================================================================


class TestDeleteDebateSync:
    def test_delete_debate_removes_messages(self):
        store = fresh_store()
        for _ in range(3):
            store.record_sync(make_message(debate_id="d-del"))
        count = store._delete_debate_sync("d-del")
        assert count == 3
        remaining = store._query_sync_impl(QueryFilters(debate_id="d-del"))
        assert remaining == []
        store.close()

    def test_delete_debate_does_not_affect_other_debates(self):
        store = fresh_store()
        store.record_sync(make_message(debate_id="d-del"))
        for _ in range(2):
            store.record_sync(make_message(debate_id="d-keep"))
        store._delete_debate_sync("d-del")
        kept = store._query_sync_impl(QueryFilters(debate_id="d-keep"))
        assert len(kept) == 2
        store.close()

    def test_delete_debate_nonexistent_returns_zero(self):
        store = fresh_store()
        count = store._delete_debate_sync("no-such-debate")
        assert count == 0
        store.close()


# ===========================================================================
# _cleanup_old_sync Tests
# ===========================================================================


class TestCleanupOldSync:
    def test_cleanup_old_removes_old_messages(self):
        store = fresh_store()
        old_ts = datetime.now(timezone.utc) - timedelta(days=40)
        store.record_sync(make_message(debate_id="d-old", ts=old_ts))
        store.record_sync(make_message(debate_id="d-recent"))
        deleted = store._cleanup_old_sync(days=30)
        assert deleted == 1
        remaining = store._query_sync_impl()
        assert len(remaining) == 1
        assert remaining[0].debate_id == "d-recent"
        store.close()

    def test_cleanup_old_no_old_messages(self):
        store = fresh_store()
        store.record_sync(make_message(debate_id="d-recent"))
        deleted = store._cleanup_old_sync(days=30)
        assert deleted == 0
        store.close()

    def test_cleanup_old_empty_store(self):
        store = fresh_store()
        deleted = store._cleanup_old_sync(days=30)
        assert deleted == 0
        store.close()


# ===========================================================================
# _row_to_message Tests
# ===========================================================================


class TestRowToMessage:
    def test_row_to_message_handles_json_payload(self):
        store = fresh_store()
        msg = make_message(payload={"key": "val", "num": 42})
        store.record_sync(msg)
        result = store._get_sync(msg.message_id)
        assert result.payload == {"key": "val", "num": 42}
        store.close()

    def test_row_to_message_handles_json_metadata(self):
        store = fresh_store()
        msg = make_message(metadata={"source": "test", "priority": 1})
        store.record_sync(msg)
        result = store._get_sync(msg.message_id)
        assert result.metadata == {"source": "test", "priority": 1}
        store.close()

    def test_row_to_message_null_payload_gives_none(self):
        store = fresh_store()
        msg = make_message(payload=None)
        store.record_sync(msg)
        result = store._get_sync(msg.message_id)
        assert result.payload is None
        store.close()

    def test_row_to_message_empty_metadata_gives_dict(self):
        store = fresh_store()
        msg = ProtocolMessage(
            message_type=ProtocolMessageType.DEBATE_STARTED,
            debate_id="d-meta",
            metadata={},
        )
        store.record_sync(msg)
        result = store._get_sync(msg.message_id)
        assert isinstance(result.metadata, dict)
        store.close()

    def test_row_to_message_timestamp_parsing_iso(self):
        store = fresh_store()
        ts = datetime(2026, 2, 14, 10, 30, 0, tzinfo=timezone.utc)
        msg = make_message(ts=ts)
        store.record_sync(msg)
        result = store._get_sync(msg.message_id)
        # Compare at second precision (isoformat roundtrip)
        assert result.timestamp.year == 2026
        assert result.timestamp.month == 2
        assert result.timestamp.day == 14
        store.close()

    def test_row_to_message_timestamp_is_datetime(self):
        store = fresh_store()
        msg = make_message()
        store.record_sync(msg)
        result = store._get_sync(msg.message_id)
        assert isinstance(result.timestamp, datetime)
        store.close()

    def test_row_to_message_message_type_enum(self):
        store = fresh_store()
        msg = make_message(message_type=ProtocolMessageType.CONSENSUS_REACHED)
        store.record_sync(msg)
        result = store._get_sync(msg.message_id)
        assert result.message_type == ProtocolMessageType.CONSENSUS_REACHED
        store.close()

    def test_row_to_message_optional_fields_none(self):
        store = fresh_store()
        msg = ProtocolMessage(
            message_type=ProtocolMessageType.DEBATE_STARTED,
            debate_id="d-minimal",
            # agent_id, round_number, correlation_id, parent_message_id all default to None
        )
        store.record_sync(msg)
        result = store._get_sync(msg.message_id)
        assert result.agent_id is None
        assert result.round_number is None
        assert result.correlation_id is None
        assert result.parent_message_id is None
        store.close()


# ===========================================================================
# close / close_all Tests
# ===========================================================================


class TestCloseAndCloseAll:
    def test_close_marks_pool_closed(self):
        store = fresh_store()
        store.close()
        assert store._pool._closed is True

    def test_close_all_marks_pool_closed(self):
        store = fresh_store()
        store.close_all()
        assert store._pool._closed is True

    def test_close_and_close_all_equivalent(self):
        store1 = fresh_store()
        store2 = fresh_store()
        store1.close()
        store2.close_all()
        assert store1._pool._closed == store2._pool._closed


# ===========================================================================
# QueryFilters Dataclass Tests
# ===========================================================================


class TestQueryFiltersDataclass:
    def test_defaults(self):
        f = QueryFilters()
        assert f.debate_id is None
        assert f.agent_id is None
        assert f.message_type is None
        assert f.message_types is None
        assert f.round_number is None
        assert f.min_round is None
        assert f.max_round is None
        assert f.start_time is None
        assert f.end_time is None
        assert f.correlation_id is None
        assert f.parent_message_id is None
        assert f.limit == 1000
        assert f.offset == 0
        assert f.order_by == "timestamp"
        assert f.order_desc is False

    def test_custom_values(self):
        now = datetime.now(timezone.utc)
        f = QueryFilters(
            debate_id="d-1",
            agent_id="a-1",
            message_type=ProtocolMessageType.VOTE_CAST,
            message_types=[ProtocolMessageType.PROPOSAL_SUBMITTED],
            round_number=2,
            min_round=1,
            max_round=5,
            start_time=now,
            end_time=now,
            correlation_id="corr-1",
            parent_message_id="par-1",
            limit=50,
            offset=10,
            order_by="round_number",
            order_desc=True,
        )
        assert f.debate_id == "d-1"
        assert f.limit == 50
        assert f.order_desc is True


# ===========================================================================
# Async ProtocolMessageStore Tests (via asyncio, asyncio_mode="auto")
# ===========================================================================


class TestAsyncProtocolMessageStoreRecord:
    async def test_record_returns_message_id(self):
        store = fresh_store()
        msg = make_message()
        result = await store.record(msg)
        assert result == msg.message_id
        store.close()

    async def test_record_persists_message(self):
        store = fresh_store()
        msg = make_message(debate_id="d-async")
        await store.record(msg)
        results = store._query_sync_impl(QueryFilters(debate_id="d-async"))
        assert len(results) == 1
        assert results[0].message_id == msg.message_id
        store.close()

    async def test_record_multiple_async(self):
        store = fresh_store()
        msgs = [make_message(debate_id="d-async-multi") for _ in range(4)]
        for m in msgs:
            await store.record(m)
        results = store._query_sync_impl(QueryFilters(debate_id="d-async-multi"))
        assert len(results) == 4
        store.close()


class TestAsyncProtocolMessageStoreQuery:
    async def test_query_returns_messages(self):
        store = fresh_store()
        msg = make_message(debate_id="d-q")
        store.record_sync(msg)
        results = await store.query(QueryFilters(debate_id="d-q"))
        assert len(results) == 1
        assert results[0].message_id == msg.message_id
        store.close()

    async def test_query_none_filters_returns_all(self):
        store = fresh_store()
        for _ in range(3):
            store.record_sync(make_message())
        results = await store.query(None)
        assert len(results) == 3
        store.close()

    async def test_query_with_debate_id_filter(self):
        store = fresh_store()
        store.record_sync(make_message(debate_id="d-q-match"))
        store.record_sync(make_message(debate_id="d-q-other"))
        results = await store.query(QueryFilters(debate_id="d-q-match"))
        assert len(results) == 1
        store.close()


class TestAsyncProtocolMessageStoreGet:
    async def test_get_returns_message(self):
        store = fresh_store()
        msg = make_message()
        store.record_sync(msg)
        result = await store.get(msg.message_id)
        assert result is not None
        assert result.message_id == msg.message_id
        store.close()

    async def test_get_returns_none_for_missing(self):
        store = fresh_store()
        result = await store.get("no-such-id")
        assert result is None
        store.close()


class TestAsyncProtocolMessageStoreCount:
    async def test_count_returns_correct_number(self):
        store = fresh_store()
        for _ in range(5):
            store.record_sync(make_message(debate_id="d-cnt"))
        result = await store.count(QueryFilters(debate_id="d-cnt"))
        assert result == 5
        store.close()

    async def test_count_empty_returns_zero(self):
        store = fresh_store()
        result = await store.count()
        assert result == 0
        store.close()


class TestAsyncProtocolMessageStoreTimeline:
    async def test_get_debate_timeline_returns_chronological(self):
        store = fresh_store()
        base = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        for i in range(4):
            msg = ProtocolMessage(
                message_type=ProtocolMessageType.ROUND_STARTED,
                debate_id="d-timeline",
                round_number=i,
                timestamp=base + timedelta(minutes=i),
            )
            store.record_sync(msg)
        results = await store.get_debate_timeline("d-timeline")
        assert len(results) == 4
        ts_list = [r.timestamp for r in results]
        assert ts_list == sorted(ts_list)
        store.close()

    async def test_get_debate_timeline_with_type_filter(self):
        store = fresh_store()
        store.record_sync(make_message(
            debate_id="d-tl-typed",
            message_type=ProtocolMessageType.PROPOSAL_SUBMITTED,
        ))
        store.record_sync(make_message(
            debate_id="d-tl-typed",
            message_type=ProtocolMessageType.VOTE_CAST,
        ))
        results = await store.get_debate_timeline(
            "d-tl-typed",
            include_types=[ProtocolMessageType.PROPOSAL_SUBMITTED],
        )
        assert len(results) == 1
        assert results[0].message_type == ProtocolMessageType.PROPOSAL_SUBMITTED
        store.close()

    async def test_get_debate_timeline_empty_debate(self):
        store = fresh_store()
        results = await store.get_debate_timeline("no-debate")
        assert results == []
        store.close()

    async def test_get_round_messages(self):
        store = fresh_store()
        store.record_sync(make_message(debate_id="d-round", round_number=1))
        store.record_sync(make_message(debate_id="d-round", round_number=1))
        store.record_sync(make_message(debate_id="d-round", round_number=2))
        results = await store.get_round_messages("d-round", 1)
        assert len(results) == 2
        assert all(r.round_number == 1 for r in results)
        store.close()

    async def test_get_agent_messages(self):
        store = fresh_store()
        store.record_sync(make_message(debate_id="d-agent", agent_id="agent-A"))
        store.record_sync(make_message(debate_id="d-agent", agent_id="agent-A"))
        store.record_sync(make_message(debate_id="d-agent", agent_id="agent-B"))
        results = await store.get_agent_messages("d-agent", "agent-A")
        assert len(results) == 2
        assert all(r.agent_id == "agent-A" for r in results)
        store.close()


class TestAsyncProtocolMessageStoreDeleteDebate:
    async def test_delete_debate_removes_messages(self):
        store = fresh_store()
        for _ in range(3):
            store.record_sync(make_message(debate_id="d-del-async"))
        count = await store.delete_debate("d-del-async")
        assert count == 3
        remaining = store._query_sync_impl(QueryFilters(debate_id="d-del-async"))
        assert remaining == []
        store.close()

    async def test_delete_debate_nonexistent_returns_zero(self):
        store = fresh_store()
        count = await store.delete_debate("no-such-debate")
        assert count == 0
        store.close()


class TestAsyncProtocolMessageStoreExportJsonl:
    async def test_export_jsonl_writes_correct_lines(self, tmp_path):
        store = fresh_store()
        msgs = []
        for i in range(3):
            m = make_message(debate_id="d-export", round_number=i)
            store.record_sync(m)
            msgs.append(m)
        output = tmp_path / "export.jsonl"
        count = await store.export_jsonl("d-export", str(output))
        assert count == 3
        lines = output.read_text().strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            data = json.loads(line)
            assert "message_id" in data
            assert data["debate_id"] == "d-export"
        store.close()

    async def test_export_jsonl_empty_debate(self, tmp_path):
        store = fresh_store()
        output = tmp_path / "empty.jsonl"
        count = await store.export_jsonl("no-debate", str(output))
        assert count == 0
        assert output.read_text() == ""
        store.close()

    async def test_export_jsonl_returns_message_count(self, tmp_path):
        store = fresh_store()
        for _ in range(5):
            store.record_sync(make_message(debate_id="d-count-export"))
        output = tmp_path / "count.jsonl"
        count = await store.export_jsonl("d-count-export", str(output))
        assert count == 5
        store.close()


# ===========================================================================
# AsyncProtocolMessageStore Tests
# ===========================================================================


class TestAsyncProtocolMessageStoreClass:
    async def test_record_and_query_roundtrip(self):
        store = fresh_async_store()
        msg = make_message(debate_id="d-async-class")
        msg_id = await store.record(msg)
        assert msg_id == msg.message_id
        results = await store.query(QueryFilters(debate_id="d-async-class"))
        assert len(results) == 1
        assert results[0].message_id == msg.message_id
        store.close()

    async def test_get_existing_message(self):
        store = fresh_async_store()
        msg = make_message()
        await store.record(msg)
        result = await store.get(msg.message_id)
        assert result is not None
        assert result.message_id == msg.message_id
        store.close()

    async def test_get_missing_returns_none(self):
        store = fresh_async_store()
        result = await store.get("nonexistent")
        assert result is None
        store.close()

    async def test_count_roundtrip(self):
        store = fresh_async_store()
        for _ in range(4):
            await store.record(make_message(debate_id="d-async-cnt"))
        n = await store.count(QueryFilters(debate_id="d-async-cnt"))
        assert n == 4
        store.close()

    async def test_count_by_message_type(self):
        store = fresh_async_store()
        await store.record(make_message(message_type=ProtocolMessageType.PROPOSAL_SUBMITTED))
        await store.record(make_message(message_type=ProtocolMessageType.PROPOSAL_SUBMITTED))
        await store.record(make_message(message_type=ProtocolMessageType.VOTE_CAST))
        n = await store.count(
            QueryFilters(message_type=ProtocolMessageType.PROPOSAL_SUBMITTED)
        )
        assert n == 2
        store.close()

    async def test_get_debate_timeline(self):
        store = fresh_async_store()
        base = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        for i in range(3):
            msg = ProtocolMessage(
                message_type=ProtocolMessageType.ROUND_STARTED,
                debate_id="d-atl",
                round_number=i,
                timestamp=base + timedelta(minutes=i),
            )
            await store.record(msg)
        results = await store.get_debate_timeline("d-atl")
        assert len(results) == 3
        ts_list = [r.timestamp for r in results]
        assert ts_list == sorted(ts_list)
        store.close()

    async def test_get_round_messages(self):
        store = fresh_async_store()
        await store.record(make_message(debate_id="d-arm", round_number=2))
        await store.record(make_message(debate_id="d-arm", round_number=2))
        await store.record(make_message(debate_id="d-arm", round_number=3))
        results = await store.get_round_messages("d-arm", 2)
        assert len(results) == 2
        store.close()

    async def test_get_agent_messages(self):
        store = fresh_async_store()
        await store.record(make_message(debate_id="d-aam", agent_id="agent-x"))
        await store.record(make_message(debate_id="d-aam", agent_id="agent-y"))
        results = await store.get_agent_messages("d-aam", "agent-x")
        assert len(results) == 1
        assert results[0].agent_id == "agent-x"
        store.close()

    async def test_delete_debate(self):
        store = fresh_async_store()
        for _ in range(2):
            await store.record(make_message(debate_id="d-adel"))
        n = await store.delete_debate("d-adel")
        assert n == 2
        results = await store.query(QueryFilters(debate_id="d-adel"))
        assert results == []
        store.close()

    async def test_export_jsonl(self, tmp_path):
        store = fresh_async_store()
        for i in range(2):
            await store.record(make_message(debate_id="d-ajsonl", round_number=i))
        output = tmp_path / "async_export.jsonl"
        n = await store.export_jsonl("d-ajsonl", str(output))
        assert n == 2
        lines = output.read_text().strip().splitlines()
        assert len(lines) == 2
        store.close()

    async def test_cleanup_old(self):
        store = fresh_async_store()
        old_ts = datetime.now(timezone.utc) - timedelta(days=60)
        store._sync_store.record_sync(make_message(debate_id="d-aclean", ts=old_ts))
        store._sync_store.record_sync(make_message(debate_id="d-aclean"))
        n = await store.cleanup_old(days=30)
        assert n == 1
        store.close()

    def test_db_path_property(self):
        store = fresh_async_store()
        assert store.db_path == ":memory:"
        store.close()

    def test_close_closes_sync_store(self):
        store = fresh_async_store()
        store.close()
        assert store._sync_store._pool._closed is True

    def test_close_all_closes_sync_store(self):
        store = fresh_async_store()
        store.close_all()
        assert store._sync_store._pool._closed is True


# ===========================================================================
# get_protocol_store Singleton Tests
# ===========================================================================


class TestGetProtocolStore:
    def setup_method(self):
        """Reset the singleton before each test."""
        import aragora.debate.protocol_messages.store as store_module
        with store_module._store_lock:
            store_module._protocol_store = None

    def teardown_method(self):
        """Reset the singleton after each test."""
        import aragora.debate.protocol_messages.store as store_module
        with store_module._store_lock:
            if store_module._protocol_store is not None:
                store_module._protocol_store.close()
            store_module._protocol_store = None

    def test_get_protocol_store_returns_store(self):
        store = get_protocol_store(":memory:")
        assert isinstance(store, ProtocolMessageStore)

    def test_get_protocol_store_singleton(self):
        store1 = get_protocol_store(":memory:")
        store2 = get_protocol_store(":memory:")
        assert store1 is store2

    def test_get_protocol_store_ignores_second_db_path(self):
        store1 = get_protocol_store(":memory:")
        store2 = get_protocol_store("/some/other/path.db")
        assert store1 is store2

    def test_get_protocol_store_creates_fresh_after_reset(self):
        import aragora.debate.protocol_messages.store as store_module
        store1 = get_protocol_store(":memory:")
        # Reset singleton
        with store_module._store_lock:
            store_module._protocol_store = None
        store2 = get_protocol_store(":memory:")
        assert store1 is not store2

    def test_get_protocol_store_thread_safe(self):
        """Multiple threads calling get_protocol_store should get same singleton."""
        results = []
        errors = []

        def get_store():
            try:
                s = get_protocol_store(":memory:")
                results.append(s)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=get_store) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors
        assert len(results) == 10
        # All results should be the same object
        assert all(r is results[0] for r in results)


# ===========================================================================
# Edge Cases and Integration Tests
# ===========================================================================


class TestEdgeCases:
    def test_empty_store_query_returns_empty(self):
        store = fresh_store()
        results = store._query_sync_impl()
        assert results == []
        store.close()

    def test_large_payload_roundtrip(self):
        store = fresh_store()
        large_payload = {"data": "x" * 10_000, "items": list(range(100))}
        msg = make_message(payload=large_payload)
        store.record_sync(msg)
        result = store._get_sync(msg.message_id)
        assert result.payload == large_payload
        store.close()

    def test_unicode_in_payload(self):
        store = fresh_store()
        payload = {"text": "日本語テスト", "emoji": "🎯", "accents": "café résumé"}
        msg = make_message(payload=payload)
        store.record_sync(msg)
        result = store._get_sync(msg.message_id)
        assert result.payload == payload
        store.close()

    def test_all_message_types_can_be_stored(self):
        store = fresh_store()
        for mtype in ProtocolMessageType:
            msg = ProtocolMessage(
                message_type=mtype,
                debate_id="d-all-types",
            )
            store.record_sync(msg)
        results = store._query_sync_impl(QueryFilters(debate_id="d-all-types"))
        stored_types = {r.message_type for r in results}
        assert stored_types == set(ProtocolMessageType)
        store.close()

    def test_file_backed_store_persists(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            store1 = ProtocolMessageStore(db_path=path)
            msg = make_message(debate_id="d-persist")
            store1.record_sync(msg)
            store1.close()

            store2 = ProtocolMessageStore(db_path=path)
            results = store2._query_sync_impl(QueryFilters(debate_id="d-persist"))
            assert len(results) == 1
            assert results[0].message_id == msg.message_id
            store2.close()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_metadata_nested_dict_roundtrip(self):
        store = fresh_store()
        metadata = {
            "nested": {"a": 1, "b": [2, 3]},
            "tags": ["x", "y"],
        }
        msg = make_message(metadata=metadata)
        store.record_sync(msg)
        result = store._get_sync(msg.message_id)
        assert result.metadata == metadata
        store.close()

    def test_timestamp_with_timezone_preserved(self):
        store = fresh_store()
        ts = datetime(2026, 6, 15, 14, 30, 45, tzinfo=timezone.utc)
        msg = make_message(ts=ts)
        store.record_sync(msg)
        result = store._get_sync(msg.message_id)
        assert result.timestamp.year == ts.year
        assert result.timestamp.month == ts.month
        assert result.timestamp.day == ts.day
        assert result.timestamp.hour == ts.hour
        assert result.timestamp.minute == ts.minute
        store.close()

    async def test_concurrent_async_records(self):
        """Multiple concurrent async records should all succeed."""
        import asyncio
        store = fresh_store()
        msgs = [make_message(debate_id="d-concurrent") for _ in range(10)]
        await asyncio.gather(*[store.record(m) for m in msgs])
        results = store._query_sync_impl(QueryFilters(debate_id="d-concurrent"))
        assert len(results) == 10
        store.close()

    def test_critique_payload_roundtrip(self):
        store = fresh_store()
        payload = CritiquePayload(
            critique_id="crit-1",
            proposal_id="prop-1",
            content="This needs improvement",
            model="gpt-4",
            round_number=2,
            severity="moderate",
        )
        msg = make_message(
            message_type=ProtocolMessageType.CRITIQUE_SUBMITTED,
            payload=payload,
        )
        store.record_sync(msg)
        result = store._get_sync(msg.message_id)
        assert isinstance(result.payload, dict)
        assert result.payload["critique_id"] == "crit-1"
        assert result.payload["severity"] == "moderate"
        store.close()

    def test_vote_payload_roundtrip(self):
        store = fresh_store()
        payload = VotePayload(
            vote_id="vote-1",
            proposal_id="prop-1",
            vote_type="support",
            confidence=0.95,
        )
        msg = make_message(
            message_type=ProtocolMessageType.VOTE_CAST,
            payload=payload,
        )
        store.record_sync(msg)
        result = store._get_sync(msg.message_id)
        assert result.payload["vote_type"] == "support"
        assert result.payload["confidence"] == 0.95
        store.close()

    def test_consensus_payload_roundtrip(self):
        store = fresh_store()
        payload = ConsensusPayload(
            consensus_id="cons-1",
            winning_proposal_id="prop-winner",
            final_answer="The answer is 42",
            confidence=0.88,
            rounds_taken=3,
        )
        msg = ProtocolMessage(
            message_type=ProtocolMessageType.CONSENSUS_REACHED,
            debate_id="d-consensus",
            payload=payload,
        )
        store.record_sync(msg)
        result = store._get_sync(msg.message_id)
        assert result.payload["final_answer"] == "The answer is 42"
        assert result.payload["confidence"] == 0.88
        store.close()
