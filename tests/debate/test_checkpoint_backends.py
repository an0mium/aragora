"""Tests for debate checkpoint storage backends.

Covers FileCheckpointStore and DatabaseCheckpointStore — the two backends
that have zero external dependencies and can be tested fully locally.
"""

from __future__ import annotations

import json
import gzip
from pathlib import Path

import pytest

from aragora.debate.checkpoint import (
    AgentState,
    CheckpointStatus,
    DebateCheckpoint,
)
from aragora.debate.checkpoint_backends import (
    DatabaseCheckpointStore,
    FileCheckpointStore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_checkpoint(
    checkpoint_id: str = "cp-test-001",
    debate_id: str = "debate-abc",
    current_round: int = 2,
    total_rounds: int = 5,
    task: str = "Design a rate limiter",
    phase: str = "critique",
    messages: list | None = None,
    status: CheckpointStatus = CheckpointStatus.COMPLETE,
    expires_at: str | None = None,
) -> DebateCheckpoint:
    return DebateCheckpoint(
        checkpoint_id=checkpoint_id,
        debate_id=debate_id,
        task=task,
        current_round=current_round,
        total_rounds=total_rounds,
        phase=phase,
        messages=messages or [{"agent": "claude", "content": "proposal text"}],
        critiques=[],
        votes=[],
        agent_states=[
            AgentState(
                agent_name="claude",
                agent_model="claude-3-opus",
                agent_role="proposer",
                system_prompt="You are a helpful assistant.",
                stance="in_favor",
            ),
        ],
        status=status,
        expires_at=expires_at,
    )


# ---------------------------------------------------------------------------
# FileCheckpointStore
# ---------------------------------------------------------------------------


class TestFileCheckpointStore:

    @pytest.fixture
    def store(self, tmp_path: Path) -> FileCheckpointStore:
        return FileCheckpointStore(base_dir=str(tmp_path / "checkpoints"))

    @pytest.fixture
    def uncompressed_store(self, tmp_path: Path) -> FileCheckpointStore:
        return FileCheckpointStore(
            base_dir=str(tmp_path / "checkpoints"),
            compress=False,
        )

    @pytest.mark.asyncio
    async def test_save_and_load(self, store):
        cp = _make_checkpoint()
        path = await store.save(cp)
        assert path.endswith(".json.gz")

        loaded = await store.load("cp-test-001")
        assert loaded is not None
        assert loaded.checkpoint_id == "cp-test-001"
        assert loaded.debate_id == "debate-abc"
        assert loaded.current_round == 2

    @pytest.mark.asyncio
    async def test_save_uncompressed(self, uncompressed_store):
        cp = _make_checkpoint()
        path = await uncompressed_store.save(cp)
        assert path.endswith(".json")

        loaded = await uncompressed_store.load("cp-test-001")
        assert loaded is not None
        assert loaded.debate_id == "debate-abc"

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, store):
        result = await store.load("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, store):
        cp = _make_checkpoint()
        await store.save(cp)

        deleted = await store.delete("cp-test-001")
        assert deleted is True

        loaded = await store.load("cp-test-001")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        deleted = await store.delete("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, store):
        cp1 = _make_checkpoint(checkpoint_id="cp-001", debate_id="d1")
        cp2 = _make_checkpoint(checkpoint_id="cp-002", debate_id="d1")
        cp3 = _make_checkpoint(checkpoint_id="cp-003", debate_id="d2")

        await store.save(cp1)
        await store.save(cp2)
        await store.save(cp3)

        all_cps = await store.list_checkpoints()
        assert len(all_cps) == 3

        d1_cps = await store.list_checkpoints(debate_id="d1")
        assert len(d1_cps) == 2

    @pytest.mark.asyncio
    async def test_list_with_limit(self, store):
        for i in range(5):
            await store.save(_make_checkpoint(checkpoint_id=f"cp-{i:03d}"))

        limited = await store.list_checkpoints(limit=2)
        assert len(limited) <= 2

    @pytest.mark.asyncio
    async def test_roundtrip_preserves_data(self, store):
        cp = _make_checkpoint(
            messages=[{"agent": "gpt4", "content": "important proposal"}],
            phase="vote",
            current_round=4,
        )
        await store.save(cp)
        loaded = await store.load(cp.checkpoint_id)

        assert loaded is not None
        assert loaded.messages == [{"agent": "gpt4", "content": "important proposal"}]
        assert loaded.phase == "vote"
        assert loaded.current_round == 4
        assert len(loaded.agent_states) == 1
        assert loaded.agent_states[0].agent_name == "claude"

    def test_sanitize_checkpoint_id(self, store):
        assert store._sanitize_checkpoint_id("valid-id_123") == "valid-id_123"
        assert store._sanitize_checkpoint_id("has/slash") == "has_slash"
        # ".." replaced by "_", then regex reduces further
        sanitized = store._sanitize_checkpoint_id("has..dots")
        assert "/" not in sanitized
        assert ".." not in sanitized

    def test_sanitize_removes_dangerous_chars(self, store):
        sanitized = store._sanitize_checkpoint_id("id<>with|bad;chars")
        assert "<" not in sanitized
        assert ">" not in sanitized
        assert "|" not in sanitized

    def test_path_traversal_prevention(self, store):
        """Sanitization prevents path traversal."""
        sanitized = store._sanitize_checkpoint_id("../../etc/passwd")
        assert ".." not in sanitized
        assert "/" not in sanitized

    @pytest.mark.asyncio
    async def test_corrupted_file_returns_none(self, store):
        """Corrupted gzip file returns None instead of crashing."""
        cp = _make_checkpoint()
        path = store._get_path(cp.checkpoint_id)
        path.write_bytes(b"not valid gzip data")

        loaded = await store.load(cp.checkpoint_id)
        assert loaded is None

    @pytest.mark.asyncio
    async def test_creates_base_dir(self, tmp_path):
        """Store creates base directory if it doesn't exist."""
        new_dir = tmp_path / "deep" / "nested" / "checkpoints"
        store = FileCheckpointStore(base_dir=str(new_dir))
        assert new_dir.exists()


# ---------------------------------------------------------------------------
# DatabaseCheckpointStore
# ---------------------------------------------------------------------------


class TestDatabaseCheckpointStore:

    @pytest.fixture
    def store(self, tmp_path: Path) -> DatabaseCheckpointStore:
        db_path = str(tmp_path / "checkpoints.db")
        return DatabaseCheckpointStore(db_path=db_path, compress=True)

    @pytest.fixture
    def uncompressed_store(self, tmp_path: Path) -> DatabaseCheckpointStore:
        db_path = str(tmp_path / "checkpoints_uc.db")
        return DatabaseCheckpointStore(db_path=db_path, compress=False)

    @pytest.mark.asyncio
    async def test_save_and_load(self, store):
        cp = _make_checkpoint()
        result = await store.save(cp)
        assert result == "db:cp-test-001"

        loaded = await store.load("cp-test-001")
        assert loaded is not None
        assert loaded.checkpoint_id == "cp-test-001"
        assert loaded.debate_id == "debate-abc"

    @pytest.mark.asyncio
    async def test_save_uncompressed(self, uncompressed_store):
        cp = _make_checkpoint()
        await uncompressed_store.save(cp)

        loaded = await uncompressed_store.load("cp-test-001")
        assert loaded is not None
        assert loaded.debate_id == "debate-abc"

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, store):
        result = await store.load("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, store):
        cp = _make_checkpoint()
        await store.save(cp)

        deleted = await store.delete("cp-test-001")
        assert deleted is True

        loaded = await store.load("cp-test-001")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        deleted = await store.delete("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, store):
        cp1 = _make_checkpoint(checkpoint_id="cp-001", debate_id="d1")
        cp2 = _make_checkpoint(checkpoint_id="cp-002", debate_id="d1")
        cp3 = _make_checkpoint(checkpoint_id="cp-003", debate_id="d2")

        await store.save(cp1)
        await store.save(cp2)
        await store.save(cp3)

        all_cps = await store.list_checkpoints()
        assert len(all_cps) == 3

    @pytest.mark.asyncio
    async def test_list_filtered_by_debate(self, store):
        await store.save(_make_checkpoint(checkpoint_id="cp-001", debate_id="d1"))
        await store.save(_make_checkpoint(checkpoint_id="cp-002", debate_id="d2"))

        d1_only = await store.list_checkpoints(debate_id="d1")
        assert len(d1_only) == 1
        assert d1_only[0]["debate_id"] == "d1"

    @pytest.mark.asyncio
    async def test_list_with_limit(self, store):
        for i in range(5):
            await store.save(_make_checkpoint(checkpoint_id=f"cp-{i:03d}"))

        limited = await store.list_checkpoints(limit=2)
        assert len(limited) == 2

    @pytest.mark.asyncio
    async def test_upsert_on_save(self, store):
        """Saving same checkpoint_id overwrites previous."""
        cp1 = _make_checkpoint(current_round=1)
        cp2 = _make_checkpoint(current_round=3)

        await store.save(cp1)
        await store.save(cp2)

        loaded = await store.load("cp-test-001")
        assert loaded is not None
        assert loaded.current_round == 3

    @pytest.mark.asyncio
    async def test_get_stats(self, store):
        await store.save(_make_checkpoint(checkpoint_id="cp-001"))
        await store.save(_make_checkpoint(checkpoint_id="cp-002", debate_id="d2"))

        stats = await store.get_stats()
        assert stats["total_checkpoints"] == 2
        assert stats["unique_debates"] == 2
        assert stats["total_bytes"] > 0
        assert "db_path" in stats

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, store):
        # Non-expired
        await store.save(_make_checkpoint(checkpoint_id="fresh"))
        # Expired (date in the past)
        await store.save(_make_checkpoint(
            checkpoint_id="old",
            expires_at="2020-01-01T00:00:00",
        ))

        deleted = await store.cleanup_expired()
        assert deleted == 1

        remaining = await store.list_checkpoints()
        assert len(remaining) == 1
        assert remaining[0]["checkpoint_id"] == "fresh"

    @pytest.mark.asyncio
    async def test_pool_stats(self, store):
        stats = store.get_pool_stats()
        assert "db_path" in stats
        assert stats["max_pool_size"] == 5

    @pytest.mark.asyncio
    async def test_roundtrip_preserves_data(self, store):
        cp = _make_checkpoint(
            messages=[{"agent": "gpt4", "content": "important"}],
            phase="synthesis",
            current_round=5,
            total_rounds=5,
        )
        await store.save(cp)
        loaded = await store.load(cp.checkpoint_id)

        assert loaded is not None
        assert loaded.messages == [{"agent": "gpt4", "content": "important"}]
        assert loaded.phase == "synthesis"
        assert loaded.current_round == 5
        assert loaded.total_rounds == 5
        assert loaded.agent_states[0].agent_name == "claude"
        assert loaded.status == CheckpointStatus.COMPLETE

    @pytest.mark.asyncio
    async def test_task_truncated_in_db(self, store):
        """Long tasks are truncated to 500 chars in the DB column."""
        long_task = "x" * 1000
        cp = _make_checkpoint(task=long_task)
        await store.save(cp)

        # The full data blob preserves the complete task
        loaded = await store.load(cp.checkpoint_id)
        assert loaded is not None
        assert loaded.task == long_task  # Full task from BLOB

        # But list_checkpoints shows truncated version
        listed = await store.list_checkpoints()
        assert len(listed[0]["task"]) <= 100
