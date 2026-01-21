"""
Tests for DebateEmbeddingsDatabase.
"""

import asyncio
import tempfile
import os
import socket
from datetime import datetime, timezone

import pytest

from aragora.debate.embeddings import DebateEmbeddingsDatabase
from aragora.persistence.models import DebateArtifact


def is_ollama_running() -> bool:
    """Check if Ollama is running on localhost:11434."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", 11434))
        sock.close()
        return result == 0
    except Exception:
        return False


# Skip tests that require Ollama if it's not running
requires_ollama = pytest.mark.skipif(
    not is_ollama_running(), reason="Ollama not running on localhost:11434"
)


@requires_ollama
@pytest.mark.asyncio
async def test_index_and_search_debate():
    """Test indexing a debate and searching for similar ones."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = DebateEmbeddingsDatabase(db_path)

        # Create a sample debate
        debate = DebateArtifact(
            loop_id="test_loop",
            cycle_number=1,
            phase="debate",
            task="Test task about improving AI",
            agents=["Claude", "Gemini"],
            transcript=[
                {"agent": "Claude", "content": "We should add more features", "type": "proposal"},
                {"agent": "Gemini", "content": "But keep it simple", "type": "critique"},
                {"agent": "Claude", "content": "Agreed, balance complexity", "type": "revision"},
            ],
            consensus_reached=True,
            confidence=0.8,
            winning_proposal="Balanced approach",
            vote_tally={"Claude": 1, "Gemini": 1},
            created_at=datetime.now(timezone.utc),
        )

        # Index the debate
        await db.index_debate(debate)

        # Search for similar debates
        results = await db.find_similar_debates("improving AI features", limit=5)

        assert len(results) == 1
        debate_id, excerpt, similarity = results[0]
        assert debate_id == "test_loop_1_debate"
        assert similarity > 0.5  # Should be similar

        # Test search with different query
        results2 = await db.find_similar_debates("something unrelated", limit=5)
        assert len(results2) == 0 or results2[0][2] < 0.5  # Low similarity

    finally:
        os.unlink(db_path)


@requires_ollama
@pytest.mark.asyncio
async def test_multiple_debates():
    """Test indexing multiple debates and retrieving the most similar."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = DebateEmbeddingsDatabase(db_path)

        # Debate 1: About AI safety
        debate1 = DebateArtifact(
            loop_id="loop1",
            cycle_number=1,
            phase="debate",
            task="AI safety measures",
            agents=["Claude", "Gemini"],
            transcript=[
                {"agent": "Claude", "content": "Need safety protocols", "type": "proposal"}
            ],
            consensus_reached=True,
            confidence=0.9,
            winning_proposal="Implement safety checks",
            created_at=datetime.now(timezone.utc),
        )

        # Debate 2: About AI features
        debate2 = DebateArtifact(
            loop_id="loop1",
            cycle_number=2,
            phase="debate",
            task="Add new AI features",
            agents=["Claude", "Gemini"],
            transcript=[
                {"agent": "Gemini", "content": "Add embeddings for search", "type": "proposal"}
            ],
            consensus_reached=True,
            confidence=0.7,
            winning_proposal="Add embeddings database",
            created_at=datetime.now(timezone.utc),
        )

        await db.index_debate(debate1)
        await db.index_debate(debate2)

        # Search for safety-related
        results = await db.find_similar_debates("safety protocols", limit=5)
        assert len(results) >= 1
        # Should find debate1 as most similar
        assert any("loop1_1_debate" in r[0] for r in results)

        # Search for feature-related
        results2 = await db.find_similar_debates("new features", limit=5)
        assert len(results2) >= 1
        assert any("loop1_2_debate" in r[0] for r in results2)

    finally:
        os.unlink(db_path)


def test_stats():
    """Test getting database statistics."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = DebateEmbeddingsDatabase(db_path)
        stats = db.get_stats()
        assert "total_embeddings" in stats
        assert stats["total_embeddings"] == 0

    finally:
        os.unlink(db_path)
