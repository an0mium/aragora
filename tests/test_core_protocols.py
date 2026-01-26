"""
Tests for core protocol definitions.

Verifies that protocol interfaces work correctly with runtime_checkable
and that mock implementations satisfy the protocol constraints.
"""

import pytest
from typing import Any, Optional, runtime_checkable, Protocol


class TestStorageBackendProtocol:
    """Tests for StorageBackend protocol."""

    def test_protocol_is_importable(self):
        """StorageBackend should be importable."""
        from aragora.core_protocols import StorageBackend

        assert StorageBackend is not None

    def test_protocol_is_runtime_checkable(self):
        """StorageBackend should be runtime checkable (isinstance works)."""
        from aragora.core_protocols import StorageBackend

        # Runtime checkable protocols can be used with isinstance
        class MockStorage:
            def save(
                self,
                debate_id: str,
                task: str,
                agents: list[str],
                artifact: dict[str, Any],
                consensus_reached: bool = False,
                confidence: float = 0.0,
            ) -> str:
                return "slug"

            def get(self, slug: str) -> Optional[dict[str, Any]]:
                return None

            def list_debates(self, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
                return []

            def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
                return []

        mock = MockStorage()
        assert isinstance(mock, StorageBackend)

    def test_incomplete_implementation_fails_protocol(self):
        """An incomplete implementation should not satisfy the protocol."""
        from aragora.core_protocols import StorageBackend

        class IncompleteStorage:
            def save(self, debate_id: str, task: str, agents: list, artifact: dict) -> str:
                return "slug"

            # Missing: get, list_debates, search

        incomplete = IncompleteStorage()
        # Should fail isinstance check since methods are missing
        assert not isinstance(incomplete, StorageBackend)


class TestMemoryBackendProtocol:
    """Tests for MemoryBackend protocol."""

    def test_protocol_is_importable(self):
        """MemoryBackend should be importable."""
        from aragora.core_protocols import MemoryBackend

        assert MemoryBackend is not None

    def test_mock_implementation_satisfies_protocol(self):
        """A mock implementation should satisfy the protocol."""
        from aragora.core_protocols import MemoryBackend

        class MockMemory:
            def store(
                self,
                content: str,
                importance: float = 0.5,
                metadata: Optional[dict[str, Any]] = None,
            ) -> str:
                return "mem-123"

            def retrieve(
                self,
                query: str,
                limit: int = 10,
                tier: Optional[str] = None,
            ) -> list[dict[str, Any]]:
                return []

            def promote(self, memory_id: str, reason: str) -> bool:
                return True

            def decay(self) -> int:
                return 0

        mock = MockMemory()
        assert isinstance(mock, MemoryBackend)


class TestEloBackendProtocol:
    """Tests for EloBackend protocol."""

    def test_protocol_is_importable(self):
        """EloBackend should be importable."""
        from aragora.core_protocols import EloBackend

        assert EloBackend is not None

    def test_mock_implementation_satisfies_protocol(self):
        """A mock implementation should satisfy the protocol."""
        from aragora.core_protocols import EloBackend

        class MockElo:
            def get_rating(self, agent: str) -> float:
                return 1500.0

            def update_ratings(
                self,
                debate_id: str,
                winner: Optional[str],
                participants: list[str],
                scores: dict[str, float],
            ) -> dict[str, float]:
                return {p: 0.0 for p in participants}

            def get_leaderboard(
                self, limit: int = 20, domain: Optional[str] = None
            ) -> list[dict[str, Any]]:
                return []

            def get_history(self, agent: str, limit: int = 50) -> list[dict[str, Any]]:
                return []

        mock = MockElo()
        assert isinstance(mock, EloBackend)


class TestConsensusBackendProtocol:
    """Tests for ConsensusBackend protocol."""

    def test_protocol_is_importable(self):
        """ConsensusBackend should be importable."""
        from aragora.core_protocols import ConsensusBackend

        assert ConsensusBackend is not None

    def test_mock_implementation_satisfies_protocol(self):
        """A mock implementation should satisfy the protocol."""
        from aragora.core_protocols import ConsensusBackend

        class MockConsensus:
            def record_consensus(
                self,
                topic: str,
                position: str,
                confidence: float,
                supporting_agents: list[str],
                evidence: list[str],
                debate_id: Optional[str] = None,
            ) -> int:
                return 123

            def get_consensus(self, topic: str) -> Optional[dict[str, Any]]:
                return None

            def record_dissent(
                self,
                consensus_id: int,
                agent: str,
                position: str,
                reasoning: str,
            ) -> int:
                return 456

            def get_dissents(self, consensus_id: int) -> list[dict[str, Any]]:
                return []

        mock = MockConsensus()
        assert isinstance(mock, ConsensusBackend)


class TestEmbeddingBackendProtocol:
    """Tests for EmbeddingBackend protocol."""

    def test_protocol_is_importable(self):
        """EmbeddingBackend should be importable."""
        from aragora.core_protocols import EmbeddingBackend

        assert EmbeddingBackend is not None

    def test_mock_implementation_satisfies_protocol(self):
        """A mock implementation should satisfy the protocol."""
        from aragora.core_protocols import EmbeddingBackend

        class MockEmbedding:
            def embed(self, text: str) -> list[float]:
                return [0.1, 0.2, 0.3]

            def store_embedding(
                self,
                id: str,
                text: str,
                embedding: list[float],
            ) -> None:
                pass

            def search_similar(
                self,
                query_embedding: list[float],
                limit: int = 10,
                threshold: float = 0.7,
            ) -> list[tuple[str, float]]:
                return []

        mock = MockEmbedding()
        assert isinstance(mock, EmbeddingBackend)


class TestHTTPRequestHandlerProtocol:
    """Tests for HTTPRequestHandler protocol."""

    def test_protocol_is_importable(self):
        """HTTPRequestHandler should be importable."""
        from aragora.core_protocols import HTTPRequestHandler

        assert HTTPRequestHandler is not None


class TestAgentProtocol:
    """Tests for Agent protocol."""

    def test_protocol_is_importable(self):
        """Agent should be importable."""
        from aragora.core_protocols import Agent

        assert Agent is not None

    def test_mock_implementation_satisfies_protocol(self):
        """A mock implementation should satisfy the protocol."""
        from aragora.core_protocols import Agent

        class MockAgent:
            name = "test-agent"

            def generate(self, prompt: str, **kwargs: Any) -> str:
                return "Generated response"

        mock = MockAgent()
        assert isinstance(mock, Agent)


class TestTypeAliases:
    """Tests for type aliases."""

    def test_debate_record_is_dict(self):
        """DebateRecord should be dict type alias."""
        from aragora.core_protocols import DebateRecord

        assert DebateRecord == dict[str, Any]

    def test_memory_record_is_dict(self):
        """MemoryRecord should be dict type alias."""
        from aragora.core_protocols import MemoryRecord

        assert MemoryRecord == dict[str, Any]

    def test_agent_record_is_dict(self):
        """AgentRecord should be dict type alias."""
        from aragora.core_protocols import AgentRecord

        assert AgentRecord == dict[str, Any]

    def test_query_params_is_dict(self):
        """QueryParams should be dict type alias."""
        from aragora.core_protocols import QueryParams

        assert QueryParams == dict[str, Any]

    def test_path_segments_is_list(self):
        """PathSegments should be list type alias."""
        from aragora.core_protocols import PathSegments

        assert PathSegments == list[str]
