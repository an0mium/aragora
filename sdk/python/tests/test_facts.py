"""Tests for Facts namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestFactsCRUD:
    """Tests for fact create, read, update, delete operations."""

    def test_create_fact_minimal(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "fact_1", "content": "Water boils at 100C"}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.facts.create_fact("Water boils at 100C")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/facts",
                json={"content": "Water boils at 100C"},
            )
            assert result["id"] == "fact_1"
            client.close()

    def test_create_fact_with_all_options(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "fact_2", "content": "Python is interpreted"}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.facts.create_fact(
                "Python is interpreted",
                source="docs",
                confidence=0.95,
                tags=["programming", "python"],
                metadata={"verified": True},
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/facts",
                json={
                    "content": "Python is interpreted",
                    "source": "docs",
                    "confidence": 0.95,
                    "tags": ["programming", "python"],
                    "metadata": {"verified": True},
                },
            )
            assert result["id"] == "fact_2"
            client.close()

    def test_get_fact(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "fact_1", "content": "Earth orbits the Sun"}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.facts.get_fact("fact_1")
            mock_request.assert_called_once_with("GET", "/api/v1/facts/fact_1")
            assert result["content"] == "Earth orbits the Sun"
            client.close()

    def test_update_fact(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "fact_1", "confidence": 0.99}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.facts.update_fact("fact_1", confidence=0.99, content="Updated content")
            mock_request.assert_called_once_with(
                "PATCH",
                "/api/v1/facts/fact_1",
                json={"confidence": 0.99, "content": "Updated content"},
            )
            assert result["confidence"] == 0.99
            client.close()

    def test_delete_fact(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"deleted": True}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.facts.delete_fact("fact_1")
            mock_request.assert_called_once_with("DELETE", "/api/v1/facts/fact_1")
            assert result["deleted"] is True
            client.close()

    def test_list_facts_no_filters(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"facts": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.facts.list_facts()
            mock_request.assert_called_once_with("GET", "/api/v1/facts", params=None)
            assert result["total"] == 0
            client.close()

    def test_list_facts_with_filters(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"facts": [{"id": "f1"}], "total": 1}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.facts.list_facts(
                limit=10,
                offset=5,
                tag="science",
                source="wikipedia",
                min_confidence=0.8,
            )
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/facts",
                params={
                    "limit": 10,
                    "offset": 5,
                    "tag": "science",
                    "source": "wikipedia",
                    "min_confidence": 0.8,
                },
            )
            assert result["total"] == 1
            client.close()


class TestFactsSearchAndExists:
    """Tests for search and existence check operations."""

    def test_search_facts_basic(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"results": [{"id": "f1", "score": 0.92}]}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.facts.search_facts("boiling point")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/facts/search",
                params={"query": "boiling point"},
            )
            assert result["results"][0]["score"] == 0.92
            client.close()

    def test_search_facts_with_options(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"results": []}
            client = AragoraClient(base_url="https://api.aragora.ai")
            client.facts.search_facts("quantum physics", limit=5, min_score=0.7)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/facts/search",
                params={"query": "quantum physics", "limit": 5, "min_score": 0.7},
            )
            client.close()

    def test_exists(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {}
            client = AragoraClient(base_url="https://api.aragora.ai")
            client.facts.exists("fact_42")
            mock_request.assert_called_once_with("HEAD", "/api/v1/facts/fact_42")
            client.close()


class TestFactsRelationships:
    """Tests for relationship management between facts."""

    def test_create_relationship_minimal(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "rel_1", "rel_type": "supports"}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.facts.create_relationship("fact_1", "fact_2", "supports")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/facts/relationships",
                json={
                    "source_id": "fact_1",
                    "target_id": "fact_2",
                    "rel_type": "supports",
                },
            )
            assert result["rel_type"] == "supports"
            client.close()

    def test_create_relationship_with_strength(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "rel_2", "strength": 0.85}
            client = AragoraClient(base_url="https://api.aragora.ai")
            client.facts.create_relationship("fact_3", "fact_4", "contradicts", strength=0.85)
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/facts/relationships",
                json={
                    "source_id": "fact_3",
                    "target_id": "fact_4",
                    "rel_type": "contradicts",
                    "strength": 0.85,
                },
            )
            client.close()

    def test_get_relationship(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "rel_1", "rel_type": "supports"}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.facts.get_relationship("rel_1")
            mock_request.assert_called_once_with("GET", "/api/v1/facts/relationships/rel_1")
            assert result["id"] == "rel_1"
            client.close()

    def test_update_relationship(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "rel_1", "strength": 0.95}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.facts.update_relationship("rel_1", strength=0.95)
            mock_request.assert_called_once_with(
                "PATCH",
                "/api/v1/facts/relationships/rel_1",
                json={"strength": 0.95},
            )
            assert result["strength"] == 0.95
            client.close()

    def test_delete_relationship(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"deleted": True}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.facts.delete_relationship("rel_1")
            mock_request.assert_called_once_with("DELETE", "/api/v1/facts/relationships/rel_1")
            assert result["deleted"] is True
            client.close()

    def test_get_relationships_no_filters(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"relationships": []}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.facts.get_relationships("fact_1")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/facts/fact_1/relationships",
                params=None,
            )
            assert result["relationships"] == []
            client.close()

    def test_get_relationships_with_filters(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"relationships": [{"id": "rel_5"}]}
            client = AragoraClient(base_url="https://api.aragora.ai")
            client.facts.get_relationships("fact_1", direction="outgoing", rel_type="supports")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/facts/fact_1/relationships",
                params={"direction": "outgoing", "rel_type": "supports"},
            )
            client.close()

    def test_get_related_facts_no_filters(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"facts": [{"id": "fact_7"}]}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.facts.get_related_facts("fact_1")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/facts/fact_1/related",
                params=None,
            )
            assert len(result["facts"]) == 1
            client.close()

    def test_get_related_facts_with_options(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"facts": []}
            client = AragoraClient(base_url="https://api.aragora.ai")
            client.facts.get_related_facts("fact_1", max_depth=3, min_strength=0.5)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/facts/fact_1/related",
                params={"max_depth": 3, "min_strength": 0.5},
            )
            client.close()


class TestFactsBatchAndUtility:
    """Tests for batch operations, stats, validation, and merging."""

    def test_batch_create(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"created": 2, "ids": ["f1", "f2"]}
            client = AragoraClient(base_url="https://api.aragora.ai")
            facts_data = [
                {"content": "Fact A"},
                {"content": "Fact B", "confidence": 0.9},
            ]
            result = client.facts.batch_create(facts_data)
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/facts/batch",
                json={"facts": facts_data},
            )
            assert result["created"] == 2
            client.close()

    def test_batch_delete(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"deleted": 3}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.facts.batch_delete(["f1", "f2", "f3"])
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/facts/batch/delete",
                json={"ids": ["f1", "f2", "f3"]},
            )
            assert result["deleted"] == 3
            client.close()

    def test_get_stats(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"total_facts": 1500, "avg_confidence": 0.87}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.facts.get_stats()
            mock_request.assert_called_once_with("GET", "/api/v1/facts/stats")
            assert result["total_facts"] == 1500
            client.close()

    def test_validate_content(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"valid": True, "issues": []}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.facts.validate_content("The sky is blue")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/facts/validate",
                json={"content": "The sky is blue"},
            )
            assert result["valid"] is True
            client.close()

    def test_merge_facts_minimal(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"merged_id": "fact_merged", "status": "ok"}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.facts.merge_facts("fact_1", "fact_2")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/facts/merge",
                json={"source_id": "fact_1", "target_id": "fact_2"},
            )
            assert result["merged_id"] == "fact_merged"
            client.close()

    def test_merge_facts_with_strategy(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"merged_id": "fact_merged"}
            client = AragoraClient(base_url="https://api.aragora.ai")
            client.facts.merge_facts("fact_1", "fact_2", strategy="prefer_source")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/facts/merge",
                json={
                    "source_id": "fact_1",
                    "target_id": "fact_2",
                    "strategy": "prefer_source",
                },
            )
            client.close()


class TestAsyncFacts:
    """Tests for async facts methods."""

    @pytest.mark.asyncio
    async def test_create_fact(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"id": "fact_1", "content": "Async fact"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.facts.create_fact("Async fact", source="test", confidence=0.9)
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/facts",
                json={"content": "Async fact", "source": "test", "confidence": 0.9},
            )
            assert result["id"] == "fact_1"
            await client.close()

    @pytest.mark.asyncio
    async def test_get_fact(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"id": "fact_1", "content": "Async retrieved"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.facts.get_fact("fact_1")
            mock_request.assert_called_once_with("GET", "/api/v1/facts/fact_1")
            assert result["content"] == "Async retrieved"
            await client.close()

    @pytest.mark.asyncio
    async def test_list_facts_with_filters(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"facts": [{"id": "f1"}], "total": 1}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.facts.list_facts(tag="science", limit=20)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/facts",
                params={"tag": "science", "limit": 20},
            )
            assert result["total"] == 1
            await client.close()

    @pytest.mark.asyncio
    async def test_search_facts(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"results": [{"id": "f1", "score": 0.88}]}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.facts.search_facts("gravity", min_score=0.5)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/facts/search",
                params={"query": "gravity", "min_score": 0.5},
            )
            assert result["results"][0]["score"] == 0.88
            await client.close()

    @pytest.mark.asyncio
    async def test_create_relationship(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"id": "rel_1"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.facts.create_relationship(
                "fact_1", "fact_2", "supports", strength=0.9
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/facts/relationships",
                json={
                    "source_id": "fact_1",
                    "target_id": "fact_2",
                    "rel_type": "supports",
                    "strength": 0.9,
                },
            )
            assert result["id"] == "rel_1"
            await client.close()

    @pytest.mark.asyncio
    async def test_batch_create(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"created": 2}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            facts_data = [{"content": "A"}, {"content": "B"}]
            result = await client.facts.batch_create(facts_data)
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/facts/batch",
                json={"facts": facts_data},
            )
            assert result["created"] == 2
            await client.close()

    @pytest.mark.asyncio
    async def test_merge_facts(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"merged_id": "fact_merged"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.facts.merge_facts("f1", "f2", strategy="combine")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/facts/merge",
                json={"source_id": "f1", "target_id": "f2", "strategy": "combine"},
            )
            assert result["merged_id"] == "fact_merged"
            await client.close()

    @pytest.mark.asyncio
    async def test_get_stats(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"total_facts": 500}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.facts.get_stats()
            mock_request.assert_called_once_with("GET", "/api/v1/facts/stats")
            assert result["total_facts"] == 500
            await client.close()
