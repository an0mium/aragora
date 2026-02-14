"""Tests for Facts namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestFactsCRUD:
    """Tests for fact create and list operations."""

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
