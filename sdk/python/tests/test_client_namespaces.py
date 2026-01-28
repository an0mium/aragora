"""Client namespace wiring tests for the Python SDK."""

from __future__ import annotations

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient
from aragora.namespaces.belief import AsyncBeliefAPI, BeliefAPI
from aragora.namespaces.genesis import AsyncGenesisAPI, GenesisAPI
from aragora.namespaces.relationships import AsyncRelationshipsAPI, RelationshipsAPI


def test_sync_client_includes_new_namespaces() -> None:
    client = AragoraClient(base_url="http://localhost")

    assert isinstance(client.belief, BeliefAPI)
    assert isinstance(client.genesis, GenesisAPI)
    assert isinstance(client.relationships, RelationshipsAPI)


@pytest.mark.asyncio
async def test_async_client_includes_new_namespaces() -> None:
    async with AragoraAsyncClient(base_url="http://localhost") as client:
        assert isinstance(client.belief, AsyncBeliefAPI)
        assert isinstance(client.genesis, AsyncGenesisAPI)
        assert isinstance(client.relationships, AsyncRelationshipsAPI)
