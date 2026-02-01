"""
Tests for the Aragora Python SDK Hybrid Debates namespace.

Covers: HybridDebatesAPI (sync) and AsyncHybridDebatesAPI (async) methods
for creating, retrieving, and listing hybrid debates.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from sdk.python.aragora.namespaces.hybrid_debates import (
    AsyncHybridDebatesAPI,
    HybridDebatesAPI,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def sync_client() -> MagicMock:
    """Create a mock synchronous client."""
    client = MagicMock()
    client.request = MagicMock()
    return client


@pytest.fixture
def async_client() -> MagicMock:
    """Create a mock asynchronous client."""
    client = MagicMock()
    client.request = AsyncMock()
    return client


@pytest.fixture
def sync_api(sync_client: MagicMock) -> HybridDebatesAPI:
    """Create a sync HybridDebatesAPI instance."""
    return HybridDebatesAPI(sync_client)


@pytest.fixture
def async_api(async_client: MagicMock) -> AsyncHybridDebatesAPI:
    """Create an async AsyncHybridDebatesAPI instance."""
    return AsyncHybridDebatesAPI(async_client)


# ===========================================================================
# HybridDebatesAPI (Sync) - create
# ===========================================================================


class TestHybridDebatesAPICreate:
    """Tests for the synchronous create method."""

    def test_create_with_required_params(
        self, sync_api: HybridDebatesAPI, sync_client: MagicMock
    ) -> None:
        """create() sends POST with required fields."""
        sync_client.request.return_value = {
            "debate_id": "hybrid_abc123",
            "status": "completed",
        }

        result = sync_api.create(
            task="Should we adopt microservices?",
            external_agent="crewai-infra-team",
        )

        sync_client.request.assert_called_once_with(
            "POST",
            "/api/v1/debates/hybrid",
            json={
                "task": "Should we adopt microservices?",
                "external_agent": "crewai-infra-team",
                "consensus_threshold": 0.7,
                "max_rounds": 3,
                "domain": "general",
            },
        )
        assert result["debate_id"] == "hybrid_abc123"

    def test_create_with_all_params(
        self, sync_api: HybridDebatesAPI, sync_client: MagicMock
    ) -> None:
        """create() sends POST with all optional fields."""
        sync_client.request.return_value = {"debate_id": "hybrid_xyz789"}

        result = sync_api.create(
            task="Evaluate cloud migration strategy",
            external_agent="langraph-cloud-team",
            consensus_threshold=0.85,
            max_rounds=5,
            verification_agents=["claude", "gpt4"],
            domain="infrastructure",
            config={"timeout": 300},
        )

        sync_client.request.assert_called_once_with(
            "POST",
            "/api/v1/debates/hybrid",
            json={
                "task": "Evaluate cloud migration strategy",
                "external_agent": "langraph-cloud-team",
                "consensus_threshold": 0.85,
                "max_rounds": 5,
                "verification_agents": ["claude", "gpt4"],
                "domain": "infrastructure",
                "config": {"timeout": 300},
            },
        )
        assert result["debate_id"] == "hybrid_xyz789"

    def test_create_omits_none_optional_fields(
        self, sync_api: HybridDebatesAPI, sync_client: MagicMock
    ) -> None:
        """create() omits verification_agents and config when None."""
        sync_client.request.return_value = {"debate_id": "hybrid_000"}

        sync_api.create(
            task="Test task",
            external_agent="agent-1",
            verification_agents=None,
            config=None,
        )

        call_json = sync_client.request.call_args[1]["json"]
        assert "verification_agents" not in call_json
        assert "config" not in call_json

    def test_create_includes_nonempty_optional_fields(
        self, sync_api: HybridDebatesAPI, sync_client: MagicMock
    ) -> None:
        """create() includes verification_agents and config when provided."""
        sync_client.request.return_value = {"debate_id": "hybrid_111"}

        sync_api.create(
            task="Test task",
            external_agent="agent-1",
            verification_agents=["claude"],
            config={"key": "value"},
        )

        call_json = sync_client.request.call_args[1]["json"]
        assert call_json["verification_agents"] == ["claude"]
        assert call_json["config"] == {"key": "value"}


# ===========================================================================
# HybridDebatesAPI (Sync) - get
# ===========================================================================


class TestHybridDebatesAPIGet:
    """Tests for the synchronous get method."""

    def test_get_debate(
        self, sync_api: HybridDebatesAPI, sync_client: MagicMock
    ) -> None:
        """get() sends GET to the correct path."""
        sync_client.request.return_value = {
            "debate_id": "hybrid_abc123",
            "task": "Should we adopt microservices?",
            "status": "completed",
            "consensus_reached": True,
            "confidence": 0.92,
        }

        result = sync_api.get("hybrid_abc123")

        sync_client.request.assert_called_once_with(
            "GET", "/api/v1/debates/hybrid/hybrid_abc123"
        )
        assert result["debate_id"] == "hybrid_abc123"
        assert result["consensus_reached"] is True

    def test_get_debate_with_different_id(
        self, sync_api: HybridDebatesAPI, sync_client: MagicMock
    ) -> None:
        """get() correctly interpolates debate_id into the URL."""
        sync_client.request.return_value = {"debate_id": "hybrid_foobar"}

        sync_api.get("hybrid_foobar")

        sync_client.request.assert_called_once_with(
            "GET", "/api/v1/debates/hybrid/hybrid_foobar"
        )


# ===========================================================================
# HybridDebatesAPI (Sync) - list
# ===========================================================================


class TestHybridDebatesAPIList:
    """Tests for the synchronous list method."""

    def test_list_no_params(
        self, sync_api: HybridDebatesAPI, sync_client: MagicMock
    ) -> None:
        """list() sends GET with no params when using defaults."""
        sync_client.request.return_value = {"debates": [], "total": 0}

        result = sync_api.list()

        sync_client.request.assert_called_once_with(
            "GET", "/api/v1/debates/hybrid", params={}
        )
        assert result["total"] == 0

    def test_list_with_status_filter(
        self, sync_api: HybridDebatesAPI, sync_client: MagicMock
    ) -> None:
        """list() includes status param when provided."""
        sync_client.request.return_value = {"debates": [], "total": 0}

        sync_api.list(status="completed")

        sync_client.request.assert_called_once_with(
            "GET", "/api/v1/debates/hybrid", params={"status": "completed"}
        )

    def test_list_with_custom_limit(
        self, sync_api: HybridDebatesAPI, sync_client: MagicMock
    ) -> None:
        """list() includes limit param when not default."""
        sync_client.request.return_value = {"debates": [], "total": 0}

        sync_api.list(limit=50)

        sync_client.request.assert_called_once_with(
            "GET", "/api/v1/debates/hybrid", params={"limit": 50}
        )

    def test_list_with_all_params(
        self, sync_api: HybridDebatesAPI, sync_client: MagicMock
    ) -> None:
        """list() includes all params when provided."""
        sync_client.request.return_value = {
            "debates": [{"debate_id": "hybrid_1"}],
            "total": 1,
        }

        result = sync_api.list(status="pending", limit=10)

        sync_client.request.assert_called_once_with(
            "GET",
            "/api/v1/debates/hybrid",
            params={"status": "pending", "limit": 10},
        )
        assert result["total"] == 1

    def test_list_default_limit_not_sent(
        self, sync_api: HybridDebatesAPI, sync_client: MagicMock
    ) -> None:
        """list() does not send limit param when it equals the default (20)."""
        sync_client.request.return_value = {"debates": [], "total": 0}

        sync_api.list(limit=20)

        call_params = sync_client.request.call_args[1]["params"]
        assert "limit" not in call_params


# ===========================================================================
# AsyncHybridDebatesAPI - create
# ===========================================================================


class TestAsyncHybridDebatesAPICreate:
    """Tests for the asynchronous create method."""

    @pytest.mark.asyncio
    async def test_create_with_required_params(
        self, async_api: AsyncHybridDebatesAPI, async_client: MagicMock
    ) -> None:
        """create() sends POST with required fields."""
        async_client.request.return_value = {
            "debate_id": "hybrid_abc123",
            "status": "completed",
        }

        result = await async_api.create(
            task="Should we adopt microservices?",
            external_agent="crewai-infra-team",
        )

        async_client.request.assert_called_once_with(
            "POST",
            "/api/v1/debates/hybrid",
            json={
                "task": "Should we adopt microservices?",
                "external_agent": "crewai-infra-team",
                "consensus_threshold": 0.7,
                "max_rounds": 3,
                "domain": "general",
            },
        )
        assert result["debate_id"] == "hybrid_abc123"

    @pytest.mark.asyncio
    async def test_create_with_all_params(
        self, async_api: AsyncHybridDebatesAPI, async_client: MagicMock
    ) -> None:
        """create() sends POST with all optional fields."""
        async_client.request.return_value = {"debate_id": "hybrid_xyz789"}

        result = await async_api.create(
            task="Evaluate cloud migration strategy",
            external_agent="langraph-cloud-team",
            consensus_threshold=0.85,
            max_rounds=5,
            verification_agents=["claude", "gpt4"],
            domain="infrastructure",
            config={"timeout": 300},
        )

        async_client.request.assert_called_once_with(
            "POST",
            "/api/v1/debates/hybrid",
            json={
                "task": "Evaluate cloud migration strategy",
                "external_agent": "langraph-cloud-team",
                "consensus_threshold": 0.85,
                "max_rounds": 5,
                "verification_agents": ["claude", "gpt4"],
                "domain": "infrastructure",
                "config": {"timeout": 300},
            },
        )
        assert result["debate_id"] == "hybrid_xyz789"

    @pytest.mark.asyncio
    async def test_create_omits_none_optional_fields(
        self, async_api: AsyncHybridDebatesAPI, async_client: MagicMock
    ) -> None:
        """create() omits verification_agents and config when None."""
        async_client.request.return_value = {"debate_id": "hybrid_000"}

        await async_api.create(
            task="Test task",
            external_agent="agent-1",
        )

        call_json = async_client.request.call_args[1]["json"]
        assert "verification_agents" not in call_json
        assert "config" not in call_json


# ===========================================================================
# AsyncHybridDebatesAPI - get
# ===========================================================================


class TestAsyncHybridDebatesAPIGet:
    """Tests for the asynchronous get method."""

    @pytest.mark.asyncio
    async def test_get_debate(
        self, async_api: AsyncHybridDebatesAPI, async_client: MagicMock
    ) -> None:
        """get() sends GET to the correct path."""
        async_client.request.return_value = {
            "debate_id": "hybrid_abc123",
            "status": "completed",
        }

        result = await async_api.get("hybrid_abc123")

        async_client.request.assert_called_once_with(
            "GET", "/api/v1/debates/hybrid/hybrid_abc123"
        )
        assert result["debate_id"] == "hybrid_abc123"


# ===========================================================================
# AsyncHybridDebatesAPI - list
# ===========================================================================


class TestAsyncHybridDebatesAPIList:
    """Tests for the asynchronous list method."""

    @pytest.mark.asyncio
    async def test_list_no_params(
        self, async_api: AsyncHybridDebatesAPI, async_client: MagicMock
    ) -> None:
        """list() sends GET with no params when using defaults."""
        async_client.request.return_value = {"debates": [], "total": 0}

        result = await async_api.list()

        async_client.request.assert_called_once_with(
            "GET", "/api/v1/debates/hybrid", params={}
        )
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_list_with_status_filter(
        self, async_api: AsyncHybridDebatesAPI, async_client: MagicMock
    ) -> None:
        """list() includes status param when provided."""
        async_client.request.return_value = {"debates": [], "total": 0}

        await async_api.list(status="completed")

        async_client.request.assert_called_once_with(
            "GET", "/api/v1/debates/hybrid", params={"status": "completed"}
        )

    @pytest.mark.asyncio
    async def test_list_with_custom_limit(
        self, async_api: AsyncHybridDebatesAPI, async_client: MagicMock
    ) -> None:
        """list() includes limit param when not default."""
        async_client.request.return_value = {"debates": [], "total": 0}

        await async_api.list(limit=50)

        async_client.request.assert_called_once_with(
            "GET", "/api/v1/debates/hybrid", params={"limit": 50}
        )

    @pytest.mark.asyncio
    async def test_list_with_all_params(
        self, async_api: AsyncHybridDebatesAPI, async_client: MagicMock
    ) -> None:
        """list() includes all params when provided."""
        async_client.request.return_value = {
            "debates": [{"debate_id": "hybrid_1"}],
            "total": 1,
        }

        result = await async_api.list(status="pending", limit=10)

        async_client.request.assert_called_once_with(
            "GET",
            "/api/v1/debates/hybrid",
            params={"status": "pending", "limit": 10},
        )
        assert result["total"] == 1


# ===========================================================================
# Class structure and initialization
# ===========================================================================


class TestHybridDebatesAPIStructure:
    """Tests for class structure and initialization."""

    def test_sync_api_stores_client(self, sync_client: MagicMock) -> None:
        """HybridDebatesAPI stores the client reference."""
        api = HybridDebatesAPI(sync_client)
        assert api._client is sync_client

    def test_async_api_stores_client(self, async_client: MagicMock) -> None:
        """AsyncHybridDebatesAPI stores the client reference."""
        api = AsyncHybridDebatesAPI(async_client)
        assert api._client is async_client

    def test_sync_api_has_expected_methods(self) -> None:
        """HybridDebatesAPI exposes create, get, list methods."""
        public_methods = [
            m for m in dir(HybridDebatesAPI) if not m.startswith("_")
        ]
        assert "create" in public_methods
        assert "get" in public_methods
        assert "list" in public_methods

    def test_async_api_has_expected_methods(self) -> None:
        """AsyncHybridDebatesAPI exposes create, get, list methods."""
        public_methods = [
            m for m in dir(AsyncHybridDebatesAPI) if not m.startswith("_")
        ]
        assert "create" in public_methods
        assert "get" in public_methods
        assert "list" in public_methods

    def test_sync_and_async_method_parity(self) -> None:
        """Sync and async APIs have the same public methods."""
        sync_methods = {
            m for m in dir(HybridDebatesAPI) if not m.startswith("_")
        }
        async_methods = {
            m for m in dir(AsyncHybridDebatesAPI) if not m.startswith("_")
        }
        assert sync_methods == async_methods


# ===========================================================================
# Import from namespaces __init__
# ===========================================================================


class TestHybridDebatesImports:
    """Tests for module imports."""

    def test_import_from_namespaces_package(self) -> None:
        """HybridDebatesAPI and AsyncHybridDebatesAPI are importable from the package."""
        from sdk.python.aragora.namespaces import (
            AsyncHybridDebatesAPI,
            HybridDebatesAPI,
        )

        assert HybridDebatesAPI is not None
        assert AsyncHybridDebatesAPI is not None
