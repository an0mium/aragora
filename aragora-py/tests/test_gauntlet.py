"""Tests for Aragora SDK Gauntlet API.

Comprehensive tests covering:
- Gauntlet run and wait operations
- Receipt management and verification
- Persona listing
- Results and heatmaps
- Comparison operations
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora_client.client import AragoraClient, GauntletAPI
from aragora_client.exceptions import AragoraNotFoundError, AragoraTimeoutError
from aragora_client.types import GauntletReceipt

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock AragoraClient."""
    client = MagicMock(spec=AragoraClient)
    client._get = AsyncMock()
    client._post = AsyncMock()
    client._delete = AsyncMock()
    client._get_raw = AsyncMock()
    return client


@pytest.fixture
def gauntlet_api(mock_client: MagicMock) -> GauntletAPI:
    """Create GauntletAPI with mock client."""
    return GauntletAPI(mock_client)


@pytest.fixture
def receipt_response() -> dict[str, Any]:
    """Standard gauntlet receipt response."""
    return {
        "id": "receipt-123",
        "score": 0.85,
        "findings": [
            {
                "id": "finding-1",
                "severity": "medium",
                "category": "security",
                "description": "SQL injection risk in user input handler",
                "location": "handlers/user.py:42",
            }
        ],
        "persona": "security",
        "created_at": "2026-01-01T00:00:00Z",
        "hash": "sha256:abc123def456",
    }


@pytest.fixture
def run_response() -> dict[str, Any]:
    """Standard gauntlet run response."""
    return {
        "gauntlet_id": "gauntlet-123",
        "status": "running",
        "persona": "security",
        "created_at": "2026-01-01T00:00:00Z",
    }


@pytest.fixture
def heatmap_response() -> dict[str, Any]:
    """Standard heatmap response."""
    return {
        "id": "heatmap-123",
        "gauntlet_id": "gauntlet-123",
        "categories": {
            "security": {"score": 0.3, "findings": 5},
            "performance": {"score": 0.7, "findings": 2},
            "reliability": {"score": 0.9, "findings": 1},
        },
        "overall_risk": "medium",
    }


# =============================================================================
# Run Tests
# =============================================================================


class TestGauntletAPIRun:
    """Tests for GauntletAPI.run()."""

    @pytest.mark.asyncio
    async def test_run_basic(
        self,
        gauntlet_api: GauntletAPI,
        mock_client: MagicMock,
        run_response: dict[str, Any],
    ) -> None:
        """Test running gauntlet validation."""
        mock_client._post.return_value = run_response

        result = await gauntlet_api.run("API spec content here")

        mock_client._post.assert_called_once()
        call_args = mock_client._post.call_args[0]
        assert call_args[0] == "/api/v1/gauntlet/run"
        assert call_args[1]["input_content"] == "API spec content here"
        assert result["gauntlet_id"] == "gauntlet-123"

    @pytest.mark.asyncio
    async def test_run_with_persona(
        self,
        gauntlet_api: GauntletAPI,
        mock_client: MagicMock,
        run_response: dict[str, Any],
    ) -> None:
        """Test running with a specific persona."""
        mock_client._post.return_value = run_response

        await gauntlet_api.run(
            "Test input",
            input_type="code",
            persona="performance",
        )

        call_args = mock_client._post.call_args[0][1]
        assert call_args["input_type"] == "code"
        assert call_args["persona"] == "performance"


class TestGauntletAPIRunAndWait:
    """Tests for GauntletAPI.run_and_wait()."""

    @pytest.mark.asyncio
    async def test_run_and_wait_completes(
        self,
        gauntlet_api: GauntletAPI,
        mock_client: MagicMock,
        run_response: dict[str, Any],
        receipt_response: dict[str, Any],
    ) -> None:
        """Test run_and_wait when gauntlet completes."""
        mock_client._post.return_value = run_response
        mock_client._get.return_value = receipt_response

        result = await gauntlet_api.run_and_wait(
            "Test input",
            poll_interval=0.01,
        )

        assert isinstance(result, GauntletReceipt)
        assert result.id == "receipt-123"
        assert result.score == 0.85

    @pytest.mark.asyncio
    async def test_run_and_wait_timeout(
        self,
        gauntlet_api: GauntletAPI,
        mock_client: MagicMock,
        run_response: dict[str, Any],
    ) -> None:
        """Test run_and_wait timeout."""
        mock_client._post.return_value = run_response
        mock_client._get.side_effect = AragoraNotFoundError("Receipt", "gauntlet-123")

        with pytest.raises(AragoraTimeoutError):
            await gauntlet_api.run_and_wait(
                "Test",
                poll_interval=0.01,
                timeout=0.05,
            )


# =============================================================================
# Receipt Tests
# =============================================================================


class TestGauntletAPIReceipts:
    """Tests for GauntletAPI receipt operations."""

    @pytest.mark.asyncio
    async def test_get_receipt(
        self,
        gauntlet_api: GauntletAPI,
        mock_client: MagicMock,
        receipt_response: dict[str, Any],
    ) -> None:
        """Test getting a receipt."""
        mock_client._get.return_value = receipt_response

        result = await gauntlet_api.get_receipt("gauntlet-123")

        mock_client._get.assert_called_once_with(
            "/api/v1/gauntlet/gauntlet-123/receipt"
        )
        assert isinstance(result, GauntletReceipt)
        assert result.persona == "security"

    @pytest.mark.asyncio
    async def test_list_receipts(
        self,
        gauntlet_api: GauntletAPI,
        mock_client: MagicMock,
        receipt_response: dict[str, Any],
    ) -> None:
        """Test listing receipts."""
        mock_client._get.return_value = {"receipts": [receipt_response]}

        result = await gauntlet_api.list_receipts()

        assert len(result) == 1
        assert isinstance(result[0], GauntletReceipt)

    @pytest.mark.asyncio
    async def test_list_receipts_by_verdict(
        self, gauntlet_api: GauntletAPI, mock_client: MagicMock
    ) -> None:
        """Test listing receipts filtered by verdict."""
        mock_client._get.return_value = {"receipts": []}

        await gauntlet_api.list_receipts(verdict="pass")

        call_args = mock_client._get.call_args
        assert call_args[1]["params"]["verdict"] == "pass"

    @pytest.mark.asyncio
    async def test_list_receipts_pagination(
        self, gauntlet_api: GauntletAPI, mock_client: MagicMock
    ) -> None:
        """Test listing receipts with pagination."""
        mock_client._get.return_value = {"receipts": []}

        await gauntlet_api.list_receipts(limit=10, offset=20)

        call_args = mock_client._get.call_args
        assert call_args[1]["params"]["limit"] == 10
        assert call_args[1]["params"]["offset"] == 20

    @pytest.mark.asyncio
    async def test_verify_receipt(
        self, gauntlet_api: GauntletAPI, mock_client: MagicMock
    ) -> None:
        """Test verifying receipt integrity."""
        mock_client._get.return_value = {
            "valid": True,
            "hash_match": True,
            "chain_verified": True,
        }

        result = await gauntlet_api.verify_receipt("receipt-123")

        mock_client._get.assert_called_once_with(
            "/api/v1/gauntlet/receipts/receipt-123/verify"
        )
        assert result["valid"] is True

    @pytest.mark.asyncio
    async def test_export_receipt(
        self, gauntlet_api: GauntletAPI, mock_client: MagicMock
    ) -> None:
        """Test exporting a receipt."""
        mock_client._get_raw.return_value = b'{"receipt": "data"}'

        result = await gauntlet_api.export_receipt("receipt-123", format="json")

        mock_client._get_raw.assert_called_once()
        assert b"receipt" in result

    @pytest.mark.asyncio
    async def test_export_receipt_pdf(
        self, gauntlet_api: GauntletAPI, mock_client: MagicMock
    ) -> None:
        """Test exporting receipt as PDF."""
        mock_client._get_raw.return_value = b"%PDF-1.4"

        await gauntlet_api.export_receipt("receipt-123", format="pdf")

        call_args = mock_client._get_raw.call_args
        assert call_args[1]["params"]["format"] == "pdf"


# =============================================================================
# CRUD Tests
# =============================================================================


class TestGauntletAPICRUD:
    """Tests for GauntletAPI CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_gauntlet(
        self, gauntlet_api: GauntletAPI, mock_client: MagicMock
    ) -> None:
        """Test getting gauntlet run details."""
        mock_client._get.return_value = {
            "id": "gauntlet-123",
            "status": "completed",
            "input_type": "spec",
        }

        result = await gauntlet_api.get("gauntlet-123")

        mock_client._get.assert_called_once_with("/api/v1/gauntlet/gauntlet-123")
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_delete_gauntlet(
        self, gauntlet_api: GauntletAPI, mock_client: MagicMock
    ) -> None:
        """Test deleting a gauntlet run."""
        result = await gauntlet_api.delete("gauntlet-123")

        mock_client._delete.assert_called_once_with("/api/v1/gauntlet/gauntlet-123")
        assert result["deleted"] is True


# =============================================================================
# Persona Tests
# =============================================================================


class TestGauntletAPIPersonas:
    """Tests for GauntletAPI persona operations."""

    @pytest.mark.asyncio
    async def test_list_personas(
        self, gauntlet_api: GauntletAPI, mock_client: MagicMock
    ) -> None:
        """Test listing available personas."""
        mock_client._get.return_value = {
            "personas": [
                {"id": "security", "name": "Security Auditor", "category": "security"},
                {"id": "performance", "name": "Performance Analyst", "category": "ops"},
            ]
        }

        result = await gauntlet_api.list_personas()

        assert len(result) == 2
        assert result[0]["id"] == "security"

    @pytest.mark.asyncio
    async def test_list_personas_by_category(
        self, gauntlet_api: GauntletAPI, mock_client: MagicMock
    ) -> None:
        """Test listing personas filtered by category."""
        mock_client._get.return_value = {"personas": []}

        await gauntlet_api.list_personas(category="security")

        call_args = mock_client._get.call_args
        assert call_args[1]["params"]["category"] == "security"

    @pytest.mark.asyncio
    async def test_list_personas_enabled_only(
        self, gauntlet_api: GauntletAPI, mock_client: MagicMock
    ) -> None:
        """Test listing only enabled personas."""
        mock_client._get.return_value = {"personas": []}

        await gauntlet_api.list_personas(enabled=True)

        call_args = mock_client._get.call_args
        assert call_args[1]["params"]["enabled"] is True


# =============================================================================
# Results Tests
# =============================================================================


class TestGauntletAPIResults:
    """Tests for GauntletAPI results operations."""

    @pytest.mark.asyncio
    async def test_list_results(
        self, gauntlet_api: GauntletAPI, mock_client: MagicMock
    ) -> None:
        """Test listing gauntlet results."""
        mock_client._get.return_value = {
            "results": [{"id": "result-1", "status": "passed", "score": 0.92}]
        }

        result = await gauntlet_api.list_results()

        assert len(result) == 1
        assert result[0]["status"] == "passed"

    @pytest.mark.asyncio
    async def test_list_results_by_status(
        self, gauntlet_api: GauntletAPI, mock_client: MagicMock
    ) -> None:
        """Test listing results filtered by status."""
        mock_client._get.return_value = {"results": []}

        await gauntlet_api.list_results(status="failed")

        call_args = mock_client._get.call_args
        assert call_args[1]["params"]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_list_results_by_gauntlet_id(
        self, gauntlet_api: GauntletAPI, mock_client: MagicMock
    ) -> None:
        """Test listing results for a specific gauntlet."""
        mock_client._get.return_value = {"results": []}

        await gauntlet_api.list_results(gauntlet_id="gauntlet-123")

        call_args = mock_client._get.call_args
        assert call_args[1]["params"]["gauntlet_id"] == "gauntlet-123"


# =============================================================================
# Heatmap Tests
# =============================================================================


class TestGauntletAPIHeatmaps:
    """Tests for GauntletAPI heatmap operations."""

    @pytest.mark.asyncio
    async def test_get_heatmap(
        self,
        gauntlet_api: GauntletAPI,
        mock_client: MagicMock,
        heatmap_response: dict[str, Any],
    ) -> None:
        """Test getting a risk heatmap."""
        mock_client._get.return_value = heatmap_response

        result = await gauntlet_api.get_heatmap("gauntlet-123")

        assert result["overall_risk"] == "medium"
        assert "security" in result["categories"]

    @pytest.mark.asyncio
    async def test_get_heatmap_svg_format(
        self, gauntlet_api: GauntletAPI, mock_client: MagicMock
    ) -> None:
        """Test getting heatmap in SVG format."""
        mock_client._get.return_value = {"svg": "<svg>...</svg>"}

        await gauntlet_api.get_heatmap("gauntlet-123", format="svg")

        call_args = mock_client._get.call_args
        assert call_args[1]["params"]["format"] == "svg"

    @pytest.mark.asyncio
    async def test_list_heatmaps(
        self, gauntlet_api: GauntletAPI, mock_client: MagicMock
    ) -> None:
        """Test listing heatmaps."""
        mock_client._get.return_value = {
            "heatmaps": [{"id": "heatmap-1"}, {"id": "heatmap-2"}]
        }

        result = await gauntlet_api.list_heatmaps()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_heatmaps_pagination(
        self, gauntlet_api: GauntletAPI, mock_client: MagicMock
    ) -> None:
        """Test listing heatmaps with pagination."""
        mock_client._get.return_value = {"heatmaps": []}

        await gauntlet_api.list_heatmaps(limit=5, offset=10)

        call_args = mock_client._get.call_args
        assert call_args[1]["params"]["limit"] == 5

    @pytest.mark.asyncio
    async def test_get_risk_heatmap(
        self, gauntlet_api: GauntletAPI, mock_client: MagicMock
    ) -> None:
        """Test getting a specific risk heatmap by ID."""
        mock_client._get.return_value = {"id": "heatmap-123", "risk_level": "high"}

        result = await gauntlet_api.get_risk_heatmap("heatmap-123")

        mock_client._get.assert_called_once_with(
            "/api/v1/gauntlet/heatmaps/heatmap-123"
        )
        assert result["risk_level"] == "high"


# =============================================================================
# Comparison Tests
# =============================================================================


class TestGauntletAPICompare:
    """Tests for GauntletAPI comparison operations."""

    @pytest.mark.asyncio
    async def test_compare(
        self, gauntlet_api: GauntletAPI, mock_client: MagicMock
    ) -> None:
        """Test comparing two gauntlet runs."""
        mock_client._get.return_value = {
            "run_a": "gauntlet-1",
            "run_b": "gauntlet-2",
            "score_diff": 0.15,
            "new_findings": 3,
            "resolved_findings": 1,
            "regression": False,
        }

        result = await gauntlet_api.compare("gauntlet-1", "gauntlet-2")

        mock_client._get.assert_called_once_with(
            "/api/v1/gauntlet/compare/gauntlet-1/gauntlet-2"
        )
        assert result["score_diff"] == 0.15
        assert result["regression"] is False
