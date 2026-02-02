"""
Tests for E-commerce Platform Handler.

Stability: STABLE (graduated from EXPERIMENTAL)

Covers:
- Platform listing and connection status
- Platform connection/disconnection
- Orders: listing, filtering, cross-platform aggregation
- Products: listing, details, cross-platform aggregation
- Inventory: retrieval and sync operations
- Fulfillment: status and shipment creation
- Metrics aggregation
- Circuit breaker functionality
- Rate limiting integration
- Input validation
- RBAC permission checks
- Error handling
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.ecommerce import (
    EcommerceHandler,
    EcommerceCircuitBreaker,
    SUPPORTED_PLATFORMS,
    UnifiedOrder,
    UnifiedProduct,
    get_ecommerce_circuit_breaker,
    reset_ecommerce_circuit_breaker,
    _validate_platform_id,
    _validate_resource_id,
    _validate_sku,
    _validate_url,
    _validate_quantity,
    _platform_credentials,
    _platform_connectors,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def server_context():
    """Create a mock server context."""
    return {"config": {"debug": True}}


@pytest.fixture
def ecommerce_handler(server_context):
    """Create an EcommerceHandler instance."""
    reset_ecommerce_circuit_breaker()
    # Clear global state
    _platform_credentials.clear()
    _platform_connectors.clear()
    return EcommerceHandler(server_context=server_context)


@pytest.fixture
def mock_request():
    """Create a mock HTTP request."""
    request = MagicMock()
    request.method = "GET"
    request.path = "/api/v1/ecommerce/platforms"
    request.query = {}
    request.headers = {
        "Authorization": "Bearer test-token",
        "Content-Type": "application/json",
    }
    return request


@pytest.fixture
def mock_auth_context():
    """Create a mock authorization context."""
    ctx = MagicMock()
    ctx.user_id = "test-user"
    ctx.workspace_id = "test-workspace"
    ctx.roles = ["admin"]
    return ctx


@pytest.fixture(autouse=True)
def cleanup_global_state():
    """Clean up global state before and after each test."""
    _platform_credentials.clear()
    _platform_connectors.clear()
    reset_ecommerce_circuit_breaker()
    yield
    _platform_credentials.clear()
    _platform_connectors.clear()
    reset_ecommerce_circuit_breaker()


# -----------------------------------------------------------------------------
# Test: Supported Platforms Configuration
# -----------------------------------------------------------------------------


class TestSupportedPlatforms:
    """Tests for e-commerce platform configuration."""

    def test_all_platforms_defined(self):
        """Test that e-commerce platforms are configured."""
        assert "shopify" in SUPPORTED_PLATFORMS
        assert "shipstation" in SUPPORTED_PLATFORMS
        assert "walmart" in SUPPORTED_PLATFORMS
        assert len(SUPPORTED_PLATFORMS) >= 3

    def test_platform_has_required_fields(self):
        """Test that all platforms have required configuration."""
        for platform_id, config in SUPPORTED_PLATFORMS.items():
            assert "name" in config, f"Platform {platform_id} missing 'name'"
            assert "description" in config, f"Platform {platform_id} missing 'description'"
            assert "features" in config, f"Platform {platform_id} missing 'features'"
            assert isinstance(config["features"], list), (
                f"Platform {platform_id} features should be list"
            )

    def test_shopify_features(self):
        """Test Shopify has expected features."""
        shopify = SUPPORTED_PLATFORMS["shopify"]
        assert "orders" in shopify["features"]
        assert "products" in shopify["features"]
        assert "inventory" in shopify["features"]

    def test_shipstation_features(self):
        """Test ShipStation has expected features."""
        shipstation = SUPPORTED_PLATFORMS["shipstation"]
        assert "shipments" in shipstation["features"]
        assert "tracking" in shipstation["features"]

    def test_walmart_features(self):
        """Test Walmart has expected features."""
        walmart = SUPPORTED_PLATFORMS["walmart"]
        assert "orders" in walmart["features"]
        assert "inventory" in walmart["features"]


# -----------------------------------------------------------------------------
# Test: Input Validation Functions
# -----------------------------------------------------------------------------


class TestInputValidation:
    """Tests for input validation functions."""

    def test_validate_platform_id_valid(self):
        """Test valid platform IDs."""
        assert _validate_platform_id("shopify") == (True, None)
        assert _validate_platform_id("my_platform") == (True, None)
        assert _validate_platform_id("platform123") == (True, None)

    def test_validate_platform_id_invalid(self):
        """Test invalid platform IDs."""
        is_valid, msg = _validate_platform_id("")
        assert not is_valid
        assert "required" in msg.lower()

        is_valid, msg = _validate_platform_id("a" * 100)
        assert not is_valid
        assert "too long" in msg.lower()

        is_valid, msg = _validate_platform_id("invalid-platform")
        assert not is_valid
        assert "invalid" in msg.lower()

        is_valid, msg = _validate_platform_id("123platform")
        assert not is_valid

    def test_validate_resource_id_valid(self):
        """Test valid resource IDs."""
        assert _validate_resource_id("order123") == (True, None)
        assert _validate_resource_id("prod-123-abc") == (True, None)
        assert _validate_resource_id("item_456") == (True, None)

    def test_validate_resource_id_invalid(self):
        """Test invalid resource IDs."""
        is_valid, msg = _validate_resource_id("")
        assert not is_valid

        is_valid, msg = _validate_resource_id("a" * 200)
        assert not is_valid
        assert "too long" in msg.lower()

        is_valid, msg = _validate_resource_id("-invalid")
        assert not is_valid

    def test_validate_sku_valid(self):
        """Test valid SKUs."""
        assert _validate_sku("SKU123") == (True, None)
        assert _validate_sku("product-abc-001") == (True, None)
        assert _validate_sku("item.v2.0") == (True, None)

    def test_validate_sku_invalid(self):
        """Test invalid SKUs."""
        is_valid, msg = _validate_sku("")
        assert not is_valid

        is_valid, msg = _validate_sku("a" * 200)
        assert not is_valid

    def test_validate_url_valid(self):
        """Test valid URLs."""
        assert _validate_url("https://example.com") == (True, None)
        assert _validate_url("http://shop.example.com") == (True, None)

    def test_validate_url_invalid(self):
        """Test invalid URLs."""
        is_valid, msg = _validate_url("")
        assert not is_valid

        is_valid, msg = _validate_url("not-a-url")
        assert not is_valid

        is_valid, msg = _validate_url("ftp://example.com")
        assert not is_valid

    def test_validate_quantity_valid(self):
        """Test valid quantities."""
        is_valid, msg, qty = _validate_quantity(100)
        assert is_valid
        assert qty == 100

        is_valid, msg, qty = _validate_quantity("50")
        assert is_valid
        assert qty == 50

        is_valid, msg, qty = _validate_quantity(0)
        assert is_valid
        assert qty == 0

    def test_validate_quantity_invalid(self):
        """Test invalid quantities."""
        is_valid, msg, qty = _validate_quantity(None)
        assert not is_valid

        is_valid, msg, qty = _validate_quantity(-10)
        assert not is_valid

        is_valid, msg, qty = _validate_quantity("not-a-number")
        assert not is_valid

        is_valid, msg, qty = _validate_quantity(10_000_000_000)
        assert not is_valid


# -----------------------------------------------------------------------------
# Test: Circuit Breaker
# -----------------------------------------------------------------------------


class TestEcommerceCircuitBreaker:
    """Tests for the circuit breaker pattern."""

    def test_initial_state_closed(self):
        """Test circuit breaker starts in closed state."""
        cb = EcommerceCircuitBreaker()
        assert cb.state == "closed"
        assert cb.can_proceed()

    def test_opens_after_failure_threshold(self):
        """Test circuit opens after reaching failure threshold."""
        cb = EcommerceCircuitBreaker(failure_threshold=3)

        for _ in range(3):
            cb.record_failure()

        assert cb.state == "open"
        assert not cb.can_proceed()

    def test_remains_closed_below_threshold(self):
        """Test circuit stays closed below failure threshold."""
        cb = EcommerceCircuitBreaker(failure_threshold=5)

        for _ in range(3):
            cb.record_failure()

        assert cb.state == "closed"
        assert cb.can_proceed()

    def test_success_resets_failure_count(self):
        """Test success resets failure count in closed state."""
        cb = EcommerceCircuitBreaker(failure_threshold=5)

        cb.record_failure()
        cb.record_failure()
        cb.record_success()

        # Should be able to withstand more failures
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "closed"

    def test_transitions_to_half_open_after_cooldown(self):
        """Test transition to half-open after cooldown."""
        cb = EcommerceCircuitBreaker(failure_threshold=1, cooldown_seconds=0.1)

        cb.record_failure()
        assert cb.state == "open"

        time.sleep(0.15)
        assert cb.state == "half_open"

    def test_closes_after_successful_half_open_calls(self):
        """Test circuit closes after successful calls in half-open."""
        cb = EcommerceCircuitBreaker(
            failure_threshold=1,
            cooldown_seconds=0.1,
            half_open_max_calls=2,
        )

        cb.record_failure()
        time.sleep(0.15)

        # Now in half-open state
        assert cb.can_proceed()
        cb.record_success()
        cb.record_success()

        assert cb.state == "closed"

    def test_reopens_on_half_open_failure(self):
        """Test circuit reopens on failure during half-open."""
        cb = EcommerceCircuitBreaker(
            failure_threshold=1,
            cooldown_seconds=0.1,
        )

        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == "half_open"

        cb.record_failure()
        assert cb.state == "open"

    def test_get_status(self):
        """Test circuit breaker status reporting."""
        cb = EcommerceCircuitBreaker()
        status = cb.get_status()

        assert "state" in status
        assert "failure_count" in status
        assert "failure_threshold" in status
        assert "cooldown_seconds" in status

    def test_reset(self):
        """Test circuit breaker reset."""
        cb = EcommerceCircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == "open"

        cb.reset()
        assert cb.state == "closed"
        assert cb.can_proceed()

    def test_global_circuit_breaker(self):
        """Test global circuit breaker access."""
        cb = get_ecommerce_circuit_breaker()
        assert cb is not None

        reset_ecommerce_circuit_breaker()
        assert cb.state == "closed"


# -----------------------------------------------------------------------------
# Test: Handler Creation and Configuration
# -----------------------------------------------------------------------------


class TestEcommerceHandler:
    """Tests for EcommerceHandler class."""

    def test_handler_creation(self, server_context):
        """Test creating handler instance."""
        handler = EcommerceHandler(server_context=server_context)
        assert handler is not None
        assert handler.ctx == server_context

    def test_handler_creation_with_ctx(self):
        """Test creating handler with ctx parameter."""
        handler = EcommerceHandler(ctx={"key": "value"})
        assert handler.ctx == {"key": "value"}

    def test_handler_has_routes(self, ecommerce_handler):
        """Test that handler has route definitions."""
        assert hasattr(ecommerce_handler, "ROUTES")
        assert len(ecommerce_handler.ROUTES) > 0
        assert "/api/v1/ecommerce/platforms" in ecommerce_handler.ROUTES

    def test_can_handle_matching_paths(self, ecommerce_handler):
        """Test can_handle returns True for matching paths."""
        assert ecommerce_handler.can_handle("/api/v1/ecommerce/platforms")
        assert ecommerce_handler.can_handle("/api/v1/ecommerce/orders")
        assert ecommerce_handler.can_handle("/api/v1/ecommerce/shopify/orders")

    def test_can_handle_non_matching_paths(self, ecommerce_handler):
        """Test can_handle returns False for non-matching paths."""
        assert not ecommerce_handler.can_handle("/api/v1/other/path")
        assert not ecommerce_handler.can_handle("/api/crm/contacts")

    def test_circuit_breaker_status(self, ecommerce_handler):
        """Test circuit breaker status endpoint."""
        status = ecommerce_handler.get_circuit_breaker_status()
        assert "state" in status
        assert status["state"] == "closed"


# -----------------------------------------------------------------------------
# Test: Platform Management Endpoints
# -----------------------------------------------------------------------------


class TestPlatformManagement:
    """Tests for platform connection management."""

    @pytest.mark.asyncio
    async def test_list_platforms(self, ecommerce_handler, mock_request):
        """Test listing supported platforms."""
        mock_request.path = "/api/v1/ecommerce/platforms"

        result = await ecommerce_handler._list_platforms(mock_request)

        assert result["status_code"] == 200
        assert "platforms" in result["body"]
        assert len(result["body"]["platforms"]) >= 3
        assert result["body"]["connected_count"] == 0

    @pytest.mark.asyncio
    async def test_list_platforms_with_connected(self, ecommerce_handler, mock_request):
        """Test listing platforms shows connection status."""
        _platform_credentials["shopify"] = {
            "credentials": {"shop_url": "https://test.myshopify.com", "access_token": "test"},
            "connected_at": "2024-01-01T00:00:00Z",
        }

        result = await ecommerce_handler._list_platforms(mock_request)

        assert result["status_code"] == 200
        assert result["body"]["connected_count"] == 1

        shopify = next(p for p in result["body"]["platforms"] if p["id"] == "shopify")
        assert shopify["connected"] is True

    @pytest.mark.asyncio
    async def test_connect_platform_success(self, ecommerce_handler, mock_request):
        """Test successful platform connection."""
        mock_request.method = "POST"
        mock_request.path = "/api/v1/ecommerce/connect"

        with patch.object(ecommerce_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "platform": "shopify",
                "credentials": {
                    "shop_url": "https://test.myshopify.com",
                    "access_token": "test-token",
                },
            }

            with patch.object(
                ecommerce_handler, "_get_connector", new_callable=AsyncMock
            ) as mock_conn:
                mock_conn.return_value = None

                result = await ecommerce_handler._connect_platform(mock_request)

                assert result["status_code"] == 200
                assert "shopify" in _platform_credentials

    @pytest.mark.asyncio
    async def test_connect_platform_missing_platform(self, ecommerce_handler, mock_request):
        """Test connect with missing platform."""
        with patch.object(ecommerce_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"credentials": {"key": "value"}}

            result = await ecommerce_handler._connect_platform(mock_request)

            assert result["status_code"] == 400
            assert "required" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_connect_platform_invalid_platform(self, ecommerce_handler, mock_request):
        """Test connect with invalid platform ID."""
        with patch.object(ecommerce_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "platform": "invalid-platform!@#",
                "credentials": {"key": "value"},
            }

            result = await ecommerce_handler._connect_platform(mock_request)

            assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_connect_platform_unsupported(self, ecommerce_handler, mock_request):
        """Test connect with unsupported platform."""
        with patch.object(ecommerce_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "platform": "unknown_platform",
                "credentials": {"key": "value"},
            }

            result = await ecommerce_handler._connect_platform(mock_request)

            assert result["status_code"] == 400
            assert "unsupported" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_connect_platform_missing_credentials(self, ecommerce_handler, mock_request):
        """Test connect with missing credentials."""
        with patch.object(ecommerce_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "platform": "shopify",
                "credentials": {},
            }

            result = await ecommerce_handler._connect_platform(mock_request)

            assert result["status_code"] == 400
            assert "credentials" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_connect_platform_invalid_credentials_type(self, ecommerce_handler, mock_request):
        """Test connect with invalid credentials type."""
        with patch.object(ecommerce_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "platform": "shopify",
                "credentials": "not-a-dict",
            }

            result = await ecommerce_handler._connect_platform(mock_request)

            assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_disconnect_platform_success(self, ecommerce_handler, mock_request):
        """Test successful platform disconnection."""
        _platform_credentials["shopify"] = {
            "credentials": {"shop_url": "https://test.myshopify.com", "access_token": "test"},
            "connected_at": "2024-01-01T00:00:00Z",
        }

        result = await ecommerce_handler._disconnect_platform(mock_request, "shopify")

        assert result["status_code"] == 200
        assert "shopify" not in _platform_credentials

    @pytest.mark.asyncio
    async def test_disconnect_platform_not_connected(self, ecommerce_handler, mock_request):
        """Test disconnect when platform not connected."""
        result = await ecommerce_handler._disconnect_platform(mock_request, "shopify")

        assert result["status_code"] == 404


# -----------------------------------------------------------------------------
# Test: Order Endpoints
# -----------------------------------------------------------------------------


class TestOrderEndpoints:
    """Tests for order management endpoints."""

    @pytest.mark.asyncio
    async def test_list_all_orders_no_platforms(self, ecommerce_handler, mock_request):
        """Test listing orders with no connected platforms."""
        result = await ecommerce_handler._list_all_orders(mock_request)

        assert result["status_code"] == 200
        assert result["body"]["orders"] == []
        assert result["body"]["total"] == 0

    @pytest.mark.asyncio
    async def test_list_platform_orders_not_connected(self, ecommerce_handler, mock_request):
        """Test listing orders for non-connected platform."""
        result = await ecommerce_handler._list_platform_orders(mock_request, "shopify")

        assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_get_order_not_connected(self, ecommerce_handler, mock_request):
        """Test getting order for non-connected platform."""
        result = await ecommerce_handler._get_order(mock_request, "shopify", "order123")

        assert result["status_code"] == 404


# -----------------------------------------------------------------------------
# Test: Product Endpoints
# -----------------------------------------------------------------------------


class TestProductEndpoints:
    """Tests for product management endpoints."""

    @pytest.mark.asyncio
    async def test_list_all_products_no_platforms(self, ecommerce_handler, mock_request):
        """Test listing products with no connected platforms."""
        result = await ecommerce_handler._list_all_products(mock_request)

        assert result["status_code"] == 200
        assert result["body"]["products"] == []
        assert result["body"]["total"] == 0

    @pytest.mark.asyncio
    async def test_list_platform_products_not_connected(self, ecommerce_handler, mock_request):
        """Test listing products for non-connected platform."""
        result = await ecommerce_handler._list_platform_products(mock_request, "shopify")

        assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_get_product_not_connected(self, ecommerce_handler, mock_request):
        """Test getting product for non-connected platform."""
        result = await ecommerce_handler._get_product(mock_request, "shopify", "prod123")

        assert result["status_code"] == 404


# -----------------------------------------------------------------------------
# Test: Inventory Endpoints
# -----------------------------------------------------------------------------


class TestInventoryEndpoints:
    """Tests for inventory management endpoints."""

    @pytest.mark.asyncio
    async def test_get_inventory_no_platforms(self, ecommerce_handler, mock_request):
        """Test getting inventory with no connected platforms."""
        result = await ecommerce_handler._get_inventory(mock_request)

        assert result["status_code"] == 200
        assert result["body"]["inventory"] == {}

    @pytest.mark.asyncio
    async def test_sync_inventory_missing_sku(self, ecommerce_handler, mock_request):
        """Test sync inventory with missing SKU."""
        with patch.object(ecommerce_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"quantity": 100}

            result = await ecommerce_handler._sync_inventory(mock_request)

            assert result["status_code"] == 400
            assert "sku" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_sync_inventory_invalid_sku(self, ecommerce_handler, mock_request):
        """Test sync inventory with invalid SKU."""
        with patch.object(ecommerce_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"sku": "", "quantity": 100}

            result = await ecommerce_handler._sync_inventory(mock_request)

            assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_sync_inventory_invalid_quantity(self, ecommerce_handler, mock_request):
        """Test sync inventory with invalid quantity."""
        with patch.object(ecommerce_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"sku": "SKU123", "quantity": -10}

            result = await ecommerce_handler._sync_inventory(mock_request)

            assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_sync_inventory_missing_quantity_and_source(
        self, ecommerce_handler, mock_request
    ):
        """Test sync inventory with neither quantity nor source."""
        with patch.object(ecommerce_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"sku": "SKU123"}

            result = await ecommerce_handler._sync_inventory(mock_request)

            assert result["status_code"] == 400


# -----------------------------------------------------------------------------
# Test: Shipment Endpoints
# -----------------------------------------------------------------------------


class TestShipmentEndpoints:
    """Tests for shipment creation endpoints."""

    @pytest.mark.asyncio
    async def test_create_shipment_missing_order_id(self, ecommerce_handler, mock_request):
        """Test shipment creation with missing order ID."""
        with patch.object(ecommerce_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"platform": "shipstation"}

            result = await ecommerce_handler._create_shipment(mock_request)

            assert result["status_code"] == 400
            assert "order_id" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_create_shipment_platform_not_connected(self, ecommerce_handler, mock_request):
        """Test shipment creation for non-connected platform."""
        with patch.object(ecommerce_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "platform": "shipstation",
                "order_id": "order123",
            }

            result = await ecommerce_handler._create_shipment(mock_request)

            assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_create_shipment_invalid_carrier(self, ecommerce_handler, mock_request):
        """Test shipment creation with invalid carrier."""
        with patch.object(ecommerce_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "platform": "shipstation",
                "order_id": "order123",
                "carrier": "invalid carrier!@#",
            }

            result = await ecommerce_handler._create_shipment(mock_request)

            assert result["status_code"] == 400


# -----------------------------------------------------------------------------
# Test: Metrics Endpoint
# -----------------------------------------------------------------------------


class TestMetricsEndpoint:
    """Tests for metrics endpoint."""

    @pytest.mark.asyncio
    async def test_get_metrics_no_platforms(self, ecommerce_handler, mock_request):
        """Test getting metrics with no connected platforms."""
        result = await ecommerce_handler._get_metrics(mock_request)

        assert result["status_code"] == 200
        assert result["body"]["platforms"] == {}
        assert result["body"]["totals"]["total_orders"] == 0


# -----------------------------------------------------------------------------
# Test: Circuit Breaker Integration
# -----------------------------------------------------------------------------


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration with handler."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_when_open(self, ecommerce_handler):
        """Test requests are blocked when circuit is open."""
        # Open the circuit breaker
        cb = ecommerce_handler._circuit_breaker
        for _ in range(5):
            cb.record_failure()

        assert cb.state == "open"

        result = await ecommerce_handler._with_circuit_breaker(AsyncMock())

        assert result["status_code"] == 503
        assert "unavailable" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_success(self, ecommerce_handler):
        """Test successful calls record success."""
        mock_func = AsyncMock(return_value={"status_code": 200, "body": {}})

        result = await ecommerce_handler._with_circuit_breaker(mock_func)

        assert result["status_code"] == 200
        mock_func.assert_called_once()

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_connection_failure(self, ecommerce_handler):
        """Test connection errors are recorded as failures."""
        mock_func = AsyncMock(side_effect=ConnectionError("Connection refused"))

        result = await ecommerce_handler._with_circuit_breaker(mock_func)

        assert result["status_code"] == 503
        assert ecommerce_handler._circuit_breaker.get_status()["failure_count"] == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_endpoint(self, ecommerce_handler, mock_request):
        """Test circuit breaker status endpoint."""
        mock_request.path = "/api/v1/ecommerce/circuit-breaker"

        result = await ecommerce_handler.handle_request(mock_request)

        assert result["status_code"] == 200
        assert "state" in result["body"]


# -----------------------------------------------------------------------------
# Test: Request Routing
# -----------------------------------------------------------------------------


class TestRequestRouting:
    """Tests for request routing in handle_request."""

    @pytest.mark.asyncio
    async def test_route_platforms(self, ecommerce_handler, mock_request):
        """Test routing to platforms endpoint."""
        mock_request.path = "/api/v1/ecommerce/platforms"
        mock_request.method = "GET"

        result = await ecommerce_handler.handle_request(mock_request)

        assert result["status_code"] == 200
        assert "platforms" in result["body"]

    @pytest.mark.asyncio
    async def test_route_not_found(self, ecommerce_handler, mock_request):
        """Test routing returns 404 for unknown endpoints."""
        mock_request.path = "/api/v1/ecommerce/unknown"
        mock_request.method = "GET"

        result = await ecommerce_handler.handle_request(mock_request)

        assert result["status_code"] == 404


# -----------------------------------------------------------------------------
# Test: Data Models
# -----------------------------------------------------------------------------


class TestUnifiedOrder:
    """Tests for UnifiedOrder dataclass."""

    def test_order_to_dict(self):
        """Test UnifiedOrder.to_dict() conversion."""
        from datetime import datetime, timezone

        order = UnifiedOrder(
            id="order-123",
            platform="shopify",
            order_number="1001",
            status="fulfilled",
            financial_status="paid",
            fulfillment_status="fulfilled",
            customer_email="test@example.com",
            customer_name="Test User",
            total_price=99.99,
            subtotal=89.99,
            shipping_price=10.00,
            tax=0.00,
            currency="USD",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        result = order.to_dict()

        assert result["id"] == "order-123"
        assert result["platform"] == "shopify"
        assert result["total_price"] == 99.99
        assert result["created_at"] == "2024-01-01T00:00:00+00:00"


class TestUnifiedProduct:
    """Tests for UnifiedProduct dataclass."""

    def test_product_to_dict(self):
        """Test UnifiedProduct.to_dict() conversion."""
        product = UnifiedProduct(
            id="prod-123",
            platform="shopify",
            title="Test Product",
            sku="SKU123",
            barcode="1234567890",
            price=29.99,
            compare_at_price=39.99,
            inventory_quantity=100,
            status="active",
            vendor="Test Vendor",
            product_type="widget",
            tags=["sale", "new"],
        )

        result = product.to_dict()

        assert result["id"] == "prod-123"
        assert result["title"] == "Test Product"
        assert result["price"] == 29.99
        assert result["tags"] == ["sale", "new"]


# -----------------------------------------------------------------------------
# Test: Error Handling
# -----------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_invalid_json_body(self, ecommerce_handler, mock_request):
        """Test handling of invalid JSON body."""
        with patch.object(ecommerce_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.side_effect = ValueError("Invalid JSON")

            result = await ecommerce_handler._connect_platform(mock_request)

            assert result["status_code"] == 400
            assert "invalid" in result["body"]["error"].lower()

    def test_error_response_format(self, ecommerce_handler):
        """Test error response format."""
        result = ecommerce_handler._error_response(400, "Test error message")

        assert result["status_code"] == 400
        assert result["body"]["error"] == "Test error message"

    def test_json_response_format(self, ecommerce_handler):
        """Test JSON response format."""
        result = ecommerce_handler._json_response(200, {"key": "value"})

        assert result["status_code"] == 200
        assert result["body"]["key"] == "value"
        assert result["headers"]["Content-Type"] == "application/json"
