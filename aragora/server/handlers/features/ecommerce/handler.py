"""E-commerce Platform API Handler.

Stability: STABLE

Unified API for e-commerce platforms:
- Shopify (orders, products, customers, inventory)
- ShipStation (shipping, fulfillment, tracking)
- Walmart Marketplace (listings, orders, inventory)

Usage:
    GET    /api/v1/ecommerce/platforms            - List connected platforms
    POST   /api/v1/ecommerce/connect              - Connect a platform
    DELETE /api/v1/ecommerce/{platform}           - Disconnect platform

    GET    /api/v1/ecommerce/orders               - List orders (cross-platform)
    GET    /api/v1/ecommerce/{platform}/orders    - Platform orders
    GET    /api/v1/ecommerce/{platform}/orders/{id} - Get order details

    GET    /api/v1/ecommerce/products             - List products
    GET    /api/v1/ecommerce/inventory            - Get inventory levels
    POST   /api/v1/ecommerce/sync-inventory       - Sync inventory across platforms

    GET    /api/v1/ecommerce/fulfillment          - Get fulfillment status
    POST   /api/v1/ecommerce/ship                 - Create shipment

Features:
- Circuit breaker pattern for platform API resilience
- Rate limiting (60 requests/minute)
- RBAC permission checks (ecommerce:read, ecommerce:write, ecommerce:configure)
- Comprehensive input validation with safe ID patterns
- Error isolation (platform failures handled gracefully)
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, cast

from aragora.server.handlers.secure import SecureHandler, ForbiddenError, UnauthorizedError
from aragora.server.handlers.utils.responses import error_response
from aragora.server.handlers.utils import parse_json_body
from aragora.server.handlers.utils.rate_limit import rate_limit
from aragora.server.validation.query_params import safe_query_int

from .circuit_breaker import get_ecommerce_circuit_breaker
from .models import SUPPORTED_PLATFORMS, _platform_credentials, _platform_connectors
from .validation import (
    ALLOWED_CURRENCY_CODES,
    MAX_CARRIER_LENGTH,
    MAX_CREDENTIAL_VALUE_LENGTH,
    MAX_SERVICE_LENGTH,
    MAX_TRACKING_NUMBER_LENGTH,
    _validate_platform_id,
    _validate_resource_id,
    _validate_sku,
    _validate_url,
    _validate_quantity,
    _validate_financial_amount,
    _validate_currency_code,
    _validate_product_id,
    _validate_pagination,
)

logger = logging.getLogger(__name__)


class EcommerceHandler(SecureHandler):
    """Handler for e-commerce platform API endpoints with RBAC protection.

    Stability: STABLE

    Features:
    - Circuit breaker pattern for platform API resilience
    - Rate limiting (60 requests/minute)
    - RBAC permission checks (ecommerce:read, ecommerce:write, ecommerce:configure)
    - Comprehensive input validation with safe ID patterns
    """

    # Input validation constants
    MAX_LIMIT = 1000
    MAX_DAYS = 365
    DEFAULT_LIMIT = 100
    DEFAULT_DAYS = 30

    def __init__(self, ctx: dict | None = None, server_context: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = server_context or ctx or {}
        self._circuit_breaker = get_ecommerce_circuit_breaker()

    RESOURCE_TYPE = "ecommerce"

    ROUTES = [
        "/api/v1/ecommerce/platforms",
        "/api/v1/ecommerce/connect",
        "/api/v1/ecommerce/{platform}",
        "/api/v1/ecommerce/orders",
        "/api/v1/ecommerce/{platform}/orders",
        "/api/v1/ecommerce/{platform}/orders/{order_id}",
        "/api/v1/ecommerce/products",
        "/api/v1/ecommerce/{platform}/products",
        "/api/v1/ecommerce/{platform}/products/{product_id}",
        "/api/v1/ecommerce/inventory",
        "/api/v1/ecommerce/sync-inventory",
        "/api/v1/ecommerce/fulfillment",
        "/api/v1/ecommerce/ship",
        "/api/v1/ecommerce/metrics",
        "/api/v1/ecommerce/circuit-breaker",
    ]

    def get_circuit_breaker_status(self) -> dict[str, Any]:
        """Get the current status of the circuit breaker."""
        return self._circuit_breaker.get_status()

    async def _check_permission(self, request: Any, permission: str) -> Any:
        """Check if user has the required permission using RBAC system."""
        try:
            auth_context = await self.get_auth_context(request, require_auth=True)
            self.check_permission(auth_context, permission)
            return None
        except UnauthorizedError:
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            logger.warning("Handler error: %s", e)
            return error_response("Permission denied", 403)

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can handle the given path."""
        return path.startswith("/api/v1/ecommerce/")

    @rate_limit(requests_per_minute=60)
    async def handle_request(self, request: Any) -> dict[str, Any]:
        """Route request to appropriate handler.

        All endpoints are rate limited to 60 requests per minute.
        Platform API operations are protected by circuit breaker.
        """
        method = request.method
        path = str(request.path)

        # Parse path components
        platform = None
        resource_id = None

        parts = path.replace("/api/v1/ecommerce/", "").split("/")
        if parts and parts[0] in SUPPORTED_PLATFORMS:
            platform = parts[0]
            if len(parts) > 2:
                resource_id = parts[2]

        # Validate platform ID if present
        if platform:
            is_valid, err_msg = _validate_platform_id(platform)
            if not is_valid:
                return self._error_response(400, err_msg or "Invalid platform")

        # Validate resource ID if present
        if resource_id:
            is_valid, err_msg = _validate_resource_id(resource_id, "Resource ID")
            if not is_valid:
                return self._error_response(400, err_msg or "Invalid resource ID")

        # Circuit breaker status endpoint (no auth required)
        if path.endswith("/circuit-breaker") and method == "GET":
            return self._json_response(200, self.get_circuit_breaker_status())

        # Route to handlers
        if path.endswith("/platforms") and method == "GET":
            return await self._list_platforms(request)

        elif path.endswith("/connect") and method == "POST":
            if err := await self._check_permission(request, "ecommerce:configure"):
                return err
            return await self._connect_platform(request)

        elif platform and path.endswith(f"/{platform}") and method == "DELETE":
            if err := await self._check_permission(request, "ecommerce:configure"):
                return err
            return await self._disconnect_platform(request, platform)

        # Orders - these require circuit breaker protection
        elif path.endswith("/orders") and not platform and method == "GET":
            if err := await self._check_permission(request, "ecommerce:read"):
                return err
            return await self._with_circuit_breaker(self._list_all_orders, request)

        elif platform and "orders" in path:
            if method == "GET" and not resource_id:
                if err := await self._check_permission(request, "ecommerce:read"):
                    return err
                return await self._with_circuit_breaker(
                    self._list_platform_orders, request, platform
                )
            elif method == "GET" and resource_id:
                if err := await self._check_permission(request, "ecommerce:read"):
                    return err
                return await self._with_circuit_breaker(
                    self._get_order, request, platform, resource_id
                )

        # Products - circuit breaker protected
        elif path.endswith("/products") and not platform and method == "GET":
            if err := await self._check_permission(request, "ecommerce:read"):
                return err
            return await self._with_circuit_breaker(self._list_all_products, request)

        elif platform and "products" in path:
            if method == "GET" and not resource_id:
                if err := await self._check_permission(request, "ecommerce:read"):
                    return err
                return await self._with_circuit_breaker(
                    self._list_platform_products, request, platform
                )
            elif method == "GET" and resource_id:
                if err := await self._check_permission(request, "ecommerce:read"):
                    return err
                return await self._with_circuit_breaker(
                    self._get_product, request, platform, resource_id
                )

        # Inventory - circuit breaker protected
        elif path.endswith("/inventory") and method == "GET":
            if err := await self._check_permission(request, "ecommerce:read"):
                return err
            return await self._with_circuit_breaker(self._get_inventory, request)

        elif path.endswith("/sync-inventory") and method == "POST":
            if err := await self._check_permission(request, "ecommerce:write"):
                return err
            return await self._with_circuit_breaker(self._sync_inventory, request)

        # Fulfillment - circuit breaker protected
        elif path.endswith("/fulfillment") and method == "GET":
            if err := await self._check_permission(request, "ecommerce:read"):
                return err
            return await self._with_circuit_breaker(self._get_fulfillment_status, request)

        elif path.endswith("/ship") and method == "POST":
            if err := await self._check_permission(request, "ecommerce:write"):
                return err
            return await self._with_circuit_breaker(self._create_shipment, request)

        # Metrics - circuit breaker protected
        elif path.endswith("/metrics") and method == "GET":
            if err := await self._check_permission(request, "ecommerce:read"):
                return err
            return await self._with_circuit_breaker(self._get_metrics, request)

        return self._error_response(404, "Endpoint not found")

    async def _with_circuit_breaker(self, func: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Execute a function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Function result or error response if circuit is open
        """
        if not self._circuit_breaker.can_proceed():
            logger.warning("Ecommerce circuit breaker is open, rejecting request")
            return self._error_response(
                503, "E-commerce service temporarily unavailable. Please try again later."
            )

        try:
            result = await func(*args, **kwargs)
            self._circuit_breaker.record_success()
            return result
        except (ConnectionError, TimeoutError, OSError) as e:
            self._circuit_breaker.record_failure()
            logger.error("Ecommerce platform connection error: %s", e)
            return self._error_response(503, "E-commerce platform unavailable")
        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as e:
            # Don't count application-level errors as circuit breaker failures
            logger.error("Ecommerce handler error: %s", e)
            raise

    async def _list_platforms(self, request: Any) -> dict[str, Any]:
        """List all supported e-commerce platforms and connection status."""
        platforms = []
        for platform_id, meta in SUPPORTED_PLATFORMS.items():
            connected = platform_id in _platform_credentials
            platforms.append(
                {
                    "id": platform_id,
                    "name": meta["name"],
                    "description": meta["description"],
                    "features": meta["features"],
                    "connected": connected,
                    "connected_at": _platform_credentials.get(platform_id, {}).get("connected_at"),
                }
            )

        return self._json_response(
            200,
            {
                "platforms": platforms,
                "connected_count": sum(1 for p in platforms if p["connected"]),
            },
        )

    async def _connect_platform(self, request: Any) -> dict[str, Any]:
        """Connect an e-commerce platform with credentials."""
        try:
            body = await self._get_json_body(request)
        except (ValueError, TypeError, KeyError) as e:
            logger.warning("Ecommerce connect_platform: invalid JSON body: %s", e)
            return self._error_response(400, "Invalid request body")

        platform = body.get("platform")
        if not platform:
            return self._error_response(400, "Platform is required")

        # Validate platform ID format
        is_valid, err_msg = _validate_platform_id(platform)
        if not is_valid:
            return self._error_response(400, err_msg or "Invalid platform")

        if platform not in SUPPORTED_PLATFORMS:
            return self._error_response(400, f"Unsupported platform: {platform}")

        credentials = body.get("credentials", {})
        if not credentials:
            return self._error_response(400, "Credentials are required")

        if not isinstance(credentials, dict):
            return self._error_response(400, "Credentials must be an object")

        required_fields = self._get_required_credentials(platform)
        missing = [f for f in required_fields if f not in credentials]
        if missing:
            return self._error_response(400, f"Missing required credentials: {', '.join(missing)}")

        # Validate credential values
        for field_name, value in credentials.items():
            if not isinstance(value, str):
                return self._error_response(400, f"Credential '{field_name}' must be a string")
            if len(value) > MAX_CREDENTIAL_VALUE_LENGTH:
                return self._error_response(
                    400, f"Credential '{field_name}' exceeds maximum length"
                )
            if not value.strip():
                return self._error_response(400, f"Credential '{field_name}' cannot be empty")

        # Platform-specific validation
        if platform == "shopify":
            shop_url = credentials.get("shop_url", "")
            is_valid, err_msg = _validate_url(shop_url, "Shop URL")
            if not is_valid:
                return self._error_response(400, err_msg or "Invalid shop URL")

        _platform_credentials[platform] = {
            "credentials": credentials,
            "connected_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            connector = await self._get_connector(platform)
            if connector:
                _platform_connectors[platform] = connector
        except (ImportError, ConnectionError, TimeoutError, OSError, TypeError, ValueError) as e:
            logger.warning("Could not initialize %s connector: %s", platform, e)

        logger.info("Connected e-commerce platform: %s", platform)

        return self._json_response(
            200,
            {
                "message": f"Successfully connected to {SUPPORTED_PLATFORMS[platform]['name']}",
                "platform": platform,
                "connected_at": _platform_credentials[platform]["connected_at"],
            },
        )

    async def _disconnect_platform(self, request: Any, platform: str) -> dict[str, Any]:
        """Disconnect an e-commerce platform."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        if platform in _platform_connectors:
            connector = _platform_connectors[platform]
            if hasattr(connector, "close"):
                await connector.close()
            del _platform_connectors[platform]

        del _platform_credentials[platform]

        logger.info("Disconnected e-commerce platform: %s", platform)

        return self._json_response(
            200,
            {
                "message": f"Disconnected from {SUPPORTED_PLATFORMS[platform]['name']}",
                "platform": platform,
            },
        )

    # Order operations

    async def _list_all_orders(self, request: Any) -> dict[str, Any]:
        """List orders from all connected platforms."""
        status = request.query.get("status")
        days = safe_query_int(request.query, "days", default=30, max_val=365)

        # Validate pagination parameters
        is_valid, err_msg, limit, offset = _validate_pagination(
            request.query.get("limit"), request.query.get("offset")
        )
        if not is_valid:
            return self._error_response(400, err_msg or "Invalid pagination parameters")

        all_orders: list[dict[str, Any]] = []

        tasks = []
        for platform in _platform_credentials.keys():
            tasks.append(self._fetch_platform_orders(platform, status, limit + offset, days))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for platform, result in zip(_platform_credentials.keys(), results):
            if isinstance(result, BaseException):
                logger.error("Error fetching orders from %s: %s", platform, result)
                continue
            all_orders.extend(result)

        # Sort by created_at descending
        all_orders.sort(key=lambda o: o.get("created_at") or "", reverse=True)

        # Apply pagination
        paginated_orders = all_orders[offset : offset + limit]

        return self._json_response(
            200,
            {
                "orders": paginated_orders,
                "total": len(all_orders),
                "limit": limit,
                "offset": offset,
                "platforms_queried": list(_platform_credentials.keys()),
            },
        )

    async def _fetch_platform_orders(
        self,
        platform: str,
        status: str | None = None,
        limit: int = 100,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """Fetch orders from a specific platform."""
        connector = await self._get_connector(platform)
        if not connector:
            return []

        try:
            if platform == "shopify":
                orders = await connector.get_orders(
                    status=status,
                    limit=limit,
                    created_at_min=datetime.now(timezone.utc) - timedelta(days=days),
                )
                return [self._normalize_shopify_order(o) for o in orders]

            elif platform == "shipstation":
                orders = await connector.get_orders(
                    order_status=status,
                    page_size=limit,
                )
                return [self._normalize_shipstation_order(o) for o in orders]

            elif platform == "walmart":
                orders = await connector.get_orders(
                    status=status,
                    limit=limit,
                )
                return [self._normalize_walmart_order(o) for o in orders]

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error("Connection error fetching %s orders: %s", platform, e)
        except ValueError as e:
            logger.error("Data error fetching %s orders: %s", platform, e)
        except (TypeError, AttributeError, KeyError, RuntimeError) as e:
            logger.error("Unexpected error fetching %s orders: %s: %s", platform, type(e).__name__, e)

        return []

    async def _list_platform_orders(self, request: Any, platform: str) -> dict[str, Any]:
        """List orders from a specific platform."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        status = request.query.get("status")
        days = safe_query_int(request.query, "days", default=30, max_val=365)

        # Validate pagination parameters
        is_valid, err_msg, limit, offset = _validate_pagination(
            request.query.get("limit"), request.query.get("offset")
        )
        if not is_valid:
            return self._error_response(400, err_msg or "Invalid pagination parameters")

        orders = await self._fetch_platform_orders(platform, status, limit + offset, days)

        # Apply pagination
        paginated_orders = orders[offset : offset + limit]

        return self._json_response(
            200,
            {
                "orders": paginated_orders,
                "total": len(orders),
                "limit": limit,
                "offset": offset,
                "platform": platform,
            },
        )

    async def _get_order(self, request: Any, platform: str, order_id: str) -> dict[str, Any]:
        """Get a specific order with details."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        connector = await self._get_connector(platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {platform} connector")

        try:
            if platform == "shopify":
                order = await connector.get_order(int(order_id))
                return self._json_response(200, self._normalize_shopify_order(order))

            elif platform == "shipstation":
                order = await connector.get_order(int(order_id))
                return self._json_response(200, self._normalize_shipstation_order(order))

            elif platform == "walmart":
                order = await connector.get_order(order_id)
                return self._json_response(200, self._normalize_walmart_order(order))

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error("Connection error fetching order %s from %s: %s", order_id, platform, e)
            return self._error_response(503, f"Platform {platform} temporarily unavailable")
        except ValueError as e:
            return self._error_response(400, "Invalid order request")
        except (TypeError, AttributeError, KeyError, RuntimeError) as e:
            logger.error(
                "Error fetching order %s from %s: %s: %s", order_id, platform, type(e).__name__, e
            )
            return self._error_response(404, "Order not found")

        return self._error_response(400, "Unsupported platform")

    # Product operations

    async def _list_all_products(self, request: Any) -> dict[str, Any]:
        """List products from all connected platforms."""
        status = request.query.get("status")

        # Validate pagination parameters
        is_valid, err_msg, limit, offset = _validate_pagination(
            request.query.get("limit"), request.query.get("offset")
        )
        if not is_valid:
            return self._error_response(400, err_msg or "Invalid pagination parameters")

        all_products: list[dict[str, Any]] = []

        tasks = []
        for platform in _platform_credentials.keys():
            tasks.append(self._fetch_platform_products(platform, status, limit + offset))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for platform, result in zip(_platform_credentials.keys(), results):
            if isinstance(result, BaseException):
                logger.error("Error fetching products from %s: %s", platform, result)
                continue
            all_products.extend(result)

        # Apply pagination
        paginated_products = all_products[offset : offset + limit]

        return self._json_response(
            200,
            {
                "products": paginated_products,
                "total": len(all_products),
                "limit": limit,
                "offset": offset,
            },
        )

    async def _fetch_platform_products(
        self,
        platform: str,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch products from a specific platform."""
        connector = await self._get_connector(platform)
        if not connector:
            return []

        try:
            if platform == "shopify":
                products = await connector.get_products(status=status, limit=limit)
                return [self._normalize_shopify_product(p) for p in products]

            elif platform == "walmart":
                items = await connector.get_items(limit=limit)
                return [self._normalize_walmart_item(i) for i in items]

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error("Connection error fetching %s products: %s", platform, e)
        except ValueError as e:
            logger.error("Data error fetching %s products: %s", platform, e)
        except (TypeError, AttributeError, KeyError, RuntimeError) as e:
            logger.error("Unexpected error fetching %s products: %s: %s", platform, type(e).__name__, e)

        return []

    async def _list_platform_products(self, request: Any, platform: str) -> dict[str, Any]:
        """List products from a specific platform."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        status = request.query.get("status")

        # Validate pagination parameters
        is_valid, err_msg, limit, offset = _validate_pagination(
            request.query.get("limit"), request.query.get("offset")
        )
        if not is_valid:
            return self._error_response(400, err_msg or "Invalid pagination parameters")

        products = await self._fetch_platform_products(platform, status, limit + offset)

        # Apply pagination
        paginated_products = products[offset : offset + limit]

        return self._json_response(
            200,
            {
                "products": paginated_products,
                "total": len(products),
                "limit": limit,
                "offset": offset,
                "platform": platform,
            },
        )

    async def _get_product(self, request: Any, platform: str, product_id: str) -> dict[str, Any]:
        """Get a specific product."""
        # Validate product ID with strict pattern matching
        is_valid, err_msg = _validate_product_id(product_id)
        if not is_valid:
            return self._error_response(400, err_msg or "Invalid product ID")

        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        connector = await self._get_connector(platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {platform} connector")

        try:
            if platform == "shopify":
                product = await connector.get_product(int(product_id))
                return self._json_response(200, self._normalize_shopify_product(product))

            elif platform == "walmart":
                item = await connector.get_item(product_id)
                return self._json_response(200, self._normalize_walmart_item(item))

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error("Connection error fetching product %s from %s: %s", product_id, platform, e)
            return self._error_response(503, f"Platform {platform} temporarily unavailable")
        except ValueError as e:
            return self._error_response(400, "Invalid product request")
        except (TypeError, AttributeError, KeyError, RuntimeError) as e:
            logger.error(
                "Error fetching product %s from %s: %s: %s", product_id, platform, type(e).__name__, e
            )
            return self._error_response(404, "Product not found")

        return self._error_response(400, "Unsupported platform")

    # Inventory operations

    async def _get_inventory(self, request: Any) -> dict[str, Any]:
        """Get inventory levels across platforms."""
        sku = request.query.get("sku")

        # Validate SKU if provided
        if sku:
            is_valid, err_msg = _validate_sku(sku)
            if not is_valid:
                return self._error_response(400, err_msg or "Invalid SKU")

        inventory: dict[str, list[dict[str, Any]]] = {}

        for platform in _platform_credentials.keys():
            connector = await self._get_connector(platform)
            if not connector:
                continue

            try:
                if platform == "shopify":
                    levels = await connector.get_inventory_levels()
                    inventory["shopify"] = [
                        {
                            "inventory_item_id": str(level.inventory_item_id),
                            "location_id": str(level.location_id),
                            "available": level.available,
                        }
                        for level in levels
                    ]

                elif platform == "walmart":
                    inv = await connector.get_inventory(sku=sku)
                    inventory["walmart"] = [
                        {
                            "sku": i.sku,
                            "quantity": i.quantity,
                        }
                        for i in inv
                    ]

            except (ConnectionError, TimeoutError, OSError, ValueError, AttributeError) as e:
                logger.error("Error fetching %s inventory: %s", platform, e)
                inventory[platform] = [{"error": "Failed to fetch inventory"}]

        return self._json_response(
            200,
            {
                "inventory": inventory,
                "platforms": list(inventory.keys()),
            },
        )

    async def _sync_inventory(self, request: Any) -> dict[str, Any]:
        """Sync inventory across platforms."""
        try:
            body = await self._get_json_body(request)
        except (ValueError, TypeError, KeyError) as e:
            logger.warning("Ecommerce sync_inventory: invalid JSON body: %s", e)
            return self._error_response(400, "Invalid request body")

        sku = body.get("sku")
        quantity = body.get("quantity")
        source_platform = body.get("source_platform")
        target_platforms = body.get("target_platforms", list(_platform_credentials.keys()))

        # Validate SKU
        if not sku:
            return self._error_response(400, "SKU is required")
        is_valid, err_msg = _validate_sku(sku)
        if not is_valid:
            return self._error_response(400, err_msg or "Invalid SKU")

        # Validate quantity if provided
        if quantity is not None:
            is_valid, err_msg, parsed_qty = _validate_quantity(quantity)
            if not is_valid:
                return self._error_response(400, err_msg or "Invalid quantity")
            quantity = parsed_qty

        # Validate source platform if provided
        if source_platform:
            is_valid, err_msg = _validate_platform_id(source_platform)
            if not is_valid:
                return self._error_response(400, err_msg or "Invalid source platform")
            if source_platform not in SUPPORTED_PLATFORMS:
                return self._error_response(400, f"Unsupported source platform: {source_platform}")

        # Validate target platforms
        if not isinstance(target_platforms, list):
            return self._error_response(400, "target_platforms must be a list")
        if len(target_platforms) > 50:
            return self._error_response(400, "Too many target platforms (max 50)")
        for tp in target_platforms:
            is_valid, err_msg = _validate_platform_id(tp)
            if not is_valid:
                return self._error_response(400, f"Invalid target platform: {err_msg}")

        if quantity is None and not source_platform:
            return self._error_response(400, "Either quantity or source_platform is required")

        # If source_platform provided, fetch quantity from there
        if source_platform and quantity is None:
            connector = await self._get_connector(source_platform)
            if connector:
                try:
                    if source_platform == "shopify":
                        # Get inventory from Shopify
                        pass
                    elif source_platform == "walmart":
                        inv = await connector.get_inventory(sku=sku)
                        if inv:
                            quantity = inv[0].quantity
                except (ConnectionError, TimeoutError, OSError) as e:
                    logger.error(
                        "Connection error fetching source inventory from %s for SKU %s: %s", source_platform, sku, e
                    )
                except (ValueError, TypeError, AttributeError, KeyError) as e:
                    logger.error(
                        "Error fetching source inventory from %s for SKU %s: %s: %s", source_platform, sku, type(e).__name__, e
                    )

        if quantity is None:
            return self._error_response(400, "Could not determine quantity")

        # Update inventory on target platforms
        results: dict[str, dict[str, Any]] = {}
        for platform in target_platforms:
            if platform == source_platform:
                continue

            if platform not in _platform_credentials:
                results[platform] = {"error": "Not connected"}
                continue

            connector = await self._get_connector(platform)
            if not connector:
                results[platform] = {"error": "Connector not available"}
                continue

            try:
                if platform == "walmart":
                    await connector.update_inventory(sku, quantity)
                    results[platform] = {"success": True, "quantity": quantity}

            except (ConnectionError, TimeoutError, OSError) as e:
                logger.error(
                    "Connection error syncing inventory to %s (SKU: %s, quantity: %s): %s", platform, sku, quantity, e
                )
                results[platform] = {"error": "Platform temporarily unavailable"}
            except (ValueError, TypeError, AttributeError, KeyError, RuntimeError) as e:
                logger.error(
                    "Error syncing inventory to %s (SKU: %s, quantity: %s): %s: %s", platform, sku, quantity, type(e).__name__, e
                )
                results[platform] = {"error": "Inventory sync failed"}

        return self._json_response(
            200,
            {
                "sku": sku,
                "quantity": quantity,
                "results": results,
            },
        )

    # Fulfillment operations

    async def _get_fulfillment_status(self, request: Any) -> dict[str, Any]:
        """Get fulfillment status for orders."""
        order_id = request.query.get("order_id")
        platform = request.query.get("platform")

        # Validate order_id if provided
        if order_id:
            is_valid, err_msg = _validate_resource_id(str(order_id), "Order ID")
            if not is_valid:
                return self._error_response(400, err_msg or "Invalid order ID")

        # Validate platform filter if provided
        if platform:
            is_valid, err_msg = _validate_platform_id(platform)
            if not is_valid:
                return self._error_response(400, err_msg or "Invalid platform")

        fulfillments: list[dict[str, Any]] = []

        platforms_to_query = [platform] if platform else list(_platform_credentials.keys())

        for p in platforms_to_query:
            if p not in _platform_credentials:
                continue

            connector = await self._get_connector(p)
            if not connector:
                continue

            try:
                if p == "shopify" and order_id:
                    order = await connector.get_order(int(order_id))
                    if hasattr(order, "fulfillments"):
                        for f in order.fulfillments:
                            fulfillments.append(
                                {
                                    "platform": p,
                                    "order_id": order_id,
                                    "fulfillment_id": str(f.id),
                                    "status": f.status,
                                    "tracking_number": f.tracking_number,
                                    "tracking_url": f.tracking_url,
                                    "carrier": f.tracking_company,
                                }
                            )

                elif p == "shipstation":
                    shipments = await connector.get_shipments(order_id=order_id)
                    for s in shipments:
                        fulfillments.append(
                            {
                                "platform": p,
                                "order_id": str(s.order_id),
                                "shipment_id": str(s.shipment_id),
                                "status": s.shipment_status,
                                "tracking_number": s.tracking_number,
                                "carrier": s.carrier_code,
                            }
                        )

            except (ConnectionError, TimeoutError, OSError, ValueError, AttributeError) as e:
                logger.error("Error fetching %s fulfillment: %s", p, e)

        return self._json_response(
            200,
            {
                "fulfillments": fulfillments,
                "total": len(fulfillments),
            },
        )

    async def _create_shipment(self, request: Any) -> dict[str, Any]:
        """Create a shipment for an order."""
        try:
            body = await self._get_json_body(request)
        except (ValueError, TypeError, KeyError) as e:
            logger.warning("Ecommerce create_shipment: invalid JSON body: %s", e)
            return self._error_response(400, "Invalid request body")

        platform = body.get("platform", "shipstation")
        order_id = body.get("order_id")
        carrier = body.get("carrier")
        service = body.get("service")
        tracking_number = body.get("tracking_number")

        # Validate declared_value if provided (financial amount)
        declared_value = body.get("declared_value")
        if declared_value is not None:
            is_valid, err_msg, _ = _validate_financial_amount(
                declared_value, "Declared value", allow_zero=False
            )
            if not is_valid:
                return self._error_response(400, err_msg or "Invalid declared value")

        # Validate currency if provided
        currency = body.get("currency")
        if currency is not None:
            is_valid, err_msg = _validate_currency_code(currency)
            if not is_valid:
                return self._error_response(400, err_msg or "Invalid currency code")

        # Validate platform
        is_valid, err_msg = _validate_platform_id(platform)
        if not is_valid:
            return self._error_response(400, err_msg or "Invalid platform")

        # Validate order_id
        if not order_id:
            return self._error_response(400, "order_id is required")
        is_valid, err_msg = _validate_resource_id(str(order_id), "Order ID")
        if not is_valid:
            return self._error_response(400, err_msg or "Invalid order ID")

        # Validate carrier if provided
        if carrier:
            if not isinstance(carrier, str) or len(carrier) > MAX_CARRIER_LENGTH:
                return self._error_response(400, "Invalid carrier format")
            if not re.match(r"^[a-zA-Z0-9_\-]+$", carrier):
                return self._error_response(400, "Carrier contains invalid characters")

        # Validate service if provided
        if service:
            if not isinstance(service, str) or len(service) > MAX_SERVICE_LENGTH:
                return self._error_response(400, "Invalid service format")
            if not re.match(r"^[a-zA-Z0-9_\-]+$", service):
                return self._error_response(400, "Service contains invalid characters")

        # Validate tracking number if provided
        if tracking_number:
            if (
                not isinstance(tracking_number, str)
                or len(tracking_number) > MAX_TRACKING_NUMBER_LENGTH
            ):
                return self._error_response(400, "Invalid tracking number format")

        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        connector = await self._get_connector(platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {platform} connector")

        try:
            if platform == "shipstation":
                label = await connector.create_label(
                    order_id=int(order_id),
                    carrier_code=carrier,
                    service_code=service,
                )
                return self._json_response(
                    201,
                    {
                        "shipment_id": (
                            str(label.shipment_id) if hasattr(label, "shipment_id") else None
                        ),
                        "tracking_number": (
                            label.tracking_number if hasattr(label, "tracking_number") else None
                        ),
                        "label_url": label.label_data if hasattr(label, "label_data") else None,
                        "platform": platform,
                    },
                )

            elif platform == "shopify":
                fulfillment = await connector.create_fulfillment(
                    order_id=int(order_id),
                    tracking_number=body.get("tracking_number"),
                    tracking_company=carrier,
                )
                return self._json_response(
                    201,
                    {
                        "fulfillment_id": str(fulfillment.id),
                        "status": fulfillment.status,
                        "platform": platform,
                    },
                )

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(
                "Connection error creating shipment on %s (order: %s): %s", platform, order_id, e
            )
            return self._error_response(503, f"Platform {platform} temporarily unavailable")
        except (ValueError, TypeError, AttributeError, KeyError, RuntimeError) as e:
            logger.error(
                "Error creating shipment on %s (order: %s): %s: %s", platform, order_id, type(e).__name__, e
            )
            logger.warning("Handler error: %s", e)
            return self._error_response(500, "Failed to create shipment")

        return self._error_response(400, "Unsupported platform")

    # Metrics

    async def _get_metrics(self, request: Any) -> dict[str, Any]:
        """Get e-commerce metrics overview."""
        days = safe_query_int(request.query, "days", default=30, max_val=365)

        metrics: dict[str, Any] = {
            "period_days": days,
            "platforms": {},
            "totals": {
                "total_orders": 0,
                "total_revenue": 0,
                "average_order_value": 0,
                "orders_pending_fulfillment": 0,
            },
        }

        for platform in _platform_credentials.keys():
            try:
                platform_metrics = await self._fetch_platform_metrics(platform, days)
                metrics["platforms"][platform] = platform_metrics

                metrics["totals"]["total_orders"] += platform_metrics.get("total_orders", 0)
                metrics["totals"]["total_revenue"] += platform_metrics.get("total_revenue", 0)
                metrics["totals"]["orders_pending_fulfillment"] += platform_metrics.get(
                    "pending_fulfillment", 0
                )

            except (ConnectionError, TimeoutError, OSError, ValueError, AttributeError) as e:
                logger.error("Error fetching %s metrics: %s", platform, e)
                metrics["platforms"][platform] = {"error": "Failed to fetch platform metrics"}

        if metrics["totals"]["total_orders"] > 0:
            metrics["totals"]["average_order_value"] = round(
                metrics["totals"]["total_revenue"] / metrics["totals"]["total_orders"], 2
            )

        return self._json_response(200, metrics)

    async def _fetch_platform_metrics(self, platform: str, days: int) -> dict[str, Any]:
        """Fetch metrics from a specific platform."""
        orders = await self._fetch_platform_orders(platform, limit=1000, days=days)

        total_revenue = sum(o.get("total_price", 0) for o in orders)
        pending = sum(
            1 for o in orders if o.get("fulfillment_status") in [None, "unfulfilled", "partial"]
        )

        return {
            "total_orders": len(orders),
            "total_revenue": round(total_revenue, 2),
            "average_order_value": round(total_revenue / len(orders), 2) if orders else 0,
            "pending_fulfillment": pending,
            "fulfilled": len(orders) - pending,
        }

    # Helper methods

    def _get_required_credentials(self, platform: str) -> list[str]:
        """Get required credential fields for a platform."""
        requirements = {
            "shopify": ["shop_url", "access_token"],
            "shipstation": ["api_key", "api_secret"],
            "walmart": ["client_id", "client_secret"],
        }
        return requirements.get(platform, [])

    async def _get_connector(self, platform: str) -> Any | None:
        """Get or create a connector for a platform."""
        if platform in _platform_connectors:
            return _platform_connectors[platform]

        if platform not in _platform_credentials:
            return None

        creds = _platform_credentials[platform]["credentials"]

        try:
            if platform == "shopify":
                from aragora.connectors.ecommerce.shopify import (
                    ShopifyConnector,
                    ShopifyCredentials,
                )

                connector: Any = cast(type, ShopifyConnector)(ShopifyCredentials(**creds))

            elif platform == "shipstation":
                from aragora.connectors.ecommerce.shipstation import (
                    ShipStationConnector,
                    ShipStationCredentials,
                )

                connector = ShipStationConnector(ShipStationCredentials(**creds))

            elif platform == "walmart":
                from aragora.connectors.marketplace.walmart import (
                    WalmartConnector,
                    WalmartCredentials,
                )

                connector = WalmartConnector(WalmartCredentials(**creds))

            else:
                return None

            _platform_connectors[platform] = connector
            return connector

        except ImportError as e:
            logger.error("Missing dependency for %s connector: %s", platform, e)
            return None
        except (ValueError, TypeError) as e:
            logger.error("Invalid credentials for %s connector: %s", platform, e)
            return None
        except (ConnectionError, TimeoutError, OSError, AttributeError, RuntimeError) as e:
            logger.error("Failed to create %s connector: %s: %s", platform, type(e).__name__, e)
            return None

    def _sanitize_financial_amount(self, value: Any) -> float:
        """Sanitize a financial amount from platform data.

        Clamps to non-negative and caps at reasonable bounds.
        Returns 0 if the value cannot be parsed.
        """
        try:
            amount = float(value)
        except (ValueError, TypeError):
            return 0.0
        if amount < 0:
            return 0.0
        if amount > 99_999_999.99:
            return 99_999_999.99
        return round(amount, 2)

    def _sanitize_currency_code(self, currency: Any) -> str:
        """Sanitize a currency code from platform data.

        Returns the uppercase code if it is in the allowed whitelist,
        otherwise returns 'USD' as a safe default.
        """
        if not currency or not isinstance(currency, str):
            return "USD"
        code = currency.strip().upper()[:3]
        if code in ALLOWED_CURRENCY_CODES:
            return code
        return "USD"

    def _normalize_shopify_order(self, order: Any) -> dict[str, Any]:
        """Normalize Shopify order."""
        return {
            "id": str(order.id),
            "platform": "shopify",
            "order_number": str(order.order_number),
            "status": order.fulfillment_status or "unfulfilled",
            "financial_status": order.financial_status,
            "fulfillment_status": order.fulfillment_status,
            "customer_email": order.email,
            "customer_name": (
                f"{order.customer.first_name} {order.customer.last_name}".strip()
                if hasattr(order, "customer") and order.customer
                else None
            ),
            "total_price": self._sanitize_financial_amount(order.total_price),
            "subtotal": self._sanitize_financial_amount(order.subtotal_price),
            "shipping_price": (
                self._sanitize_financial_amount(order.total_shipping_price_set.shop_money.amount)
                if hasattr(order, "total_shipping_price_set")
                else 0
            ),
            "tax": self._sanitize_financial_amount(order.total_tax),
            "currency": self._sanitize_currency_code(order.currency),
            "line_items": [
                {
                    "title": item.title,
                    "quantity": item.quantity,
                    "price": float(item.price),
                    "sku": item.sku,
                }
                for item in (order.line_items if hasattr(order, "line_items") else [])
            ],
            "created_at": order.created_at.isoformat() if order.created_at else None,
            "updated_at": order.updated_at.isoformat() if order.updated_at else None,
        }

    def _normalize_shipstation_order(self, order: Any) -> dict[str, Any]:
        """Normalize ShipStation order."""
        return {
            "id": str(order.order_id),
            "platform": "shipstation",
            "order_number": order.order_number,
            "status": order.order_status,
            "financial_status": order.payment_status if hasattr(order, "payment_status") else None,
            "fulfillment_status": order.order_status,
            "customer_email": order.customer_email if hasattr(order, "customer_email") else None,
            "customer_name": (
                order.ship_to.name if hasattr(order, "ship_to") and order.ship_to else None
            ),
            "total_price": (
                self._sanitize_financial_amount(order.order_total)
                if hasattr(order, "order_total")
                else 0
            ),
            "subtotal": (
                self._sanitize_financial_amount(order.order_total)
                if hasattr(order, "order_total")
                else 0
            ),
            "shipping_price": (
                self._sanitize_financial_amount(order.shipping_amount)
                if hasattr(order, "shipping_amount")
                else 0
            ),
            "tax": (
                self._sanitize_financial_amount(order.tax_amount)
                if hasattr(order, "tax_amount")
                else 0
            ),
            "currency": "USD",
            "created_at": (
                order.order_date.isoformat()
                if hasattr(order, "order_date") and order.order_date
                else None
            ),
        }

    def _normalize_walmart_order(self, order: Any) -> dict[str, Any]:
        """Normalize Walmart order."""
        return {
            "id": order.purchase_order_id,
            "platform": "walmart",
            "order_number": order.customer_order_id,
            "status": (
                order.order_status.value
                if hasattr(order.order_status, "value")
                else order.order_status
            ),
            "financial_status": None,
            "fulfillment_status": (
                order.order_status.value
                if hasattr(order.order_status, "value")
                else order.order_status
            ),
            "customer_email": order.customer_email if hasattr(order, "customer_email") else None,
            "customer_name": (
                order.shipping_info.postal_address.name
                if hasattr(order, "shipping_info") and order.shipping_info
                else None
            ),
            "total_price": (
                self._sanitize_financial_amount(order.order_total)
                if hasattr(order, "order_total")
                else 0
            ),
            "subtotal": (
                self._sanitize_financial_amount(order.order_total)
                if hasattr(order, "order_total")
                else 0
            ),
            "shipping_price": 0,
            "tax": 0,
            "currency": "USD",
            "created_at": (
                order.order_date.isoformat()
                if hasattr(order, "order_date") and order.order_date
                else None
            ),
        }

    def _normalize_shopify_product(self, product: Any) -> dict[str, Any]:
        """Normalize Shopify product."""
        variant = product.variants[0] if hasattr(product, "variants") and product.variants else None
        return {
            "id": str(product.id),
            "platform": "shopify",
            "title": product.title,
            "sku": variant.sku if variant else None,
            "barcode": variant.barcode if variant else None,
            "price": self._sanitize_financial_amount(variant.price) if variant else 0,
            "compare_at_price": (
                self._sanitize_financial_amount(variant.compare_at_price)
                if variant and variant.compare_at_price
                else None
            ),
            "inventory_quantity": variant.inventory_quantity if variant else 0,
            "status": product.status,
            "vendor": product.vendor,
            "product_type": product.product_type,
            "tags": product.tags.split(", ") if product.tags else [],
            "images": [img.src for img in product.images] if hasattr(product, "images") else [],
            "created_at": product.created_at.isoformat() if product.created_at else None,
        }

    def _normalize_walmart_item(self, item: Any) -> dict[str, Any]:
        """Normalize Walmart item."""
        return {
            "id": item.sku,
            "platform": "walmart",
            "title": item.product_name,
            "sku": item.sku,
            "barcode": item.gtin if hasattr(item, "gtin") else None,
            "price": self._sanitize_financial_amount(item.price.amount)
            if hasattr(item, "price")
            else 0,
            "compare_at_price": None,
            "inventory_quantity": item.quantity if hasattr(item, "quantity") else 0,
            "status": (
                item.lifecycle_status.value
                if hasattr(item.lifecycle_status, "value")
                else item.lifecycle_status
            ),
            "vendor": item.brand if hasattr(item, "brand") else None,
            "product_type": item.product_type if hasattr(item, "product_type") else None,
            "tags": [],
            "images": [],
        }

    async def _get_json_body(self, request: Any) -> dict[str, Any]:
        """Parse JSON body from request."""
        body, _err = await parse_json_body(request, context="ecommerce._get_json_body")
        return body if body is not None else {}

    def _json_response(self, status: int, data: Any) -> dict[str, Any]:
        """Create a JSON response."""
        return {
            "status_code": status,
            "headers": {"Content-Type": "application/json"},
            "body": data,
        }

    def _error_response(self, status: int, message: str) -> dict[str, Any]:
        """Create an error response."""
        return self._json_response(status, {"error": message})
