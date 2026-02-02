"""
E-commerce Platform API Handlers.

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
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, cast

from aragora.server.handlers.secure import SecureHandler, ForbiddenError, UnauthorizedError
from aragora.server.handlers.utils.responses import error_response
from aragora.server.handlers.utils import parse_json_body
from aragora.server.handlers.utils.rate_limit import rate_limit
from aragora.server.validation.query_params import safe_query_int

logger = logging.getLogger(__name__)


# =============================================================================
# Input Validation Constants
# =============================================================================

# Platform ID validation: alphanumeric and underscores only
SAFE_PLATFORM_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,49}$")

# Order/Product/SKU ID validation: alphanumeric, hyphens, underscores
SAFE_RESOURCE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,127}$")

# Max lengths for input validation
MAX_SHOP_URL_LENGTH = 512
MAX_CREDENTIAL_VALUE_LENGTH = 1024
MAX_SKU_LENGTH = 128
MAX_ORDER_ID_LENGTH = 128
MAX_CARRIER_LENGTH = 64
MAX_SERVICE_LENGTH = 64
MAX_TRACKING_NUMBER_LENGTH = 128


def _validate_platform_id(platform: str) -> tuple[bool, str | None]:
    """Validate a platform ID.

    Args:
        platform: Platform identifier to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not platform:
        return False, "Platform is required"
    if len(platform) > 50:
        return False, "Platform name too long (max 50 characters)"
    if not SAFE_PLATFORM_PATTERN.match(platform):
        return False, "Invalid platform format (alphanumeric and underscores only)"
    return True, None


def _validate_resource_id(resource_id: str, resource_type: str = "ID") -> tuple[bool, str | None]:
    """Validate a resource ID (order, product, etc.).

    Args:
        resource_id: Resource identifier to validate
        resource_type: Type name for error messages

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not resource_id:
        return False, f"{resource_type} is required"
    if len(resource_id) > 128:
        return False, f"{resource_type} too long (max 128 characters)"
    if not SAFE_RESOURCE_ID_PATTERN.match(resource_id):
        return False, f"Invalid {resource_type.lower()} format"
    return True, None


def _validate_sku(sku: str) -> tuple[bool, str | None]:
    """Validate a SKU.

    Args:
        sku: SKU to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not sku:
        return False, "SKU is required"
    if len(sku) > MAX_SKU_LENGTH:
        return False, f"SKU too long (max {MAX_SKU_LENGTH} characters)"
    # SKU can contain alphanumeric, hyphens, underscores, dots
    if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_\-\.]{0,127}$", sku):
        return False, "Invalid SKU format"
    return True, None


def _validate_url(url: str, field_name: str = "URL") -> tuple[bool, str | None]:
    """Validate a URL field.

    Args:
        url: URL to validate
        field_name: Field name for error messages

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url:
        return False, f"{field_name} is required"
    if len(url) > MAX_SHOP_URL_LENGTH:
        return False, f"{field_name} too long (max {MAX_SHOP_URL_LENGTH} characters)"
    # Basic URL validation
    if not url.startswith(("http://", "https://")):
        return False, f"Invalid {field_name} format (must start with http:// or https://)"
    return True, None


def _validate_quantity(quantity: Any) -> tuple[bool, str | None, int | None]:
    """Validate a quantity value.

    Args:
        quantity: Quantity value to validate

    Returns:
        Tuple of (is_valid, error_message, parsed_value)
    """
    if quantity is None:
        return False, "Quantity is required", None
    try:
        qty = int(quantity)
        if qty < 0:
            return False, "Quantity cannot be negative", None
        if qty > 1_000_000_000:
            return False, "Quantity too large", None
        return True, None, qty
    except (ValueError, TypeError):
        return False, "Invalid quantity format", None


# =============================================================================
# Circuit Breaker for E-commerce Platform Access
# =============================================================================


class EcommerceCircuitBreaker:
    """Circuit breaker for e-commerce platform API access.

    Prevents cascading failures when external platform APIs are unavailable.
    Uses a simple state machine: CLOSED -> OPEN -> HALF_OPEN -> CLOSED.
    """

    # State constants
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        half_open_max_calls: int = 2,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            cooldown_seconds: Time to wait before allowing test calls
            half_open_max_calls: Number of test calls in half-open state
        """
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.half_open_max_calls = half_open_max_calls

        self._state = self.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """Get current circuit state."""
        with self._lock:
            return self._check_state()

    def _check_state(self) -> str:
        """Check and potentially transition state (must hold lock)."""
        if self._state == self.OPEN:
            # Check if cooldown has elapsed
            if (
                self._last_failure_time is not None
                and time.time() - self._last_failure_time >= self.cooldown_seconds
            ):
                self._state = self.HALF_OPEN
                self._half_open_calls = 0
                logger.info("Ecommerce circuit breaker transitioning to HALF_OPEN")
        return self._state

    def can_proceed(self) -> bool:
        """Check if a call can proceed.

        Returns:
            True if call is allowed, False if circuit is open
        """
        with self._lock:
            state = self._check_state()
            if state == self.CLOSED:
                return True
            elif state == self.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            else:  # OPEN
                return False

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == self.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_calls:
                    self._state = self.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("Ecommerce circuit breaker closed after successful recovery")
            elif self._state == self.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == self.HALF_OPEN:
                # Any failure in half-open state reopens the circuit
                self._state = self.OPEN
                self._success_count = 0
                logger.warning("Ecommerce circuit breaker reopened after failure in HALF_OPEN")
            elif self._state == self.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = self.OPEN
                    logger.warning(
                        f"Ecommerce circuit breaker opened after {self._failure_count} failures"
                    )

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status."""
        with self._lock:
            return {
                "state": self._check_state(),
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "failure_threshold": self.failure_threshold,
                "cooldown_seconds": self.cooldown_seconds,
                "last_failure_time": self._last_failure_time,
            }

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._state = self.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0


# Global circuit breaker instance for e-commerce platform access
_circuit_breaker = EcommerceCircuitBreaker()
_circuit_breaker_lock = threading.Lock()


def get_ecommerce_circuit_breaker() -> EcommerceCircuitBreaker:
    """Get the global circuit breaker for e-commerce platform access."""
    return _circuit_breaker


def reset_ecommerce_circuit_breaker() -> None:
    """Reset the global circuit breaker (for testing)."""
    with _circuit_breaker_lock:
        _circuit_breaker.reset()


# Platform credentials storage
_platform_credentials: dict[str, dict[str, Any]] = {}
_platform_connectors: dict[str, Any] = {}


SUPPORTED_PLATFORMS = {
    "shopify": {
        "name": "Shopify",
        "description": "E-commerce platform for online stores",
        "features": ["orders", "products", "customers", "inventory", "fulfillment"],
    },
    "shipstation": {
        "name": "ShipStation",
        "description": "Shipping and fulfillment platform",
        "features": ["shipments", "orders", "carriers", "labels", "tracking"],
    },
    "walmart": {
        "name": "Walmart Marketplace",
        "description": "Walmart seller marketplace",
        "features": ["orders", "items", "inventory", "pricing", "reports"],
    },
}


@dataclass
class UnifiedOrder:
    """Unified order representation across platforms."""

    id: str
    platform: str
    order_number: str
    status: str
    financial_status: str | None
    fulfillment_status: str | None
    customer_email: str | None
    customer_name: str | None
    total_price: float
    subtotal: float
    shipping_price: float
    tax: float
    currency: str
    line_items: list[dict[str, Any]] = field(default_factory=list)
    shipping_address: dict[str, Any] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "platform": self.platform,
            "order_number": self.order_number,
            "status": self.status,
            "financial_status": self.financial_status,
            "fulfillment_status": self.fulfillment_status,
            "customer_email": self.customer_email,
            "customer_name": self.customer_name,
            "total_price": self.total_price,
            "subtotal": self.subtotal,
            "shipping_price": self.shipping_price,
            "tax": self.tax,
            "currency": self.currency,
            "line_items": self.line_items,
            "shipping_address": self.shipping_address,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass
class UnifiedProduct:
    """Unified product representation."""

    id: str
    platform: str
    title: str
    sku: str | None
    barcode: str | None
    price: float
    compare_at_price: float | None
    inventory_quantity: int
    status: str
    vendor: str | None
    product_type: str | None
    tags: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    created_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "platform": self.platform,
            "title": self.title,
            "sku": self.sku,
            "barcode": self.barcode,
            "price": self.price,
            "compare_at_price": self.compare_at_price,
            "inventory_quantity": self.inventory_quantity,
            "status": self.status,
            "vendor": self.vendor,
            "product_type": self.product_type,
            "tags": self.tags,
            "images": self.images,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


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
            return error_response(str(e), 403)

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
            logger.error(f"Ecommerce platform connection error: {e}")
            return self._error_response(503, "E-commerce platform unavailable")
        except Exception as e:
            # Don't count application-level errors as circuit breaker failures
            logger.error(f"Ecommerce handler error: {e}")
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
        except Exception as e:
            logger.warning("Ecommerce connect_platform: invalid JSON body: %s", e)
            return self._error_response(400, f"Invalid JSON body: {e}")

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
        except Exception as e:
            logger.warning(f"Could not initialize {platform} connector: {e}")

        logger.info(f"Connected e-commerce platform: {platform}")

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

        logger.info(f"Disconnected e-commerce platform: {platform}")

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
        limit = safe_query_int(request.query, "limit", default=100, max_val=1000)
        days = safe_query_int(request.query, "days", default=30, max_val=365)

        all_orders: list[dict[str, Any]] = []

        tasks = []
        for platform in _platform_credentials.keys():
            tasks.append(self._fetch_platform_orders(platform, status, limit, days))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for platform, result in zip(_platform_credentials.keys(), results):
            if isinstance(result, BaseException):
                logger.error(f"Error fetching orders from {platform}: {result}")
                continue
            all_orders.extend(result)

        # Sort by created_at descending
        all_orders.sort(key=lambda o: o.get("created_at") or "", reverse=True)

        return self._json_response(
            200,
            {
                "orders": all_orders[:limit],
                "total": len(all_orders),
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

        except Exception as e:
            logger.error(f"Error fetching {platform} orders: {e}")

        return []

    async def _list_platform_orders(self, request: Any, platform: str) -> dict[str, Any]:
        """List orders from a specific platform."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        status = request.query.get("status")
        limit = safe_query_int(request.query, "limit", default=100, max_val=1000)
        days = safe_query_int(request.query, "days", default=30, max_val=365)

        orders = await self._fetch_platform_orders(platform, status, limit, days)

        return self._json_response(
            200,
            {
                "orders": orders,
                "total": len(orders),
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

        except Exception as e:
            return self._error_response(404, f"Order not found: {e}")

        return self._error_response(400, "Unsupported platform")

    # Product operations

    async def _list_all_products(self, request: Any) -> dict[str, Any]:
        """List products from all connected platforms."""
        limit = safe_query_int(request.query, "limit", default=100, max_val=1000)
        status = request.query.get("status")

        all_products: list[dict[str, Any]] = []

        tasks = []
        for platform in _platform_credentials.keys():
            tasks.append(self._fetch_platform_products(platform, status, limit))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for platform, result in zip(_platform_credentials.keys(), results):
            if isinstance(result, BaseException):
                logger.error(f"Error fetching products from {platform}: {result}")
                continue
            all_products.extend(result)

        return self._json_response(
            200,
            {
                "products": all_products[:limit],
                "total": len(all_products),
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

        except Exception as e:
            logger.error(f"Error fetching {platform} products: {e}")

        return []

    async def _list_platform_products(self, request: Any, platform: str) -> dict[str, Any]:
        """List products from a specific platform."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        limit = safe_query_int(request.query, "limit", default=100, max_val=1000)
        status = request.query.get("status")

        products = await self._fetch_platform_products(platform, status, limit)

        return self._json_response(
            200,
            {
                "products": products,
                "total": len(products),
                "platform": platform,
            },
        )

    async def _get_product(self, request: Any, platform: str, product_id: str) -> dict[str, Any]:
        """Get a specific product."""
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

        except Exception as e:
            return self._error_response(404, f"Product not found: {e}")

        return self._error_response(400, "Unsupported platform")

    # Inventory operations

    async def _get_inventory(self, request: Any) -> dict[str, Any]:
        """Get inventory levels across platforms."""
        sku = request.query.get("sku")

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

            except Exception as e:
                logger.error(f"Error fetching {platform} inventory: {e}")
                inventory[platform] = [{"error": str(e)}]

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
        except Exception as e:
            logger.warning("Ecommerce sync_inventory: invalid JSON body: %s", e)
            return self._error_response(400, f"Invalid JSON body: {e}")

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
                except Exception as e:
                    logger.error(f"Error fetching source inventory: {e}")

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

            except Exception as e:
                results[platform] = {"error": str(e)}

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

            except Exception as e:
                logger.error(f"Error fetching {p} fulfillment: {e}")

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
        except Exception as e:
            logger.warning("Ecommerce create_shipment: invalid JSON body: %s", e)
            return self._error_response(400, f"Invalid JSON body: {e}")

        platform = body.get("platform", "shipstation")
        order_id = body.get("order_id")
        carrier = body.get("carrier")
        service = body.get("service")
        tracking_number = body.get("tracking_number")

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

        except Exception as e:
            return self._error_response(500, f"Failed to create shipment: {e}")

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

            except Exception as e:
                logger.error(f"Error fetching {platform} metrics: {e}")
                metrics["platforms"][platform] = {"error": str(e)}

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

        except Exception as e:
            logger.error(f"Failed to create {platform} connector: {e}")
            return None

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
            "total_price": float(order.total_price),
            "subtotal": float(order.subtotal_price),
            "shipping_price": (
                float(order.total_shipping_price_set.shop_money.amount)
                if hasattr(order, "total_shipping_price_set")
                else 0
            ),
            "tax": float(order.total_tax),
            "currency": order.currency,
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
            "total_price": float(order.order_total) if hasattr(order, "order_total") else 0,
            "subtotal": float(order.order_total) if hasattr(order, "order_total") else 0,
            "shipping_price": (
                float(order.shipping_amount) if hasattr(order, "shipping_amount") else 0
            ),
            "tax": float(order.tax_amount) if hasattr(order, "tax_amount") else 0,
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
            "total_price": float(order.order_total) if hasattr(order, "order_total") else 0,
            "subtotal": float(order.order_total) if hasattr(order, "order_total") else 0,
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
            "price": float(variant.price) if variant else 0,
            "compare_at_price": (
                float(variant.compare_at_price) if variant and variant.compare_at_price else None
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
            "price": float(item.price.amount) if hasattr(item, "price") else 0,
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


__all__ = [
    "EcommerceHandler",
    "EcommerceCircuitBreaker",
    "get_ecommerce_circuit_breaker",
    "reset_ecommerce_circuit_breaker",
    "SUPPORTED_PLATFORMS",
    "UnifiedOrder",
    "UnifiedProduct",
]
