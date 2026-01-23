"""
E-commerce Platform API Handlers.

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
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from aragora.server.handlers.secure import SecureHandler
from aragora.server.handlers.utils.decorators import has_permission

logger = logging.getLogger(__name__)


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
    """Handler for e-commerce platform API endpoints."""

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
    ]

    def _check_permission(self, request: Any, permission: str) -> dict[str, Any] | None:
        """Check if user has the required permission."""
        user = self.get_current_user(request)
        if user:
            user_role = user.role if hasattr(user, "role") else None
            if not has_permission(user_role, permission):
                return self._error_response(403, f"Permission denied: {permission} required")
        return None

    async def handle_request(self, request: Any) -> Any:
        """Route request to appropriate handler."""
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

        # Route to handlers
        if path.endswith("/platforms") and method == "GET":
            return await self._list_platforms(request)

        elif path.endswith("/connect") and method == "POST":
            if err := self._check_permission(request, "ecommerce:configure"):
                return err
            return await self._connect_platform(request)

        elif platform and path.endswith(f"/{platform}") and method == "DELETE":
            if err := self._check_permission(request, "ecommerce:configure"):
                return err
            return await self._disconnect_platform(request, platform)

        # Orders
        elif path.endswith("/orders") and not platform and method == "GET":
            if err := self._check_permission(request, "ecommerce:read"):
                return err
            return await self._list_all_orders(request)

        elif platform and "orders" in path:
            if method == "GET" and not resource_id:
                if err := self._check_permission(request, "ecommerce:read"):
                    return err
                return await self._list_platform_orders(request, platform)
            elif method == "GET" and resource_id:
                if err := self._check_permission(request, "ecommerce:read"):
                    return err
                return await self._get_order(request, platform, resource_id)

        # Products
        elif path.endswith("/products") and not platform and method == "GET":
            if err := self._check_permission(request, "ecommerce:read"):
                return err
            return await self._list_all_products(request)

        elif platform and "products" in path:
            if method == "GET" and not resource_id:
                if err := self._check_permission(request, "ecommerce:read"):
                    return err
                return await self._list_platform_products(request, platform)
            elif method == "GET" and resource_id:
                if err := self._check_permission(request, "ecommerce:read"):
                    return err
                return await self._get_product(request, platform, resource_id)

        # Inventory
        elif path.endswith("/inventory") and method == "GET":
            if err := self._check_permission(request, "ecommerce:read"):
                return err
            return await self._get_inventory(request)

        elif path.endswith("/sync-inventory") and method == "POST":
            if err := self._check_permission(request, "ecommerce:write"):
                return err
            return await self._sync_inventory(request)

        # Fulfillment
        elif path.endswith("/fulfillment") and method == "GET":
            if err := self._check_permission(request, "ecommerce:read"):
                return err
            return await self._get_fulfillment_status(request)

        elif path.endswith("/ship") and method == "POST":
            if err := self._check_permission(request, "ecommerce:write"):
                return err
            return await self._create_shipment(request)

        # Metrics
        elif path.endswith("/metrics") and method == "GET":
            if err := self._check_permission(request, "ecommerce:read"):
                return err
            return await self._get_metrics(request)

        return self._error_response(404, "Endpoint not found")

    async def _list_platforms(self, request: Any) -> dict[str, Any]:
        """List all supported e-commerce platforms and connection status."""
        platforms = []
        for platform_id, meta in SUPPORTED_PLATFORMS.items():
            connected = platform_id in _platform_credentials
            platforms.append({
                "id": platform_id,
                "name": meta["name"],
                "description": meta["description"],
                "features": meta["features"],
                "connected": connected,
                "connected_at": _platform_credentials.get(platform_id, {}).get("connected_at"),
            })

        return self._json_response(200, {
            "platforms": platforms,
            "connected_count": sum(1 for p in platforms if p["connected"]),
        })

    async def _connect_platform(self, request: Any) -> dict[str, Any]:
        """Connect an e-commerce platform with credentials."""
        try:
            body = await self._get_json_body(request)
        except Exception as e:
            return self._error_response(400, f"Invalid JSON body: {e}")

        platform = body.get("platform")
        if not platform:
            return self._error_response(400, "Platform is required")

        if platform not in SUPPORTED_PLATFORMS:
            return self._error_response(400, f"Unsupported platform: {platform}")

        credentials = body.get("credentials", {})
        if not credentials:
            return self._error_response(400, "Credentials are required")

        required_fields = self._get_required_credentials(platform)
        missing = [f for f in required_fields if f not in credentials]
        if missing:
            return self._error_response(400, f"Missing required credentials: {', '.join(missing)}")

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

        return self._json_response(200, {
            "message": f"Successfully connected to {SUPPORTED_PLATFORMS[platform]['name']}",
            "platform": platform,
            "connected_at": _platform_credentials[platform]["connected_at"],
        })

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

        return self._json_response(200, {
            "message": f"Disconnected from {SUPPORTED_PLATFORMS[platform]['name']}",
            "platform": platform,
        })

    # Order operations

    async def _list_all_orders(self, request: Any) -> dict[str, Any]:
        """List orders from all connected platforms."""
        status = request.query.get("status")
        limit = int(request.query.get("limit", 100))
        days = int(request.query.get("days", 30))

        all_orders: list[dict[str, Any]] = []

        tasks = []
        for platform in _platform_credentials.keys():
            tasks.append(self._fetch_platform_orders(platform, status, limit, days))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for platform, result in zip(_platform_credentials.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching orders from {platform}: {result}")
                continue
            all_orders.extend(result)

        # Sort by created_at descending
        all_orders.sort(key=lambda o: o.get("created_at") or "", reverse=True)

        return self._json_response(200, {
            "orders": all_orders[:limit],
            "total": len(all_orders),
            "platforms_queried": list(_platform_credentials.keys()),
        })

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
        limit = int(request.query.get("limit", 100))
        days = int(request.query.get("days", 30))

        orders = await self._fetch_platform_orders(platform, status, limit, days)

        return self._json_response(200, {
            "orders": orders,
            "total": len(orders),
            "platform": platform,
        })

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
        limit = int(request.query.get("limit", 100))
        status = request.query.get("status")

        all_products: list[dict[str, Any]] = []

        tasks = []
        for platform in _platform_credentials.keys():
            tasks.append(self._fetch_platform_products(platform, status, limit))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for platform, result in zip(_platform_credentials.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching products from {platform}: {result}")
                continue
            all_products.extend(result)

        return self._json_response(200, {
            "products": all_products[:limit],
            "total": len(all_products),
        })

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

        limit = int(request.query.get("limit", 100))
        status = request.query.get("status")

        products = await self._fetch_platform_products(platform, status, limit)

        return self._json_response(200, {
            "products": products,
            "total": len(products),
            "platform": platform,
        })

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
                            "inventory_item_id": str(l.inventory_item_id),
                            "location_id": str(l.location_id),
                            "available": l.available,
                        }
                        for l in levels
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

        return self._json_response(200, {
            "inventory": inventory,
            "platforms": list(inventory.keys()),
        })

    async def _sync_inventory(self, request: Any) -> dict[str, Any]:
        """Sync inventory across platforms."""
        try:
            body = await self._get_json_body(request)
        except Exception as e:
            return self._error_response(400, f"Invalid JSON body: {e}")

        sku = body.get("sku")
        quantity = body.get("quantity")
        source_platform = body.get("source_platform")
        target_platforms = body.get("target_platforms", list(_platform_credentials.keys()))

        if not sku:
            return self._error_response(400, "SKU is required")

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
        results = {}
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

        return self._json_response(200, {
            "sku": sku,
            "quantity": quantity,
            "results": results,
        })

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
                            fulfillments.append({
                                "platform": p,
                                "order_id": order_id,
                                "fulfillment_id": str(f.id),
                                "status": f.status,
                                "tracking_number": f.tracking_number,
                                "tracking_url": f.tracking_url,
                                "carrier": f.tracking_company,
                            })

                elif p == "shipstation":
                    shipments = await connector.get_shipments(order_id=order_id)
                    for s in shipments:
                        fulfillments.append({
                            "platform": p,
                            "order_id": str(s.order_id),
                            "shipment_id": str(s.shipment_id),
                            "status": s.shipment_status,
                            "tracking_number": s.tracking_number,
                            "carrier": s.carrier_code,
                        })

            except Exception as e:
                logger.error(f"Error fetching {p} fulfillment: {e}")

        return self._json_response(200, {
            "fulfillments": fulfillments,
            "total": len(fulfillments),
        })

    async def _create_shipment(self, request: Any) -> dict[str, Any]:
        """Create a shipment for an order."""
        try:
            body = await self._get_json_body(request)
        except Exception as e:
            return self._error_response(400, f"Invalid JSON body: {e}")

        platform = body.get("platform", "shipstation")
        order_id = body.get("order_id")
        carrier = body.get("carrier")
        service = body.get("service")

        if not order_id:
            return self._error_response(400, "order_id is required")

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
                return self._json_response(201, {
                    "shipment_id": str(label.shipment_id) if hasattr(label, "shipment_id") else None,
                    "tracking_number": label.tracking_number if hasattr(label, "tracking_number") else None,
                    "label_url": label.label_data if hasattr(label, "label_data") else None,
                    "platform": platform,
                })

            elif platform == "shopify":
                fulfillment = await connector.create_fulfillment(
                    order_id=int(order_id),
                    tracking_number=body.get("tracking_number"),
                    tracking_company=carrier,
                )
                return self._json_response(201, {
                    "fulfillment_id": str(fulfillment.id),
                    "status": fulfillment.status,
                    "platform": platform,
                })

        except Exception as e:
            return self._error_response(500, f"Failed to create shipment: {e}")

        return self._error_response(400, "Unsupported platform")

    # Metrics

    async def _get_metrics(self, request: Any) -> dict[str, Any]:
        """Get e-commerce metrics overview."""
        days = int(request.query.get("days", 30))

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
                metrics["totals"]["orders_pending_fulfillment"] += platform_metrics.get("pending_fulfillment", 0)

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
        pending = sum(1 for o in orders if o.get("fulfillment_status") in [None, "unfulfilled", "partial"])

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
                connector = ShopifyConnector(ShopifyCredentials(**creds))

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
            "customer_name": f"{order.customer.first_name} {order.customer.last_name}".strip() if hasattr(order, "customer") and order.customer else None,
            "total_price": float(order.total_price),
            "subtotal": float(order.subtotal_price),
            "shipping_price": float(order.total_shipping_price_set.shop_money.amount) if hasattr(order, "total_shipping_price_set") else 0,
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
            "customer_name": order.ship_to.name if hasattr(order, "ship_to") and order.ship_to else None,
            "total_price": float(order.order_total) if hasattr(order, "order_total") else 0,
            "subtotal": float(order.order_total) if hasattr(order, "order_total") else 0,
            "shipping_price": float(order.shipping_amount) if hasattr(order, "shipping_amount") else 0,
            "tax": float(order.tax_amount) if hasattr(order, "tax_amount") else 0,
            "currency": "USD",
            "created_at": order.order_date.isoformat() if hasattr(order, "order_date") and order.order_date else None,
        }

    def _normalize_walmart_order(self, order: Any) -> dict[str, Any]:
        """Normalize Walmart order."""
        return {
            "id": order.purchase_order_id,
            "platform": "walmart",
            "order_number": order.customer_order_id,
            "status": order.order_status.value if hasattr(order.order_status, "value") else order.order_status,
            "financial_status": None,
            "fulfillment_status": order.order_status.value if hasattr(order.order_status, "value") else order.order_status,
            "customer_email": order.customer_email if hasattr(order, "customer_email") else None,
            "customer_name": order.shipping_info.postal_address.name if hasattr(order, "shipping_info") and order.shipping_info else None,
            "total_price": float(order.order_total) if hasattr(order, "order_total") else 0,
            "subtotal": float(order.order_total) if hasattr(order, "order_total") else 0,
            "shipping_price": 0,
            "tax": 0,
            "currency": "USD",
            "created_at": order.order_date.isoformat() if hasattr(order, "order_date") and order.order_date else None,
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
            "compare_at_price": float(variant.compare_at_price) if variant and variant.compare_at_price else None,
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
            "status": item.lifecycle_status.value if hasattr(item.lifecycle_status, "value") else item.lifecycle_status,
            "vendor": item.brand if hasattr(item, "brand") else None,
            "product_type": item.product_type if hasattr(item, "product_type") else None,
            "tags": [],
            "images": [],
        }

    async def _get_json_body(self, request: Any) -> dict[str, Any]:
        """Parse JSON body from request."""
        body = await request.json()
        return body if isinstance(body, dict) else {}

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


__all__ = ["EcommerceHandler"]
