"""Tests for the e-commerce data models module.

Covers all public symbols in aragora.server.handlers.features.ecommerce.models:
- SUPPORTED_PLATFORMS constant (structure, keys, features)
- _platform_credentials / _platform_connectors module-level dicts
- UnifiedOrder dataclass (construction, defaults, to_dict, edge cases)
- UnifiedProduct dataclass (construction, defaults, to_dict, edge cases)
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any

import pytest

from aragora.server.handlers.features.ecommerce.models import (
    SUPPORTED_PLATFORMS,
    UnifiedOrder,
    UnifiedProduct,
    _platform_connectors,
    _platform_credentials,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_order(**overrides: Any) -> UnifiedOrder:
    """Create a UnifiedOrder with minimal required fields."""
    defaults: dict[str, Any] = {
        "id": "order-1",
        "platform": "shopify",
        "order_number": "1001",
        "status": "open",
        "financial_status": None,
        "fulfillment_status": None,
        "customer_email": None,
        "customer_name": None,
        "total_price": 99.99,
        "subtotal": 89.99,
        "shipping_price": 5.00,
        "tax": 5.00,
        "currency": "USD",
    }
    defaults.update(overrides)
    return UnifiedOrder(**defaults)


def _minimal_product(**overrides: Any) -> UnifiedProduct:
    """Create a UnifiedProduct with minimal required fields."""
    defaults: dict[str, Any] = {
        "id": "prod-1",
        "platform": "shopify",
        "title": "Widget",
        "sku": None,
        "barcode": None,
        "price": 29.99,
        "compare_at_price": None,
        "inventory_quantity": 100,
        "status": "active",
        "vendor": None,
        "product_type": None,
    }
    defaults.update(overrides)
    return UnifiedProduct(**defaults)


# ===================================================================
# SUPPORTED_PLATFORMS
# ===================================================================


class TestSupportedPlatforms:
    """Tests for the SUPPORTED_PLATFORMS constant."""

    def test_contains_shopify(self):
        assert "shopify" in SUPPORTED_PLATFORMS

    def test_contains_shipstation(self):
        assert "shipstation" in SUPPORTED_PLATFORMS

    def test_contains_walmart(self):
        assert "walmart" in SUPPORTED_PLATFORMS

    def test_exactly_three_platforms(self):
        assert len(SUPPORTED_PLATFORMS) == 3

    def test_shopify_has_name(self):
        assert SUPPORTED_PLATFORMS["shopify"]["name"] == "Shopify"

    def test_shipstation_has_name(self):
        assert SUPPORTED_PLATFORMS["shipstation"]["name"] == "ShipStation"

    def test_walmart_has_name(self):
        assert SUPPORTED_PLATFORMS["walmart"]["name"] == "Walmart Marketplace"

    def test_each_platform_has_description(self):
        for key, info in SUPPORTED_PLATFORMS.items():
            assert "description" in info, f"{key} missing description"
            assert isinstance(info["description"], str)
            assert len(info["description"]) > 0

    def test_each_platform_has_features_list(self):
        for key, info in SUPPORTED_PLATFORMS.items():
            assert "features" in info, f"{key} missing features"
            assert isinstance(info["features"], list)
            assert len(info["features"]) > 0

    def test_shopify_features_include_orders(self):
        assert "orders" in SUPPORTED_PLATFORMS["shopify"]["features"]

    def test_shipstation_features_include_tracking(self):
        assert "tracking" in SUPPORTED_PLATFORMS["shipstation"]["features"]

    def test_walmart_features_include_pricing(self):
        assert "pricing" in SUPPORTED_PLATFORMS["walmart"]["features"]

    def test_all_keys_are_lowercase(self):
        for key in SUPPORTED_PLATFORMS:
            assert key == key.lower(), f"Platform key {key!r} is not lowercase"


# ===================================================================
# Module-level dicts
# ===================================================================


class TestModuleLevelDicts:
    """Tests for _platform_credentials and _platform_connectors."""

    def test_platform_credentials_is_dict(self):
        assert isinstance(_platform_credentials, dict)

    def test_platform_connectors_is_dict(self):
        assert isinstance(_platform_connectors, dict)

    def test_credentials_mutable(self):
        """Module-level dict should be mutable (used for runtime storage)."""
        key = "__test_sentinel__"
        _platform_credentials[key] = {"token": "abc"}
        assert key in _platform_credentials
        del _platform_credentials[key]

    def test_connectors_mutable(self):
        key = "__test_sentinel__"
        _platform_connectors[key] = object()
        assert key in _platform_connectors
        del _platform_connectors[key]


# ===================================================================
# UnifiedOrder
# ===================================================================


class TestUnifiedOrderConstruction:
    """Tests for creating UnifiedOrder instances."""

    def test_minimal_construction(self):
        order = _minimal_order()
        assert order.id == "order-1"
        assert order.platform == "shopify"

    def test_all_required_fields(self):
        order = _minimal_order()
        assert order.order_number == "1001"
        assert order.status == "open"
        assert order.total_price == 99.99
        assert order.subtotal == 89.99
        assert order.shipping_price == 5.00
        assert order.tax == 5.00
        assert order.currency == "USD"

    def test_optional_fields_default_none(self):
        order = _minimal_order()
        assert order.financial_status is None
        assert order.fulfillment_status is None
        assert order.customer_email is None
        assert order.customer_name is None
        assert order.shipping_address is None
        assert order.created_at is None
        assert order.updated_at is None

    def test_line_items_default_empty_list(self):
        order = _minimal_order()
        assert order.line_items == []
        assert isinstance(order.line_items, list)

    def test_line_items_default_factory_isolation(self):
        """Each instance should get its own list."""
        a = _minimal_order()
        b = _minimal_order()
        a.line_items.append({"sku": "X"})
        assert b.line_items == []

    def test_with_all_optional_fields(self):
        now = datetime.now(timezone.utc)
        order = _minimal_order(
            financial_status="paid",
            fulfillment_status="fulfilled",
            customer_email="user@example.com",
            customer_name="Jane Doe",
            line_items=[{"sku": "A", "qty": 2}],
            shipping_address={"city": "Portland"},
            created_at=now,
            updated_at=now,
        )
        assert order.financial_status == "paid"
        assert order.fulfillment_status == "fulfilled"
        assert order.customer_email == "user@example.com"
        assert order.customer_name == "Jane Doe"
        assert len(order.line_items) == 1
        assert order.shipping_address["city"] == "Portland"
        assert order.created_at == now
        assert order.updated_at == now

    def test_zero_price_order(self):
        order = _minimal_order(total_price=0.0, subtotal=0.0, shipping_price=0.0, tax=0.0)
        assert order.total_price == 0.0

    def test_negative_price_allowed(self):
        """Dataclass does not enforce validation; negative values are accepted."""
        order = _minimal_order(total_price=-10.0)
        assert order.total_price == -10.0


class TestUnifiedOrderToDict:
    """Tests for UnifiedOrder.to_dict()."""

    def test_returns_dict(self):
        order = _minimal_order()
        result = order.to_dict()
        assert isinstance(result, dict)

    def test_all_keys_present(self):
        expected_keys = {
            "id",
            "platform",
            "order_number",
            "status",
            "financial_status",
            "fulfillment_status",
            "customer_email",
            "customer_name",
            "total_price",
            "subtotal",
            "shipping_price",
            "tax",
            "currency",
            "line_items",
            "shipping_address",
            "created_at",
            "updated_at",
        }
        result = _minimal_order().to_dict()
        assert set(result.keys()) == expected_keys

    def test_none_datetimes_serialize_as_none(self):
        result = _minimal_order().to_dict()
        assert result["created_at"] is None
        assert result["updated_at"] is None

    def test_created_at_serialized_as_isoformat(self):
        dt = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        order = _minimal_order(created_at=dt)
        result = order.to_dict()
        assert result["created_at"] == dt.isoformat()

    def test_updated_at_serialized_as_isoformat(self):
        dt = datetime(2025, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
        order = _minimal_order(updated_at=dt)
        result = order.to_dict()
        assert result["updated_at"] == dt.isoformat()

    def test_line_items_preserved(self):
        items = [{"sku": "A", "qty": 1}, {"sku": "B", "qty": 3}]
        order = _minimal_order(line_items=items)
        result = order.to_dict()
        assert result["line_items"] == items

    def test_shipping_address_preserved(self):
        addr = {"street": "123 Main", "city": "Portland", "zip": "97201"}
        order = _minimal_order(shipping_address=addr)
        result = order.to_dict()
        assert result["shipping_address"] == addr

    def test_numeric_fields_exact(self):
        order = _minimal_order(total_price=123.45, subtotal=100.00, shipping_price=15.00, tax=8.45)
        result = order.to_dict()
        assert result["total_price"] == 123.45
        assert result["subtotal"] == 100.00
        assert result["shipping_price"] == 15.00
        assert result["tax"] == 8.45

    def test_to_dict_does_not_share_references(self):
        """Modifying the returned dict should not affect the original order."""
        order = _minimal_order()
        d = order.to_dict()
        d["id"] = "changed"
        assert order.id == "order-1"

    def test_roundtrip_values(self):
        """Every scalar field in to_dict matches the order attribute."""
        order = _minimal_order(
            financial_status="paid",
            customer_email="a@b.com",
            customer_name="Test",
            currency="EUR",
        )
        d = order.to_dict()
        assert d["id"] == order.id
        assert d["platform"] == order.platform
        assert d["order_number"] == order.order_number
        assert d["status"] == order.status
        assert d["financial_status"] == order.financial_status
        assert d["customer_email"] == order.customer_email
        assert d["customer_name"] == order.customer_name
        assert d["currency"] == order.currency

    def test_different_platforms(self):
        for platform in ("shopify", "shipstation", "walmart"):
            order = _minimal_order(platform=platform)
            assert order.to_dict()["platform"] == platform


# ===================================================================
# UnifiedProduct
# ===================================================================


class TestUnifiedProductConstruction:
    """Tests for creating UnifiedProduct instances."""

    def test_minimal_construction(self):
        product = _minimal_product()
        assert product.id == "prod-1"
        assert product.platform == "shopify"
        assert product.title == "Widget"

    def test_all_required_fields(self):
        product = _minimal_product()
        assert product.price == 29.99
        assert product.inventory_quantity == 100
        assert product.status == "active"

    def test_optional_fields_default_none(self):
        product = _minimal_product()
        assert product.sku is None
        assert product.barcode is None
        assert product.compare_at_price is None
        assert product.vendor is None
        assert product.product_type is None
        assert product.created_at is None

    def test_tags_default_empty_list(self):
        product = _minimal_product()
        assert product.tags == []

    def test_images_default_empty_list(self):
        product = _minimal_product()
        assert product.images == []

    def test_tags_default_factory_isolation(self):
        a = _minimal_product()
        b = _minimal_product()
        a.tags.append("sale")
        assert b.tags == []

    def test_images_default_factory_isolation(self):
        a = _minimal_product()
        b = _minimal_product()
        a.images.append("http://img.png")
        assert b.images == []

    def test_with_all_optional_fields(self):
        now = datetime.now(timezone.utc)
        product = _minimal_product(
            sku="WDG-001",
            barcode="1234567890",
            compare_at_price=39.99,
            vendor="Acme Co",
            product_type="Hardware",
            tags=["sale", "new"],
            images=["https://cdn/img1.jpg"],
            created_at=now,
        )
        assert product.sku == "WDG-001"
        assert product.barcode == "1234567890"
        assert product.compare_at_price == 39.99
        assert product.vendor == "Acme Co"
        assert product.product_type == "Hardware"
        assert product.tags == ["sale", "new"]
        assert product.images == ["https://cdn/img1.jpg"]
        assert product.created_at == now

    def test_zero_inventory(self):
        product = _minimal_product(inventory_quantity=0)
        assert product.inventory_quantity == 0

    def test_negative_inventory_allowed(self):
        product = _minimal_product(inventory_quantity=-5)
        assert product.inventory_quantity == -5

    def test_zero_price(self):
        product = _minimal_product(price=0.0)
        assert product.price == 0.0


class TestUnifiedProductToDict:
    """Tests for UnifiedProduct.to_dict()."""

    def test_returns_dict(self):
        product = _minimal_product()
        result = product.to_dict()
        assert isinstance(result, dict)

    def test_all_keys_present(self):
        expected_keys = {
            "id",
            "platform",
            "title",
            "sku",
            "barcode",
            "price",
            "compare_at_price",
            "inventory_quantity",
            "status",
            "vendor",
            "product_type",
            "tags",
            "images",
            "created_at",
        }
        result = _minimal_product().to_dict()
        assert set(result.keys()) == expected_keys

    def test_none_created_at_serialized_as_none(self):
        result = _minimal_product().to_dict()
        assert result["created_at"] is None

    def test_created_at_serialized_as_isoformat(self):
        dt = datetime(2025, 3, 20, 8, 0, 0, tzinfo=timezone.utc)
        product = _minimal_product(created_at=dt)
        result = product.to_dict()
        assert result["created_at"] == dt.isoformat()

    def test_tags_preserved(self):
        tags = ["electronics", "sale", "featured"]
        product = _minimal_product(tags=tags)
        result = product.to_dict()
        assert result["tags"] == tags

    def test_images_preserved(self):
        images = ["https://cdn/a.jpg", "https://cdn/b.jpg"]
        product = _minimal_product(images=images)
        result = product.to_dict()
        assert result["images"] == images

    def test_numeric_fields_exact(self):
        product = _minimal_product(price=49.99, compare_at_price=59.99, inventory_quantity=42)
        result = product.to_dict()
        assert result["price"] == 49.99
        assert result["compare_at_price"] == 59.99
        assert result["inventory_quantity"] == 42

    def test_roundtrip_scalar_fields(self):
        product = _minimal_product(
            sku="SKU-1",
            barcode="999",
            vendor="V",
            product_type="PT",
        )
        d = product.to_dict()
        assert d["id"] == product.id
        assert d["platform"] == product.platform
        assert d["title"] == product.title
        assert d["sku"] == product.sku
        assert d["barcode"] == product.barcode
        assert d["status"] == product.status
        assert d["vendor"] == product.vendor
        assert d["product_type"] == product.product_type

    def test_to_dict_does_not_share_references(self):
        product = _minimal_product()
        d = product.to_dict()
        d["id"] = "changed"
        assert product.id == "prod-1"

    def test_different_platforms(self):
        for platform in ("shopify", "shipstation", "walmart"):
            product = _minimal_product(platform=platform)
            assert product.to_dict()["platform"] == platform

    def test_empty_tags_and_images(self):
        product = _minimal_product()
        d = product.to_dict()
        assert d["tags"] == []
        assert d["images"] == []


# ===================================================================
# Cross-cutting edge cases
# ===================================================================


class TestEdgeCases:
    """Edge cases and integration-style checks for the models module."""

    def test_order_equality_same_values(self):
        """Dataclasses support equality by default."""
        a = _minimal_order()
        b = _minimal_order()
        assert a == b

    def test_product_equality_same_values(self):
        a = _minimal_product()
        b = _minimal_product()
        assert a == b

    def test_order_inequality_different_id(self):
        a = _minimal_order(id="1")
        b = _minimal_order(id="2")
        assert a != b

    def test_product_inequality_different_id(self):
        a = _minimal_product(id="1")
        b = _minimal_product(id="2")
        assert a != b

    def test_order_with_unicode_customer_name(self):
        order = _minimal_order(customer_name="Jorg Muller")
        assert order.to_dict()["customer_name"] == "Jorg Muller"

    def test_product_with_unicode_title(self):
        product = _minimal_product(title="Cafe Latte Maker")
        assert product.to_dict()["title"] == "Cafe Latte Maker"

    def test_order_large_line_items(self):
        items = [{"sku": f"item-{i}", "qty": i} for i in range(100)]
        order = _minimal_order(line_items=items)
        d = order.to_dict()
        assert len(d["line_items"]) == 100

    def test_product_many_tags(self):
        tags = [f"tag-{i}" for i in range(50)]
        product = _minimal_product(tags=tags)
        d = product.to_dict()
        assert len(d["tags"]) == 50

    def test_naive_datetime_order(self):
        """Naive datetimes (no tz) should still serialize."""
        dt = datetime(2025, 1, 1, 0, 0, 0)
        order = _minimal_order(created_at=dt)
        assert order.to_dict()["created_at"] == "2025-01-01T00:00:00"

    def test_naive_datetime_product(self):
        dt = datetime(2025, 1, 1, 0, 0, 0)
        product = _minimal_product(created_at=dt)
        assert product.to_dict()["created_at"] == "2025-01-01T00:00:00"

    def test_order_empty_string_fields(self):
        order = _minimal_order(id="", order_number="", status="")
        d = order.to_dict()
        assert d["id"] == ""
        assert d["order_number"] == ""
        assert d["status"] == ""

    def test_product_empty_string_fields(self):
        product = _minimal_product(id="", title="", status="")
        d = product.to_dict()
        assert d["id"] == ""
        assert d["title"] == ""
        assert d["status"] == ""
