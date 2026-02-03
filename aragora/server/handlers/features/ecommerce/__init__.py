"""E-commerce Platform API Handlers.

Stability: STABLE

Unified API for e-commerce platforms:
- Shopify (orders, products, customers, inventory)
- ShipStation (shipping, fulfillment, tracking)
- Walmart Marketplace (listings, orders, inventory)

Features:
- Circuit breaker pattern for platform API resilience
- Rate limiting (60 requests/minute)
- RBAC permission checks (ecommerce:read, ecommerce:write, ecommerce:configure)
- Comprehensive input validation with safe ID patterns
- Error isolation (platform failures handled gracefully)
"""

from .circuit_breaker import EcommerceCircuitBreaker  # noqa: F401
from .circuit_breaker import get_ecommerce_circuit_breaker  # noqa: F401
from .circuit_breaker import reset_ecommerce_circuit_breaker  # noqa: F401
from .handler import EcommerceHandler  # noqa: F401
from .models import SUPPORTED_PLATFORMS  # noqa: F401
from .models import UnifiedOrder  # noqa: F401
from .models import UnifiedProduct  # noqa: F401
from .models import _platform_credentials  # noqa: F401
from .models import _platform_connectors  # noqa: F401
from .validation import _validate_platform_id  # noqa: F401
from .validation import _validate_resource_id  # noqa: F401
from .validation import _validate_sku  # noqa: F401
from .validation import _validate_url  # noqa: F401
from .validation import _validate_quantity  # noqa: F401

__all__ = [
    "EcommerceHandler",
    "EcommerceCircuitBreaker",
    "get_ecommerce_circuit_breaker",
    "reset_ecommerce_circuit_breaker",
    "SUPPORTED_PLATFORMS",
    "UnifiedOrder",
    "UnifiedProduct",
    "_validate_platform_id",
    "_validate_resource_id",
    "_validate_sku",
    "_validate_url",
    "_validate_quantity",
    "_platform_credentials",
    "_platform_connectors",
]
