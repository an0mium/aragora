"""Input validation utilities for e-commerce handlers.

Provides constants and functions for validating platform IDs, resource IDs,
SKUs, URLs, quantities, financial amounts, currency codes, product IDs,
and pagination parameters used across e-commerce endpoints.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any


# =============================================================================
# Input Validation Constants
# =============================================================================

# Platform ID validation: alphanumeric and underscores only
SAFE_PLATFORM_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,49}$")

# Order/Product/SKU ID validation: alphanumeric, hyphens, underscores
SAFE_RESOURCE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,127}$")

# Product ID validation: alphanumeric, hyphens, underscores, dots (stricter subset)
SAFE_PRODUCT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-\.]{0,127}$")

# Max lengths for input validation
MAX_SHOP_URL_LENGTH = 512
MAX_CREDENTIAL_VALUE_LENGTH = 1024
MAX_SKU_LENGTH = 128
MAX_ORDER_ID_LENGTH = 128
MAX_CARRIER_LENGTH = 64
MAX_SERVICE_LENGTH = 64
MAX_TRACKING_NUMBER_LENGTH = 128

# Financial amount constraints
MAX_FINANCIAL_AMOUNT = Decimal("99_999_999.99")  # ~100M reasonable upper bound
MAX_DECIMAL_PLACES = 2  # Standard currency precision

# Pagination constraints
MAX_PAGINATION_LIMIT = 1000
MAX_PAGINATION_OFFSET = 1_000_000
DEFAULT_PAGINATION_LIMIT = 100
DEFAULT_PAGINATION_OFFSET = 0

# ISO 4217 currency code whitelist (commonly used codes)
ALLOWED_CURRENCY_CODES: frozenset[str] = frozenset(
    {
        "USD",
        "EUR",
        "GBP",
        "JPY",
        "CAD",
        "AUD",
        "CHF",
        "CNY",
        "HKD",
        "NZD",
        "SEK",
        "NOK",
        "DKK",
        "SGD",
        "KRW",
        "INR",
        "BRL",
        "MXN",
        "ZAR",
        "TRY",
        "PLN",
        "CZK",
        "HUF",
        "ILS",
        "THB",
        "MYR",
        "PHP",
        "IDR",
        "TWD",
        "AED",
        "SAR",
        "CLP",
        "COP",
        "PEN",
        "ARS",
        "VND",
        "EGP",
        "NGN",
        "KES",
        "GHS",
    }
)


def validate_platform_id(platform: str) -> tuple[bool, str | None]:
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


def validate_resource_id(resource_id: str, resource_type: str = "ID") -> tuple[bool, str | None]:
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


def validate_sku(sku: str) -> tuple[bool, str | None]:
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


def validate_url(url: str, field_name: str = "URL") -> tuple[bool, str | None]:
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


def validate_quantity(quantity: Any) -> tuple[bool, str | None, int | None]:
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


def validate_financial_amount(
    amount: Any,
    field_name: str = "Amount",
    *,
    allow_zero: bool = True,
    max_amount: Decimal | None = None,
) -> tuple[bool, str | None, Decimal | None]:
    """Validate a financial amount value.

    Checks that the amount is non-negative, within reasonable bounds,
    and has at most 2 decimal places of precision.

    Args:
        amount: The amount to validate (str, int, float, or Decimal)
        field_name: Field name for error messages
        allow_zero: Whether zero is a valid amount
        max_amount: Custom upper bound (defaults to MAX_FINANCIAL_AMOUNT)

    Returns:
        Tuple of (is_valid, error_message, parsed_decimal_value)
    """
    if amount is None:
        return False, f"{field_name} is required", None

    upper_bound = max_amount if max_amount is not None else MAX_FINANCIAL_AMOUNT

    try:
        # Convert to Decimal for precise arithmetic
        if isinstance(amount, float):
            # Avoid float precision issues by converting via string
            dec_amount = Decimal(str(amount))
        else:
            dec_amount = Decimal(str(amount))
    except (InvalidOperation, ValueError, TypeError):
        return False, f"Invalid {field_name.lower()} format", None

    # Reject special values (NaN, Infinity)
    if not dec_amount.is_finite():
        return False, f"{field_name} must be a finite number", None

    # Non-negative check
    if dec_amount < 0:
        return False, f"{field_name} cannot be negative", None

    # Zero check
    if not allow_zero and dec_amount == 0:
        return False, f"{field_name} must be greater than zero", None

    # Upper bound check
    if dec_amount > upper_bound:
        return False, f"{field_name} exceeds maximum allowed value ({upper_bound})", None

    # Decimal precision check (max 2 decimal places for currency)
    if dec_amount != dec_amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP):
        # Check if there are more than MAX_DECIMAL_PLACES digits after decimal
        sign, digits, exponent = dec_amount.as_tuple()
        if isinstance(exponent, int) and exponent < -MAX_DECIMAL_PLACES:
            return (
                False,
                f"{field_name} must have at most {MAX_DECIMAL_PLACES} decimal places",
                None,
            )

    return True, None, dec_amount


def validate_currency_code(currency: Any) -> tuple[bool, str | None]:
    """Validate a currency code against the ISO 4217 whitelist.

    Args:
        currency: Currency code to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not currency:
        return False, "Currency code is required"
    if not isinstance(currency, str):
        return False, "Currency code must be a string"
    code = currency.strip().upper()
    if len(code) != 3:
        return False, "Currency code must be exactly 3 characters"
    if not code.isalpha():
        return False, "Currency code must contain only letters"
    if code not in ALLOWED_CURRENCY_CODES:
        return False, f"Unsupported currency code: {code}"
    return True, None


def validate_product_id(product_id: str) -> tuple[bool, str | None]:
    """Validate a product ID with strict alphanumeric pattern matching.

    Product IDs must start with an alphanumeric character and contain only
    alphanumeric characters, hyphens, underscores, and dots.

    Args:
        product_id: Product identifier to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not product_id:
        return False, "Product ID is required"
    if not isinstance(product_id, str):
        return False, "Product ID must be a string"
    if len(product_id) > 128:
        return False, "Product ID too long (max 128 characters)"
    if not SAFE_PRODUCT_ID_PATTERN.match(product_id):
        return (
            False,
            "Invalid product ID format (must start with alphanumeric, "
            "contain only alphanumeric, hyphens, underscores, dots)",
        )
    return True, None


def validate_pagination(
    limit: Any = None,
    offset: Any = None,
) -> tuple[bool, str | None, int, int]:
    """Validate and sanitize pagination parameters.

    Ensures limit and offset are within safe bounds to prevent
    resource exhaustion or excessively large result sets.

    Args:
        limit: Maximum number of results (default: DEFAULT_PAGINATION_LIMIT)
        offset: Number of results to skip (default: 0)

    Returns:
        Tuple of (is_valid, error_message, parsed_limit, parsed_offset)
    """
    parsed_limit = DEFAULT_PAGINATION_LIMIT
    parsed_offset = DEFAULT_PAGINATION_OFFSET

    if limit is not None:
        try:
            parsed_limit = int(limit)
        except (ValueError, TypeError):
            return False, "Limit must be an integer", 0, 0
        if parsed_limit < 1:
            return False, "Limit must be at least 1", 0, 0
        if parsed_limit > MAX_PAGINATION_LIMIT:
            return (
                False,
                f"Limit exceeds maximum ({MAX_PAGINATION_LIMIT})",
                0,
                0,
            )

    if offset is not None:
        try:
            parsed_offset = int(offset)
        except (ValueError, TypeError):
            return False, "Offset must be an integer", 0, 0
        if parsed_offset < 0:
            return False, "Offset cannot be negative", 0, 0
        if parsed_offset > MAX_PAGINATION_OFFSET:
            return (
                False,
                f"Offset exceeds maximum ({MAX_PAGINATION_OFFSET})",
                0,
                0,
            )

    return True, None, parsed_limit, parsed_offset


# Backward-compatible aliases
_validate_platform_id = validate_platform_id
_validate_resource_id = validate_resource_id
_validate_sku = validate_sku
_validate_url = validate_url
_validate_quantity = validate_quantity
_validate_financial_amount = validate_financial_amount
_validate_currency_code = validate_currency_code
_validate_product_id = validate_product_id
_validate_pagination = validate_pagination
