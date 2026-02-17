"""
Shared utilities for invoice storage backends (AP and AR).

This module consolidates common code between invoice_store.py and ar_invoice_store.py:
- DecimalEncoder: JSON encoder for Decimal types
- decimal_decoder: JSON decoder hook for decimal strings
- parse_date: Flexible date parsing for Postgres operations
- Common payment recording logic
- Common list result processing

Usage:
    from aragora.storage.invoice_common import (
        DecimalEncoder,
        decimal_decoder,
        parse_date,
        calculate_payment_update,
        deserialize_invoice_row,
    )
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)

# Fields that should be converted to Decimal when deserializing
DECIMAL_FIELDS = frozenset(
    [
        "subtotal",
        "tax_amount",
        "total_amount",
        "amount_paid",
        "balance_due",
        "amount",  # For payment records
    ]
)


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types.

    Converts Decimal objects to strings to preserve precision in JSON storage.

    Example:
        json.dumps({"amount": Decimal("123.45")}, cls=DecimalEncoder)
        # Returns: '{"amount": "123.45"}'
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return str(obj)
        return super().default(obj)


def decimal_decoder(dct: dict[str, Any]) -> dict[str, Any]:
    """JSON decoder hook that converts decimal strings back to Decimal.

    Use as object_hook in json.loads() to restore Decimal precision.

    Example:
        json.loads('{"amount": "123.45"}', object_hook=decimal_decoder)
        # Returns: {"amount": Decimal("123.45")}
    """
    for key in DECIMAL_FIELDS:
        if key in dct and isinstance(dct[key], str):
            try:
                dct[key] = Decimal(dct[key])
            except (ValueError, TypeError, ArithmeticError) as e:
                logger.debug("Failed to convert field '%s' to Decimal: %s", key, e)
    return dct


def parse_date(value: Any) -> datetime | None:
    """Parse various date formats into datetime objects.

    Handles:
    - datetime objects (returned as-is)
    - ISO format strings (with or without timezone)
    - None values

    Used primarily for Postgres timestamp conversion.

    Args:
        value: Date value to parse (datetime, str, or None)

    Returns:
        datetime object or None if parsing fails
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            # Try ISO format with timezone
            if value.endswith("Z"):
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            return datetime.fromisoformat(value)
        except ValueError:
            try:
                # Try common date format
                return datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                logger.debug("Failed to parse date: %s", value)
                return None
    return None


def calculate_payment_update(
    current_data: dict[str, Any],
    payment_amount: Decimal,
    payment_date: str,
    payment_method: str,
    reference: str | None = None,
) -> dict[str, Any]:
    """Calculate updated invoice data after recording a payment.

    This implements the common payment recording logic shared between
    AP and AR invoice stores.

    Args:
        current_data: Current invoice data dict
        payment_amount: Amount of the payment
        payment_date: Date of payment (ISO format string)
        payment_method: Payment method identifier
        reference: Optional payment reference/check number

    Returns:
        Updated invoice data with new payment recorded and balances updated
    """
    # Get or initialize payments list
    payments = current_data.get("payments", [])
    if payments is None:
        payments = []

    # Create payment record
    payment_record = {
        "amount": payment_amount,
        "date": payment_date,
        "method": payment_method,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    if reference:
        payment_record["reference"] = reference

    payments.append(payment_record)

    # Calculate totals
    total_paid = sum(Decimal(str(p.get("amount", 0))) for p in payments)
    total_amount = Decimal(str(current_data.get("total_amount", 0)))
    balance_due = total_amount - total_paid

    # Update invoice data
    updated_data = current_data.copy()
    updated_data["payments"] = payments
    updated_data["amount_paid"] = total_paid
    updated_data["balance_due"] = balance_due
    updated_data["updated_at"] = datetime.now(timezone.utc).isoformat()

    # Mark as paid if fully paid
    if balance_due <= Decimal("0"):
        updated_data["status"] = "paid"

    return updated_data


def deserialize_invoice_row(
    row: dict[str, Any] | Any,
    json_field: str = "data_json",
) -> dict[str, Any] | None:
    """Deserialize a database row containing invoice JSON data.

    Handles both dict-like rows (asyncpg) and sqlite3.Row objects.
    Applies decimal_decoder to restore Decimal precision.

    Args:
        row: Database row containing JSON data
        json_field: Name of the field containing JSON (default: "data_json")

    Returns:
        Deserialized invoice dict or None if row is None
    """
    if row is None:
        return None

    # Handle dict-like access
    if hasattr(row, "__getitem__"):
        data = row[json_field]
    else:
        data = getattr(row, json_field, None)

    if data is None:
        return None

    if isinstance(data, str):
        return json.loads(data, object_hook=decimal_decoder)
    elif isinstance(data, dict):
        return decimal_decoder(data.copy())

    return data


def deserialize_invoice_rows(
    rows: list[Any],
    json_field: str = "data_json",
) -> list[dict[str, Any]]:
    """Deserialize multiple database rows containing invoice JSON data.

    Args:
        rows: List of database rows
        json_field: Name of the field containing JSON (default: "data_json")

    Returns:
        List of deserialized invoice dicts
    """
    results = []
    for row in rows:
        data = deserialize_invoice_row(row, json_field)
        if data is not None:
            results.append(data)
    return results


def serialize_invoice(data: dict[str, Any]) -> str:
    """Serialize invoice data to JSON string.

    Uses DecimalEncoder to handle Decimal types.

    Args:
        data: Invoice data dict

    Returns:
        JSON string
    """
    return json.dumps(data, cls=DecimalEncoder)
