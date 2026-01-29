"""
Database connector ID encoding/decoding utilities.

Provides reversible evidence ID generation for database connectors,
enabling the fetch() method to retrieve original rows by primary key.

ID Format:
    {prefix}:{database}:{table}:{pk_type}:{encoded_pk}

    - prefix: Connector type (pg, mysql, mssql, mongo, sf)
    - database: Database name
    - table: Table or collection name
    - pk_type: Type indicator (i=int, s=string, u=UUID, c=composite)
    - encoded_pk: Encoded primary key value

    Snowflake adds account: sf:{account}:{database}:{table}:{pk_type}:{pk}

Legacy hash-based IDs (4 parts for most, 5 for snowflake) are detected
and handled gracefully - fetch() returns None for these.
"""

from __future__ import annotations

import base64
import json
import re
from enum import Enum
from typing import Any


class PKType(Enum):
    """Primary key type indicators for evidence ID encoding."""

    INTEGER = "i"
    STRING = "s"
    UUID = "u"
    COMPOSITE = "c"


def detect_pk_type(pk_value: Any) -> PKType:
    """Detect the type of a primary key value."""
    if isinstance(pk_value, int):
        return PKType.INTEGER
    if isinstance(pk_value, (list, tuple)):
        return PKType.COMPOSITE
    pk_str = str(pk_value)
    if re.match(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        pk_str,
        re.IGNORECASE,
    ):
        return PKType.UUID
    return PKType.STRING


def encode_pk(pk_value: Any, pk_type: PKType | None = None) -> tuple[str, str]:
    """
    Encode a primary key value for inclusion in an evidence ID.

    Returns:
        Tuple of (type_indicator, encoded_value)
    """
    if pk_type is None:
        pk_type = detect_pk_type(pk_value)

    if pk_type == PKType.INTEGER:
        return pk_type.value, str(pk_value)

    if pk_type == PKType.UUID:
        return pk_type.value, str(pk_value).replace("-", "").lower()

    if pk_type == PKType.COMPOSITE:
        json_str = json.dumps(list(pk_value), separators=(",", ":"))
        encoded = base64.urlsafe_b64encode(json_str.encode()).decode().rstrip("=")
        return pk_type.value, encoded

    # String: base64url encode to handle special characters
    encoded = base64.urlsafe_b64encode(str(pk_value).encode()).decode().rstrip("=")
    return pk_type.value, encoded


def decode_pk(pk_type_str: str, encoded_value: str) -> Any:
    """
    Decode a primary key from an evidence ID.

    Args:
        pk_type_str: Type indicator (i, s, u, c)
        encoded_value: Encoded primary key value

    Returns:
        Decoded primary key value
    """
    if pk_type_str == PKType.INTEGER.value:
        return int(encoded_value)

    if pk_type_str == PKType.UUID.value:
        v = encoded_value
        return f"{v[:8]}-{v[8:12]}-{v[12:16]}-{v[16:20]}-{v[20:]}"

    if pk_type_str == PKType.COMPOSITE.value:
        padded = _pad_base64(encoded_value)
        json_str = base64.urlsafe_b64decode(padded).decode()
        return json.loads(json_str)

    # String
    padded = _pad_base64(encoded_value)
    return base64.urlsafe_b64decode(padded).decode()


def _pad_base64(s: str) -> str:
    """Add padding to base64url string."""
    padding = 4 - (len(s) % 4)
    if padding != 4:
        s += "=" * padding
    return s


def generate_evidence_id(
    prefix: str,
    database: str,
    table: str,
    pk_value: Any,
    account: str | None = None,
) -> str:
    """
    Generate a reversible evidence ID for a database row.

    Args:
        prefix: Connector prefix (pg, mysql, mssql, mongo, sf)
        database: Database name
        table: Table or collection name
        pk_value: Primary key value
        account: Optional account (for Snowflake)

    Returns:
        Evidence ID string
    """
    pk_type_str, encoded_pk = encode_pk(pk_value)

    if prefix == "sf" and account:
        return f"{prefix}:{account}:{database}:{table}:{pk_type_str}:{encoded_pk}"

    return f"{prefix}:{database}:{table}:{pk_type_str}:{encoded_pk}"


def parse_evidence_id(evidence_id: str) -> dict[str, Any] | None:
    """
    Parse an evidence ID to extract components.

    Handles both legacy hash-based and new reversible formats.

    Returns:
        Dict with keys: prefix, database, table, pk_type, pk_value, is_legacy
        Or None if parsing fails
    """
    parts = evidence_id.split(":")

    # Legacy 4-part format: prefix:db:table:hash (pg, mongo)
    if len(parts) == 4:
        prefix, db, table, last = parts
        if _is_hex_hash(last):
            return {
                "prefix": prefix,
                "database": db,
                "table": table,
                "pk_type": None,
                "pk_value": None,
                "pk_hash": last,
                "is_legacy": True,
            }

    # New 5-part format: prefix:db:table:pk_type:encoded_pk
    if len(parts) == 5:
        prefix, db, table, pk_type, encoded_pk = parts

        # Legacy snowflake format: sf:account:db:table:hash
        if prefix == "sf" and _is_hex_hash(encoded_pk):
            return {
                "prefix": prefix,
                "account": db,
                "database": table,
                "table": pk_type,
                "pk_type": None,
                "pk_value": None,
                "pk_hash": encoded_pk,
                "is_legacy": True,
            }

        # New format
        if pk_type in ("i", "s", "u", "c"):
            try:
                pk_value = decode_pk(pk_type, encoded_pk)
                return {
                    "prefix": prefix,
                    "database": db,
                    "table": table,
                    "pk_type": pk_type,
                    "pk_value": pk_value,
                    "is_legacy": False,
                }
            except Exception:
                return None

    # New Snowflake 6-part: sf:account:db:table:pk_type:encoded_pk
    if len(parts) == 6:
        prefix, account, db, table, pk_type, encoded_pk = parts
        if pk_type in ("i", "s", "u", "c"):
            try:
                pk_value = decode_pk(pk_type, encoded_pk)
                return {
                    "prefix": prefix,
                    "account": account,
                    "database": db,
                    "table": table,
                    "pk_type": pk_type,
                    "pk_value": pk_value,
                    "is_legacy": False,
                }
            except Exception:
                return None

    return None


def is_legacy_id(evidence_id: str) -> bool:
    """Check if an evidence ID uses the old hash-based format."""
    parsed = parse_evidence_id(evidence_id)
    return parsed is not None and parsed.get("is_legacy", True)


def _is_hex_hash(s: str) -> bool:
    """Check if a string looks like a hex hash (12-16 chars)."""
    return len(s) in (12, 16) and all(c in "0123456789abcdef" for c in s)
