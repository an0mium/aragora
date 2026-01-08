"""
Handler utility modules.

This package contains focused utility modules extracted from base.py
for better organization and maintainability.

All utilities are re-exported from base.py for backwards compatibility.
"""

from aragora.server.handlers.utils.safe_data import (
    safe_get,
    safe_get_nested,
    safe_json_parse,
)
from aragora.server.handlers.utils.database import (
    get_db_connection,
    table_exists,
)

__all__ = [
    # Safe data access
    "safe_get",
    "safe_get_nested",
    "safe_json_parse",
    # Database utilities
    "get_db_connection",
    "table_exists",
]
