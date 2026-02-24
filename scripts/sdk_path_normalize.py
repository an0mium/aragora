"""Shared SDK path normalization.

Single source of truth used by sdk_codegen.py, verify_sdk_contracts.py,
check_sdk_parity.py, batch_add_openapi_stubs.py, and cross-parity checks.
"""

from __future__ import annotations

import re


def normalize_sdk_path(path: str) -> str:
    """Normalize an SDK path for consistent comparison.

    - Strip query string
    - Strip version prefix (/api/v1/, /api/v2/, etc.)
    - Normalize param styles: :param, {named}, ${expr}, * -> {param}
    - Strip trailing slash
    - Lowercase
    """
    # Strip query string
    path = path.split("?", 1)[0]
    # Strip version prefix
    path = re.sub(r"^/api/v\d+/", "/api/", path)
    # Template literal expressions ${...} -> {param}
    path = re.sub(r"\$\{[^}]+\}", "{param}", path)
    # Express-style :param -> {param}
    path = re.sub(r":([a-zA-Z_][a-zA-Z0-9_]*)", "{param}", path)
    # Wildcard segments
    path = path.replace("/*", "/{param}")
    # All named path parameters {session_id} etc. -> {param}
    path = re.sub(r"\{[^}]+\}", "{param}", path)
    # Strip trailing slash (but keep bare "/")
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return path.lower()
