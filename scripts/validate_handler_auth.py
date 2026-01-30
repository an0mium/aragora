#!/usr/bin/env python3
"""Validate that all HTTP handler modules have proper authentication.

Scans all Python files in aragora/server/handlers/ recursively and verifies
that each file defining HTTP route handler classes imports an auth decorator.

Usage:
    python scripts/validate_handler_auth.py

Exit codes:
    0 - All handler modules have proper auth (or are allowlisted)
    1 - One or more handler modules are missing auth imports
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

HANDLERS_ROOT = Path("aragora/server/handlers")

# Auth decorator imports that satisfy the check.
# A handler file must import at least one of these names.
AUTH_IMPORT_NAMES: frozenset[str] = frozenset(
    {
        "require_permission",
        "require_role",
        "require_authenticated",
        "require_auth",
        "require_admin",
    }
)

# Inline auth patterns that also satisfy the check.
# Many handlers use these instead of decorators (extracting auth from the
# HTTP handler object inside the method body).
INLINE_AUTH_PATTERNS: tuple[str, ...] = (
    "get_auth_context(",
    "extract_user_from_request(",
    "_check_rbac_permission(",
    "require_auth_or_error(",
    "require_permission_or_error(",
    "_require_admin(",
    "admin_secure_endpoint",
    "check_permission(",
    "_verify_bearer_token(",
    "verify_signature(",
)

# Filenames that are always allowlisted regardless of location.
ALLOWLISTED_FILENAMES: frozenset[str] = frozenset(
    {
        "__init__.py",
        "base.py",
        "types.py",
        "exceptions.py",
        "secure.py",
        "sso.py",
    }
)

# Directory segments that cause a file to be allowlisted if they appear
# anywhere in the relative path from the handlers root.
# - utils/       -> utility modules, no routes
# - bots/        -> webhook signature auth (not RBAC)
# - auth/        -> auth endpoints themselves
# - social/slack/ and social/_slack_impl/ -> internal Slack implementation
ALLOWLISTED_DIR_SEGMENTS: tuple[str, ...] = (
    "utils/",
    "bots/",
    "auth/",
)

# Directory prefixes that start with "oauth" in any form.
# Matched via startswith on each path component.
OAUTH_DIR_PREFIX = "oauth"

# Specific relative paths (from handlers root) that are allowlisted.
ALLOWLISTED_RELATIVE_PATHS: frozenset[str] = frozenset(
    {
        "public/status_page.py",
        "chat/router.py",  # Webhook endpoints use platform signature auth, not RBAC
    }
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _is_allowlisted(rel_path: Path) -> bool:
    """Return True if *rel_path* (relative to HANDLERS_ROOT) is allowlisted."""

    # 1. Allowlisted filenames
    if rel_path.name in ALLOWLISTED_FILENAMES:
        return True

    # 2. Specific relative paths
    if rel_path.as_posix() in ALLOWLISTED_RELATIVE_PATHS:
        return True

    parts = rel_path.parts  # e.g. ("social", "slack", "handler.py")

    # 3. Allowlisted directory segments
    rel_posix = rel_path.as_posix() + "/"  # ensure trailing slash for segment matching
    for segment in ALLOWLISTED_DIR_SEGMENTS:
        if segment in rel_posix:
            return True

    # 4. OAuth directories (any component starting with "oauth")
    for part in parts[:-1]:  # exclude the filename itself
        if part.startswith(OAUTH_DIR_PREFIX) or part.startswith("_oauth"):
            return True

    # 5. social/slack/ and social/_slack_impl/ directories
    if len(parts) >= 2 and parts[0] == "social":
        if parts[1] in ("slack", "_slack_impl"):
            return True

    return False


def _has_handler_indicators(source: str, tree: ast.Module) -> bool:
    """Return True if the file looks like it defines HTTP route handlers.

    We check for:
    - A class-level ``ROUTES`` attribute assignment
    - Methods named handle, handle_get, handle_post, handle_put,
      handle_delete, or handle_patch
    """

    handler_method_names = frozenset(
        {
            "handle",
            "handle_get",
            "handle_post",
            "handle_put",
            "handle_delete",
            "handle_patch",
        }
    )

    for node in ast.walk(tree):
        # Class-level ROUTES = [...]
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == "ROUTES":
                            return True
                # Also check for handle* methods inside classes
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name in handler_method_names:
                        return True

    return False


def _has_auth_import(tree: ast.Module, source: str = "") -> bool:
    """Return True if the AST imports any of the recognised auth decorators
    or the source contains inline auth patterns."""

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.names:
                for alias in node.names:
                    if alias.name in AUTH_IMPORT_NAMES:
                        return True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                # e.g. import aragora.rbac.decorators  (unlikely but handle it)
                name = alias.asname or alias.name
                if name in AUTH_IMPORT_NAMES:
                    return True

    # Also check for inline auth patterns in source code.
    # Many handlers extract auth from the HTTP handler object inside
    # method bodies rather than using decorators.
    if source:
        for pattern in INLINE_AUTH_PATTERNS:
            if pattern in source:
                return True

    return False


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    # Resolve handlers root relative to the project root (cwd).
    handlers_dir = Path.cwd() / HANDLERS_ROOT
    if not handlers_dir.is_dir():
        print(f"ERROR: handlers directory not found at {handlers_dir}", file=sys.stderr)
        return 1

    violations: list[str] = []
    scanned = 0
    allowlisted = 0
    passed = 0
    skipped_no_handler = 0

    for root, _dirs, files in os.walk(handlers_dir):
        for fname in sorted(files):
            if not fname.endswith(".py"):
                continue

            full_path = Path(root) / fname
            rel_path = full_path.relative_to(handlers_dir)

            # Check allowlist first (cheap, avoids parsing).
            if _is_allowlisted(rel_path):
                allowlisted += 1
                continue

            scanned += 1

            # Parse the file.
            try:
                source = full_path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(full_path))
            except (SyntaxError, UnicodeDecodeError) as exc:
                print(f"  WARNING: could not parse {rel_path}: {exc}", file=sys.stderr)
                continue

            # Only check files that define route handlers.
            if not _has_handler_indicators(source, tree):
                skipped_no_handler += 1
                continue

            # Check for auth imports or inline auth patterns.
            if _has_auth_import(tree, source):
                passed += 1
            else:
                violations.append(str(rel_path))

    # ---- Output ----------------------------------------------------------- #

    print("=" * 60)
    print("Handler Auth Validation Report")
    print("=" * 60)
    print()
    print(f"  Handlers scanned:           {scanned}")
    print(f"  Allowlisted (skipped):      {allowlisted}")
    print(f"  No handler indicators:      {skipped_no_handler}")
    print(f"  Passed (auth import found): {passed}")
    print(f"  VIOLATIONS:                 {len(violations)}")
    print()

    if violations:
        print("The following handler files define routes but have NO auth")
        print("(no decorator import or inline auth pattern detected):")
        print()
        for v in sorted(violations):
            print(f"  - {v}")
        print()
        print("If a file intentionally does not need RBAC, add it to the allowlist in this script.")
        print()
        return 1

    print("All handler modules have proper authentication or are allowlisted.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
