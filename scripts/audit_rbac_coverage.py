#!/usr/bin/env python3
"""
Audit RBAC permission coverage across server handlers.

Scans handler files for @require_permission and @require_role decorators
to identify gaps in access control coverage.

Usage:
    python scripts/audit_rbac_coverage.py              # Full report
    python scripts/audit_rbac_coverage.py --json       # JSON output
    python scripts/audit_rbac_coverage.py --summary    # Summary only
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class HandlerInfo:
    """Information about a handler function."""

    file: str
    function: str
    line: int
    has_permission: bool
    has_role: bool
    permission_name: Optional[str] = None
    role_name: Optional[str] = None
    http_method: Optional[str] = None
    route: Optional[str] = None
    protection_type: Optional[str] = None  # 'decorator', 'admin_secure', 'manual_check'


def find_handlers(handler_dir: Path) -> list[HandlerInfo]:
    """Find all handler functions and their RBAC decorators."""
    handlers = []

    # Decorator patterns
    func_pattern = re.compile(r"^(?:async\s+)?def\s+(\w+)\s*\(")
    require_permission = re.compile(r'@require_permission\s*\(\s*["\']([^"\']+)["\']')
    require_role = re.compile(r'@require_role\s*\(\s*["\']([^"\']+)["\']')
    admin_secure = re.compile(r"@admin_secure_endpoint\s*\(")
    admin_secure_perm = re.compile(
        r'@admin_secure_endpoint\s*\([^)]*permission\s*=\s*["\']([^"\']+)["\']'
    )
    route_pattern = re.compile(
        r'@(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']', re.IGNORECASE
    )
    # Pattern for check_permission method calls within function body
    manual_check_pattern = re.compile(
        r"\bcheck_permission\s*\(|self\.check_permission\s*\(|_check_permission\s*\("
    )

    for py_file in handler_dir.rglob("*.py"):
        if py_file.name.startswith("_"):
            continue

        try:
            content = py_file.read_text()
            lines = content.split("\n")

            i = 0
            while i < len(lines):
                line = lines[i]
                func_match = func_pattern.match(line.strip())

                if func_match:
                    func_name = func_match.group(1)

                    # Look back for decorators
                    decorator_lines = []
                    j = i - 1
                    while j >= 0 and (lines[j].strip().startswith("@") or not lines[j].strip()):
                        if lines[j].strip().startswith("@"):
                            decorator_lines.append(lines[j])
                        j -= 1

                    decorator_text = "\n".join(decorator_lines)

                    # Look forward for function body (next 30 lines or until next function)
                    body_lines = []
                    k = i + 1
                    indent_level = len(line) - len(line.lstrip())
                    while k < min(i + 30, len(lines)):
                        body_line = lines[k]
                        if body_line.strip() and not body_line.startswith(" " * (indent_level + 1)):
                            if func_pattern.match(body_line.strip()):
                                break
                        body_lines.append(body_line)
                        k += 1
                    body_text = "\n".join(body_lines)

                    perm_match = require_permission.search(decorator_text)
                    role_match = require_role.search(decorator_text)
                    admin_secure_match = admin_secure.search(decorator_text)
                    admin_secure_perm_match = admin_secure_perm.search(decorator_text)
                    route_match = route_pattern.search(decorator_text)
                    manual_check_match = manual_check_pattern.search(body_text)

                    # Determine protection type
                    protection_type = None
                    has_perm = False
                    perm_name = None

                    if perm_match:
                        has_perm = True
                        protection_type = "decorator"
                        perm_name = perm_match.group(1)
                    elif role_match:
                        has_perm = True
                        protection_type = "role_decorator"
                    elif admin_secure_match:
                        has_perm = True
                        protection_type = "admin_secure"
                        if admin_secure_perm_match:
                            perm_name = admin_secure_perm_match.group(1)
                    elif manual_check_match:
                        has_perm = True
                        protection_type = "manual_check"

                    handler = HandlerInfo(
                        file=str(py_file.relative_to(handler_dir.parent.parent)),
                        function=func_name,
                        line=i + 1,
                        has_permission=has_perm,
                        has_role=role_match is not None,
                        permission_name=perm_name,
                        role_name=role_match.group(1) if role_match else None,
                        http_method=route_match.group(1).upper() if route_match else None,
                        route=route_match.group(2) if route_match else None,
                        protection_type=protection_type,
                    )

                    # Only include if it's a route handler
                    if route_match or func_name.startswith(
                        ("get_", "post_", "put_", "delete_", "handle_")
                    ):
                        handlers.append(handler)

                i += 1

        except Exception as e:
            print(f"Error processing {py_file}: {e}", file=sys.stderr)

    return handlers


def generate_report(handlers: list[HandlerInfo], output_format: str = "text") -> str:
    """Generate the audit report."""
    total = len(handlers)
    with_permission = sum(1 for h in handlers if h.has_permission)
    with_role = sum(1 for h in handlers if h.has_role)
    protected = sum(1 for h in handlers if h.has_permission or h.has_role)
    unprotected = total - protected

    coverage = (protected / total * 100) if total > 0 else 0

    # Group by module
    by_module = defaultdict(list)
    for h in handlers:
        parts = h.file.split("/")
        module = parts[3] if len(parts) > 3 else "root"
        by_module[module].append(h)

    if output_format == "json":
        return json.dumps(
            {
                "summary": {
                    "total_handlers": total,
                    "protected_handlers": protected,
                    "unprotected_handlers": unprotected,
                    "coverage_percent": round(coverage, 1),
                    "with_permission": with_permission,
                    "with_role": with_role,
                },
                "by_module": {
                    m: {
                        "total": len(hs),
                        "protected": sum(1 for h in hs if h.has_permission or h.has_role),
                        "handlers": [asdict(h) for h in hs],
                    }
                    for m, hs in sorted(by_module.items())
                },
                "unprotected": [
                    asdict(h) for h in handlers if not (h.has_permission or h.has_role)
                ],
            },
            indent=2,
        )

    # Text report
    lines = [
        "=" * 60,
        "RBAC HANDLER COVERAGE AUDIT",
        "=" * 60,
        "",
        "SUMMARY",
        "-" * 40,
        f"Total handlers:      {total}",
        f"Protected:           {protected} ({coverage:.1f}%)",
        f"  @require_permission: {with_permission}",
        f"  @require_role:       {with_role}",
        f"Unprotected:         {unprotected}",
        "",
        "BY MODULE",
        "-" * 40,
    ]

    for module, hs in sorted(by_module.items(), key=lambda x: -len(x[1])):
        prot = sum(1 for h in hs if h.has_permission or h.has_role)
        pct = (prot / len(hs) * 100) if hs else 0
        lines.append(f"  {module}: {prot}/{len(hs)} ({pct:.0f}%)")

    lines.extend(
        [
            "",
            "UNPROTECTED HANDLERS (requiring attention)",
            "-" * 40,
        ]
    )

    for h in sorted(handlers, key=lambda x: x.file)[:30]:
        if not (h.has_permission or h.has_role):
            method = h.http_method or "?"
            lines.append(f"  [{method}] {h.file}:{h.line} - {h.function}")

    if unprotected > 30:
        lines.append(f"  ... and {unprotected - 30} more")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Audit RBAC coverage")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--summary", action="store_true", help="Summary only")
    parser.add_argument("--min-coverage", type=float, help="Fail if below threshold")
    args = parser.parse_args()

    handler_dir = Path("aragora/server/handlers")
    if not handler_dir.exists():
        print(f"Handler directory not found: {handler_dir}", file=sys.stderr)
        return 1

    handlers = find_handlers(handler_dir)

    if args.summary:
        total = len(handlers)
        protected = sum(1 for h in handlers if h.has_permission or h.has_role)
        coverage = (protected / total * 100) if total > 0 else 0
        print(f"{protected}/{total} handlers protected ({coverage:.1f}%)")
    else:
        output_format = "json" if args.json else "text"
        print(generate_report(handlers, output_format))

    if args.min_coverage is not None:
        total = len(handlers)
        protected = sum(1 for h in handlers if h.has_permission or h.has_role)
        coverage = (protected / total * 100) if total > 0 else 0
        if coverage < args.min_coverage:
            print(
                f"\n::error::RBAC coverage ({coverage:.1f}%) below minimum ({args.min_coverage}%)"
            )
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
