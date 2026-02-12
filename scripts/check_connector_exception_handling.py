#!/usr/bin/env python3
"""
Guardrail for connector exception handling hygiene.

Fails when connector code silently swallows broad exceptions via:
  - except Exception: pass
  - except Exception as e: pass
  - bare except: pass
  - except Exception: return None
  - except Exception: continue/break
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Finding:
    path: Path
    lineno: int
    message: str


def _is_catch_all(handler: ast.ExceptHandler) -> bool:
    """True when handler catches Exception/BaseException (or bare except)."""
    if handler.type is None:
        return True

    if isinstance(handler.type, ast.Name):
        return handler.type.id in {"Exception", "BaseException"}

    if isinstance(handler.type, ast.Tuple):
        names = {elt.id for elt in handler.type.elts if isinstance(elt, ast.Name)}
        return bool({"Exception", "BaseException"} & names)

    return False


def _is_silent_action(node: ast.stmt) -> tuple[bool, str]:
    if isinstance(node, ast.Pass):
        return True, "pass"
    if isinstance(node, ast.Continue):
        return True, "continue"
    if isinstance(node, ast.Break):
        return True, "break"
    if isinstance(node, ast.Return):
        if node.value is None:
            return True, "return"
        if isinstance(node.value, ast.Constant) and node.value.value is None:
            return True, "return None"
    return False, ""


def _scan_file(path: Path) -> list[Finding]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    findings: list[Finding] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            if not _is_catch_all(handler):
                continue
            if len(handler.body) != 1:
                continue
            is_silent, action = _is_silent_action(handler.body[0])
            if is_silent:
                findings.append(
                    Finding(
                        path=path,
                        lineno=handler.lineno,
                        message=f"silent broad exception handler ({action})",
                    )
                )
    return findings


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*.py")
        if "__pycache__" not in p.parts and ".venv" not in p.parts and "node_modules" not in p.parts
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Check connector exception handling hygiene")
    parser.add_argument(
        "--path",
        default="aragora/connectors",
        help="Path to scan (default: aragora/connectors)",
    )
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        print(f"Path does not exist: {root}", file=sys.stderr)
        return 2

    findings: list[Finding] = []
    parse_errors: list[tuple[Path, str]] = []

    for file_path in _iter_python_files(root):
        try:
            findings.extend(_scan_file(file_path))
        except SyntaxError as exc:
            parse_errors.append((file_path, f"SyntaxError: {exc.msg} (line {exc.lineno})"))
        except OSError as exc:
            parse_errors.append((file_path, f"OSError: {exc}"))

    if parse_errors:
        print("Failed to parse one or more files:")
        for path, msg in parse_errors:
            print(f"  - {path}: {msg}")
        return 2

    if findings:
        print("Found silent broad exception handlers in connectors:")
        for finding in findings:
            print(f"  - {finding.path}:{finding.lineno}: {finding.message}")
        print("\nReplace silent handlers with targeted exceptions + logging.")
        return 1

    print(f"Connector exception handling check passed ({root})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
