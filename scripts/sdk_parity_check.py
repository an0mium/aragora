#!/usr/bin/env python3
"""
SDK Parity Validation Tool.

Compares Python and TypeScript SDK namespaces to detect drift.
Returns non-zero exit code if parity exceeds threshold.

Usage:
    python scripts/sdk_parity_check.py
    python scripts/sdk_parity_check.py --threshold 10  # Allow 10% drift
    python scripts/sdk_parity_check.py --verbose       # Show all namespaces
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def normalize_name(name: str) -> str:
    """Normalize namespace name for comparison (Python uses _, TS uses -)."""
    return name.replace("_", "-").replace("--", "-").lower()


def get_python_namespaces(sdk_path: Path) -> set[str]:
    """Get Python SDK namespace names."""
    ns_path = sdk_path / "python" / "aragora_sdk" / "namespaces"
    if not ns_path.exists():
        return set()

    namespaces = set()
    for f in ns_path.glob("*.py"):
        name = f.stem
        if name.startswith("_"):
            continue  # Skip __init__.py and private modules
        namespaces.add(normalize_name(name))
    return namespaces


def get_typescript_namespaces(sdk_path: Path) -> set[str]:
    """Get TypeScript SDK namespace names."""
    ns_path = sdk_path / "typescript" / "src" / "namespaces"
    if not ns_path.exists():
        return set()

    namespaces = set()
    for f in ns_path.glob("*.ts"):
        name = f.stem
        if name.startswith("_") or name == "index":
            continue  # Skip index.ts and private modules
        namespaces.add(normalize_name(name))
    return namespaces


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SDK namespace parity")
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Maximum allowed drift percentage (default: 5%%)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed namespace lists",
    )
    parser.add_argument(
        "--sdk-path",
        type=Path,
        default=Path(__file__).parent.parent / "sdk",
        help="Path to SDK directory",
    )
    args = parser.parse_args()

    sdk_path = args.sdk_path.resolve()

    if not sdk_path.exists():
        print(f"Error: SDK path not found: {sdk_path}")
        return 1

    python_ns = get_python_namespaces(sdk_path)
    ts_ns = get_typescript_namespaces(sdk_path)

    python_only = python_ns - ts_ns
    ts_only = ts_ns - python_ns
    common = python_ns & ts_ns

    total_unique = len(python_ns | ts_ns)
    drift_count = len(python_only) + len(ts_only)
    drift_pct = (drift_count / total_unique * 100) if total_unique > 0 else 0

    # Print results
    print("=" * 60)
    print("SDK NAMESPACE PARITY CHECK")
    print("=" * 60)
    print(f"Python namespaces:     {len(python_ns):>4}")
    print(f"TypeScript namespaces: {len(ts_ns):>4}")
    print(f"Common namespaces:     {len(common):>4}")
    print(f"Drift count:           {drift_count:>4}")
    print(f"Drift percentage:      {drift_pct:>6.1f}%")
    print(f"Threshold:             {args.threshold:>6.1f}%")
    print("=" * 60)

    if python_only:
        print(f"\nPython-only namespaces ({len(python_only)}):")
        for ns in sorted(python_only):
            print(f"  - {ns}")

    if ts_only:
        print(f"\nTypeScript-only namespaces ({len(ts_only)}):")
        for ns in sorted(ts_only):
            print(f"  - {ns}")

    if args.verbose:
        print(f"\nCommon namespaces ({len(common)}):")
        for ns in sorted(common):
            print(f"  - {ns}")

    # Determine pass/fail
    if drift_pct > args.threshold:
        print(f"\n❌ FAIL: Drift ({drift_pct:.1f}%) exceeds threshold ({args.threshold}%)")
        return 1
    else:
        print(f"\n✅ PASS: Drift ({drift_pct:.1f}%) within threshold ({args.threshold}%)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
