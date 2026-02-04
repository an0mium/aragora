#!/usr/bin/env python3
"""
Generate API reference documentation for both SDKs.

Generates:
- TypeScript SDK docs via TypeDoc (HTML + Markdown)
- Python SDK docs via pdoc (HTML)

Usage:
    python scripts/generate_sdk_docs.py
    python scripts/generate_sdk_docs.py --python-only
    python scripts/generate_sdk_docs.py --typescript-only
    python scripts/generate_sdk_docs.py --output-dir docs-site/static/sdk-api
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TS_SDK = ROOT / "sdk" / "typescript"
PY_SDK = ROOT / "sdk" / "python"
DEFAULT_OUTPUT = ROOT / "docs-site" / "static" / "sdk-api"


def run(cmd: list[str], cwd: Path, label: str) -> bool:
    """Run a command and report success/failure."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  cwd: {cwd}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"\n  FAILED: {label} (exit code {result.returncode})")
        return False
    print(f"\n  OK: {label}")
    return True


def generate_typescript_docs(output_dir: Path) -> bool:
    """Generate TypeScript SDK docs via TypeDoc."""
    ts_output = output_dir / "typescript"

    # Check if typedoc is installed
    npx = shutil.which("npx")
    if not npx:
        print("  SKIP: npx not found (install Node.js)")
        return False

    # Install dev deps if needed
    if not (TS_SDK / "node_modules" / "typedoc").exists():
        print("  Installing TypeDoc...")
        ok = run(["npm", "install", "--save-dev"], cwd=TS_SDK, label="npm install (TypeScript SDK)")
        if not ok:
            return False

    # Generate HTML docs
    ok = run(
        ["npx", "typedoc", "--out", str(ts_output)],
        cwd=TS_SDK,
        label="TypeDoc HTML generation",
    )
    if not ok:
        return False

    # Generate Markdown docs for Docusaurus integration
    ts_md_output = output_dir / "typescript-md"
    run(
        [
            "npx",
            "typedoc",
            "--plugin",
            "typedoc-plugin-markdown",
            "--out",
            str(ts_md_output),
        ],
        cwd=TS_SDK,
        label="TypeDoc Markdown generation",
    )

    return True


def generate_python_docs(output_dir: Path) -> bool:
    """Generate Python SDK docs via pdoc."""
    py_output = output_dir / "python"

    # Check if pdoc is available
    try:
        import pdoc  # noqa: F401
    except ImportError:
        print("  SKIP: pdoc not installed (pip install pdoc)")
        print("  Install with: pip install 'aragora-sdk[docs]' or pip install pdoc")
        return False

    ok = run(
        [
            sys.executable,
            "-m",
            "pdoc",
            "--html",
            "--output-dir",
            str(py_output),
            "aragora_sdk",
        ],
        cwd=PY_SDK,
        label="pdoc HTML generation",
    )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SDK API reference docs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for generated docs",
    )
    parser.add_argument("--typescript-only", action="store_true", help="Only generate TS docs")
    parser.add_argument("--python-only", action="store_true", help="Only generate Python docs")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    if not args.python_only:
        results["TypeScript"] = generate_typescript_docs(output_dir)

    if not args.typescript_only:
        results["Python"] = generate_python_docs(output_dir)

    print(f"\n{'=' * 60}")
    print("  SDK Documentation Generation Summary")
    print(f"{'=' * 60}")
    for sdk, ok in results.items():
        status = "OK" if ok else "FAILED/SKIPPED"
        print(f"  {sdk:15s} {status}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}\n")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
