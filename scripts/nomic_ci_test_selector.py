#!/usr/bin/env python3
"""Select and run tests relevant to changed files for nomic CI.

Maps changed source files to their corresponding test files using
the same logic as AutonomousOrchestrator._infer_test_paths().

Usage:
    python scripts/nomic_ci_test_selector.py --changed-files aragora/foo/bar.py aragora/baz/qux.py --run
    python scripts/nomic_ci_test_selector.py --changed-files aragora/foo/bar.py --dry-run
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def infer_test_paths(changed_files: list[str]) -> list[str]:
    """Map source files to test files."""
    test_paths = []
    for path in changed_files:
        if not path.strip():
            continue
        if path.startswith("tests/"):
            test_paths.append(path)
            continue
        if path.startswith("aragora/"):
            rel = path[len("aragora/"):]
            parts = rel.rsplit("/", 1)
            if len(parts) == 2:
                directory, filename = parts
                if filename.endswith(".py"):
                    test_file = f"tests/{directory}/test_{filename}"
                    if Path(test_file).exists():
                        test_paths.append(test_file)
            elif len(parts) == 1 and parts[0].endswith(".py"):
                test_file = f"tests/test_{parts[0]}"
                if Path(test_file).exists():
                    test_paths.append(test_file)
    # Deduplicate
    return list(dict.fromkeys(test_paths))


def main():
    parser = argparse.ArgumentParser(description="Nomic CI test selector")
    parser.add_argument("--changed-files", nargs="*", default=[])
    parser.add_argument("--run", action="store_true", help="Run the selected tests")
    parser.add_argument("--dry-run", action="store_true", help="Print tests without running")
    args = parser.parse_args()

    test_paths = infer_test_paths(args.changed_files)

    result = {
        "changed_files": args.changed_files,
        "test_paths": test_paths,
        "test_count": len(test_paths),
    }

    if not test_paths:
        print("No matching test files found for changed files")
        result["status"] = "skipped"
        Path(".nomic-ci-result.json").write_text(json.dumps(result, indent=2))
        return 0

    print(f"Selected {len(test_paths)} test files for {len(args.changed_files)} changed files:")
    for tp in test_paths:
        print(f"  {tp}")

    if args.dry_run:
        result["status"] = "dry_run"
        Path(".nomic-ci-result.json").write_text(json.dumps(result, indent=2))
        return 0

    if args.run:
        cmd = [
            sys.executable, "-m", "pytest",
            *test_paths,
            "--timeout=120", "-v", "--tb=short",
            "--junit-xml=.nomic-ci-junit.xml",
        ]
        proc = subprocess.run(cmd)
        result["status"] = "passed" if proc.returncode == 0 else "failed"
        result["exit_code"] = proc.returncode
        Path(".nomic-ci-result.json").write_text(json.dumps(result, indent=2))
        return proc.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
