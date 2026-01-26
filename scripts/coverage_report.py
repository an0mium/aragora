#!/usr/bin/env python3
"""
Module-level coverage reporting for Aragora.

Generates coverage reports broken down by module to identify
gaps in test coverage for critical paths.

Usage:
    python scripts/coverage_report.py
    python scripts/coverage_report.py --json
    python scripts/coverage_report.py --critical-only
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


# Critical modules that MUST have high coverage (80%+)
CRITICAL_MODULES = [
    "aragora/debate",
    "aragora/consensus",
    "aragora/auth",
    "aragora/rbac",
    "aragora/verification",
    "aragora/server/handlers",
    "aragora/billing",
]

# Target thresholds by tier
COVERAGE_THRESHOLDS = {
    "tier1_critical": 80,  # debate, consensus, auth, rbac, verification
    "tier2_important": 70,  # server, billing, knowledge
    "tier3_standard": 50,  # other modules
}


def run_coverage() -> dict:
    """Run pytest with coverage and return results."""
    cmd = [
        "python",
        "-m",
        "pytest",
        "tests/",
        "-m",
        "not slow and not load and not e2e",
        "--cov=aragora",
        "--cov-report=json:coverage.json",
        "--cov-report=term-missing",
        "-q",
        "--timeout=120",
    ]

    print("Running coverage analysis...")
    subprocess.run(cmd, capture_output=True, text=True)

    # Load coverage data
    coverage_file = Path("coverage.json")
    if not coverage_file.exists():
        print("Error: coverage.json not found. Run pytest with --cov first.")
        sys.exit(1)

    with open(coverage_file) as f:
        return json.load(f)


def analyze_coverage(coverage_data: dict) -> dict:
    """Analyze coverage by module."""
    files = coverage_data.get("files", {})

    module_coverage = {}
    for filepath, data in files.items():
        if not filepath.startswith("aragora/"):
            continue

        # Extract module name (first two path components)
        parts = filepath.split("/")
        if len(parts) >= 2:
            module = "/".join(parts[:2])
        else:
            module = parts[0]

        if module not in module_coverage:
            module_coverage[module] = {
                "files": 0,
                "covered_lines": 0,
                "total_lines": 0,
                "missing_lines": [],
            }

        summary = data.get("summary", {})
        module_coverage[module]["files"] += 1
        module_coverage[module]["covered_lines"] += summary.get("covered_lines", 0)
        module_coverage[module]["total_lines"] += summary.get("num_statements", 0)

        # Track files with low coverage
        pct = summary.get("percent_covered", 0)
        if pct < 50:
            module_coverage[module]["missing_lines"].append(
                {
                    "file": filepath,
                    "coverage": pct,
                    "missing": summary.get("missing_lines", 0),
                }
            )

    # Calculate percentages
    for module, data in module_coverage.items():
        if data["total_lines"] > 0:
            data["percent"] = round((data["covered_lines"] / data["total_lines"]) * 100, 1)
        else:
            data["percent"] = 0

        # Determine tier
        if any(module.startswith(cm) for cm in CRITICAL_MODULES[:5]):
            data["tier"] = "tier1_critical"
            data["threshold"] = COVERAGE_THRESHOLDS["tier1_critical"]
        elif any(module.startswith(cm) for cm in CRITICAL_MODULES[5:]):
            data["tier"] = "tier2_important"
            data["threshold"] = COVERAGE_THRESHOLDS["tier2_important"]
        else:
            data["tier"] = "tier3_standard"
            data["threshold"] = COVERAGE_THRESHOLDS["tier3_standard"]

        data["passing"] = data["percent"] >= data["threshold"]

    return module_coverage


def print_report(module_coverage: dict, critical_only: bool = False) -> bool:
    """Print coverage report and return True if all thresholds met."""
    all_passing = True

    # Sort by coverage percentage
    sorted_modules = sorted(module_coverage.items(), key=lambda x: (x[1]["tier"], -x[1]["percent"]))

    print("\n" + "=" * 70)
    print("ARAGORA MODULE COVERAGE REPORT")
    print("=" * 70)

    current_tier = None
    for module, data in sorted_modules:
        if critical_only and data["tier"] == "tier3_standard":
            continue

        if data["tier"] != current_tier:
            current_tier = data["tier"]
            tier_label = {
                "tier1_critical": "TIER 1: CRITICAL (80% required)",
                "tier2_important": "TIER 2: IMPORTANT (70% required)",
                "tier3_standard": "TIER 3: STANDARD (50% required)",
            }[current_tier]
            print(f"\n{tier_label}")
            print("-" * 70)

        status = "PASS" if data["passing"] else "FAIL"
        status_icon = "\u2705" if data["passing"] else "\u274c"

        print(
            f"{status_icon} {module:40} "
            f"{data['percent']:5.1f}% ({data['covered_lines']}/{data['total_lines']}) "
            f"[{status}]"
        )

        if not data["passing"]:
            all_passing = False
            # Show worst files
            for bad_file in sorted(data["missing_lines"], key=lambda x: x["coverage"])[:3]:
                print(f"     \u2514\u2500 {bad_file['file']}: {bad_file['coverage']:.1f}%")

    print("\n" + "=" * 70)

    # Summary
    total_modules = len(module_coverage)
    passing_modules = sum(1 for m in module_coverage.values() if m["passing"])
    critical_failing = [
        m for m, d in module_coverage.items() if d["tier"] == "tier1_critical" and not d["passing"]
    ]

    print(f"SUMMARY: {passing_modules}/{total_modules} modules meeting threshold")
    if critical_failing:
        print(f"CRITICAL FAILURES: {', '.join(critical_failing)}")
        print("\nAction required: Increase test coverage for critical modules.")

    return all_passing


def main():
    parser = argparse.ArgumentParser(description="Generate module-level coverage report")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--critical-only", action="store_true", help="Only show critical modules")
    parser.add_argument("--no-run", action="store_true", help="Use existing coverage.json")
    args = parser.parse_args()

    if args.no_run:
        coverage_file = Path("coverage.json")
        if not coverage_file.exists():
            print("Error: coverage.json not found. Run without --no-run first.")
            sys.exit(1)
        with open(coverage_file) as f:
            coverage_data = json.load(f)
    else:
        coverage_data = run_coverage()

    module_coverage = analyze_coverage(coverage_data)

    if args.json:
        print(json.dumps(module_coverage, indent=2))
    else:
        all_passing = print_report(module_coverage, args.critical_only)
        sys.exit(0 if all_passing else 1)


if __name__ == "__main__":
    main()
