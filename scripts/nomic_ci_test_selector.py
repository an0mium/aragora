#!/usr/bin/env python3
"""Select and run tests relevant to changed files for nomic CI.

Maps changed source files to their corresponding test files.

Usage:
    python scripts/nomic_ci_test_selector.py --changed-files aragora/foo/bar.py aragora/baz/qux.py --run
    python scripts/nomic_ci_test_selector.py --changed-files aragora/foo/bar.py --dry-run
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Auto-generated from the three P1 tests-migration commits (PRs #8387, #8404,
# #8415).  Maps the pre-migration root test path to its post-migration
# subdirectory home.  Used to resolve a top-level ``aragora/<x>.py`` whose
# legacy ``tests/test_<x>.py`` was relocated.
_MIGRATED_TEST_MAP = {  # {old_root_path: new_subdir_path}
    "tests/test_agent_anthropic.py": "tests/agents/test_agent_anthropic.py",
    "tests/test_agent_api_common.py": "tests/agents/test_agent_api_common.py",
    "tests/test_agent_error_classifier.py": "tests/agents/test_agent_error_classifier.py",
    "tests/test_agent_error_decorators.py": "tests/agents/test_agent_error_decorators.py",
    "tests/test_agent_fallback_integration.py": "tests/agents/test_agent_fallback_integration.py",
    "tests/test_agent_gemini.py": "tests/agents/test_agent_gemini.py",
    "tests/test_agent_grok.py": "tests/agents/test_agent_grok.py",
    "tests/test_agent_laboratory.py": "tests/agents/test_agent_laboratory.py",
    "tests/test_agent_lm_studio.py": "tests/agents/test_agent_lm_studio.py",
    "tests/test_agent_local.py": "tests/agents/test_agent_local.py",
    "tests/test_agent_mistral.py": "tests/agents/test_agent_mistral.py",
    "tests/test_agent_ollama.py": "tests/agents/test_agent_ollama.py",
    "tests/test_agent_openai.py": "tests/agents/test_agent_openai.py",
    "tests/test_agent_openrouter.py": "tests/agents/test_agent_openrouter.py",
    "tests/test_agent_performance_monitor.py": "tests/agents/test_agent_performance_monitor.py",
    "tests/test_agent_positions.py": "tests/agents/test_agent_positions.py",
    "tests/test_agents_airlock.py": "tests/agents/test_agents_airlock.py",
    "tests/test_agents_telemetry.py": "tests/agents/test_agents_telemetry.py",
    "tests/test_agents_unit.py": "tests/agents/test_agents_unit.py",
    "tests/test_api_agent_base.py": "tests/agents/test_api_agent_base.py",
    "tests/test_api_agents.py": "tests/agents/test_api_agents.py",
    "tests/test_api_agents_extended.py": "tests/agents/test_api_agents_extended.py",
    "tests/test_api_agents_rate_limiter.py": "tests/agents/test_api_agents_rate_limiter.py",
    "tests/test_cli_agents_sanitization.py": "tests/agents/test_cli_agents_sanitization.py",
    "tests/test_cli_fallback.py": "tests/agents/test_cli_fallback.py",
    "tests/test_error_classification.py": "tests/agents/test_error_classification.py",
    "tests/test_error_classifier.py": "tests/agents/test_error_classifier.py",
    "tests/test_exceptions.py": "tests/agents/test_exceptions.py",
    "tests/test_fallback_chain.py": "tests/agents/test_fallback_chain.py",
    "tests/test_moment_detector.py": "tests/agents/test_moment_detector.py",
    "tests/test_tournament.py": "tests/ranking/test_tournament.py",
}


def _relocated_test_path(old_root_test: str) -> str | None:
    """Return the post-migration path for a relocated root test, or None."""
    return _MIGRATED_TEST_MAP.get(old_root_test)


def _repo_path_exists(repo_relative_path: str) -> bool:
    """Return whether a repo-relative path exists, independent of caller cwd."""
    return (REPO_ROOT / repo_relative_path).exists()


def infer_test_paths(changed_files: list[str]) -> list[str]:
    """Map source files to test files.

    For a subdirectory file ``aragora/<module>/<file>.py`` the primary
    mapping is ``tests/<module>/test_<file>.py``; the ``_root``-suffixed
    variant ``tests/<module>/test_<file>_root.py`` (batch-3 convention)
    is also probed when it exists.

    For a top-level ``aragora/<x>.py`` the legacy root mapping
    ``tests/test_<x>.py`` is tried first.  When that file was relocated
    by the P1 tests-migration the selector follows the rename via a
    pre-computed migration map to the new subdirectory home.
    """
    test_paths = []
    for path in changed_files:
        if not path.strip():
            continue
        if path.startswith("tests/"):
            test_paths.append(path)
            continue
        if path.startswith("aragora/"):
            rel = path[len("aragora/") :]
            parts = rel.rsplit("/", 1)
            if len(parts) == 2:
                directory, filename = parts
                if filename.endswith(".py"):
                    basename = filename[:-3]  # strip ".py"
                    # Primary subdirectory mapping
                    test_file = f"tests/{directory}/test_{filename}"
                    if _repo_path_exists(test_file):
                        test_paths.append(test_file)
                    # _root-suffixed variant (batch 3 convention)
                    root_test = f"tests/{directory}/test_{basename}_root.py"
                    if _repo_path_exists(root_test):
                        test_paths.append(root_test)
            elif len(parts) == 1 and parts[0].endswith(".py"):
                # Legacy root path
                test_file = f"tests/test_{parts[0]}"
                if _repo_path_exists(test_file):
                    test_paths.append(test_file)
                # Check the migration map for relocated path.  Include this
                # even if a legacy root stub still exists so stale root files
                # cannot hide the real post-migration test.
                relocated = _relocated_test_path(test_file)
                if relocated and _repo_path_exists(relocated):
                    test_paths.append(relocated)
    # Deduplicate
    return list(dict.fromkeys(test_paths))


def changed_python_files(changed_files: list[str]) -> list[str]:
    """Return changed Aragora Python source files relevant to PR-scoped coverage."""
    return [
        path
        for path in changed_files
        if path.strip() and path.startswith("aragora/") and path.endswith(".py")
    ]


def main():
    parser = argparse.ArgumentParser(description="Nomic CI test selector")
    parser.add_argument("--changed-files", nargs="*", default=[])
    parser.add_argument("--run", action="store_true", help="Run the selected tests")
    parser.add_argument("--dry-run", action="store_true", help="Print tests without running")
    args = parser.parse_args()

    test_paths = infer_test_paths(args.changed_files)
    python_files = changed_python_files(args.changed_files)

    result = {
        "changed_files": args.changed_files,
        "changed_python_files": python_files,
        "test_paths": test_paths,
        "test_count": len(test_paths),
    }

    if not test_paths:
        if python_files:
            print("No mapped test files found for changed Python files")
            for path in python_files:
                print(f"::error::untested new Python module: {path}")
            result["status"] = "unmapped_python_changes"
            result["exit_code"] = 1
            Path(".nomic-ci-result.json").write_text(json.dumps(result, indent=2))
            return 1
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
            sys.executable,
            "-m",
            "pytest",
            *test_paths,
            "--timeout=120",
            "-v",
            "--tb=short",
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
