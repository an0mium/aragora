#!/usr/bin/env python3
"""
Check version alignment across all package manifests.

This script validates that all version sources in the repository are aligned.
It fails with exit code 1 if any version mismatch is detected.

Usage:
    python scripts/check_version_alignment.py
    python scripts/check_version_alignment.py --fix  # Auto-fix mismatches
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def get_canonical_version() -> str:
    """Get the canonical version from aragora/__version__.py."""
    version_file = Path("aragora/__version__.py")
    if not version_file.exists():
        raise FileNotFoundError("aragora/__version__.py not found")

    content = version_file.read_text()

    # Extract version components
    major = re.search(r"VERSION_MAJOR\s*=\s*(\d+)", content)
    minor = re.search(r"VERSION_MINOR\s*=\s*(\d+)", content)
    patch = re.search(r"VERSION_PATCH\s*=\s*(\d+)", content)

    if major is None or minor is None or patch is None:
        raise ValueError("Could not parse version from aragora/__version__.py")

    return f"{major.group(1)}.{minor.group(1)}.{patch.group(1)}"


def get_pyproject_version(path: Path) -> str | None:
    """Extract version from a pyproject.toml file."""
    if not path.exists():
        return None

    content = path.read_text()
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    return match.group(1) if match else None


def get_package_json_version(path: Path) -> str | None:
    """Extract version from a package.json file."""
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
        return data.get("version")
    except json.JSONDecodeError:
        return None


def fix_pyproject_version(path: Path, new_version: str) -> bool:
    """Update version in a pyproject.toml file."""
    if not path.exists():
        return False

    content = path.read_text()
    new_content = re.sub(
        r'^(version\s*=\s*["\'])([^"\']+)(["\'])',
        rf"\g<1>{new_version}\g<3>",
        content,
        flags=re.MULTILINE,
    )

    if new_content != content:
        path.write_text(new_content)
        return True
    return False


def fix_package_json_version(path: Path, new_version: str) -> bool:
    """Update version in a package.json file."""
    if not path.exists():
        return False

    try:
        data = json.loads(path.read_text())
        if data.get("version") != new_version:
            data["version"] = new_version
            path.write_text(json.dumps(data, indent=2) + "\n")
            return True
    except json.JSONDecodeError:
        pass
    return False


def get_canonical_release_date() -> str | None:
    """Get RELEASE_DATE from aragora/__version__.py, if declared."""
    version_file = Path("aragora/__version__.py")
    if not version_file.exists():
        return None
    match = re.search(r'RELEASE_DATE\s*=\s*["\'](\d{4}-\d{2}-\d{2})["\']', version_file.read_text())
    return match.group(1) if match else None


def _version_group(pattern: str) -> str | int:
    """Doc patterns carry the version in group 2 unless they name a ``version`` group.

    Patterns that also capture a ``date`` group cannot keep the version in
    position 2 (named groups are numbered too), so they name it instead.
    """
    return "version" if "(?P<version>" in pattern else 2


def get_doc_versions(path: Path, pattern: str) -> list[str]:
    """Extract every version string a regex pattern matches in a documentation file.

    A pattern may legitimately match several spots in one file (for example an
    image tag repeated in a compose example and an env-var example); each match
    is returned so a single stale occurrence cannot hide behind a fresh one.
    """
    if not path.exists():
        return []
    content = path.read_text()
    group = _version_group(pattern)
    return [match.group(group) for match in re.finditer(pattern, content, re.MULTILINE)]


def fix_doc_version(
    path: Path, pattern: str, new_version: str, release_date: str | None = None
) -> bool:
    """Update a version string in documentation using a regex pattern.

    When the pattern names a ``date`` group and a release date is supplied, the
    date is rewritten in the same pass so "Last Updated" style lines cannot end
    up carrying the new version next to the previous release's date.
    """
    if not path.exists():
        return False
    content = path.read_text()
    group = _version_group(pattern)

    def _rewrite(match: re.Match[str]) -> str:
        text = match.group(0)
        base = match.start()
        edits = [(match.span(group), new_version)]
        if release_date and "date" in match.re.groupindex and match.group("date"):
            edits.append((match.span("date"), release_date))
        for (start, end), value in sorted(edits, reverse=True):
            text = text[: start - base] + value + text[end - base :]
        return text

    new_content = re.sub(pattern, _rewrite, content, flags=re.MULTILINE)
    if new_content != content:
        path.write_text(new_content)
        return True
    return False


def get_python_version(path: Path) -> str | None:
    """Extract __version__ from a Python source file."""
    if not path.exists():
        return None
    content = path.read_text()
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    return match.group(1) if match else None


def fix_python_version(path: Path, new_version: str) -> bool:
    """Update __version__ in a Python source file."""
    if not path.exists():
        return False
    content = path.read_text()
    new_content = re.sub(
        r'^(__version__\s*=\s*["\'])([^"\']+)(["\'])',
        rf"\g<1>{new_version}\g<3>",
        content,
        flags=re.MULTILINE,
    )
    if new_content != content:
        path.write_text(new_content)
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Check version alignment across packages")
    parser.add_argument("--fix", action="store_true", help="Auto-fix version mismatches")
    args = parser.parse_args()

    # Get canonical version
    try:
        canonical = get_canonical_version()
        print(f"Canonical version (aragora/__version__.py): {canonical}")
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    release_date = get_canonical_release_date()

    # Define all version sources
    version_sources: list[tuple[str, Path, str]] = [
        ("pyproject.toml", Path("pyproject.toml"), "pyproject"),
        ("sdk/python/pyproject.toml", Path("sdk/python/pyproject.toml"), "pyproject"),
        ("aragora-js/package.json", Path("aragora-js/package.json"), "package"),
        ("aragora/live/package.json", Path("aragora/live/package.json"), "package"),
        ("sdk/typescript/package.json", Path("sdk/typescript/package.json"), "package"),
        ("ide/vscode-aragora/package.json", Path("ide/vscode-aragora/package.json"), "package"),
        (
            "ide/vscode-aragora/webview-ui/package.json",
            Path("ide/vscode-aragora/webview-ui/package.json"),
            "package",
        ),
    ]
    python_version_sources: list[tuple[str, Path]] = [
        ("sdk/python/aragora_sdk/__init__.py", Path("sdk/python/aragora_sdk/__init__.py")),
    ]
    doc_sources: list[tuple[str, Path, str]] = [
        (
            "ROADMAP.md",
            Path("ROADMAP.md"),
            r"^(\*\*Current Version:\*\*\s*)(\d+\.\d+\.\d+)(.*)$",
        ),
        (
            "docs/status/STATUS.md",
            Path("docs/status/STATUS.md"),
            r"^(Current released version is \*\*v?)(\d+\.\d+\.\d+)(\*\*\.)$",
        ),
        (
            "docs/guides/GETTING_STARTED.md",
            Path("docs/guides/GETTING_STARTED.md"),
            r"^(\s*aragora:\s*)(\d+\.\d+\.\d+)(.*)$",
        ),
        (
            "docs/deployment/SCALING.md",
            Path("docs/deployment/SCALING.md"),
            r'(\s*"version":\s*")(\d+\.\d+\.\d+)(",)',
        ),
        (
            "docs/api/API_REFERENCE.md",
            Path("docs/api/API_REFERENCE.md"),
            r"^(\|\s*TypeScript\s*\([^)]+\)\s*\|\s*)(\d+\.\d+\.\d+)(\s*\|.*)$",
        ),
        (
            "docs/api/API_REFERENCE.md (Python SDK)",
            Path("docs/api/API_REFERENCE.md"),
            r"^(\|\s*Python\s*\([^)]+\)\s*\|\s*)(\d+\.\d+\.\d+)(\s*\|.*)$",
        ),
        (
            "docs/CANONICAL_GOALS.md",
            Path("docs/CANONICAL_GOALS.md"),
            r"^(\|\s*Version\s*\|\s*)(\d+\.\d+\.\d+)(\s*\|.*)$",
        ),
        (
            "docs/DEPLOYMENT.md",
            Path("docs/DEPLOYMENT.md"),
            r"(`)(\d+\.\d+\.\d+)(` \(version from pyproject\.toml\))",
        ),
        (
            "docs/DEPLOYMENT.md (git tag)",
            Path("docs/DEPLOYMENT.md"),
            r"(`v)(\d+\.\d+\.\d+)(` \(git tag\))",
        ),
        (
            "docs/DEPLOYMENT.md (image pin)",
            Path("docs/DEPLOYMENT.md"),
            r"(ghcr\.io/synaptent/aragora/backend:)(\d+\.\d+\.\d+)(\b)",
        ),
        (
            "docs/deployment/GO_LIVE_CHECKLIST.md",
            Path("docs/deployment/GO_LIVE_CHECKLIST.md"),
            r"(ghcr\.io/synaptent/aragora/(?:backend|frontend):)(\d+\.\d+\.\d+)(\b)",
        ),
        (
            "docs/reference/INSTALL_MATRIX.md",
            Path("docs/reference/INSTALL_MATRIX.md"),
            r"^(\|\s*Root platform\s*\|[^|]*\|[^|]*\|\s*\*\*)(\d+\.\d+\.\d+)(\*\*\s*\|.*)$",
        ),
        (
            "docs/reference/INSTALL_MATRIX.md (Python SDK)",
            Path("docs/reference/INSTALL_MATRIX.md"),
            r"^(\|\s*Python SDK\s*\|[^|]*\|[^|]*\|\s*\*\*)(\d+\.\d+\.\d+)(\*\*\s*\|.*)$",
        ),
        (
            "docs/reference/INSTALL_MATRIX.md (checkout)",
            Path("docs/reference/INSTALL_MATRIX.md"),
            r"^(\|\s*This checkout \(`pip install \./sdk/python`\)\s*\|\s*)(\d+\.\d+\.\d+)(\s*\|.*)$",
        ),
        (
            "docs/api/API_REFERENCE.md (last updated)",
            Path("docs/api/API_REFERENCE.md"),
            r"^(> \*\*Last Updated:\*\* )(?P<date>\d{4}-\d{2}-\d{2})( \(v)(?P<version>\d+\.\d+\.\d+)"
            r"( alignment with repo versions\))$",
        ),
        (
            "docs/STATUS.md",
            Path("docs/STATUS.md"),
            r"^(Current released version is \*\*v?)(\d+\.\d+\.\d+)(\*\*\.)$",
        ),
        (
            "docs/STATUS.md (version bullet)",
            Path("docs/STATUS.md"),
            r"^(- \*\*Version\*\*: v)(\d+\.\d+\.\d+)()$",
        ),
        (
            "docs/status/STATUS.md (version bullet)",
            Path("docs/status/STATUS.md"),
            r"^(- \*\*Version\*\*: v)(\d+\.\d+\.\d+)()$",
        ),
        (
            "docs/guides/SELF_HOSTED_QUICKSTART.md",
            Path("docs/guides/SELF_HOSTED_QUICKSTART.md"),
            r"^(\*Updated: )(?P<date>\d{4}-\d{2}-\d{2})(\*\n\*Version: )(?P<version>\d+\.\d+\.\d+)(\*)$",
        ),
        (
            "docs/guides/SELF_HOSTED_QUICKSTART.md (health response)",
            Path("docs/guides/SELF_HOSTED_QUICKSTART.md"),
            r'(\{"status": "healthy", "version": ")(\d+\.\d+\.\d+)("\})',
        ),
        (
            "docs/guides/SELF_HOSTED_COMPLETE_GUIDE.md",
            Path("docs/guides/SELF_HOSTED_COMPLETE_GUIDE.md"),
            r"^(\*Version: )(?P<version>\d+\.\d+\.\d+)( \| Updated: )(?P<date>\d{4}-\d{2}-\d{2})(\*)$",
        ),
        (
            "docs/guides/SELF_HOSTED_COMPLETE_GUIDE.md (header)",
            Path("docs/guides/SELF_HOSTED_COMPLETE_GUIDE.md"),
            r"^(\*\*Version:\*\* )(?P<version>\d+\.\d+\.\d+)(\n\*\*Last Updated:\*\* )(?P<date>\d{4}-\d{2}-\d{2})()$",
        ),
        (
            "docs/guides/SELF_HOSTED_COMPLETE_GUIDE.md (health response)",
            Path("docs/guides/SELF_HOSTED_COMPLETE_GUIDE.md"),
            r'^(\s*"version": ")(\d+\.\d+\.\d+)(",)$',
        ),
        (
            "docs-site/docs/getting-started/overview.md",
            Path("docs-site/docs/getting-started/overview.md"),
            r"^(\s*aragora:\s*)(\d+\.\d+\.\d+)(.*)$",
        ),
        (
            "docs-site/docs/deployment/scaling.md",
            Path("docs-site/docs/deployment/scaling.md"),
            r'(\s*"version":\s*")(\d+\.\d+\.\d+)(",)',
        ),
        (
            "docs-site/docs/api/reference.md",
            Path("docs-site/docs/api/reference.md"),
            r"^(\|\s*TypeScript\s*\([^)]+\)\s*\|\s*)(\d+\.\d+\.\d+)(\s*\|.*)$",
        ),
        (
            "docs-site/docs/api/reference.md (Python SDK)",
            Path("docs-site/docs/api/reference.md"),
            r"^(\|\s*Python\s*\([^)]+\)\s*\|\s*)(\d+\.\d+\.\d+)(\s*\|.*)$",
        ),
        (
            "docs-site/docs/deployment/overview.md",
            Path("docs-site/docs/deployment/overview.md"),
            r"(`)(\d+\.\d+\.\d+)(` \(version from pyproject\.toml\))",
        ),
        (
            "docs-site/docs/deployment/overview.md (git tag)",
            Path("docs-site/docs/deployment/overview.md"),
            r"(`v)(\d+\.\d+\.\d+)(` \(git tag\))",
        ),
        (
            "docs-site/docs/deployment/overview.md (image pin)",
            Path("docs-site/docs/deployment/overview.md"),
            r"(ghcr\.io/synaptent/aragora/backend:)(\d+\.\d+\.\d+)(\b)",
        ),
        (
            "docs-site/docs/reference/install-matrix.md",
            Path("docs-site/docs/reference/install-matrix.md"),
            r"^(\|\s*Root platform\s*\|[^|]*\|[^|]*\|\s*\*\*)(\d+\.\d+\.\d+)(\*\*\s*\|.*)$",
        ),
        (
            "docs-site/docs/reference/install-matrix.md (Python SDK)",
            Path("docs-site/docs/reference/install-matrix.md"),
            r"^(\|\s*Python SDK\s*\|[^|]*\|[^|]*\|\s*\*\*)(\d+\.\d+\.\d+)(\*\*\s*\|.*)$",
        ),
        (
            "docs-site/docs/reference/install-matrix.md (checkout)",
            Path("docs-site/docs/reference/install-matrix.md"),
            r"^(\|\s*This checkout \(`pip install \./sdk/python`\)\s*\|\s*)(\d+\.\d+\.\d+)(\s*\|.*)$",
        ),
        (
            "docs-site/docs/api/reference.md (last updated)",
            Path("docs-site/docs/api/reference.md"),
            r"^(> \*\*Last Updated:\*\* )(?P<date>\d{4}-\d{2}-\d{2})( \(v)(?P<version>\d+\.\d+\.\d+)"
            r"( alignment with repo versions\))$",
        ),
        (
            "docs-site/docs/contributing/status.md",
            Path("docs-site/docs/contributing/status.md"),
            r"^(Current released version is \*\*v?)(\d+\.\d+\.\d+)(\*\*\.)$",
        ),
        (
            "docs-site/docs/contributing/status.md (version bullet)",
            Path("docs-site/docs/contributing/status.md"),
            r"^(- \*\*Version\*\*: v)(\d+\.\d+\.\d+)()$",
        ),
        (
            "docs/migration/V3_MIGRATION_GUIDE.md",
            Path("docs/migration/V3_MIGRATION_GUIDE.md"),
            r"^(> \*\*Current version:\*\* v)(\d+\.\d+\.\d+)()$",
        ),
        (
            "docs/deployment/UPGRADE_ROADMAP.md",
            Path("docs/deployment/UPGRADE_ROADMAP.md"),
            r"^(\*\*Aragora v)(\d+\.\d+\.\d+)(\*\* \(released \d{4}-\d{2}-\d{2}\))$",
        ),
        (
            "docs/deployment/UPGRADE_ROADMAP.md (version check)",
            Path("docs/deployment/UPGRADE_ROADMAP.md"),
            r'^(print\(__version__\)\s*# ")(\d+\.\d+\.\d+)(")$',
        ),
        (
            "docs/guides/SELF_HOSTED_COMPLETE_GUIDE.md (image pin)",
            Path("docs/guides/SELF_HOSTED_COMPLETE_GUIDE.md"),
            r"(ghcr\.io/synaptent/aragora/backend:)(\d+\.\d+\.\d+)(\b)",
        ),
    ]

    mismatches: list[tuple[str, str | None]] = []
    fixed: list[str] = []

    print("\nChecking version alignment:")
    print("-" * 50)

    for name, path, file_type in version_sources:
        if file_type == "pyproject":
            version = get_pyproject_version(path)
        else:
            version = get_package_json_version(path)

        if version is None:
            print(f"  {name}: (not found)")
            continue

        status = "OK" if version == canonical else "MISMATCH"
        print(f"  {name}: {version} [{status}]")

        if version != canonical:
            mismatches.append((name, version))

            if args.fix:
                if file_type == "pyproject":
                    if fix_pyproject_version(path, canonical):
                        fixed.append(name)
                else:
                    if fix_package_json_version(path, canonical):
                        fixed.append(name)

    for name, path in python_version_sources:
        version = get_python_version(path)
        if version is None:
            print(f"  {name}: (not found)")
            continue

        status = "OK" if version == canonical else "MISMATCH"
        print(f"  {name}: {version} [__version__] [{status}]")

        if version != canonical:
            mismatches.append((name, version))

            if args.fix:
                if fix_python_version(path, canonical):
                    fixed.append(name)

    for name, path, pattern in doc_sources:
        versions = get_doc_versions(path, pattern)
        if not versions:
            print(f"  {name}: (version not found)")
            continue

        stale = [v for v in versions if v != canonical]
        version = stale[0] if stale else versions[0]
        status = "OK" if not stale else "MISMATCH"
        suffix = f" ({len(versions)} occurrences)" if len(versions) > 1 else ""
        print(f"  {name}: {version} [doc] [{status}]{suffix}")

        if stale:
            mismatches.append((name, version))

            if args.fix:
                if fix_doc_version(path, pattern, canonical, release_date):
                    fixed.append(name)

    print("-" * 50)

    if fixed:
        print(f"\nFixed {len(fixed)} file(s):")
        for name in fixed:
            print(f"  - {name} -> {canonical}")

    if mismatches and not args.fix:
        print(f"\nERROR: {len(mismatches)} version mismatch(es) found!")
        print("Run with --fix to auto-fix, or manually update the files.")
        return 1

    if mismatches and fixed:
        remaining = len(mismatches) - len(fixed)
        if remaining > 0:
            print(f"\nWARNING: {remaining} mismatch(es) could not be fixed.")
            return 1

    print("\nAll versions aligned!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
