"""
Dependency Vulnerability Scanner.

Scans project dependencies for known vulnerabilities by:
1. Parsing lock files (package-lock.json, requirements.txt, etc.)
2. Querying CVE databases for each dependency
3. Generating scan reports with remediation guidance

Supports:
- npm/yarn (package.json, package-lock.json, yarn.lock)
- Python (requirements.txt, Pipfile.lock, poetry.lock)
- Go (go.mod, go.sum)
- Rust (Cargo.lock)
- Ruby (Gemfile.lock)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cve_client import CVEClient
from .models import (
    DependencyInfo,
    ScanResult,
)

logger = logging.getLogger(__name__)


class DependencyScanner:
    """
    Scanner for project dependency vulnerabilities.

    Example:
        scanner = DependencyScanner()

        # Scan a repository
        result = await scanner.scan_repository("/path/to/repo")

        # Scan specific files
        result = await scanner.scan_files([
            "/path/to/package-lock.json",
            "/path/to/requirements.txt",
        ])

        # Print summary
        print(f"Found {result.critical_count} critical vulnerabilities")
    """

    def __init__(
        self,
        cve_client: Optional[CVEClient] = None,
        skip_dev_dependencies: bool = False,
        max_concurrency: int = 20,
    ):
        """
        Initialize scanner.

        Args:
            cve_client: CVE client for vulnerability queries
            skip_dev_dependencies: Skip development dependencies
            max_concurrency: Max concurrent vulnerability queries
        """
        self.cve_client = cve_client or CVEClient()
        self.skip_dev_dependencies = skip_dev_dependencies
        self.max_concurrency = max_concurrency

        # Parsers for different lock file formats
        self._parsers = {
            "package-lock.json": self._parse_npm_lock,
            "package.json": self._parse_npm_package,
            "yarn.lock": self._parse_yarn_lock,
            "requirements.txt": self._parse_requirements,
            "Pipfile.lock": self._parse_pipfile_lock,
            "poetry.lock": self._parse_poetry_lock,
            "go.mod": self._parse_go_mod,
            "go.sum": self._parse_go_sum,
            "Cargo.lock": self._parse_cargo_lock,
            "Gemfile.lock": self._parse_gemfile_lock,
        }

    async def scan_repository(
        self,
        repo_path: str,
        branch: Optional[str] = None,
        commit_sha: Optional[str] = None,
    ) -> ScanResult:
        """
        Scan a repository for vulnerable dependencies.

        Args:
            repo_path: Path to repository root
            branch: Git branch name
            commit_sha: Git commit SHA

        Returns:
            ScanResult with findings
        """
        scan_id = f"scan_{uuid.uuid4().hex[:12]}"
        result = ScanResult(
            scan_id=scan_id,
            repository=repo_path,
            branch=branch,
            commit_sha=commit_sha,
        )

        try:
            # Find lock files
            lock_files = self._find_lock_files(repo_path)
            logger.info(f"[Scanner] Found {len(lock_files)} lock files in {repo_path}")

            if not lock_files:
                result.status = "completed"
                result.completed_at = datetime.now(timezone.utc)
                return result

            # Parse dependencies from all lock files
            all_dependencies: List[DependencyInfo] = []
            for lock_file in lock_files:
                deps = await self._parse_lock_file(lock_file)
                all_dependencies.extend(deps)

            # Deduplicate by (name, version, ecosystem)
            unique_deps = {}
            for dep in all_dependencies:
                key = (dep.name, dep.version, dep.ecosystem)
                if key not in unique_deps:
                    unique_deps[key] = dep

            all_dependencies = list(unique_deps.values())
            logger.info(f"[Scanner] Found {len(all_dependencies)} unique dependencies")

            # Query vulnerabilities for each dependency
            all_dependencies = await self._query_vulnerabilities(all_dependencies)

            # Collect all vulnerabilities
            all_vulns = []
            for dep in all_dependencies:
                all_vulns.extend(dep.vulnerabilities)

            result.dependencies = all_dependencies
            result.vulnerabilities = all_vulns
            result.calculate_summary()
            result.status = "completed"
            result.completed_at = datetime.now(timezone.utc)

            logger.info(
                f"[Scanner] Scan complete: {result.vulnerable_dependencies} vulnerable packages, "
                f"{result.critical_count} critical, {result.high_count} high"
            )

        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            result.completed_at = datetime.now(timezone.utc)
            logger.exception(f"[Scanner] Scan failed: {e}")

        return result

    async def scan_files(
        self,
        file_paths: List[str],
        repository: str = "unknown",
    ) -> ScanResult:
        """
        Scan specific lock files for vulnerabilities.

        Args:
            file_paths: Paths to lock files
            repository: Repository identifier

        Returns:
            ScanResult with findings
        """
        scan_id = f"scan_{uuid.uuid4().hex[:12]}"
        result = ScanResult(
            scan_id=scan_id,
            repository=repository,
        )

        try:
            all_dependencies: List[DependencyInfo] = []
            for file_path in file_paths:
                deps = await self._parse_lock_file(file_path)
                all_dependencies.extend(deps)

            # Deduplicate
            unique_deps = {}
            for dep in all_dependencies:
                key = (dep.name, dep.version, dep.ecosystem)
                if key not in unique_deps:
                    unique_deps[key] = dep

            all_dependencies = list(unique_deps.values())

            # Query vulnerabilities
            all_dependencies = await self._query_vulnerabilities(all_dependencies)

            # Collect all vulnerabilities
            all_vulns = []
            for dep in all_dependencies:
                all_vulns.extend(dep.vulnerabilities)

            result.dependencies = all_dependencies
            result.vulnerabilities = all_vulns
            result.calculate_summary()
            result.status = "completed"
            result.completed_at = datetime.now(timezone.utc)

        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            result.completed_at = datetime.now(timezone.utc)

        return result

    def _find_lock_files(self, repo_path: str) -> List[str]:
        """Find all lock files in repository."""
        lock_files = []
        repo = Path(repo_path)

        for filename in self._parsers.keys():
            for found in repo.rglob(filename):
                # Skip node_modules and other vendor directories
                if "node_modules" in str(found):
                    continue
                if "vendor" in str(found):
                    continue
                if ".git" in str(found):
                    continue

                lock_files.append(str(found))

        return lock_files

    async def _parse_lock_file(self, file_path: str) -> List[DependencyInfo]:
        """Parse a lock file and return dependencies."""
        filename = os.path.basename(file_path)
        parser = self._parsers.get(filename)

        if not parser:
            logger.warning(f"[Scanner] No parser for {filename}")
            return []

        try:
            with open(file_path, "r") as f:
                content = f.read()

            return parser(content, file_path)
        except Exception as e:
            logger.error(f"[Scanner] Failed to parse {file_path}: {e}")
            return []

    async def _query_vulnerabilities(
        self,
        dependencies: List[DependencyInfo],
    ) -> List[DependencyInfo]:
        """Query CVE databases for vulnerabilities."""
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def query_one(dep: DependencyInfo):
            async with semaphore:
                try:
                    vulns = await self.cve_client.query_package(
                        dep.name,
                        dep.ecosystem,
                        dep.version,
                    )
                    dep.vulnerabilities = vulns
                except Exception as e:
                    logger.warning(f"[Scanner] Failed to query {dep.name}: {e}")

        await asyncio.gather(
            *[query_one(dep) for dep in dependencies],
            return_exceptions=True,
        )

        return dependencies

    # ==========================================================================
    # Lock File Parsers
    # ==========================================================================

    def _parse_npm_lock(self, content: str, file_path: str) -> List[DependencyInfo]:
        """Parse package-lock.json."""
        deps = []
        try:
            data = json.loads(content)
            lock_version = data.get("lockfileVersion", 1)

            if lock_version >= 2:
                # v2/v3 format with packages
                packages = data.get("packages", {})
                for pkg_path, pkg_info in packages.items():
                    if not pkg_path:  # Skip root
                        continue

                    name = pkg_path.split("node_modules/")[-1]
                    version = pkg_info.get("version", "")
                    dev = pkg_info.get("dev", False)

                    if self.skip_dev_dependencies and dev:
                        continue

                    deps.append(
                        DependencyInfo(
                            name=name,
                            version=version,
                            ecosystem="npm",
                            direct="node_modules/"
                            not in pkg_path.replace("node_modules/" + name, ""),
                            dev_dependency=dev,
                            license=pkg_info.get("license"),
                            file_path=file_path,
                        )
                    )
            else:
                # v1 format with dependencies
                dependencies = data.get("dependencies", {})
                self._parse_npm_deps_recursive(dependencies, deps, file_path, True)

        except json.JSONDecodeError as e:
            logger.error(f"[Scanner] Invalid JSON in {file_path}: {e}")

        return deps

    def _parse_npm_deps_recursive(
        self,
        dependencies: Dict[str, Any],
        result: List[DependencyInfo],
        file_path: str,
        direct: bool,
    ):
        """Recursively parse npm dependencies."""
        for name, info in dependencies.items():
            version = info.get("version", "")
            dev = info.get("dev", False)

            if self.skip_dev_dependencies and dev:
                continue

            result.append(
                DependencyInfo(
                    name=name,
                    version=version,
                    ecosystem="npm",
                    direct=direct,
                    dev_dependency=dev,
                    file_path=file_path,
                )
            )

            # Parse nested dependencies
            nested = info.get("dependencies", {})
            if nested:
                self._parse_npm_deps_recursive(nested, result, file_path, False)

    def _parse_npm_package(self, content: str, file_path: str) -> List[DependencyInfo]:
        """Parse package.json (for version constraints, not exact versions)."""
        deps = []
        try:
            data = json.loads(content)

            for name, version in data.get("dependencies", {}).items():
                deps.append(
                    DependencyInfo(
                        name=name,
                        version=version.lstrip("^~"),
                        ecosystem="npm",
                        direct=True,
                        dev_dependency=False,
                        file_path=file_path,
                    )
                )

            if not self.skip_dev_dependencies:
                for name, version in data.get("devDependencies", {}).items():
                    deps.append(
                        DependencyInfo(
                            name=name,
                            version=version.lstrip("^~"),
                            ecosystem="npm",
                            direct=True,
                            dev_dependency=True,
                            file_path=file_path,
                        )
                    )

        except json.JSONDecodeError:
            pass

        return deps

    def _parse_yarn_lock(self, content: str, file_path: str) -> List[DependencyInfo]:
        """Parse yarn.lock."""
        deps: List[DependencyInfo] = []

        # Parse yarn.lock format
        current_packages: List[str] = []
        current_version = ""

        for line in content.split("\n"):
            line = line.rstrip()

            if not line or line.startswith("#"):
                continue

            if not line.startswith(" ") and line.endswith(":"):
                # New package entry
                if current_packages and current_version:
                    for pkg in current_packages:
                        deps.append(
                            DependencyInfo(
                                name=pkg,
                                version=current_version,
                                ecosystem="npm",
                                direct=True,
                                file_path=file_path,
                            )
                        )

                # Parse package names
                current_packages = []
                current_version = ""
                entry = line.rstrip(":")

                # Handle multiple packages
                for part in entry.split(", "):
                    part = part.strip().strip('"')
                    if "@" in part:
                        # Handle scoped packages
                        if part.startswith("@"):
                            at_idx = part.rfind("@", 1)
                        else:
                            at_idx = part.rfind("@")
                        if at_idx > 0:
                            name = part[:at_idx]
                            current_packages.append(name)

            elif line.startswith("  version"):
                # Version line
                match = re.search(r'"([^"]+)"', line)
                if match:
                    current_version = match.group(1)

        # Don't forget last entry
        if current_packages and current_version:
            for pkg in current_packages:
                deps.append(
                    DependencyInfo(
                        name=pkg,
                        version=current_version,
                        ecosystem="npm",
                        direct=True,
                        file_path=file_path,
                    )
                )

        return deps

    def _parse_requirements(self, content: str, file_path: str) -> List[DependencyInfo]:
        """Parse requirements.txt."""
        deps = []

        for line in content.split("\n"):
            line = line.strip()

            if not line or line.startswith("#") or line.startswith("-"):
                continue

            # Handle various formats
            # package==version
            # package>=version
            # package~=version
            # package
            match = re.match(r"^([a-zA-Z0-9_-]+)\s*([=<>~!]+)?\s*([0-9.]+)?", line)
            if match:
                name = match.group(1)
                version = match.group(3) or "unknown"

                deps.append(
                    DependencyInfo(
                        name=name.lower(),
                        version=version,
                        ecosystem="pypi",
                        direct=True,
                        file_path=file_path,
                    )
                )

        return deps

    def _parse_pipfile_lock(self, content: str, file_path: str) -> List[DependencyInfo]:
        """Parse Pipfile.lock."""
        deps = []
        try:
            data = json.loads(content)

            for name, info in data.get("default", {}).items():
                version = info.get("version", "").lstrip("=")
                deps.append(
                    DependencyInfo(
                        name=name,
                        version=version,
                        ecosystem="pypi",
                        direct=True,
                        dev_dependency=False,
                        file_path=file_path,
                    )
                )

            if not self.skip_dev_dependencies:
                for name, info in data.get("develop", {}).items():
                    version = info.get("version", "").lstrip("=")
                    deps.append(
                        DependencyInfo(
                            name=name,
                            version=version,
                            ecosystem="pypi",
                            direct=True,
                            dev_dependency=True,
                            file_path=file_path,
                        )
                    )

        except json.JSONDecodeError:
            pass

        return deps

    def _parse_poetry_lock(self, content: str, file_path: str) -> List[DependencyInfo]:
        """Parse poetry.lock (TOML format)."""
        deps = []

        # Simple TOML parsing for packages
        current_name = ""
        current_version = ""
        current_category = ""

        for line in content.split("\n"):
            line = line.strip()

            if line == "[[package]]":
                if current_name and current_version:
                    is_dev = current_category == "dev"
                    if not (self.skip_dev_dependencies and is_dev):
                        deps.append(
                            DependencyInfo(
                                name=current_name,
                                version=current_version,
                                ecosystem="pypi",
                                direct=True,
                                dev_dependency=is_dev,
                                file_path=file_path,
                            )
                        )
                current_name = ""
                current_version = ""
                current_category = ""

            elif line.startswith("name = "):
                current_name = line.split("=")[1].strip().strip('"')
            elif line.startswith("version = "):
                current_version = line.split("=")[1].strip().strip('"')
            elif line.startswith("category = "):
                current_category = line.split("=")[1].strip().strip('"')

        # Last package
        if current_name and current_version:
            is_dev = current_category == "dev"
            if not (self.skip_dev_dependencies and is_dev):
                deps.append(
                    DependencyInfo(
                        name=current_name,
                        version=current_version,
                        ecosystem="pypi",
                        direct=True,
                        dev_dependency=is_dev,
                        file_path=file_path,
                    )
                )

        return deps

    def _parse_go_mod(self, content: str, file_path: str) -> List[DependencyInfo]:
        """Parse go.mod."""
        deps = []

        in_require = False
        for line in content.split("\n"):
            line = line.strip()

            if line.startswith("require ("):
                in_require = True
                continue
            elif line == ")":
                in_require = False
                continue

            if line.startswith("require ") or in_require:
                # Remove 'require ' prefix if present
                dep_line = line.replace("require ", "").strip()
                if not dep_line or dep_line.startswith("//"):
                    continue

                parts = dep_line.split()
                if len(parts) >= 2:
                    name = parts[0]
                    version = parts[1]

                    # Skip indirect dependencies if desired
                    is_indirect = "// indirect" in line

                    deps.append(
                        DependencyInfo(
                            name=name,
                            version=version.lstrip("v"),
                            ecosystem="go",
                            direct=not is_indirect,
                            file_path=file_path,
                        )
                    )

        return deps

    def _parse_go_sum(self, content: str, file_path: str) -> List[DependencyInfo]:
        """Parse go.sum (contains checksums, but also version info)."""
        deps = []
        seen = set()

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                version = parts[1].split("/")[0].lstrip("v")

                key = (name, version)
                if key not in seen:
                    seen.add(key)
                    deps.append(
                        DependencyInfo(
                            name=name,
                            version=version,
                            ecosystem="go",
                            direct=True,
                            file_path=file_path,
                        )
                    )

        return deps

    def _parse_cargo_lock(self, content: str, file_path: str) -> List[DependencyInfo]:
        """Parse Cargo.lock (TOML format)."""
        deps = []

        current_name = ""
        current_version = ""

        for line in content.split("\n"):
            line = line.strip()

            if line == "[[package]]":
                if current_name and current_version:
                    deps.append(
                        DependencyInfo(
                            name=current_name,
                            version=current_version,
                            ecosystem="cargo",
                            direct=True,
                            file_path=file_path,
                        )
                    )
                current_name = ""
                current_version = ""

            elif line.startswith("name = "):
                current_name = line.split("=")[1].strip().strip('"')
            elif line.startswith("version = "):
                current_version = line.split("=")[1].strip().strip('"')

        # Last package
        if current_name and current_version:
            deps.append(
                DependencyInfo(
                    name=current_name,
                    version=current_version,
                    ecosystem="cargo",
                    direct=True,
                    file_path=file_path,
                )
            )

        return deps

    def _parse_gemfile_lock(self, content: str, file_path: str) -> List[DependencyInfo]:
        """Parse Gemfile.lock."""
        deps = []

        in_specs = False
        for line in content.split("\n"):
            if line.strip() == "specs:":
                in_specs = True
                continue
            elif line and not line.startswith(" "):
                in_specs = False
                continue

            if in_specs and line.startswith("    ") and not line.startswith("      "):
                # Direct dependency line: "    gem_name (version)"
                match = re.match(r"\s+(\S+)\s+\(([^)]+)\)", line)
                if match:
                    name = match.group(1)
                    version = match.group(2)

                    deps.append(
                        DependencyInfo(
                            name=name,
                            version=version,
                            ecosystem="rubygems",
                            direct=True,
                            file_path=file_path,
                        )
                    )

        return deps


__all__ = ["DependencyScanner"]
