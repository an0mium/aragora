"""
Tests for Dependency Vulnerability Scanner module.

Tests dependency scanning: lock file parsing (npm, yarn, pip, poetry, go, cargo, ruby),
CVE client integration, repository scanning, deduplication, and error handling.
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.analysis.codebase.models import (
    DependencyInfo,
    ScanResult,
    VulnerabilityFinding,
    VulnerabilitySeverity,
)
from aragora.analysis.codebase.scanner import DependencyScanner


# ============================================================
# DependencyScanner Initialization
# ============================================================


class TestDependencyScannerInit:
    """Tests for DependencyScanner initialization."""

    def test_default_initialization(self):
        """Scanner initializes with default settings."""
        scanner = DependencyScanner()
        assert scanner.skip_dev_dependencies is False
        assert scanner.max_concurrency == 20
        assert scanner.cve_client is not None

    def test_custom_cve_client(self):
        """Scanner accepts custom CVE client."""
        mock_client = MagicMock()
        scanner = DependencyScanner(cve_client=mock_client)
        assert scanner.cve_client is mock_client

    def test_skip_dev_dependencies(self):
        """Scanner can be configured to skip dev dependencies."""
        scanner = DependencyScanner(skip_dev_dependencies=True)
        assert scanner.skip_dev_dependencies is True

    def test_custom_max_concurrency(self):
        """Scanner accepts custom max concurrency."""
        scanner = DependencyScanner(max_concurrency=5)
        assert scanner.max_concurrency == 5

    def test_parsers_registered(self):
        """All expected lock file parsers are registered."""
        scanner = DependencyScanner()
        expected_parsers = [
            "package-lock.json",
            "package.json",
            "yarn.lock",
            "requirements.txt",
            "Pipfile.lock",
            "poetry.lock",
            "pyproject.toml",
            "go.mod",
            "go.sum",
            "Cargo.lock",
            "Gemfile.lock",
        ]
        for parser_name in expected_parsers:
            assert parser_name in scanner._parsers, f"Missing parser for {parser_name}"


# ============================================================
# NPM package-lock.json Parsing (v1 format)
# ============================================================


class TestNPMLockV1Parsing:
    """Tests for package-lock.json v1 format parsing."""

    def test_parse_npm_lock_v1_basic(self):
        """Parse basic v1 package-lock.json with dependencies."""
        scanner = DependencyScanner()
        content = json.dumps(
            {
                "lockfileVersion": 1,
                "dependencies": {
                    "lodash": {"version": "4.17.21", "dev": False},
                    "express": {"version": "4.18.2", "dev": False},
                },
            }
        )
        deps = scanner._parse_npm_lock(content, "/path/to/package-lock.json")
        assert len(deps) == 2
        names = {d.name for d in deps}
        assert "lodash" in names
        assert "express" in names

    def test_parse_npm_lock_v1_with_dev(self):
        """Parse v1 with dev dependencies."""
        scanner = DependencyScanner()
        content = json.dumps(
            {
                "lockfileVersion": 1,
                "dependencies": {
                    "lodash": {"version": "4.17.21", "dev": False},
                    "jest": {"version": "29.0.0", "dev": True},
                },
            }
        )
        deps = scanner._parse_npm_lock(content, "/path/to/package-lock.json")
        assert len(deps) == 2
        dev_deps = [d for d in deps if d.dev_dependency]
        assert len(dev_deps) == 1
        assert dev_deps[0].name == "jest"

    def test_parse_npm_lock_v1_skip_dev(self):
        """Skip dev dependencies when configured."""
        scanner = DependencyScanner(skip_dev_dependencies=True)
        content = json.dumps(
            {
                "lockfileVersion": 1,
                "dependencies": {
                    "lodash": {"version": "4.17.21", "dev": False},
                    "jest": {"version": "29.0.0", "dev": True},
                },
            }
        )
        deps = scanner._parse_npm_lock(content, "/path/to/package-lock.json")
        assert len(deps) == 1
        assert deps[0].name == "lodash"

    def test_parse_npm_lock_v1_nested_dependencies(self):
        """Parse v1 with nested dependencies."""
        scanner = DependencyScanner()
        content = json.dumps(
            {
                "lockfileVersion": 1,
                "dependencies": {
                    "express": {
                        "version": "4.18.2",
                        "dependencies": {
                            "body-parser": {"version": "1.20.0"},
                        },
                    },
                },
            }
        )
        deps = scanner._parse_npm_lock(content, "/path/to/package-lock.json")
        assert len(deps) == 2
        names = {d.name for d in deps}
        assert "express" in names
        assert "body-parser" in names

    def test_parse_npm_lock_v1_direct_vs_transitive(self):
        """Distinguish direct from transitive dependencies."""
        scanner = DependencyScanner()
        content = json.dumps(
            {
                "lockfileVersion": 1,
                "dependencies": {
                    "express": {
                        "version": "4.18.2",
                        "dependencies": {
                            "body-parser": {"version": "1.20.0"},
                        },
                    },
                },
            }
        )
        deps = scanner._parse_npm_lock(content, "/path/to/package-lock.json")
        express_dep = next(d for d in deps if d.name == "express")
        body_parser_dep = next(d for d in deps if d.name == "body-parser")
        assert express_dep.direct is True
        assert body_parser_dep.direct is False


# ============================================================
# NPM package-lock.json Parsing (v2/v3 format)
# ============================================================


class TestNPMLockV2Parsing:
    """Tests for package-lock.json v2/v3 format parsing."""

    def test_parse_npm_lock_v2_basic(self):
        """Parse basic v2 package-lock.json."""
        scanner = DependencyScanner()
        content = json.dumps(
            {
                "lockfileVersion": 2,
                "packages": {
                    "": {"name": "myapp", "version": "1.0.0"},
                    "node_modules/lodash": {"version": "4.17.21"},
                    "node_modules/express": {"version": "4.18.2"},
                },
            }
        )
        deps = scanner._parse_npm_lock(content, "/path/to/package-lock.json")
        # Should skip root package ""
        assert len(deps) == 2
        names = {d.name for d in deps}
        assert "lodash" in names
        assert "express" in names

    def test_parse_npm_lock_v3_format(self):
        """Parse v3 package-lock.json."""
        scanner = DependencyScanner()
        content = json.dumps(
            {
                "lockfileVersion": 3,
                "packages": {
                    "": {"name": "myapp"},
                    "node_modules/axios": {"version": "1.4.0"},
                },
            }
        )
        deps = scanner._parse_npm_lock(content, "/path/to/package-lock.json")
        assert len(deps) == 1
        assert deps[0].name == "axios"
        assert deps[0].version == "1.4.0"
        assert deps[0].ecosystem == "npm"

    def test_parse_npm_lock_v2_with_dev(self):
        """Parse v2 with dev dependencies."""
        scanner = DependencyScanner()
        content = json.dumps(
            {
                "lockfileVersion": 2,
                "packages": {
                    "": {"name": "myapp"},
                    "node_modules/lodash": {"version": "4.17.21", "dev": False},
                    "node_modules/jest": {"version": "29.0.0", "dev": True},
                },
            }
        )
        deps = scanner._parse_npm_lock(content, "/path/to/package-lock.json")
        jest_dep = next(d for d in deps if d.name == "jest")
        assert jest_dep.dev_dependency is True

    def test_parse_npm_lock_v2_with_license(self):
        """Parse v2 with license information."""
        scanner = DependencyScanner()
        content = json.dumps(
            {
                "lockfileVersion": 2,
                "packages": {
                    "": {"name": "myapp"},
                    "node_modules/lodash": {"version": "4.17.21", "license": "MIT"},
                },
            }
        )
        deps = scanner._parse_npm_lock(content, "/path/to/package-lock.json")
        assert deps[0].license == "MIT"

    def test_parse_npm_lock_v2_scoped_package(self):
        """Parse v2 with scoped packages (@scope/name)."""
        scanner = DependencyScanner()
        content = json.dumps(
            {
                "lockfileVersion": 2,
                "packages": {
                    "": {"name": "myapp"},
                    "node_modules/@types/node": {"version": "18.0.0"},
                },
            }
        )
        deps = scanner._parse_npm_lock(content, "/path/to/package-lock.json")
        assert len(deps) == 1
        assert deps[0].name == "@types/node"


# ============================================================
# NPM package.json Parsing
# ============================================================


class TestNPMPackageJsonParsing:
    """Tests for package.json parsing."""

    def test_parse_package_json_basic(self):
        """Parse basic package.json with dependencies."""
        scanner = DependencyScanner()
        content = json.dumps(
            {
                "dependencies": {
                    "lodash": "^4.17.21",
                    "express": "~4.18.2",
                },
            }
        )
        deps = scanner._parse_npm_package(content, "/path/to/package.json")
        assert len(deps) == 2
        # Version prefixes should be stripped
        lodash_dep = next(d for d in deps if d.name == "lodash")
        assert lodash_dep.version == "4.17.21"

    def test_parse_package_json_dev_dependencies(self):
        """Parse package.json with dev dependencies."""
        scanner = DependencyScanner()
        content = json.dumps(
            {
                "dependencies": {"lodash": "4.17.21"},
                "devDependencies": {"jest": "29.0.0"},
            }
        )
        deps = scanner._parse_npm_package(content, "/path/to/package.json")
        assert len(deps) == 2
        jest_dep = next(d for d in deps if d.name == "jest")
        assert jest_dep.dev_dependency is True

    def test_parse_package_json_skip_dev(self):
        """Skip dev dependencies when configured."""
        scanner = DependencyScanner(skip_dev_dependencies=True)
        content = json.dumps(
            {
                "dependencies": {"lodash": "4.17.21"},
                "devDependencies": {"jest": "29.0.0"},
            }
        )
        deps = scanner._parse_npm_package(content, "/path/to/package.json")
        assert len(deps) == 1
        assert deps[0].name == "lodash"

    def test_parse_package_json_direct_flag(self):
        """All package.json dependencies are marked as direct."""
        scanner = DependencyScanner()
        content = json.dumps(
            {
                "dependencies": {"lodash": "4.17.21"},
            }
        )
        deps = scanner._parse_npm_package(content, "/path/to/package.json")
        assert all(d.direct for d in deps)

    def test_parse_package_json_invalid_json(self):
        """Handle invalid JSON gracefully."""
        scanner = DependencyScanner()
        deps = scanner._parse_npm_package("not valid json {", "/path/to/package.json")
        assert deps == []


# ============================================================
# Yarn.lock Parsing
# ============================================================


class TestYarnLockParsing:
    """Tests for yarn.lock parsing."""

    def test_parse_yarn_lock_basic(self):
        """Parse basic yarn.lock file."""
        scanner = DependencyScanner()
        content = """
lodash@^4.17.0:
  version "4.17.21"
  resolved "https://registry.yarnpkg.com/lodash/-/lodash-4.17.21.tgz"
  integrity sha512-abc...

express@^4.18.0:
  version "4.18.2"
  resolved "https://registry.yarnpkg.com/express/-/express-4.18.2.tgz"
"""
        deps = scanner._parse_yarn_lock(content, "/path/to/yarn.lock")
        assert len(deps) == 2
        names = {d.name for d in deps}
        assert "lodash" in names
        assert "express" in names

    def test_parse_yarn_lock_scoped_package(self):
        """Parse yarn.lock with scoped packages."""
        scanner = DependencyScanner()
        content = """
"@types/node@^18.0.0":
  version "18.11.9"
  resolved "https://registry.yarnpkg.com/@types/node/-/node-18.11.9.tgz"
"""
        deps = scanner._parse_yarn_lock(content, "/path/to/yarn.lock")
        assert len(deps) == 1
        assert deps[0].name == "@types/node"
        assert deps[0].version == "18.11.9"

    def test_parse_yarn_lock_multiple_ranges(self):
        """Parse yarn.lock with multiple version ranges for same package."""
        scanner = DependencyScanner()
        content = """
"lodash@^4.0.0, lodash@^4.17.0":
  version "4.17.21"
  resolved "https://registry.yarnpkg.com/lodash/-/lodash-4.17.21.tgz"
"""
        deps = scanner._parse_yarn_lock(content, "/path/to/yarn.lock")
        # Should create entries for each aliased name
        assert len(deps) == 2
        assert all(d.version == "4.17.21" for d in deps)

    def test_parse_yarn_lock_ecosystem(self):
        """Yarn dependencies are marked as npm ecosystem."""
        scanner = DependencyScanner()
        content = """
lodash@^4.17.0:
  version "4.17.21"
"""
        deps = scanner._parse_yarn_lock(content, "/path/to/yarn.lock")
        assert deps[0].ecosystem == "npm"

    def test_parse_yarn_lock_comments_skipped(self):
        """Comments in yarn.lock are skipped."""
        scanner = DependencyScanner()
        content = """
# THIS IS AN AUTOGENERATED FILE. DO NOT EDIT THIS FILE DIRECTLY.
# yarn lockfile v1

lodash@^4.17.0:
  version "4.17.21"
"""
        deps = scanner._parse_yarn_lock(content, "/path/to/yarn.lock")
        assert len(deps) == 1
        assert deps[0].name == "lodash"


# ============================================================
# requirements.txt Parsing
# ============================================================


class TestRequirementsTxtParsing:
    """Tests for requirements.txt parsing."""

    def test_parse_requirements_basic(self):
        """Parse basic requirements.txt."""
        scanner = DependencyScanner()
        content = """
requests==2.28.1
flask>=2.0.0
django~=4.1
"""
        deps = scanner._parse_requirements(content, "/path/to/requirements.txt")
        assert len(deps) == 3
        names = {d.name.lower() for d in deps}
        assert "requests" in names
        assert "flask" in names
        assert "django" in names

    def test_parse_requirements_version_extraction(self):
        """Extract versions from various specifiers."""
        scanner = DependencyScanner()
        content = """
requests==2.28.1
flask>=2.0.0
django~=4.1.0
"""
        deps = scanner._parse_requirements(content, "/path/to/requirements.txt")
        requests_dep = next(d for d in deps if d.name.lower() == "requests")
        assert requests_dep.version == "2.28.1"
        flask_dep = next(d for d in deps if d.name.lower() == "flask")
        assert flask_dep.version == "2.0.0"

    def test_parse_requirements_comments_skipped(self):
        """Comments are skipped."""
        scanner = DependencyScanner()
        content = """
# This is a comment
requests==2.28.1
# Another comment
flask>=2.0.0
"""
        deps = scanner._parse_requirements(content, "/path/to/requirements.txt")
        assert len(deps) == 2

    def test_parse_requirements_options_skipped(self):
        """Pip options (-r, -e, etc.) are skipped."""
        scanner = DependencyScanner()
        content = """
-r other-requirements.txt
-e git+https://github.com/example/repo.git
requests==2.28.1
"""
        deps = scanner._parse_requirements(content, "/path/to/requirements.txt")
        assert len(deps) == 1
        assert deps[0].name == "requests"

    def test_parse_requirements_pypi_ecosystem(self):
        """Requirements are marked as pypi ecosystem."""
        scanner = DependencyScanner()
        content = "requests==2.28.1"
        deps = scanner._parse_requirements(content, "/path/to/requirements.txt")
        assert deps[0].ecosystem == "pypi"

    def test_parse_requirements_no_version(self):
        """Handle packages without version specifier."""
        scanner = DependencyScanner()
        content = """
requests
flask
"""
        deps = scanner._parse_requirements(content, "/path/to/requirements.txt")
        assert len(deps) == 2
        assert all(d.version == "unknown" for d in deps)


# ============================================================
# Pipfile.lock Parsing
# ============================================================


class TestPipfileLockParsing:
    """Tests for Pipfile.lock parsing."""

    def test_parse_pipfile_lock_basic(self):
        """Parse basic Pipfile.lock."""
        scanner = DependencyScanner()
        content = json.dumps(
            {
                "default": {
                    "requests": {"version": "==2.28.1"},
                    "flask": {"version": "==2.2.0"},
                },
            }
        )
        deps = scanner._parse_pipfile_lock(content, "/path/to/Pipfile.lock")
        assert len(deps) == 2
        requests_dep = next(d for d in deps if d.name == "requests")
        assert requests_dep.version == "2.28.1"  # Leading == stripped

    def test_parse_pipfile_lock_with_develop(self):
        """Parse Pipfile.lock with develop dependencies."""
        scanner = DependencyScanner()
        content = json.dumps(
            {
                "default": {"requests": {"version": "==2.28.1"}},
                "develop": {"pytest": {"version": "==7.0.0"}},
            }
        )
        deps = scanner._parse_pipfile_lock(content, "/path/to/Pipfile.lock")
        assert len(deps) == 2
        pytest_dep = next(d for d in deps if d.name == "pytest")
        assert pytest_dep.dev_dependency is True

    def test_parse_pipfile_lock_skip_dev(self):
        """Skip develop dependencies when configured."""
        scanner = DependencyScanner(skip_dev_dependencies=True)
        content = json.dumps(
            {
                "default": {"requests": {"version": "==2.28.1"}},
                "develop": {"pytest": {"version": "==7.0.0"}},
            }
        )
        deps = scanner._parse_pipfile_lock(content, "/path/to/Pipfile.lock")
        assert len(deps) == 1
        assert deps[0].name == "requests"

    def test_parse_pipfile_lock_invalid_json(self):
        """Handle invalid JSON gracefully."""
        scanner = DependencyScanner()
        deps = scanner._parse_pipfile_lock("not valid {", "/path/to/Pipfile.lock")
        assert deps == []


# ============================================================
# poetry.lock Parsing
# ============================================================


class TestPoetryLockParsing:
    """Tests for poetry.lock (TOML) parsing."""

    def test_parse_poetry_lock_basic(self):
        """Parse basic poetry.lock."""
        scanner = DependencyScanner()
        content = """
[[package]]
name = "requests"
version = "2.28.1"

[[package]]
name = "flask"
version = "2.2.0"
"""
        deps = scanner._parse_poetry_lock(content, "/path/to/poetry.lock")
        assert len(deps) == 2
        names = {d.name for d in deps}
        assert "requests" in names
        assert "flask" in names

    def test_parse_poetry_lock_with_category(self):
        """Parse poetry.lock with category (dev vs main)."""
        scanner = DependencyScanner()
        content = """
[[package]]
name = "requests"
version = "2.28.1"
category = "main"

[[package]]
name = "pytest"
version = "7.0.0"
category = "dev"
"""
        deps = scanner._parse_poetry_lock(content, "/path/to/poetry.lock")
        assert len(deps) == 2
        pytest_dep = next(d for d in deps if d.name == "pytest")
        assert pytest_dep.dev_dependency is True

    def test_parse_poetry_lock_skip_dev(self):
        """Skip dev category when configured."""
        scanner = DependencyScanner(skip_dev_dependencies=True)
        content = """
[[package]]
name = "requests"
version = "2.28.1"

[[package]]
name = "pytest"
version = "7.0.0"
category = "dev"
"""
        deps = scanner._parse_poetry_lock(content, "/path/to/poetry.lock")
        assert len(deps) == 1
        assert deps[0].name == "requests"

    def test_parse_poetry_lock_pypi_ecosystem(self):
        """Poetry dependencies are marked as pypi ecosystem."""
        scanner = DependencyScanner()
        content = """
[[package]]
name = "requests"
version = "2.28.1"
"""
        deps = scanner._parse_poetry_lock(content, "/path/to/poetry.lock")
        assert deps[0].ecosystem == "pypi"


# ============================================================
# pyproject.toml Parsing
# ============================================================


class TestPyprojectTomlParsing:
    """Tests for pyproject.toml parsing."""

    def test_parse_pyproject_poetry_deps(self):
        """Parse Poetry-style dependencies."""
        scanner = DependencyScanner()
        content = """
[tool.poetry.dependencies]
python = "^3.9"
requests = "^2.28.0"
flask = "^2.2.0"
"""
        deps = scanner._parse_pyproject_toml(content, "/path/to/pyproject.toml")
        # Python constraint should be skipped
        names = {d.name for d in deps}
        assert "python" not in names
        assert "requests" in names
        assert "flask" in names

    def test_parse_pyproject_poetry_dev_deps(self):
        """Parse Poetry dev-dependencies section."""
        scanner = DependencyScanner()
        content = """
[tool.poetry.dependencies]
requests = "^2.28.0"

[tool.poetry.dev-dependencies]
pytest = "^7.0.0"
"""
        deps = scanner._parse_pyproject_toml(content, "/path/to/pyproject.toml")
        assert len(deps) == 2
        pytest_dep = next(d for d in deps if d.name == "pytest")
        assert pytest_dep.dev_dependency is True

    def test_parse_pyproject_poetry_group_deps(self):
        """Parse Poetry group dependencies."""
        scanner = DependencyScanner()
        content = """
[tool.poetry.dependencies]
requests = "^2.28.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.0.0"
"""
        deps = scanner._parse_pyproject_toml(content, "/path/to/pyproject.toml")
        assert len(deps) == 2
        pytest_dep = next(d for d in deps if d.name == "pytest")
        assert pytest_dep.dev_dependency is True

    def test_parse_pyproject_poetry_complex_version(self):
        """Parse complex version specifications."""
        scanner = DependencyScanner()
        content = """
[tool.poetry.dependencies]
sqlalchemy = {version = "^2.0.0", extras = ["asyncio"]}
"""
        deps = scanner._parse_pyproject_toml(content, "/path/to/pyproject.toml")
        assert len(deps) == 1
        assert deps[0].name == "sqlalchemy"
        assert deps[0].version == "2.0.0"

    def test_parse_pyproject_pep621_deps(self):
        """Parse PEP 621 style dependencies."""
        scanner = DependencyScanner()
        content = """
[project]
dependencies = [
    "requests>=2.28.0",
    "flask>=2.2.0",
]
"""
        deps = scanner._parse_pyproject_toml(content, "/path/to/pyproject.toml")
        names = {d.name for d in deps}
        assert "requests" in names
        assert "flask" in names


# ============================================================
# go.mod Parsing
# ============================================================


class TestGoModParsing:
    """Tests for go.mod parsing."""

    def test_parse_go_mod_basic(self):
        """Parse basic go.mod file."""
        scanner = DependencyScanner()
        content = """
module github.com/myorg/myapp

go 1.20

require (
    github.com/gin-gonic/gin v1.9.0
    github.com/stretchr/testify v1.8.2
)
"""
        deps = scanner._parse_go_mod(content, "/path/to/go.mod")
        assert len(deps) == 2
        names = {d.name for d in deps}
        assert "github.com/gin-gonic/gin" in names
        assert "github.com/stretchr/testify" in names

    def test_parse_go_mod_single_require(self):
        """Parse go.mod with single require statement."""
        scanner = DependencyScanner()
        content = """
module github.com/myorg/myapp

go 1.20

require github.com/gin-gonic/gin v1.9.0
"""
        deps = scanner._parse_go_mod(content, "/path/to/go.mod")
        assert len(deps) == 1
        assert deps[0].name == "github.com/gin-gonic/gin"

    def test_parse_go_mod_indirect(self):
        """Parse go.mod with indirect dependencies."""
        scanner = DependencyScanner()
        content = """
module github.com/myorg/myapp

require (
    github.com/gin-gonic/gin v1.9.0
    github.com/stretchr/objx v0.5.0 // indirect
)
"""
        deps = scanner._parse_go_mod(content, "/path/to/go.mod")
        gin_dep = next(d for d in deps if "gin" in d.name)
        objx_dep = next(d for d in deps if "objx" in d.name)
        assert gin_dep.direct is True
        assert objx_dep.direct is False

    def test_parse_go_mod_version_format(self):
        """Version is stripped of 'v' prefix."""
        scanner = DependencyScanner()
        content = """
module myapp

require github.com/gin-gonic/gin v1.9.0
"""
        deps = scanner._parse_go_mod(content, "/path/to/go.mod")
        assert deps[0].version == "1.9.0"

    def test_parse_go_mod_ecosystem(self):
        """Go dependencies are marked as go ecosystem."""
        scanner = DependencyScanner()
        content = "require github.com/gin-gonic/gin v1.9.0"
        deps = scanner._parse_go_mod(content, "/path/to/go.mod")
        assert deps[0].ecosystem == "go"


# ============================================================
# go.sum Parsing
# ============================================================


class TestGoSumParsing:
    """Tests for go.sum parsing."""

    def test_parse_go_sum_basic(self):
        """Parse basic go.sum file."""
        scanner = DependencyScanner()
        content = """
github.com/gin-gonic/gin v1.9.0 h1:abc123...
github.com/gin-gonic/gin v1.9.0/go.mod h1:xyz789...
github.com/stretchr/testify v1.8.2 h1:def456...
"""
        deps = scanner._parse_go_sum(content, "/path/to/go.sum")
        # Should deduplicate same package/version
        names = {d.name for d in deps}
        assert "github.com/gin-gonic/gin" in names
        assert "github.com/stretchr/testify" in names

    def test_parse_go_sum_deduplication(self):
        """Same package+version only appears once."""
        scanner = DependencyScanner()
        content = """
github.com/gin-gonic/gin v1.9.0 h1:abc...
github.com/gin-gonic/gin v1.9.0/go.mod h1:xyz...
"""
        deps = scanner._parse_go_sum(content, "/path/to/go.sum")
        assert len(deps) == 1

    def test_parse_go_sum_version_format(self):
        """Version is stripped of 'v' prefix and /go.mod suffix."""
        scanner = DependencyScanner()
        content = "github.com/gin-gonic/gin v1.9.0/go.mod h1:abc..."
        deps = scanner._parse_go_sum(content, "/path/to/go.sum")
        assert deps[0].version == "1.9.0"


# ============================================================
# Cargo.lock Parsing
# ============================================================


class TestCargoLockParsing:
    """Tests for Cargo.lock (TOML) parsing."""

    def test_parse_cargo_lock_basic(self):
        """Parse basic Cargo.lock."""
        scanner = DependencyScanner()
        content = """
[[package]]
name = "serde"
version = "1.0.152"

[[package]]
name = "tokio"
version = "1.25.0"
"""
        deps = scanner._parse_cargo_lock(content, "/path/to/Cargo.lock")
        assert len(deps) == 2
        names = {d.name for d in deps}
        assert "serde" in names
        assert "tokio" in names

    def test_parse_cargo_lock_versions(self):
        """Versions are extracted correctly."""
        scanner = DependencyScanner()
        content = """
[[package]]
name = "serde"
version = "1.0.152"
"""
        deps = scanner._parse_cargo_lock(content, "/path/to/Cargo.lock")
        assert deps[0].version == "1.0.152"

    def test_parse_cargo_lock_ecosystem(self):
        """Cargo dependencies are marked as cargo ecosystem."""
        scanner = DependencyScanner()
        content = """
[[package]]
name = "serde"
version = "1.0.152"
"""
        deps = scanner._parse_cargo_lock(content, "/path/to/Cargo.lock")
        assert deps[0].ecosystem == "cargo"


# ============================================================
# Gemfile.lock Parsing
# ============================================================


class TestGemfileLockParsing:
    """Tests for Gemfile.lock parsing."""

    def test_parse_gemfile_lock_basic(self):
        """Parse basic Gemfile.lock."""
        scanner = DependencyScanner()
        content = """
GEM
  remote: https://rubygems.org/
  specs:
    rails (7.0.4)
    puma (6.0.0)

PLATFORMS
  ruby

DEPENDENCIES
  rails
"""
        deps = scanner._parse_gemfile_lock(content, "/path/to/Gemfile.lock")
        assert len(deps) == 2
        names = {d.name for d in deps}
        assert "rails" in names
        assert "puma" in names

    def test_parse_gemfile_lock_versions(self):
        """Versions in parentheses are extracted."""
        scanner = DependencyScanner()
        content = """
GEM
  specs:
    rails (7.0.4)
"""
        deps = scanner._parse_gemfile_lock(content, "/path/to/Gemfile.lock")
        assert deps[0].version == "7.0.4"

    def test_parse_gemfile_lock_nested_deps_skipped(self):
        """Nested dependencies (6 spaces) are skipped."""
        scanner = DependencyScanner()
        content = """
GEM
  specs:
    rails (7.0.4)
      actioncable (= 7.0.4)
      actionmailbox (= 7.0.4)
"""
        deps = scanner._parse_gemfile_lock(content, "/path/to/Gemfile.lock")
        # Only direct deps (4 spaces) are parsed
        assert len(deps) == 1
        assert deps[0].name == "rails"

    def test_parse_gemfile_lock_ecosystem(self):
        """Ruby dependencies are marked as rubygems ecosystem."""
        scanner = DependencyScanner()
        content = """
GEM
  specs:
    rails (7.0.4)
"""
        deps = scanner._parse_gemfile_lock(content, "/path/to/Gemfile.lock")
        assert deps[0].ecosystem == "rubygems"


# ============================================================
# Lock File Discovery
# ============================================================


class TestLockFileDiscovery:
    """Tests for finding lock files in a repository."""

    def test_find_lock_files_single(self):
        """Find a single lock file."""
        scanner = DependencyScanner()
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "package-lock.json")
            with open(lock_path, "w") as f:
                f.write("{}")

            lock_files = scanner._find_lock_files(tmpdir)
            assert len(lock_files) == 1
            assert "package-lock.json" in lock_files[0]

    def test_find_lock_files_multiple(self):
        """Find multiple lock files of different types."""
        scanner = DependencyScanner()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple lock files
            with open(os.path.join(tmpdir, "package-lock.json"), "w") as f:
                f.write("{}")
            with open(os.path.join(tmpdir, "requirements.txt"), "w") as f:
                f.write("")

            lock_files = scanner._find_lock_files(tmpdir)
            assert len(lock_files) == 2

    def test_find_lock_files_nested(self):
        """Find lock files in nested directories."""
        scanner = DependencyScanner()
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "backend")
            os.makedirs(subdir)
            with open(os.path.join(subdir, "requirements.txt"), "w") as f:
                f.write("")

            lock_files = scanner._find_lock_files(tmpdir)
            assert len(lock_files) == 1
            assert "backend" in lock_files[0]

    def test_find_lock_files_skip_node_modules(self):
        """Skip lock files in node_modules."""
        scanner = DependencyScanner()
        with tempfile.TemporaryDirectory() as tmpdir:
            node_modules = os.path.join(tmpdir, "node_modules", "some-package")
            os.makedirs(node_modules)
            with open(os.path.join(node_modules, "package-lock.json"), "w") as f:
                f.write("{}")

            lock_files = scanner._find_lock_files(tmpdir)
            assert len(lock_files) == 0

    def test_find_lock_files_skip_vendor(self):
        """Skip lock files in vendor directory."""
        scanner = DependencyScanner()
        with tempfile.TemporaryDirectory() as tmpdir:
            vendor = os.path.join(tmpdir, "vendor", "some-package")
            os.makedirs(vendor)
            with open(os.path.join(vendor, "go.mod"), "w") as f:
                f.write("")

            lock_files = scanner._find_lock_files(tmpdir)
            assert len(lock_files) == 0

    def test_find_lock_files_skip_git(self):
        """Skip lock files in .git directory."""
        scanner = DependencyScanner()
        with tempfile.TemporaryDirectory() as tmpdir:
            git_dir = os.path.join(tmpdir, ".git", "hooks")
            os.makedirs(git_dir)
            with open(os.path.join(git_dir, "package-lock.json"), "w") as f:
                f.write("{}")

            lock_files = scanner._find_lock_files(tmpdir)
            assert len(lock_files) == 0


# ============================================================
# Repository Scanning
# ============================================================


class TestRepositoryScan:
    """Tests for scanning a repository."""

    @pytest.mark.asyncio
    async def test_scan_repository_basic(self):
        """Scan repository with one lock file."""
        scanner = DependencyScanner()
        scanner.cve_client = MagicMock()
        scanner.cve_client.query_package = AsyncMock(return_value=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "package-lock.json")
            with open(lock_path, "w") as f:
                json.dump(
                    {
                        "lockfileVersion": 2,
                        "packages": {
                            "": {"name": "myapp"},
                            "node_modules/lodash": {"version": "4.17.21"},
                        },
                    },
                    f,
                )

            result = await scanner.scan_repository(tmpdir)

            assert result.status == "completed"
            assert result.total_dependencies == 1
            assert result.repository == tmpdir

    @pytest.mark.asyncio
    async def test_scan_repository_with_metadata(self):
        """Scan repository with branch and commit metadata."""
        scanner = DependencyScanner()
        scanner.cve_client = MagicMock()
        scanner.cve_client.query_package = AsyncMock(return_value=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "requirements.txt")
            with open(lock_path, "w") as f:
                f.write("requests==2.28.1")

            result = await scanner.scan_repository(
                tmpdir,
                branch="main",
                commit_sha="abc123",
            )

            assert result.branch == "main"
            assert result.commit_sha == "abc123"

    @pytest.mark.asyncio
    async def test_scan_repository_no_lock_files(self):
        """Scan repository with no lock files."""
        scanner = DependencyScanner()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = await scanner.scan_repository(tmpdir)

            assert result.status == "completed"
            assert result.total_dependencies == 0
            assert len(result.dependencies) == 0

    @pytest.mark.asyncio
    async def test_scan_repository_deduplication(self):
        """Duplicate dependencies are deduplicated."""
        scanner = DependencyScanner()
        scanner.cve_client = MagicMock()
        scanner.cve_client.query_package = AsyncMock(return_value=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two lock files with same dependency
            with open(os.path.join(tmpdir, "package-lock.json"), "w") as f:
                json.dump(
                    {
                        "lockfileVersion": 2,
                        "packages": {
                            "": {"name": "app1"},
                            "node_modules/lodash": {"version": "4.17.21"},
                        },
                    },
                    f,
                )
            with open(os.path.join(tmpdir, "package.json"), "w") as f:
                json.dump({"dependencies": {"lodash": "4.17.21"}}, f)

            result = await scanner.scan_repository(tmpdir)

            # Same package/version/ecosystem should be deduplicated
            assert result.total_dependencies == 1

    @pytest.mark.asyncio
    async def test_scan_repository_scan_id_generated(self):
        """Scan ID is generated for each scan."""
        scanner = DependencyScanner()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = await scanner.scan_repository(tmpdir)

            assert result.scan_id.startswith("scan_")
            assert len(result.scan_id) == 17  # "scan_" + 12 hex chars

    @pytest.mark.asyncio
    async def test_scan_repository_error_handling(self):
        """Handle errors during repository scan."""
        scanner = DependencyScanner()

        # Non-existent path won't cause failure, just empty results
        result = await scanner.scan_repository("/nonexistent/path/xyz123")
        assert result.status == "completed"
        assert result.total_dependencies == 0


# ============================================================
# File Scanning
# ============================================================


class TestFileScan:
    """Tests for scanning specific files."""

    @pytest.mark.asyncio
    async def test_scan_files_basic(self):
        """Scan specific lock files."""
        scanner = DependencyScanner()
        scanner.cve_client = MagicMock()
        scanner.cve_client.query_package = AsyncMock(return_value=[])

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="requirements"
        ) as f:
            f.write("requests==2.28.1\nflask>=2.0.0")
            f.flush()

            # Need to rename to exact filename
            import shutil

            new_path = os.path.join(os.path.dirname(f.name), "requirements.txt")
            shutil.move(f.name, new_path)

            try:
                result = await scanner.scan_files([new_path], repository="test-repo")
                assert result.status == "completed"
                assert result.repository == "test-repo"
            finally:
                if os.path.exists(new_path):
                    os.unlink(new_path)

    @pytest.mark.asyncio
    async def test_scan_files_unknown_format(self):
        """Handle unknown file format gracefully."""
        scanner = DependencyScanner()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".unknown", delete=False) as f:
            f.write("some content")
            f.flush()
            try:
                result = await scanner.scan_files([f.name])
                assert result.status == "completed"
                assert result.total_dependencies == 0
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_scan_files_multiple(self):
        """Scan multiple files."""
        scanner = DependencyScanner()
        scanner.cve_client = MagicMock()
        scanner.cve_client.query_package = AsyncMock(return_value=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            req_path = os.path.join(tmpdir, "requirements.txt")
            with open(req_path, "w") as f:
                f.write("requests==2.28.1")

            pkg_path = os.path.join(tmpdir, "package.json")
            with open(pkg_path, "w") as f:
                json.dump({"dependencies": {"lodash": "4.17.21"}}, f)

            result = await scanner.scan_files([req_path, pkg_path])
            assert result.total_dependencies == 2


# ============================================================
# Vulnerability Integration
# ============================================================


class TestVulnerabilityIntegration:
    """Tests for CVE client integration."""

    @pytest.mark.asyncio
    async def test_vulnerabilities_queried(self):
        """CVE client is queried for each dependency."""
        mock_client = MagicMock()
        mock_client.query_package = AsyncMock(return_value=[])
        scanner = DependencyScanner(cve_client=mock_client)

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "requirements.txt"), "w") as f:
                f.write("requests==2.28.1\nflask>=2.0.0")

            await scanner.scan_repository(tmpdir)

            # Should query for each unique dependency
            assert mock_client.query_package.call_count == 2

    @pytest.mark.asyncio
    async def test_vulnerabilities_attached(self):
        """Found vulnerabilities are attached to dependencies."""
        vuln = VulnerabilityFinding(
            id="CVE-2023-12345",
            title="Test Vuln",
            description="A test vulnerability",
            severity=VulnerabilitySeverity.HIGH,
        )
        mock_client = MagicMock()
        mock_client.query_package = AsyncMock(return_value=[vuln])
        scanner = DependencyScanner(cve_client=mock_client)

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "requirements.txt"), "w") as f:
                f.write("requests==2.28.1")

            result = await scanner.scan_repository(tmpdir)

            assert len(result.vulnerabilities) == 1
            assert result.vulnerabilities[0].id == "CVE-2023-12345"

    @pytest.mark.asyncio
    async def test_vulnerability_counts(self):
        """Vulnerability counts are calculated correctly."""
        vulns = [
            VulnerabilityFinding(
                id="CVE-1",
                title="Critical",
                description="",
                severity=VulnerabilitySeverity.CRITICAL,
            ),
            VulnerabilityFinding(
                id="CVE-2",
                title="High",
                description="",
                severity=VulnerabilitySeverity.HIGH,
            ),
            VulnerabilityFinding(
                id="CVE-3",
                title="Medium",
                description="",
                severity=VulnerabilitySeverity.MEDIUM,
            ),
        ]
        mock_client = MagicMock()
        mock_client.query_package = AsyncMock(return_value=vulns)
        scanner = DependencyScanner(cve_client=mock_client)

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "requirements.txt"), "w") as f:
                f.write("requests==2.28.1")

            result = await scanner.scan_repository(tmpdir)

            assert result.critical_count == 1
            assert result.high_count == 1
            assert result.medium_count == 1
            assert result.vulnerable_dependencies == 1

    @pytest.mark.asyncio
    async def test_vulnerability_query_errors_handled(self):
        """Errors during vulnerability query don't fail scan."""
        mock_client = MagicMock()
        mock_client.query_package = AsyncMock(side_effect=Exception("API Error"))
        scanner = DependencyScanner(cve_client=mock_client)

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "requirements.txt"), "w") as f:
                f.write("requests==2.28.1")

            result = await scanner.scan_repository(tmpdir)

            # Scan should complete despite query errors
            assert result.status == "completed"


# ============================================================
# Scan Result
# ============================================================


class TestScanResult:
    """Tests for ScanResult structure."""

    @pytest.mark.asyncio
    async def test_scan_result_timestamps(self):
        """Scan result has proper timestamps."""
        scanner = DependencyScanner()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = await scanner.scan_repository(tmpdir)

            assert result.started_at is not None
            assert result.completed_at is not None
            # Both timestamps should exist (can't directly compare due to timezone differences)
            assert isinstance(result.started_at, datetime)
            assert isinstance(result.completed_at, datetime)

    @pytest.mark.asyncio
    async def test_scan_result_to_dict(self):
        """Scan result can be serialized to dict."""
        scanner = DependencyScanner()
        scanner.cve_client = MagicMock()
        scanner.cve_client.query_package = AsyncMock(return_value=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "requirements.txt"), "w") as f:
                f.write("requests==2.28.1")

            result = await scanner.scan_repository(tmpdir)
            result_dict = result.to_dict()

            assert "scan_id" in result_dict
            assert "repository" in result_dict
            assert "dependencies" in result_dict
            assert "vulnerabilities" in result_dict
            assert "summary" in result_dict


# ============================================================
# PEP 508 Dependency Parsing
# ============================================================


class TestPEP508Parsing:
    """Tests for PEP 508 dependency specification parsing."""

    def test_parse_pep508_simple(self):
        """Parse simple package name."""
        scanner = DependencyScanner()
        result = scanner._parse_pep508_dependency("requests")
        assert result == ("requests", "unknown")

    def test_parse_pep508_with_version(self):
        """Parse package with version specifier."""
        scanner = DependencyScanner()
        result = scanner._parse_pep508_dependency("requests>=2.28.0")
        assert result == ("requests", "2.28.0")

    def test_parse_pep508_with_extras(self):
        """Parse package with extras."""
        scanner = DependencyScanner()
        result = scanner._parse_pep508_dependency("requests[security]>=2.28.0")
        assert result[0] == "requests"
        assert result[1] == "2.28.0"

    def test_parse_pep508_with_markers(self):
        """Parse package with environment markers."""
        scanner = DependencyScanner()
        result = scanner._parse_pep508_dependency('requests>=2.28.0; python_version >= "3.8"')
        assert result[0] == "requests"
        assert result[1] == "2.28.0"

    def test_parse_pep508_version_range(self):
        """Parse package with version range."""
        scanner = DependencyScanner()
        result = scanner._parse_pep508_dependency("requests>=2.20.0,<3.0.0")
        assert result[0] == "requests"
        assert result[1] == "2.20.0"  # Takes first version in range

    def test_parse_pep508_empty(self):
        """Empty string returns None."""
        scanner = DependencyScanner()
        result = scanner._parse_pep508_dependency("")
        assert result is None


# ============================================================
# Concurrency Control
# ============================================================


class TestConcurrencyControl:
    """Tests for concurrency limiting."""

    @pytest.mark.asyncio
    async def test_max_concurrency_respected(self):
        """Max concurrency limits parallel queries."""
        concurrent_count = 0
        max_concurrent = 0

        async def mock_query(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.01)
            concurrent_count -= 1
            return []

        mock_client = MagicMock()
        mock_client.query_package = mock_query
        scanner = DependencyScanner(cve_client=mock_client, max_concurrency=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "requirements.txt"), "w") as f:
                f.write("\n".join([f"package{i}==1.0.0" for i in range(10)]))

            await scanner.scan_repository(tmpdir)

            assert max_concurrent <= 2


# ============================================================
# Edge Cases
# ============================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_lock_file(self):
        """Handle empty lock file."""
        scanner = DependencyScanner()

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "requirements.txt"), "w") as f:
                f.write("")

            result = await scanner.scan_repository(tmpdir)
            assert result.status == "completed"
            assert result.total_dependencies == 0

    @pytest.mark.asyncio
    async def test_malformed_json_lock_file(self):
        """Handle malformed JSON in lock file."""
        scanner = DependencyScanner()

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "package-lock.json"), "w") as f:
                f.write("{ invalid json }")

            result = await scanner.scan_repository(tmpdir)
            # Should complete but with no deps from that file
            assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_unicode_in_lock_file(self):
        """Handle unicode content in lock files."""
        scanner = DependencyScanner()
        scanner.cve_client = MagicMock()
        scanner.cve_client.query_package = AsyncMock(return_value=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "requirements.txt"), "w", encoding="utf-8") as f:
                f.write("# Comment with unicode: \u4e2d\u6587\nrequests==2.28.1")

            result = await scanner.scan_repository(tmpdir)
            assert result.total_dependencies == 1

    @pytest.mark.asyncio
    async def test_very_long_version_string(self):
        """Handle very long version strings."""
        scanner = DependencyScanner()
        scanner.cve_client = MagicMock()
        scanner.cve_client.query_package = AsyncMock(return_value=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "requirements.txt"), "w") as f:
                f.write("mypackage==1.0.0.dev123456789.post987654321")

            result = await scanner.scan_repository(tmpdir)
            assert result.total_dependencies == 1

    def test_parse_npm_lock_empty_packages(self):
        """Handle package-lock.json with empty packages."""
        scanner = DependencyScanner()
        content = json.dumps(
            {
                "lockfileVersion": 2,
                "packages": {},
            }
        )
        deps = scanner._parse_npm_lock(content, "/path/to/package-lock.json")
        assert deps == []

    def test_parse_requirements_only_comments(self):
        """Handle requirements.txt with only comments."""
        scanner = DependencyScanner()
        content = """
# This is a comment
# Another comment
"""
        deps = scanner._parse_requirements(content, "/path/to/requirements.txt")
        assert deps == []

    def test_parse_go_mod_empty_require(self):
        """Handle go.mod with empty require block."""
        scanner = DependencyScanner()
        content = """
module github.com/myorg/myapp

go 1.20

require (
)
"""
        deps = scanner._parse_go_mod(content, "/path/to/go.mod")
        assert deps == []


# ============================================================
# DependencyInfo Model
# ============================================================


class TestDependencyInfoModel:
    """Tests for DependencyInfo dataclass."""

    def test_dependency_info_basic(self):
        """Create basic DependencyInfo."""
        dep = DependencyInfo(
            name="requests",
            version="2.28.1",
            ecosystem="pypi",
        )
        assert dep.name == "requests"
        assert dep.version == "2.28.1"
        assert dep.ecosystem == "pypi"
        assert dep.direct is True
        assert dep.dev_dependency is False

    def test_dependency_info_has_vulnerabilities(self):
        """has_vulnerabilities property works."""
        dep = DependencyInfo(name="pkg", version="1.0", ecosystem="npm")
        assert dep.has_vulnerabilities is False

        dep.vulnerabilities = [
            VulnerabilityFinding(
                id="CVE-1",
                title="Test",
                description="",
                severity=VulnerabilitySeverity.HIGH,
            )
        ]
        assert dep.has_vulnerabilities is True

    def test_dependency_info_highest_severity(self):
        """highest_severity property returns correct severity."""
        dep = DependencyInfo(name="pkg", version="1.0", ecosystem="npm")
        assert dep.highest_severity is None

        dep.vulnerabilities = [
            VulnerabilityFinding(
                id="CVE-1",
                title="Medium",
                description="",
                severity=VulnerabilitySeverity.MEDIUM,
            ),
            VulnerabilityFinding(
                id="CVE-2",
                title="Critical",
                description="",
                severity=VulnerabilitySeverity.CRITICAL,
            ),
        ]
        assert dep.highest_severity == VulnerabilitySeverity.CRITICAL

    def test_dependency_info_to_dict(self):
        """DependencyInfo serializes to dict."""
        dep = DependencyInfo(
            name="requests",
            version="2.28.1",
            ecosystem="pypi",
            license="MIT",
        )
        d = dep.to_dict()
        assert d["name"] == "requests"
        assert d["version"] == "2.28.1"
        assert d["ecosystem"] == "pypi"
        assert d["license"] == "MIT"


# ============================================================
# TOML Array Parsing
# ============================================================


class TestTOMLArrayParsing:
    """Tests for TOML array parsing helper."""

    def test_parse_toml_array_single_line(self):
        """Parse single-line TOML array."""
        scanner = DependencyScanner()
        result = scanner._parse_toml_array('["item1", "item2", "item3"]', "", "")
        assert result == ["item1", "item2", "item3"]

    def test_parse_toml_array_empty(self):
        """Parse empty TOML array."""
        scanner = DependencyScanner()
        result = scanner._parse_toml_array("[]", "", "")
        assert result == []

    def test_parse_toml_array_with_spaces(self):
        """Parse TOML array with extra spaces."""
        scanner = DependencyScanner()
        result = scanner._parse_toml_array('[  "item1" ,  "item2"  ]', "", "")
        assert result == ["item1", "item2"]


# ============================================================
# File Path Handling
# ============================================================


class TestFilePathHandling:
    """Tests for file path handling in results."""

    @pytest.mark.asyncio
    async def test_file_path_in_dependency(self):
        """Dependencies include source file path."""
        scanner = DependencyScanner()
        scanner.cve_client = MagicMock()
        scanner.cve_client.query_package = AsyncMock(return_value=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            req_path = os.path.join(tmpdir, "requirements.txt")
            with open(req_path, "w") as f:
                f.write("requests==2.28.1")

            result = await scanner.scan_repository(tmpdir)
            assert len(result.dependencies) == 1
            assert result.dependencies[0].file_path == req_path


# ============================================================
# Integration Tests
# ============================================================


class TestIntegration:
    """Integration tests with realistic scenarios."""

    @pytest.mark.asyncio
    async def test_monorepo_scan(self):
        """Scan a monorepo with multiple projects."""
        scanner = DependencyScanner()
        scanner.cve_client = MagicMock()
        scanner.cve_client.query_package = AsyncMock(return_value=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            # Frontend project
            frontend = os.path.join(tmpdir, "frontend")
            os.makedirs(frontend)
            with open(os.path.join(frontend, "package.json"), "w") as f:
                json.dump({"dependencies": {"react": "18.2.0"}}, f)

            # Backend project
            backend = os.path.join(tmpdir, "backend")
            os.makedirs(backend)
            with open(os.path.join(backend, "requirements.txt"), "w") as f:
                f.write("django==4.2.0")

            # Service project
            service = os.path.join(tmpdir, "service")
            os.makedirs(service)
            with open(os.path.join(service, "go.mod"), "w") as f:
                f.write("module service\nrequire github.com/gin-gonic/gin v1.9.0")

            result = await scanner.scan_repository(tmpdir)
            assert result.total_dependencies == 3
            ecosystems = {d.ecosystem for d in result.dependencies}
            assert "npm" in ecosystems
            assert "pypi" in ecosystems
            assert "go" in ecosystems

    @pytest.mark.asyncio
    async def test_scan_with_mixed_vulnerabilities(self):
        """Scan with various vulnerability severities."""
        vulns_by_package = {
            "requests": [
                VulnerabilityFinding(
                    id="CVE-CRIT",
                    title="Critical",
                    description="",
                    severity=VulnerabilitySeverity.CRITICAL,
                )
            ],
            "flask": [
                VulnerabilityFinding(
                    id="CVE-HIGH",
                    title="High",
                    description="",
                    severity=VulnerabilitySeverity.HIGH,
                ),
                VulnerabilityFinding(
                    id="CVE-MEDIUM",
                    title="Medium",
                    description="",
                    severity=VulnerabilitySeverity.MEDIUM,
                ),
            ],
        }

        async def mock_query(name, ecosystem, version):
            return vulns_by_package.get(name, [])

        mock_client = MagicMock()
        mock_client.query_package = mock_query
        scanner = DependencyScanner(cve_client=mock_client)

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "requirements.txt"), "w") as f:
                f.write("requests==2.28.1\nflask>=2.0.0")

            result = await scanner.scan_repository(tmpdir)

            assert result.critical_count == 1
            assert result.high_count == 1
            assert result.medium_count == 1
            assert result.vulnerable_dependencies == 2
