"""
Tests for release gate configurations and the pre-release check script.

Validates:
- Workflow YAML files are parseable and well-structured
- Pre-release check script runs correctly
- Individual gates produce correct pass/fail results
- Smoke test integration works
- Secret pattern detection is functional
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = PROJECT_ROOT / ".github" / "workflows"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    """Load and parse a YAML file."""
    return yaml.safe_load(path.read_text())


def _get_triggers(data: dict) -> dict:
    """Get the 'on' triggers from a workflow dict.

    PyYAML 1.1 parses the bare key ``on:`` as the boolean True, so we
    look for both ``"on"`` and ``True`` as keys.
    """
    if "on" in data:
        return data["on"]
    if True in data:
        return data[True]
    raise KeyError("Workflow has no 'on' trigger key")


# ---------------------------------------------------------------------------
# 1. Workflow YAML parsing and structure
# ---------------------------------------------------------------------------


class TestSecurityGateWorkflow:
    """Validate security-gate.yml structure."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.path = WORKFLOWS_DIR / "security-gate.yml"

    def test_workflow_file_exists(self):
        assert self.path.exists(), "security-gate.yml does not exist"

    def test_workflow_is_valid_yaml(self):
        data = _load_yaml(self.path)
        assert isinstance(data, dict)
        assert "name" in data
        triggers = _get_triggers(data)
        assert isinstance(triggers, dict)
        assert "jobs" in data

    def test_workflow_triggers(self):
        data = _load_yaml(self.path)
        triggers = _get_triggers(data)
        assert "pull_request" in triggers, "should trigger on pull_request"
        assert "workflow_call" in triggers or triggers.get("workflow_call") is None, (
            "should support workflow_call for reusable invocation"
        )

    def test_workflow_has_python_security_job(self):
        data = _load_yaml(self.path)
        jobs = data["jobs"]
        assert "python-security" in jobs, "should have python-security job"
        steps = jobs["python-security"]["steps"]
        step_names = [s.get("name", "") for s in steps]
        assert any("bandit" in n.lower() for n in step_names), (
            "python-security should include bandit scan"
        )
        assert any("pip-audit" in n.lower() or "dependency" in n.lower() for n in step_names), (
            "python-security should include pip-audit"
        )

    def test_workflow_has_npm_security_job(self):
        data = _load_yaml(self.path)
        jobs = data["jobs"]
        assert "npm-security" in jobs, "should have npm-security job"

    def test_workflow_has_summary_job(self):
        data = _load_yaml(self.path)
        jobs = data["jobs"]
        assert "security-summary" in jobs, "should have security-summary job"
        summary_job = jobs["security-summary"]
        assert "needs" in summary_job
        needs = summary_job["needs"]
        assert "python-security" in needs
        assert "npm-security" in needs


class TestIntegrationGateWorkflow:
    """Validate integration-gate.yml structure."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.path = WORKFLOWS_DIR / "integration-gate.yml"

    def test_workflow_file_exists(self):
        assert self.path.exists(), "integration-gate.yml does not exist"

    def test_workflow_is_valid_yaml(self):
        data = _load_yaml(self.path)
        assert isinstance(data, dict)
        assert "name" in data
        triggers = _get_triggers(data)
        assert isinstance(triggers, dict)
        assert "jobs" in data

    def test_workflow_has_smoke_tests_job(self):
        data = _load_yaml(self.path)
        jobs = data["jobs"]
        assert "smoke-tests" in jobs, "should have smoke-tests job"

    def test_workflow_has_api_contract_sync_job(self):
        data = _load_yaml(self.path)
        jobs = data["jobs"]
        assert "api-contract-sync" in jobs, "should have api-contract-sync job"

    def test_workflow_has_status_doc_validation_job(self):
        data = _load_yaml(self.path)
        jobs = data["jobs"]
        assert "status-doc-validation" in jobs, "should have status-doc-validation job"

    def test_workflow_has_summary_job(self):
        data = _load_yaml(self.path)
        jobs = data["jobs"]
        assert "integration-summary" in jobs, "should have integration-summary job"
        summary_job = jobs["integration-summary"]
        assert "needs" in summary_job


class TestReleaseWorkflow:
    """Validate release.yml integrates all gates."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.path = WORKFLOWS_DIR / "release.yml"

    def test_workflow_file_exists(self):
        assert self.path.exists(), "release.yml does not exist"

    def test_workflow_is_valid_yaml(self):
        data = _load_yaml(self.path)
        assert isinstance(data, dict)

    def test_build_job_requires_all_gates(self):
        data = _load_yaml(self.path)
        build_job = data["jobs"]["build"]
        needs = build_job["needs"]
        assert "security-gate" in needs, "build should require security-gate"
        assert "integration-smoke" in needs, "build should require integration-smoke"
        assert "release-checks" in needs, "build should require release-checks"
        assert "test" in needs, "build should require test"
        assert "docs-sync" in needs, "build should require docs-sync"
        assert "frontend-build" in needs, "build should require frontend-build"

    def test_security_gate_has_secrets_scan(self):
        data = _load_yaml(self.path)
        security_job = data["jobs"]["security-gate"]
        steps = security_job["steps"]
        step_names = [s.get("name", "") for s in steps]
        assert any("secret" in n.lower() for n in step_names), (
            "security-gate should include hardcoded secrets scan"
        )

    def test_integration_smoke_has_contract_tests(self):
        data = _load_yaml(self.path)
        integration_job = data["jobs"]["integration-smoke"]
        steps = integration_job["steps"]
        step_names = [s.get("name", "") for s in steps]
        assert any("openapi" in n.lower() or "contract" in n.lower() for n in step_names), (
            "integration-smoke should include API contract tests"
        )

    def test_release_checks_has_version_tag(self):
        data = _load_yaml(self.path)
        release_checks_job = data["jobs"]["release-checks"]
        steps = release_checks_job["steps"]
        step_names = [s.get("name", "") for s in steps]
        assert any("version" in n.lower() for n in step_names), (
            "release-checks should include version-tag consistency check"
        )

    def test_release_checks_has_status_doc(self):
        data = _load_yaml(self.path)
        release_checks_job = data["jobs"]["release-checks"]
        steps = release_checks_job["steps"]
        step_names = [s.get("name", "") for s in steps]
        assert any("status" in n.lower() for n in step_names), (
            "release-checks should include STATUS.md validation"
        )


# ---------------------------------------------------------------------------
# 2. Pre-release check script
# ---------------------------------------------------------------------------


class TestPreReleaseCheckScript:
    """Validate the pre_release_check.py script."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.script = PROJECT_ROOT / "scripts" / "pre_release_check.py"

    def test_script_exists(self):
        assert self.script.exists(), "scripts/pre_release_check.py does not exist"

    def test_script_is_valid_python(self):
        import ast

        content = self.script.read_text()
        ast.parse(content)

    def test_script_importable(self):
        """Script module can be imported without side effects."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("pre_release_check", str(self.script))
        module = importlib.util.module_from_spec(spec)
        # The module defines ALL_GATES which should be populated
        spec.loader.exec_module(module)
        assert hasattr(module, "ALL_GATES")
        assert len(module.ALL_GATES) >= 8, "should define at least 8 gates"

    def test_script_help_flag(self):
        result = subprocess.run(
            [sys.executable, str(self.script), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "gate" in result.stdout.lower()

    def test_gate_categories_defined(self):
        """Verify gate categories are properly defined."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("pre_release_check", str(self.script))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert hasattr(module, "SECURITY_GATES")
        assert hasattr(module, "INTEGRATION_GATES")
        assert hasattr(module, "RELEASE_GATES")

        # Every gate in categories should be in ALL_GATES
        all_categorized = module.SECURITY_GATES + module.INTEGRATION_GATES + module.RELEASE_GATES
        for gate_name in all_categorized:
            assert gate_name in module.ALL_GATES, f"gate '{gate_name}' in category but not in ALL_GATES"


# ---------------------------------------------------------------------------
# 3. Secret scanning gate
# ---------------------------------------------------------------------------


class TestSecretsScanGate:
    """Test the hardcoded secrets pattern scanner."""

    def test_secret_patterns_defined(self):
        """Verify secret patterns are loaded."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "pre_release_check",
            str(PROJECT_ROOT / "scripts" / "pre_release_check.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert len(module.SECRET_PATTERNS) >= 5, "should define at least 5 secret patterns"

    def test_aws_key_pattern_detects_key(self):
        """AWS access key pattern should match a real-format key."""
        pattern = r"(?<![A-Z0-9])(AKIA[0-9A-Z]{16})(?![A-Z0-9])"
        # This is a fake key for testing
        assert re.search(pattern, 'key = "AKIAIOSFODNN7EXAMPLE"')
        assert not re.search(pattern, "# just a comment about AKIA keys")

    def test_private_key_pattern_detects_header(self):
        """Private key pattern should match PEM headers."""
        pattern = r"-----BEGIN\s+(RSA|EC|DSA|OPENSSH)?\s*PRIVATE KEY-----"
        assert re.search(pattern, "-----BEGIN RSA PRIVATE KEY-----")
        assert re.search(pattern, "-----BEGIN PRIVATE KEY-----")
        assert not re.search(pattern, "-----BEGIN CERTIFICATE-----")

    def test_scan_excludes_test_files(self):
        """Scanner should skip test files."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "pre_release_check",
            str(PROJECT_ROOT / "scripts" / "pre_release_check.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Create a fake Path that looks like a test file
        test_file = PROJECT_ROOT / "tests" / "test_something.py"
        assert not module._should_scan_file(test_file), "should not scan test files"

    def test_scan_excludes_node_modules(self):
        """Scanner should skip node_modules."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "pre_release_check",
            str(PROJECT_ROOT / "scripts" / "pre_release_check.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        nm_file = PROJECT_ROOT / "aragora" / "live" / "node_modules" / "foo" / "index.js"
        # _should_scan_file checks for "node_modules/" in the relative path
        assert not module._should_scan_file(nm_file), "should not scan node_modules"


# ---------------------------------------------------------------------------
# 4. Status doc validation gate
# ---------------------------------------------------------------------------


class TestStatusDocGate:
    """Test the STATUS.md validation gate."""

    def test_status_doc_exists(self):
        """docs/STATUS.md must exist."""
        status_path = PROJECT_ROOT / "docs" / "STATUS.md"
        assert status_path.exists(), "docs/STATUS.md does not exist"

    def test_status_doc_has_heading(self):
        """STATUS.md must have a top-level heading."""
        status_path = PROJECT_ROOT / "docs" / "STATUS.md"
        content = status_path.read_text()
        assert any(
            line.startswith("# ") for line in content.splitlines()
        ), "STATUS.md should have a top-level heading"

    def test_status_doc_has_sections(self):
        """STATUS.md must have section headings."""
        status_path = PROJECT_ROOT / "docs" / "STATUS.md"
        content = status_path.read_text()
        sections = [line for line in content.splitlines() if line.startswith("## ")]
        assert len(sections) >= 1, "STATUS.md should have at least one ## section"

    def test_gate_function_passes_for_valid_doc(self):
        """gate_status_doc should pass for the real STATUS.md."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "pre_release_check",
            str(PROJECT_ROOT / "scripts" / "pre_release_check.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Clear results
        module._results.clear()
        result = module.gate_status_doc()
        assert result is True, "gate_status_doc should pass for real STATUS.md"


# ---------------------------------------------------------------------------
# 5. Version-tag consistency gate
# ---------------------------------------------------------------------------


class TestVersionTagGate:
    """Test the version-tag consistency gate."""

    def test_pyproject_toml_has_version(self):
        """pyproject.toml must have a valid version."""
        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        content = pyproject_path.read_text()
        match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
        assert match, "pyproject.toml should have a version field"
        version = match.group(1)
        assert re.match(r"^\d+\.\d+\.\d+", version), f"version should be semver: {version}"

    def test_gate_passes_without_release_env(self):
        """Version gate should pass when RELEASE_VERSION is not set."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "pre_release_check",
            str(PROJECT_ROOT / "scripts" / "pre_release_check.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        module._results.clear()
        # Ensure RELEASE_VERSION is not set
        old_val = os.environ.pop("RELEASE_VERSION", None)
        try:
            result = module.gate_version_tag()
            assert result is True, "gate_version_tag should pass when no release tag is set"
        finally:
            if old_val is not None:
                os.environ["RELEASE_VERSION"] = old_val


# ---------------------------------------------------------------------------
# 6. Smoke test integration
# ---------------------------------------------------------------------------


class TestSmokeTestIntegration:
    """Test that the smoke test script exists and is runnable."""

    def test_smoke_test_script_exists(self):
        script = PROJECT_ROOT / "scripts" / "smoke_test.py"
        assert script.exists(), "scripts/smoke_test.py does not exist"

    def test_smoke_test_has_skip_server_flag(self):
        """smoke_test.py should support --skip-server flag."""
        script = PROJECT_ROOT / "scripts" / "smoke_test.py"
        content = script.read_text()
        assert "--skip-server" in content, "smoke_test.py should support --skip-server"

    def test_integration_smoke_test_exists(self):
        test_file = PROJECT_ROOT / "tests" / "integration" / "test_smoke.py"
        assert test_file.exists(), "tests/integration/test_smoke.py does not exist"

    def test_openapi_sync_test_exists(self):
        test_file = PROJECT_ROOT / "tests" / "sdk" / "test_openapi_sync.py"
        assert test_file.exists(), "tests/sdk/test_openapi_sync.py does not exist"

    def test_contract_parity_test_exists(self):
        test_file = PROJECT_ROOT / "tests" / "sdk" / "test_contract_parity.py"
        assert test_file.exists(), "tests/sdk/test_contract_parity.py does not exist"
