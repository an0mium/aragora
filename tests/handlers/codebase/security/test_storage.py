"""
Tests for the security scan storage and service registry module
(aragora/server/handlers/codebase/security/storage.py).

Covers:
- safe_repo_id: path traversal validation for repository IDs
- Dependency scan storage helpers (get_or_create_repo_scans, locks, running scans)
- Secrets scan storage helpers
- SAST scan storage helpers
- SBOM storage helpers
- Service registry integration (get_scanner, get_cve_client, get_secrets_scanner,
  get_sast_scanner, get_sbom_generator)

Tests include: happy paths, error/edge cases, thread safety, registry caching,
and input validation (path traversal, injection, empty/long strings).
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.codebase.security.storage import (
    # Path validation
    safe_repo_id,
    # Dependency scan storage
    get_or_create_repo_scans,
    get_scan_lock,
    get_running_scans,
    _scan_results,
    _running_scans,
    # Secrets scan storage
    get_or_create_secrets_scans,
    get_secrets_scan_lock,
    get_running_secrets_scans,
    _secrets_scan_results,
    _running_secrets_scans,
    # SAST scan storage
    get_sast_scan_results,
    get_sast_scan_lock,
    get_running_sast_scans,
    _sast_scan_results,
    _running_sast_scans,
    # SBOM storage
    get_or_create_sbom_results,
    get_sbom_lock,
    get_running_sbom_generations,
    _sbom_results,
    _running_sbom_generations,
    # Service registry
    get_scanner,
    get_cve_client,
    get_secrets_scanner,
    get_sast_scanner,
    get_sbom_generator,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def clear_all_storage():
    """Clear all in-memory storage between tests."""
    _scan_results.clear()
    _running_scans.clear()
    _secrets_scan_results.clear()
    _running_secrets_scans.clear()
    _sast_scan_results.clear()
    _running_sast_scans.clear()
    _sbom_results.clear()
    _running_sbom_generations.clear()
    yield
    _scan_results.clear()
    _running_scans.clear()
    _secrets_scan_results.clear()
    _running_secrets_scans.clear()
    _sast_scan_results.clear()
    _running_sast_scans.clear()
    _sbom_results.clear()
    _running_sbom_generations.clear()


@pytest.fixture(autouse=True)
def reset_service_registry():
    """Reset the service registry between tests to prevent leaking state."""
    from aragora.services import ServiceRegistry

    ServiceRegistry.reset()
    yield
    ServiceRegistry.reset()


# ============================================================================
# safe_repo_id - Valid IDs
# ============================================================================


class TestSafeRepoIdValid:
    """Tests for valid repository IDs."""

    def test_simple_alphanumeric(self):
        is_valid, err = safe_repo_id("myrepo123")
        assert is_valid is True
        assert err is None

    def test_with_hyphens(self):
        is_valid, err = safe_repo_id("my-repo-123")
        assert is_valid is True
        assert err is None

    def test_with_underscores(self):
        is_valid, err = safe_repo_id("my_repo_123")
        assert is_valid is True
        assert err is None

    def test_mixed_case(self):
        is_valid, err = safe_repo_id("MyRepo")
        assert is_valid is True
        assert err is None

    def test_single_character(self):
        is_valid, err = safe_repo_id("a")
        assert is_valid is True
        assert err is None

    def test_single_digit(self):
        is_valid, err = safe_repo_id("1")
        assert is_valid is True
        assert err is None

    def test_max_length_64(self):
        repo_id = "a" * 64
        is_valid, err = safe_repo_id(repo_id)
        assert is_valid is True
        assert err is None

    def test_all_hyphens_underscores_mixed(self):
        is_valid, err = safe_repo_id("a-b_c-d_e")
        assert is_valid is True
        assert err is None

    def test_uppercase_only(self):
        is_valid, err = safe_repo_id("MYREPO")
        assert is_valid is True
        assert err is None

    def test_digits_only(self):
        is_valid, err = safe_repo_id("12345")
        assert is_valid is True
        assert err is None


# ============================================================================
# safe_repo_id - Invalid IDs
# ============================================================================


class TestSafeRepoIdInvalid:
    """Tests for invalid repository IDs."""

    def test_empty_string(self):
        is_valid, err = safe_repo_id("")
        assert is_valid is False
        assert "cannot be empty" in err

    def test_path_traversal_dotdot(self):
        is_valid, err = safe_repo_id("../etc/passwd")
        assert is_valid is False
        assert "path traversal" in err.lower()

    def test_path_traversal_embedded(self):
        is_valid, err = safe_repo_id("repo/../secret")
        assert is_valid is False
        assert "path traversal" in err.lower()

    def test_forward_slash(self):
        is_valid, err = safe_repo_id("repo/subdir")
        assert is_valid is False
        assert "path separator" in err.lower() or "path traversal" in err.lower()

    def test_backslash(self):
        is_valid, err = safe_repo_id("repo\\subdir")
        assert is_valid is False
        assert "path separator" in err.lower() or "pattern" in err.lower()

    def test_too_long(self):
        repo_id = "a" * 65
        is_valid, err = safe_repo_id(repo_id)
        assert is_valid is False
        assert "alphanumeric" in err.lower() or "1-64" in err

    def test_special_characters_dot(self):
        is_valid, err = safe_repo_id("repo.name")
        assert is_valid is False

    def test_special_characters_at(self):
        is_valid, err = safe_repo_id("repo@name")
        assert is_valid is False

    def test_special_characters_space(self):
        is_valid, err = safe_repo_id("repo name")
        assert is_valid is False

    def test_special_characters_semicolon(self):
        is_valid, err = safe_repo_id("repo;name")
        assert is_valid is False

    def test_special_characters_pipe(self):
        is_valid, err = safe_repo_id("repo|name")
        assert is_valid is False

    def test_null_byte(self):
        is_valid, err = safe_repo_id("repo\x00name")
        assert is_valid is False

    def test_only_dots(self):
        is_valid, err = safe_repo_id("..")
        assert is_valid is False
        assert "path traversal" in err.lower()

    def test_triple_dots(self):
        # "..." contains ".." so should fail path traversal
        is_valid, err = safe_repo_id("...")
        assert is_valid is False

    def test_url_encoded_traversal(self):
        # %2e%2e -> not literal ".." but still not alphanumeric
        is_valid, err = safe_repo_id("%2e%2e")
        assert is_valid is False

    def test_unicode_characters(self):
        is_valid, err = safe_repo_id("repo\u00e9")
        assert is_valid is False

    def test_newline_injection(self):
        is_valid, err = safe_repo_id("repo\nname")
        assert is_valid is False

    def test_tab_injection(self):
        is_valid, err = safe_repo_id("repo\tname")
        assert is_valid is False


# ============================================================================
# Dependency Scan Storage Helpers
# ============================================================================


class TestDependencyScanStorage:
    """Tests for dependency scan storage functions."""

    def test_get_or_create_new_repo(self):
        """First call creates a new empty dict for the repo."""
        scans = get_or_create_repo_scans("repo-1")
        assert scans == {}
        assert "repo-1" in _scan_results

    def test_get_or_create_existing_repo(self):
        """Second call returns the same dict object."""
        scans1 = get_or_create_repo_scans("repo-1")
        scans1["scan-1"] = {"status": "done"}
        scans2 = get_or_create_repo_scans("repo-1")
        assert scans2 is scans1
        assert scans2["scan-1"] == {"status": "done"}

    def test_get_or_create_multiple_repos(self):
        """Different repos get independent storage."""
        scans_a = get_or_create_repo_scans("repo-a")
        scans_b = get_or_create_repo_scans("repo-b")
        scans_a["scan-1"] = "a"
        assert "scan-1" not in scans_b

    def test_get_scan_lock_returns_lock(self):
        lock = get_scan_lock()
        assert isinstance(lock, type(threading.Lock()))

    def test_get_scan_lock_is_consistent(self):
        """Always returns the same lock object."""
        lock1 = get_scan_lock()
        lock2 = get_scan_lock()
        assert lock1 is lock2

    def test_get_running_scans_returns_dict(self):
        running = get_running_scans()
        assert isinstance(running, dict)

    def test_get_running_scans_is_mutable(self):
        """Returns the actual module-level dict, not a copy."""
        running = get_running_scans()
        running["task-1"] = "mock_task"
        assert _running_scans["task-1"] == "mock_task"

    def test_thread_safety_concurrent_repo_creation(self):
        """Multiple threads creating repos concurrently don't corrupt state."""
        results = {}
        errors = []

        def create_repo(repo_id):
            try:
                scans = get_or_create_repo_scans(repo_id)
                results[repo_id] = scans
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=create_repo, args=(f"repo-{i}",))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 20

    def test_thread_safety_same_repo_concurrent(self):
        """Multiple threads accessing same repo get the same dict."""
        results = []
        errors = []

        def access_repo():
            try:
                scans = get_or_create_repo_scans("shared-repo")
                results.append(id(scans))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=access_repo) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All threads should get the same dict object
        assert len(set(results)) == 1


# ============================================================================
# Secrets Scan Storage Helpers
# ============================================================================


class TestSecretsScanStorage:
    """Tests for secrets scan storage functions."""

    def test_get_or_create_new_repo(self):
        scans = get_or_create_secrets_scans("repo-1")
        assert scans == {}
        assert "repo-1" in _secrets_scan_results

    def test_get_or_create_existing_repo(self):
        scans1 = get_or_create_secrets_scans("repo-1")
        scans1["scan-1"] = {"status": "done"}
        scans2 = get_or_create_secrets_scans("repo-1")
        assert scans2 is scans1
        assert scans2["scan-1"] == {"status": "done"}

    def test_get_or_create_multiple_repos(self):
        scans_a = get_or_create_secrets_scans("repo-a")
        scans_b = get_or_create_secrets_scans("repo-b")
        scans_a["scan-1"] = "a"
        assert "scan-1" not in scans_b

    def test_get_secrets_scan_lock_returns_lock(self):
        lock = get_secrets_scan_lock()
        assert isinstance(lock, type(threading.Lock()))

    def test_get_secrets_scan_lock_is_consistent(self):
        lock1 = get_secrets_scan_lock()
        lock2 = get_secrets_scan_lock()
        assert lock1 is lock2

    def test_get_running_secrets_scans_returns_dict(self):
        running = get_running_secrets_scans()
        assert isinstance(running, dict)

    def test_get_running_secrets_scans_is_mutable(self):
        running = get_running_secrets_scans()
        running["task-1"] = "mock_task"
        assert _running_secrets_scans["task-1"] == "mock_task"


# ============================================================================
# SAST Scan Storage Helpers
# ============================================================================


class TestSASTScanStorage:
    """Tests for SAST scan storage functions."""

    def test_get_sast_scan_results_returns_dict(self):
        results = get_sast_scan_results()
        assert isinstance(results, dict)

    def test_get_sast_scan_results_is_module_level(self):
        """Returns the actual module-level dict."""
        results = get_sast_scan_results()
        results["repo-1"] = {"scan-1": "result"}
        assert _sast_scan_results["repo-1"] == {"scan-1": "result"}

    def test_get_sast_scan_results_persists_across_calls(self):
        results1 = get_sast_scan_results()
        results1["key"] = "value"
        results2 = get_sast_scan_results()
        assert results2["key"] == "value"
        assert results1 is results2

    def test_get_sast_scan_lock_returns_lock(self):
        lock = get_sast_scan_lock()
        assert isinstance(lock, type(threading.Lock()))

    def test_get_sast_scan_lock_is_consistent(self):
        lock1 = get_sast_scan_lock()
        lock2 = get_sast_scan_lock()
        assert lock1 is lock2

    def test_get_running_sast_scans_returns_dict(self):
        running = get_running_sast_scans()
        assert isinstance(running, dict)

    def test_get_running_sast_scans_is_mutable(self):
        running = get_running_sast_scans()
        running["task-1"] = "mock_task"
        assert _running_sast_scans["task-1"] == "mock_task"


# ============================================================================
# SBOM Storage Helpers
# ============================================================================


class TestSBOMStorage:
    """Tests for SBOM storage functions."""

    def test_get_or_create_new_repo(self):
        results = get_or_create_sbom_results("repo-1")
        assert results == {}
        assert "repo-1" in _sbom_results

    def test_get_or_create_existing_repo(self):
        results1 = get_or_create_sbom_results("repo-1")
        results1["sbom-1"] = {"format": "cyclonedx"}
        results2 = get_or_create_sbom_results("repo-1")
        assert results2 is results1
        assert results2["sbom-1"] == {"format": "cyclonedx"}

    def test_get_or_create_multiple_repos(self):
        results_a = get_or_create_sbom_results("repo-a")
        results_b = get_or_create_sbom_results("repo-b")
        results_a["sbom-1"] = "a"
        assert "sbom-1" not in results_b

    def test_get_sbom_lock_returns_lock(self):
        lock = get_sbom_lock()
        assert isinstance(lock, type(threading.Lock()))

    def test_get_sbom_lock_is_consistent(self):
        lock1 = get_sbom_lock()
        lock2 = get_sbom_lock()
        assert lock1 is lock2

    def test_get_running_sbom_generations_returns_dict(self):
        running = get_running_sbom_generations()
        assert isinstance(running, dict)

    def test_get_running_sbom_generations_is_mutable(self):
        running = get_running_sbom_generations()
        running["task-1"] = "mock_task"
        assert _running_sbom_generations["task-1"] == "mock_task"


# ============================================================================
# Service Registry - get_scanner
# ============================================================================


def _mock_registry(has_service=False, resolved_instance=None):
    """Create a mocked ServiceRegistry context.

    Returns (MockRegistry_cls, mock_reg_instance) for assertion.
    """
    mock_reg = MagicMock()
    mock_reg.has.return_value = has_service
    mock_reg.resolve.return_value = resolved_instance
    return mock_reg


class TestGetScanner:
    """Tests for the get_scanner service registry helper."""

    def test_creates_and_registers_scanner(self):
        with patch(
            "aragora.server.handlers.codebase.security.storage.DependencyScanner"
        ) as MockScanner, patch(
            "aragora.server.handlers.codebase.security.storage.ServiceRegistry"
        ) as MockRegistryCls:
            mock_instance = MagicMock()
            MockScanner.return_value = mock_instance
            mock_reg = _mock_registry(has_service=False, resolved_instance=mock_instance)
            MockRegistryCls.get.return_value = mock_reg

            scanner = get_scanner()

            MockScanner.assert_called_once()
            mock_reg.register.assert_called_once_with(MockScanner, mock_instance)
            assert scanner is mock_instance

    def test_returns_existing_scanner(self):
        """Second call returns the cached scanner via registry, no new construction."""
        existing = MagicMock()
        with patch(
            "aragora.server.handlers.codebase.security.storage.DependencyScanner"
        ) as MockScanner, patch(
            "aragora.server.handlers.codebase.security.storage.ServiceRegistry"
        ) as MockRegistryCls:
            mock_reg = _mock_registry(has_service=True, resolved_instance=existing)
            MockRegistryCls.get.return_value = mock_reg

            scanner = get_scanner()

            MockScanner.assert_not_called()
            mock_reg.register.assert_not_called()
            assert scanner is existing

    def test_uses_service_registry(self):
        """Verifies interaction with the ServiceRegistry."""
        with patch(
            "aragora.server.handlers.codebase.security.storage.DependencyScanner"
        ) as MockScanner, patch(
            "aragora.server.handlers.codebase.security.storage.ServiceRegistry"
        ) as MockRegistryCls:
            mock_instance = MagicMock()
            MockScanner.return_value = mock_instance
            mock_reg = _mock_registry(has_service=False, resolved_instance=mock_instance)
            MockRegistryCls.get.return_value = mock_reg

            scanner = get_scanner()

            mock_reg.has.assert_called_once_with(MockScanner)
            mock_reg.register.assert_called_once_with(MockScanner, mock_instance)
            mock_reg.resolve.assert_called_once_with(MockScanner)
            assert scanner is mock_instance

    def test_skips_registration_when_exists(self):
        """If scanner already registered, does not create a new one."""
        existing = MagicMock()
        with patch(
            "aragora.server.handlers.codebase.security.storage.DependencyScanner"
        ) as MockScanner, patch(
            "aragora.server.handlers.codebase.security.storage.ServiceRegistry"
        ) as MockRegistryCls:
            mock_reg = _mock_registry(has_service=True, resolved_instance=existing)
            MockRegistryCls.get.return_value = mock_reg

            scanner = get_scanner()

            mock_reg.register.assert_not_called()
            assert scanner is existing


# ============================================================================
# Service Registry - get_cve_client
# ============================================================================


class TestGetCVEClient:
    """Tests for the get_cve_client service registry helper."""

    def test_creates_and_registers_client(self):
        with patch(
            "aragora.server.handlers.codebase.security.storage.CVEClient"
        ) as MockClient, patch(
            "aragora.server.handlers.codebase.security.storage.ServiceRegistry"
        ) as MockRegistryCls:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance
            mock_reg = _mock_registry(has_service=False, resolved_instance=mock_instance)
            MockRegistryCls.get.return_value = mock_reg

            client = get_cve_client()

            MockClient.assert_called_once()
            mock_reg.register.assert_called_once_with(MockClient, mock_instance)
            assert client is mock_instance

    def test_returns_existing_client(self):
        existing = MagicMock()
        with patch(
            "aragora.server.handlers.codebase.security.storage.CVEClient"
        ) as MockClient, patch(
            "aragora.server.handlers.codebase.security.storage.ServiceRegistry"
        ) as MockRegistryCls:
            mock_reg = _mock_registry(has_service=True, resolved_instance=existing)
            MockRegistryCls.get.return_value = mock_reg

            client = get_cve_client()

            MockClient.assert_not_called()
            mock_reg.register.assert_not_called()
            assert client is existing

    def test_uses_service_registry(self):
        with patch(
            "aragora.server.handlers.codebase.security.storage.CVEClient"
        ) as MockClient, patch(
            "aragora.server.handlers.codebase.security.storage.ServiceRegistry"
        ) as MockRegistryCls:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance
            mock_reg = _mock_registry(has_service=False, resolved_instance=mock_instance)
            MockRegistryCls.get.return_value = mock_reg

            client = get_cve_client()

            mock_reg.has.assert_called_once_with(MockClient)
            mock_reg.register.assert_called_once_with(MockClient, mock_instance)
            mock_reg.resolve.assert_called_once_with(MockClient)

    def test_skips_registration_when_exists(self):
        existing = MagicMock()
        with patch(
            "aragora.server.handlers.codebase.security.storage.CVEClient"
        ) as MockClient, patch(
            "aragora.server.handlers.codebase.security.storage.ServiceRegistry"
        ) as MockRegistryCls:
            mock_reg = _mock_registry(has_service=True, resolved_instance=existing)
            MockRegistryCls.get.return_value = mock_reg

            client = get_cve_client()

            mock_reg.register.assert_not_called()
            assert client is existing


# ============================================================================
# Service Registry - get_secrets_scanner
# ============================================================================


class TestGetSecretsScanner:
    """Tests for the get_secrets_scanner service registry helper."""

    def test_creates_and_registers_scanner(self):
        with patch(
            "aragora.server.handlers.codebase.security.storage.SecretsScanner"
        ) as MockScanner, patch(
            "aragora.server.handlers.codebase.security.storage.ServiceRegistry"
        ) as MockRegistryCls:
            mock_instance = MagicMock()
            MockScanner.return_value = mock_instance
            mock_reg = _mock_registry(has_service=False, resolved_instance=mock_instance)
            MockRegistryCls.get.return_value = mock_reg

            scanner = get_secrets_scanner()

            MockScanner.assert_called_once()
            mock_reg.register.assert_called_once_with(MockScanner, mock_instance)
            assert scanner is mock_instance

    def test_returns_existing_scanner(self):
        existing = MagicMock()
        with patch(
            "aragora.server.handlers.codebase.security.storage.SecretsScanner"
        ) as MockScanner, patch(
            "aragora.server.handlers.codebase.security.storage.ServiceRegistry"
        ) as MockRegistryCls:
            mock_reg = _mock_registry(has_service=True, resolved_instance=existing)
            MockRegistryCls.get.return_value = mock_reg

            scanner = get_secrets_scanner()

            MockScanner.assert_not_called()
            mock_reg.register.assert_not_called()
            assert scanner is existing

    def test_uses_service_registry(self):
        with patch(
            "aragora.server.handlers.codebase.security.storage.SecretsScanner"
        ) as MockScanner, patch(
            "aragora.server.handlers.codebase.security.storage.ServiceRegistry"
        ) as MockRegistryCls:
            mock_instance = MagicMock()
            MockScanner.return_value = mock_instance
            mock_reg = _mock_registry(has_service=False, resolved_instance=mock_instance)
            MockRegistryCls.get.return_value = mock_reg

            scanner = get_secrets_scanner()

            mock_reg.has.assert_called_once_with(MockScanner)
            mock_reg.register.assert_called_once_with(MockScanner, mock_instance)
            mock_reg.resolve.assert_called_once_with(MockScanner)

    def test_skips_registration_when_exists(self):
        existing = MagicMock()
        with patch(
            "aragora.server.handlers.codebase.security.storage.SecretsScanner"
        ) as MockScanner, patch(
            "aragora.server.handlers.codebase.security.storage.ServiceRegistry"
        ) as MockRegistryCls:
            mock_reg = _mock_registry(has_service=True, resolved_instance=existing)
            MockRegistryCls.get.return_value = mock_reg

            scanner = get_secrets_scanner()

            mock_reg.register.assert_not_called()
            assert scanner is existing


# ============================================================================
# Service Registry - get_sast_scanner
# ============================================================================


class TestGetSASTScanner:
    """Tests for the get_sast_scanner service registry helper."""

    def test_creates_and_registers_scanner(self):
        with patch(
            "aragora.server.handlers.codebase.security.storage.SASTScanner"
        ) as MockScanner, patch(
            "aragora.server.handlers.codebase.security.storage.ServiceRegistry"
        ) as MockRegistryCls:
            mock_instance = MagicMock()
            MockScanner.return_value = mock_instance
            mock_reg = _mock_registry(has_service=False, resolved_instance=mock_instance)
            MockRegistryCls.get.return_value = mock_reg

            scanner = get_sast_scanner()

            MockScanner.assert_called_once()
            mock_reg.register.assert_called_once_with(MockScanner, mock_instance)
            assert scanner is mock_instance

    def test_returns_existing_scanner(self):
        existing = MagicMock()
        with patch(
            "aragora.server.handlers.codebase.security.storage.SASTScanner"
        ) as MockScanner, patch(
            "aragora.server.handlers.codebase.security.storage.ServiceRegistry"
        ) as MockRegistryCls:
            mock_reg = _mock_registry(has_service=True, resolved_instance=existing)
            MockRegistryCls.get.return_value = mock_reg

            scanner = get_sast_scanner()

            MockScanner.assert_not_called()
            mock_reg.register.assert_not_called()
            assert scanner is existing

    def test_uses_service_registry(self):
        with patch(
            "aragora.server.handlers.codebase.security.storage.SASTScanner"
        ) as MockScanner, patch(
            "aragora.server.handlers.codebase.security.storage.ServiceRegistry"
        ) as MockRegistryCls:
            mock_instance = MagicMock()
            MockScanner.return_value = mock_instance
            mock_reg = _mock_registry(has_service=False, resolved_instance=mock_instance)
            MockRegistryCls.get.return_value = mock_reg

            scanner = get_sast_scanner()

            mock_reg.has.assert_called_once_with(MockScanner)
            mock_reg.register.assert_called_once_with(MockScanner, mock_instance)
            mock_reg.resolve.assert_called_once_with(MockScanner)

    def test_skips_registration_when_exists(self):
        existing = MagicMock()
        with patch(
            "aragora.server.handlers.codebase.security.storage.SASTScanner"
        ) as MockScanner, patch(
            "aragora.server.handlers.codebase.security.storage.ServiceRegistry"
        ) as MockRegistryCls:
            mock_reg = _mock_registry(has_service=True, resolved_instance=existing)
            MockRegistryCls.get.return_value = mock_reg

            scanner = get_sast_scanner()

            mock_reg.register.assert_not_called()
            assert scanner is existing


# ============================================================================
# Service Registry - get_sbom_generator
# ============================================================================


class TestGetSBOMGenerator:
    """Tests for the get_sbom_generator service registry helper."""

    def test_creates_and_registers_generator(self):
        with patch(
            "aragora.server.handlers.codebase.security.storage.SBOMGenerator"
        ) as MockGenerator, patch(
            "aragora.server.handlers.codebase.security.storage.ServiceRegistry"
        ) as MockRegistryCls:
            mock_instance = MagicMock()
            MockGenerator.return_value = mock_instance
            mock_reg = _mock_registry(has_service=False, resolved_instance=mock_instance)
            MockRegistryCls.get.return_value = mock_reg

            generator = get_sbom_generator()

            MockGenerator.assert_called_once()
            mock_reg.register.assert_called_once_with(MockGenerator, mock_instance)
            assert generator is mock_instance

    def test_returns_existing_generator(self):
        existing = MagicMock()
        with patch(
            "aragora.server.handlers.codebase.security.storage.SBOMGenerator"
        ) as MockGenerator, patch(
            "aragora.server.handlers.codebase.security.storage.ServiceRegistry"
        ) as MockRegistryCls:
            mock_reg = _mock_registry(has_service=True, resolved_instance=existing)
            MockRegistryCls.get.return_value = mock_reg

            generator = get_sbom_generator()

            MockGenerator.assert_not_called()
            mock_reg.register.assert_not_called()
            assert generator is existing

    def test_uses_service_registry(self):
        with patch(
            "aragora.server.handlers.codebase.security.storage.SBOMGenerator"
        ) as MockGenerator, patch(
            "aragora.server.handlers.codebase.security.storage.ServiceRegistry"
        ) as MockRegistryCls:
            mock_instance = MagicMock()
            MockGenerator.return_value = mock_instance
            mock_reg = _mock_registry(has_service=False, resolved_instance=mock_instance)
            MockRegistryCls.get.return_value = mock_reg

            generator = get_sbom_generator()

            mock_reg.has.assert_called_once_with(MockGenerator)
            mock_reg.register.assert_called_once_with(MockGenerator, mock_instance)
            mock_reg.resolve.assert_called_once_with(MockGenerator)

    def test_skips_registration_when_exists(self):
        existing = MagicMock()
        with patch(
            "aragora.server.handlers.codebase.security.storage.SBOMGenerator"
        ) as MockGenerator, patch(
            "aragora.server.handlers.codebase.security.storage.ServiceRegistry"
        ) as MockRegistryCls:
            mock_reg = _mock_registry(has_service=True, resolved_instance=existing)
            MockRegistryCls.get.return_value = mock_reg

            generator = get_sbom_generator()

            mock_reg.register.assert_not_called()
            assert generator is existing


# ============================================================================
# Cross-Storage Isolation
# ============================================================================


class TestCrossStorageIsolation:
    """Verify that different storage types don't interfere with each other."""

    def test_repo_scans_isolated_from_secrets_scans(self):
        scans = get_or_create_repo_scans("repo-1")
        secrets = get_or_create_secrets_scans("repo-1")
        scans["scan-1"] = "dependency"
        secrets["scan-1"] = "secret"
        assert scans is not secrets
        assert scans["scan-1"] == "dependency"
        assert secrets["scan-1"] == "secret"

    def test_repo_scans_isolated_from_sbom(self):
        scans = get_or_create_repo_scans("repo-1")
        sbom = get_or_create_sbom_results("repo-1")
        scans["id-1"] = "scan"
        sbom["id-1"] = "sbom"
        assert scans is not sbom
        assert scans["id-1"] == "scan"
        assert sbom["id-1"] == "sbom"

    def test_secrets_scans_isolated_from_sbom(self):
        secrets = get_or_create_secrets_scans("repo-1")
        sbom = get_or_create_sbom_results("repo-1")
        secrets["id-1"] = "secret"
        sbom["id-1"] = "sbom"
        assert secrets is not sbom

    def test_sast_results_isolated_from_others(self):
        sast = get_sast_scan_results()
        scans = get_or_create_repo_scans("repo-1")
        sast["repo-1"] = {"s": "sast"}
        assert "s" not in scans

    def test_running_scans_isolated(self):
        """Each scan type has its own running-tasks dictionary."""
        dep_running = get_running_scans()
        sec_running = get_running_secrets_scans()
        sast_running = get_running_sast_scans()
        sbom_running = get_running_sbom_generations()

        dep_running["t1"] = "dep"
        assert "t1" not in sec_running
        assert "t1" not in sast_running
        assert "t1" not in sbom_running

    def test_locks_are_distinct(self):
        """Each scan type uses a separate lock."""
        dep_lock = get_scan_lock()
        sec_lock = get_secrets_scan_lock()
        sast_lock = get_sast_scan_lock()
        sbom_lock_obj = get_sbom_lock()

        locks = {id(dep_lock), id(sec_lock), id(sast_lock), id(sbom_lock_obj)}
        assert len(locks) == 4, "All four locks must be distinct objects"


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Edge cases for storage and validation functions."""

    def test_safe_repo_id_hyphen_only(self):
        is_valid, err = safe_repo_id("-")
        assert is_valid is True
        assert err is None

    def test_safe_repo_id_underscore_only(self):
        is_valid, err = safe_repo_id("_")
        assert is_valid is True
        assert err is None

    def test_safe_repo_id_hyphen_underscore(self):
        is_valid, err = safe_repo_id("-_")
        assert is_valid is True
        assert err is None

    def test_get_or_create_idempotent(self):
        """Calling get_or_create multiple times is idempotent."""
        for _ in range(100):
            get_or_create_repo_scans("repo-stress")
        assert len(_scan_results) == 1

    def test_storage_preserves_order_of_insertion(self):
        """Scan results within a repo preserve insertion order (dict order)."""
        scans = get_or_create_repo_scans("repo-ordered")
        for i in range(10):
            scans[f"scan-{i}"] = i
        keys = list(scans.keys())
        assert keys == [f"scan-{i}" for i in range(10)]

    def test_lock_can_be_acquired_and_released(self):
        """Locks are functional (not just objects)."""
        lock = get_scan_lock()
        assert lock.acquire(timeout=1)
        lock.release()

    def test_lock_reentrance_not_supported(self):
        """Module-level locks are threading.Lock (not RLock), so not reentrant."""
        lock = get_scan_lock()
        assert lock.acquire(timeout=1)
        # Second acquire on same thread should fail (non-reentrant)
        assert not lock.acquire(timeout=0.01)
        lock.release()
