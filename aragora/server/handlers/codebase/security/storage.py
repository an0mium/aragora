"""
In-memory storage and service registry integration for security scans.

This module provides:
- Thread-safe storage for scan results
- Service registry helpers for scanners and clients
- Path traversal validation for repo IDs
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING

from aragora.analysis.codebase import (
    CVEClient,
    DependencyScanner,
    SASTScanner,
    SBOMGenerator,
    SecretsScanner,
)
from aragora.services import ServiceRegistry
from aragora.server.validation import validate_no_path_traversal, SAFE_ID_PATTERN

if TYPE_CHECKING:
    from aragora.analysis.codebase import (
        ScanResult,
        SASTScanResult,
        SBOMResult,
        SecretsScanResult,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# Path Traversal Protection
# =============================================================================


def safe_repo_id(repo_id: str) -> tuple[bool, str | None]:
    """
    Validate a repository ID to prevent path traversal attacks.

    Ensures the repo_id:
    - Does not contain '..' (path traversal)
    - Does not contain '/' or '\\' (directory separator)
    - Matches safe alphanumeric pattern (letters, numbers, hyphens, underscores)

    Args:
        repo_id: The repository identifier to validate

    Returns:
        Tuple of (is_valid, error_message).
        If valid, returns (True, None).
        If invalid, returns (False, error_message).

    Examples:
        >>> safe_repo_id("my-repo-123")
        (True, None)
        >>> safe_repo_id("../etc/passwd")
        (False, "Invalid repo ID: path traversal not allowed")
        >>> safe_repo_id("repo/subdir")
        (False, "Invalid repo ID: must not contain path separators")
    """
    if not repo_id:
        return False, "Invalid repo ID: cannot be empty"

    # Check for path traversal sequences
    is_valid, err = validate_no_path_traversal(repo_id)
    if not is_valid:
        return False, "Invalid repo ID: path traversal not allowed"

    # Check for directory separators
    if "/" in repo_id or "\\" in repo_id:
        return False, "Invalid repo ID: must not contain path separators"

    # Validate against safe pattern (alphanumeric, hyphens, underscores)
    if not SAFE_ID_PATTERN.match(repo_id):
        return (
            False,
            "Invalid repo ID: must contain only alphanumeric characters, hyphens, and underscores (1-64 chars)",
        )

    return True, None


# =============================================================================
# In-Memory Storage (replace with database in production)
# =============================================================================

# Dependency scan storage
_scan_results: dict[str, dict[str, ScanResult]] = {}  # repo_id -> {scan_id -> result}
_scan_lock = threading.Lock()
_running_scans: dict[str, asyncio.Task] = {}

# Secrets scan storage
_secrets_scan_results: dict[str, dict[str, SecretsScanResult]] = {}
_secrets_scan_lock = threading.Lock()
_running_secrets_scans: dict[str, asyncio.Task] = {}

# SAST scan storage
_sast_scan_results: dict[str, dict[str, SASTScanResult]] = {}
_sast_scan_lock = threading.Lock()
_running_sast_scans: dict[str, asyncio.Task] = {}

# SBOM storage
_sbom_results: dict[str, dict[str, SBOMResult]] = {}  # repo_id -> {sbom_id -> result}
_sbom_lock = threading.Lock()
_running_sbom_generations: dict[str, asyncio.Task] = {}


# =============================================================================
# Dependency Scan Storage Helpers
# =============================================================================


def get_or_create_repo_scans(repo_id: str) -> dict[str, ScanResult]:
    """Get or create scan storage for a repository."""
    with _scan_lock:
        if repo_id not in _scan_results:
            _scan_results[repo_id] = {}
        return _scan_results[repo_id]


def get_scan_lock() -> threading.Lock:
    """Get the scan lock for thread-safe operations."""
    return _scan_lock


def get_running_scans() -> dict[str, asyncio.Task]:
    """Get the running scans dictionary."""
    return _running_scans


# =============================================================================
# Secrets Scan Storage Helpers
# =============================================================================


def get_or_create_secrets_scans(repo_id: str) -> dict[str, SecretsScanResult]:
    """Get or create secrets scan storage for a repository."""
    with _secrets_scan_lock:
        if repo_id not in _secrets_scan_results:
            _secrets_scan_results[repo_id] = {}
        return _secrets_scan_results[repo_id]


def get_secrets_scan_lock() -> threading.Lock:
    """Get the secrets scan lock for thread-safe operations."""
    return _secrets_scan_lock


def get_running_secrets_scans() -> dict[str, asyncio.Task]:
    """Get the running secrets scans dictionary."""
    return _running_secrets_scans


# =============================================================================
# SAST Scan Storage Helpers
# =============================================================================


def get_sast_scan_results() -> dict[str, dict[str, SASTScanResult]]:
    """Get the SAST scan results storage."""
    return _sast_scan_results


def get_sast_scan_lock() -> threading.Lock:
    """Get the SAST scan lock for thread-safe operations."""
    return _sast_scan_lock


def get_running_sast_scans() -> dict[str, asyncio.Task]:
    """Get the running SAST scans dictionary."""
    return _running_sast_scans


# =============================================================================
# SBOM Storage Helpers
# =============================================================================


def get_or_create_sbom_results(repo_id: str) -> dict[str, SBOMResult]:
    """Get or create SBOM storage for a repository."""
    with _sbom_lock:
        if repo_id not in _sbom_results:
            _sbom_results[repo_id] = {}
        return _sbom_results[repo_id]


def get_sbom_lock() -> threading.Lock:
    """Get the SBOM lock for thread-safe operations."""
    return _sbom_lock


def get_running_sbom_generations() -> dict[str, asyncio.Task]:
    """Get the running SBOM generations dictionary."""
    return _running_sbom_generations


# =============================================================================
# Service Registry Integration
# =============================================================================


def get_scanner() -> DependencyScanner:
    """Get or create DependencyScanner from service registry."""
    registry = ServiceRegistry.get()
    if not registry.has(DependencyScanner):
        scanner = DependencyScanner()
        registry.register(DependencyScanner, scanner)
        logger.info("Registered DependencyScanner with service registry")
    return registry.resolve(DependencyScanner)


def get_cve_client() -> CVEClient:
    """Get or create CVEClient from service registry."""
    registry = ServiceRegistry.get()
    if not registry.has(CVEClient):
        client = CVEClient()
        registry.register(CVEClient, client)
        logger.info("Registered CVEClient with service registry")
    return registry.resolve(CVEClient)


def get_secrets_scanner() -> SecretsScanner:
    """Get or create SecretsScanner from service registry."""
    registry = ServiceRegistry.get()
    if not registry.has(SecretsScanner):
        scanner = SecretsScanner()
        registry.register(SecretsScanner, scanner)
        logger.info("Registered SecretsScanner with service registry")
    return registry.resolve(SecretsScanner)


def get_sast_scanner() -> SASTScanner:
    """Get or create SASTScanner from service registry."""
    registry = ServiceRegistry.get()
    if not registry.has(SASTScanner):
        scanner = SASTScanner()
        registry.register(SASTScanner, scanner)
        logger.info("Registered SASTScanner with service registry")
    return registry.resolve(SASTScanner)


def get_sbom_generator() -> SBOMGenerator:
    """Get or create SBOMGenerator from service registry."""
    registry = ServiceRegistry.get()
    if not registry.has(SBOMGenerator):
        generator = SBOMGenerator()
        registry.register(SBOMGenerator, generator)
        logger.info("Registered SBOMGenerator with service registry")
    return registry.resolve(SBOMGenerator)
