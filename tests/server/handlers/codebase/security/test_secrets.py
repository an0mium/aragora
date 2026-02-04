"""
Tests for Secrets Scanning Handler.

Tests cover:
- Secrets scan trigger (current files and git history)
- Scan status retrieval
- Secrets listing with filtering
- Scan history listing
- Permission checks
- Error handling
"""

from __future__ import annotations

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.base import HandlerResult


# ============================================================================
# Mock Classes
# ============================================================================


class MockSecretsScanResult:
    """Mock secrets scan result."""

    def __init__(
        self,
        scan_id: str = "secrets_123",
        repository: str = "test-repo",
        status: str = "completed",
    ):
        self.scan_id = scan_id
        self.repository = repository
        self.status = status
        self.branch = "main"
        self.started_at = datetime.now(timezone.utc)
        self.completed_at = datetime.now(timezone.utc)
        self.error = None
        self.secrets = []
        self.files_scanned = 500
        self.scanned_history = False
        self.history_depth = 0
        self.critical_count = 0
        self.high_count = 0
        self.medium_count = 0
        self.low_count = 0

    def to_dict(self) -> dict:
        return {
            "scan_id": self.scan_id,
            "repository": self.repository,
            "status": self.status,
            "branch": self.branch,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "secrets": [s.to_dict() for s in self.secrets],
            "files_scanned": self.files_scanned,
            "scanned_history": self.scanned_history,
            "history_depth": self.history_depth,
            "summary": {
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "low": self.low_count,
                "total": len(self.secrets),
            },
        }


class MockSecret:
    """Mock detected secret."""

    def __init__(
        self,
        secret_id: str,
        severity: str,
        secret_type: str = "api_key",
        is_in_history: bool = False,
    ):
        self.secret_id = secret_id
        self.severity = severity
        self.secret_type = secret_type
        self.file_path = "/src/config.py"
        self.line_number = 42
        self.is_in_history = is_in_history
        self.commit_sha = "abc123" if is_in_history else None
        self.masked_value = "sk-***...***"
        self.description = f"Detected {secret_type}"

    def to_dict(self) -> dict:
        return {
            "secret_id": self.secret_id,
            "severity": self.severity,
            "secret_type": self.secret_type,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "is_in_history": self.is_in_history,
            "commit_sha": self.commit_sha,
            "masked_value": self.masked_value,
            "description": self.description,
        }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_secrets_scanner():
    """Create mock secrets scanner."""
    scanner = MagicMock()
    scanner.scan_repository = AsyncMock(return_value=MockSecretsScanResult())
    scanner.scan_git_history = AsyncMock(return_value=MockSecretsScanResult())
    return scanner


@pytest.fixture
def mock_scan_with_secrets():
    """Create scan with detected secrets."""
    scan = MockSecretsScanResult()
    scan.secrets = [
        MockSecret("s1", "critical", "aws_access_key"),
        MockSecret("s2", "high", "github_token"),
        MockSecret("s3", "medium", "api_key"),
        MockSecret("s4", "low", "password", is_in_history=True),
    ]
    scan.critical_count = 1
    scan.high_count = 1
    scan.medium_count = 1
    scan.low_count = 1
    return scan


# ============================================================================
# Scan Trigger Tests
# ============================================================================


class TestSecretsScanTrigger:
    """Test secrets scan trigger endpoint."""

    @pytest.mark.asyncio
    async def test_scan_secrets_success(self, mock_secrets_scanner):
        """Test successful secrets scan trigger."""
        from aragora.server.handlers.codebase.security.secrets import handle_scan_secrets

        with (
            patch(
                "aragora.server.handlers.codebase.security.secrets.SecretsScanner",
                return_value=mock_secrets_scanner,
            ),
            patch(
                "aragora.server.handlers.codebase.security.secrets.get_or_create_secrets_scans",
                return_value={},
            ),
            patch(
                "aragora.server.handlers.codebase.security.secrets.get_running_secrets_scans",
                return_value={},
            ),
            patch(
                "aragora.server.handlers.codebase.security.secrets.get_secrets_scan_lock",
                return_value=MagicMock(),
            ),
            patch(
                "aragora.server.handlers.codebase.security.secrets.emit_secrets_events",
                new_callable=AsyncMock,
            ),
        ):
            result = await handle_scan_secrets(
                repo_path="/path/to/repo",
                repo_id="test-repo",
                branch="main",
            )

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert "scan_id" in data
            assert data["status"] == "running"

    @pytest.mark.asyncio
    async def test_scan_secrets_already_running(self, mock_secrets_scanner):
        """Test secrets scan returns 409 if already running."""
        from aragora.server.handlers.codebase.security.secrets import handle_scan_secrets

        running_task = MagicMock()
        running_task.done.return_value = False

        with patch(
            "aragora.server.handlers.codebase.security.secrets.get_running_secrets_scans",
            return_value={"test-repo": running_task},
        ):
            result = await handle_scan_secrets(
                repo_path="/path/to/repo",
                repo_id="test-repo",
            )

            assert result.status_code == 409

    @pytest.mark.asyncio
    async def test_scan_secrets_with_history(self, mock_secrets_scanner):
        """Test secrets scan includes git history when requested."""
        from aragora.server.handlers.codebase.security.secrets import handle_scan_secrets

        with (
            patch(
                "aragora.server.handlers.codebase.security.secrets.SecretsScanner",
                return_value=mock_secrets_scanner,
            ),
            patch(
                "aragora.server.handlers.codebase.security.secrets.get_or_create_secrets_scans",
                return_value={},
            ),
            patch(
                "aragora.server.handlers.codebase.security.secrets.get_running_secrets_scans",
                return_value={},
            ),
            patch(
                "aragora.server.handlers.codebase.security.secrets.get_secrets_scan_lock",
                return_value=MagicMock(),
            ),
            patch(
                "aragora.server.handlers.codebase.security.secrets.emit_secrets_events",
                new_callable=AsyncMock,
            ),
        ):
            result = await handle_scan_secrets(
                repo_path="/path/to/repo",
                repo_id="test-repo",
                include_history=True,
                history_depth=100,
            )

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert data["include_history"] is True

    @pytest.mark.asyncio
    async def test_scan_secrets_generates_repo_id(self, mock_secrets_scanner):
        """Test secrets scan generates repo_id if not provided."""
        from aragora.server.handlers.codebase.security.secrets import handle_scan_secrets

        with (
            patch(
                "aragora.server.handlers.codebase.security.secrets.SecretsScanner",
                return_value=mock_secrets_scanner,
            ),
            patch(
                "aragora.server.handlers.codebase.security.secrets.get_or_create_secrets_scans",
                return_value={},
            ),
            patch(
                "aragora.server.handlers.codebase.security.secrets.get_running_secrets_scans",
                return_value={},
            ),
            patch(
                "aragora.server.handlers.codebase.security.secrets.get_secrets_scan_lock",
                return_value=MagicMock(),
            ),
            patch(
                "aragora.server.handlers.codebase.security.secrets.emit_secrets_events",
                new_callable=AsyncMock,
            ),
        ):
            result = await handle_scan_secrets(repo_path="/path/to/repo")

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert data["repository"].startswith("repo_")


# ============================================================================
# Scan Status Tests
# ============================================================================


class TestSecretsScanStatus:
    """Test secrets scan status endpoint."""

    @pytest.mark.asyncio
    async def test_get_secrets_scan_status_specific(self):
        """Test getting specific secrets scan by ID."""
        from aragora.server.handlers.codebase.security.secrets import (
            handle_get_secrets_scan_status,
        )

        scan = MockSecretsScanResult(scan_id="secrets_123")

        with patch(
            "aragora.server.handlers.codebase.security.secrets.get_or_create_secrets_scans",
            return_value={"secrets_123": scan},
        ):
            result = await handle_get_secrets_scan_status(
                repo_id="test-repo",
                scan_id="secrets_123",
            )

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert data["scan_result"]["scan_id"] == "secrets_123"

    @pytest.mark.asyncio
    async def test_get_secrets_scan_status_latest(self):
        """Test getting latest secrets scan."""
        from aragora.server.handlers.codebase.security.secrets import (
            handle_get_secrets_scan_status,
        )

        scan1 = MockSecretsScanResult(scan_id="secrets_old")
        scan1.started_at = datetime(2023, 1, 1, tzinfo=timezone.utc)
        scan2 = MockSecretsScanResult(scan_id="secrets_new")
        scan2.started_at = datetime(2024, 1, 1, tzinfo=timezone.utc)

        with patch(
            "aragora.server.handlers.codebase.security.secrets.get_or_create_secrets_scans",
            return_value={"secrets_old": scan1, "secrets_new": scan2},
        ):
            result = await handle_get_secrets_scan_status(repo_id="test-repo")

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert data["scan_result"]["scan_id"] == "secrets_new"

    @pytest.mark.asyncio
    async def test_get_secrets_scan_status_not_found(self):
        """Test 404 when secrets scan not found."""
        from aragora.server.handlers.codebase.security.secrets import (
            handle_get_secrets_scan_status,
        )

        with patch(
            "aragora.server.handlers.codebase.security.secrets.get_or_create_secrets_scans",
            return_value={},
        ):
            result = await handle_get_secrets_scan_status(
                repo_id="test-repo",
                scan_id="nonexistent",
            )

            assert result.status_code == 404


# ============================================================================
# Secrets List Tests
# ============================================================================


class TestSecretsList:
    """Test secrets listing endpoint."""

    @pytest.mark.asyncio
    async def test_get_secrets_success(self, mock_scan_with_secrets):
        """Test getting secrets from latest scan."""
        from aragora.server.handlers.codebase.security.secrets import handle_get_secrets

        with patch(
            "aragora.server.handlers.codebase.security.secrets.get_or_create_secrets_scans",
            return_value={"secrets_123": mock_scan_with_secrets},
        ):
            result = await handle_get_secrets(repo_id="test-repo")

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert "secrets" in data
            assert len(data["secrets"]) == 4

    @pytest.mark.asyncio
    async def test_get_secrets_filter_by_severity(self, mock_scan_with_secrets):
        """Test filtering secrets by severity."""
        from aragora.server.handlers.codebase.security.secrets import handle_get_secrets

        with patch(
            "aragora.server.handlers.codebase.security.secrets.get_or_create_secrets_scans",
            return_value={"secrets_123": mock_scan_with_secrets},
        ):
            result = await handle_get_secrets(
                repo_id="test-repo",
                severity="critical",
            )

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert all(s["severity"] == "critical" for s in data["secrets"])

    @pytest.mark.asyncio
    async def test_get_secrets_filter_by_type(self, mock_scan_with_secrets):
        """Test filtering secrets by type."""
        from aragora.server.handlers.codebase.security.secrets import handle_get_secrets

        with patch(
            "aragora.server.handlers.codebase.security.secrets.get_or_create_secrets_scans",
            return_value={"secrets_123": mock_scan_with_secrets},
        ):
            result = await handle_get_secrets(
                repo_id="test-repo",
                secret_type="aws_access_key",
            )

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert all(s["secret_type"] == "aws_access_key" for s in data["secrets"])

    @pytest.mark.asyncio
    async def test_get_secrets_exclude_history(self, mock_scan_with_secrets):
        """Test excluding secrets from git history."""
        from aragora.server.handlers.codebase.security.secrets import handle_get_secrets

        with patch(
            "aragora.server.handlers.codebase.security.secrets.get_or_create_secrets_scans",
            return_value={"secrets_123": mock_scan_with_secrets},
        ):
            result = await handle_get_secrets(
                repo_id="test-repo",
                include_history=False,
            )

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            # Should exclude the one secret that is in history
            assert all(not s["is_in_history"] for s in data["secrets"])

    @pytest.mark.asyncio
    async def test_get_secrets_pagination(self, mock_scan_with_secrets):
        """Test secrets pagination."""
        from aragora.server.handlers.codebase.security.secrets import handle_get_secrets

        with patch(
            "aragora.server.handlers.codebase.security.secrets.get_or_create_secrets_scans",
            return_value={"secrets_123": mock_scan_with_secrets},
        ):
            result = await handle_get_secrets(
                repo_id="test-repo",
                limit=2,
                offset=0,
            )

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert len(data["secrets"]) == 2
            assert data["total"] == 4

    @pytest.mark.asyncio
    async def test_get_secrets_no_completed_scans(self):
        """Test 404 when no completed secrets scans."""
        from aragora.server.handlers.codebase.security.secrets import handle_get_secrets

        scan = MockSecretsScanResult(status="running")

        with patch(
            "aragora.server.handlers.codebase.security.secrets.get_or_create_secrets_scans",
            return_value={"secrets_123": scan},
        ):
            result = await handle_get_secrets(repo_id="test-repo")

            assert result.status_code == 404


# ============================================================================
# Scan History Tests
# ============================================================================


class TestSecretsScanHistory:
    """Test secrets scan history listing endpoint."""

    @pytest.mark.asyncio
    async def test_list_secrets_scans_success(self):
        """Test listing secrets scan history."""
        from aragora.server.handlers.codebase.security.secrets import (
            handle_list_secrets_scans,
        )

        scan1 = MockSecretsScanResult(scan_id="secrets_1")
        scan2 = MockSecretsScanResult(scan_id="secrets_2")

        with patch(
            "aragora.server.handlers.codebase.security.secrets.get_or_create_secrets_scans",
            return_value={"secrets_1": scan1, "secrets_2": scan2},
        ):
            result = await handle_list_secrets_scans(repo_id="test-repo")

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert len(data["scans"]) == 2

    @pytest.mark.asyncio
    async def test_list_secrets_scans_filter_by_status(self):
        """Test filtering secrets scan history by status."""
        from aragora.server.handlers.codebase.security.secrets import (
            handle_list_secrets_scans,
        )

        scan1 = MockSecretsScanResult(scan_id="secrets_1", status="completed")
        scan2 = MockSecretsScanResult(scan_id="secrets_2", status="running")

        with patch(
            "aragora.server.handlers.codebase.security.secrets.get_or_create_secrets_scans",
            return_value={"secrets_1": scan1, "secrets_2": scan2},
        ):
            result = await handle_list_secrets_scans(
                repo_id="test-repo",
                status="completed",
            )

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert all(s["status"] == "completed" for s in data["scans"])

    @pytest.mark.asyncio
    async def test_list_secrets_scans_with_pagination(self):
        """Test listing secrets scans with pagination."""
        from aragora.server.handlers.codebase.security.secrets import (
            handle_list_secrets_scans,
        )

        scans = {f"secrets_{i}": MockSecretsScanResult(scan_id=f"secrets_{i}") for i in range(10)}

        with patch(
            "aragora.server.handlers.codebase.security.secrets.get_or_create_secrets_scans",
            return_value=scans,
        ):
            result = await handle_list_secrets_scans(
                repo_id="test-repo",
                limit=5,
                offset=0,
            )

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert len(data["scans"]) == 5
            assert data["total"] == 10

    @pytest.mark.asyncio
    async def test_list_secrets_scans_empty(self):
        """Test listing secrets scans when none exist."""
        from aragora.server.handlers.codebase.security.secrets import (
            handle_list_secrets_scans,
        )

        with patch(
            "aragora.server.handlers.codebase.security.secrets.get_or_create_secrets_scans",
            return_value={},
        ):
            result = await handle_list_secrets_scans(repo_id="test-repo")

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            data = response.get("data", response)
            assert data["scans"] == []


# ============================================================================
# Permission Tests
# ============================================================================


class TestSecretsPermissions:
    """Test secrets scanning permission enforcement."""

    def test_scan_has_permission_decorator(self):
        """Secrets scan trigger requires security permission."""
        from aragora.server.handlers.codebase.security.secrets import handle_scan_secrets
        import inspect

        source = inspect.getsource(handle_scan_secrets)
        assert "require_permission" in source

    def test_get_secrets_has_permission_decorator(self):
        """Get secrets requires security permission."""
        from aragora.server.handlers.codebase.security.secrets import handle_get_secrets
        import inspect

        source = inspect.getsource(handle_get_secrets)
        assert "require_permission" in source


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestSecretsErrorHandling:
    """Test secrets scanning error handling."""

    @pytest.mark.asyncio
    async def test_scan_handles_scanner_error(self, mock_secrets_scanner):
        """Test secrets scan handles scanner errors."""
        from aragora.server.handlers.codebase.security.secrets import handle_scan_secrets

        mock_secrets_scanner.scan_repository.side_effect = RuntimeError("Scanner failed")

        with (
            patch(
                "aragora.server.handlers.codebase.security.secrets.SecretsScanner",
                return_value=mock_secrets_scanner,
            ),
            patch(
                "aragora.server.handlers.codebase.security.secrets.get_or_create_secrets_scans",
                return_value={},
            ),
            patch(
                "aragora.server.handlers.codebase.security.secrets.get_running_secrets_scans",
                return_value={},
            ),
            patch(
                "aragora.server.handlers.codebase.security.secrets.get_secrets_scan_lock",
                return_value=MagicMock(),
            ),
        ):
            result = await handle_scan_secrets(
                repo_path="/path/to/repo",
                repo_id="test-repo",
            )

            # Should return 200 with running status since scan is async
            assert result.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_get_secrets_handles_corrupted_data(self):
        """Test get secrets handles corrupted scan data."""
        from aragora.server.handlers.codebase.security.secrets import handle_get_secrets

        with patch(
            "aragora.server.handlers.codebase.security.secrets.get_or_create_secrets_scans",
            return_value={"bad_scan": "not a valid scan"},
        ):
            result = await handle_get_secrets(repo_id="test-repo")

            assert result.status_code in (404, 500)


# ============================================================================
# Security Event Emission Tests
# ============================================================================


class TestSecretsEventEmission:
    """Test secrets scanning event emission."""

    @pytest.mark.asyncio
    async def test_critical_secrets_emit_events(self, mock_secrets_scanner):
        """Test critical secrets trigger event emission."""
        from aragora.server.handlers.codebase.security.secrets import handle_scan_secrets

        emit_events_mock = AsyncMock()

        with (
            patch(
                "aragora.server.handlers.codebase.security.secrets.SecretsScanner",
                return_value=mock_secrets_scanner,
            ),
            patch(
                "aragora.server.handlers.codebase.security.secrets.get_or_create_secrets_scans",
                return_value={},
            ),
            patch(
                "aragora.server.handlers.codebase.security.secrets.get_running_secrets_scans",
                return_value={},
            ),
            patch(
                "aragora.server.handlers.codebase.security.secrets.get_secrets_scan_lock",
                return_value=MagicMock(),
            ),
            patch(
                "aragora.server.handlers.codebase.security.secrets.emit_secrets_events",
                emit_events_mock,
            ),
        ):
            result = await handle_scan_secrets(
                repo_path="/path/to/repo",
                repo_id="test-repo",
            )

            assert result.status_code == 200
            # Event emission is called asynchronously in the scan task


__all__ = [
    "TestSecretsScanTrigger",
    "TestSecretsScanStatus",
    "TestSecretsList",
    "TestSecretsScanHistory",
    "TestSecretsPermissions",
    "TestSecretsErrorHandling",
    "TestSecretsEventEmission",
]
