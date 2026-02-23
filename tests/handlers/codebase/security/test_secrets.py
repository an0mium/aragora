"""
Tests for secrets scan handler functions
(aragora/server/handlers/codebase/security/secrets.py).

Covers all four handler functions:
- handle_scan_secrets: POST trigger a secrets scan
- handle_get_secrets_scan_status: GET scan status (latest or by ID)
- handle_get_secrets: GET secrets from latest completed scan
- handle_list_secrets_scans: GET scan history

Tests include: happy path, error paths, edge cases, filtering,
pagination, sorting, security (path traversal, injection), and
concurrent scan protection.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.analysis.codebase import (
    SecretFinding,
    SecretsScanResult,
    SecretType,
    VulnerabilitySeverity,
)
from aragora.server.handlers.codebase.security.secrets import (
    handle_get_secrets,
    handle_get_secrets_scan_status,
    handle_list_secrets_scans,
    handle_scan_secrets,
)


# ============================================================================
# Helpers
# ============================================================================


def _body(result) -> dict:
    """Extract the data payload from a HandlerResult.

    success_response wraps in {"success": true, "data": {...}}.
    error_response wraps in {"error": "..."}.
    """
    import json

    if isinstance(result, dict):
        raw = result
    else:
        raw = json.loads(result.body)
    if isinstance(raw, dict) and raw.get("success") and "data" in raw:
        return raw["data"]
    return raw


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _make_secret(
    secret_id: str = "sec_001",
    secret_type: SecretType = SecretType.GITHUB_TOKEN,
    file_path: str = "config.py",
    line_number: int = 42,
    severity: VulnerabilitySeverity = VulnerabilitySeverity.HIGH,
    confidence: float = 0.95,
    is_in_history: bool = False,
) -> SecretFinding:
    """Create a SecretFinding for testing."""
    return SecretFinding(
        id=secret_id,
        secret_type=secret_type,
        file_path=file_path,
        line_number=line_number,
        column_start=0,
        column_end=20,
        matched_text="ghp_****...****abcd",
        context_line=f"TOKEN = 'ghp_****...****abcd'  # line {line_number}",
        severity=severity,
        confidence=confidence,
        is_in_history=is_in_history,
    )


def _make_secrets_scan(
    scan_id: str = "secrets_001",
    repo_id: str = "test-repo",
    status: str = "completed",
    branch: str | None = "main",
    secrets: list[SecretFinding] | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    files_scanned: int = 50,
    scanned_history: bool = False,
    history_depth: int = 0,
) -> SecretsScanResult:
    """Create a SecretsScanResult for testing."""
    return SecretsScanResult(
        scan_id=scan_id,
        repository=repo_id,
        branch=branch,
        status=status,
        secrets=secrets or [],
        started_at=started_at or datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
        completed_at=completed_at or datetime(2024, 1, 15, 10, 5, tzinfo=timezone.utc),
        files_scanned=files_scanned,
        scanned_history=scanned_history,
        history_depth=history_depth,
    )


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def clear_secrets_storage():
    """Clear in-memory secrets scan stores between tests."""
    from aragora.server.handlers.codebase.security.storage import (
        _secrets_scan_results,
        _running_secrets_scans,
    )

    _secrets_scan_results.clear()
    _running_secrets_scans.clear()
    yield
    _secrets_scan_results.clear()
    _running_secrets_scans.clear()


@pytest.fixture(autouse=True)
def mock_scanner_and_events(monkeypatch):
    """Mock SecretsScanner and emit_secrets_events for all tests.

    This prevents background tasks from doing real file I/O.
    Individual tests can override with their own monkeypatch.
    """
    import aragora.server.handlers.codebase.security.secrets as secrets_mod

    mock_scanner = AsyncMock()
    mock_scanner.scan_repository.return_value = _make_secrets_scan()
    mock_scanner.scan_git_history.return_value = _make_secrets_scan(secrets=[])

    monkeypatch.setattr(secrets_mod, "SecretsScanner", lambda: mock_scanner)
    monkeypatch.setattr(secrets_mod, "emit_secrets_events", AsyncMock())
    return mock_scanner


@pytest.fixture
def repo_scans():
    """Get secrets scan storage for test-repo."""
    from aragora.server.handlers.codebase.security.storage import (
        get_or_create_secrets_scans,
    )

    return get_or_create_secrets_scans("test-repo")


@pytest.fixture
def running_scans():
    """Get the running secrets scans dictionary."""
    from aragora.server.handlers.codebase.security.storage import (
        get_running_secrets_scans,
    )

    return get_running_secrets_scans()


@pytest.fixture
def secrets_scan_lock():
    """Get the secrets scan lock."""
    from aragora.server.handlers.codebase.security.storage import (
        get_secrets_scan_lock,
    )

    return get_secrets_scan_lock()


# ============================================================================
# handle_scan_secrets tests
# ============================================================================


class TestHandleScanSecrets:
    """Tests for handle_scan_secrets."""

    @pytest.mark.asyncio
    async def test_start_scan_success(self, running_scans, monkeypatch):
        """Starting a new secrets scan returns 200 with scan_id and running status."""
        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(secrets_mod, "emit_secrets_events", AsyncMock())

        result = await handle_scan_secrets(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
            branch="main",
        )

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "running"
        assert body["repository"] == "test-repo"
        assert "scan_id" in body
        assert body["scan_id"].startswith("secrets_")

    @pytest.mark.asyncio
    async def test_start_scan_generates_repo_id_when_none(self, monkeypatch):
        """When repo_id is None, a UUID-based one is generated."""
        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(secrets_mod, "emit_secrets_events", AsyncMock())

        result = await handle_scan_secrets(repo_path="/tmp/repo")

        assert _status(result) == 200
        body = _body(result)
        assert body["repository"].startswith("repo_")

    @pytest.mark.asyncio
    async def test_scan_returns_include_history_flag(self, monkeypatch):
        """The response includes the include_history flag."""
        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(secrets_mod, "emit_secrets_events", AsyncMock())

        result = await handle_scan_secrets(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
            include_history=True,
        )

        assert _status(result) == 200
        body = _body(result)
        assert body["include_history"] is True

    @pytest.mark.asyncio
    async def test_scan_returns_include_history_false(self, monkeypatch):
        """include_history defaults to False."""
        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(secrets_mod, "emit_secrets_events", AsyncMock())

        result = await handle_scan_secrets(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        assert _status(result) == 200
        body = _body(result)
        assert body["include_history"] is False

    @pytest.mark.asyncio
    async def test_duplicate_scan_returns_409(self, running_scans):
        """If a secrets scan is already running for the repo, 409 is returned."""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        task = asyncio.ensure_future(future)
        running_scans["test-repo"] = task

        result = await handle_scan_secrets(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        assert _status(result) == 409
        body = _body(result)
        assert "already in progress" in body.get("error", "").lower()

        # Clean up
        future.set_result(None)
        await task

    @pytest.mark.asyncio
    async def test_completed_scan_allows_rescan(self, running_scans, monkeypatch):
        """If a previous scan is done, a new scan can be started."""
        completed_task = asyncio.ensure_future(asyncio.sleep(0))
        await completed_task
        running_scans["test-repo"] = completed_task

        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(secrets_mod, "emit_secrets_events", AsyncMock())

        result = await handle_scan_secrets(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_scan_stores_initial_result(self, repo_scans, monkeypatch):
        """The scan result is stored immediately with running status."""
        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(secrets_mod, "emit_secrets_events", AsyncMock())

        result = await handle_scan_secrets(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        scan_id = _body(result)["scan_id"]
        assert scan_id in repo_scans
        assert repo_scans[scan_id].status == "running"

    @pytest.mark.asyncio
    async def test_scan_with_workspace_and_user(self, monkeypatch):
        """Workspace and user IDs are passed through."""
        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(secrets_mod, "emit_secrets_events", AsyncMock())

        result = await handle_scan_secrets(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
            workspace_id="ws_001",
            user_id="user_001",
        )

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_scan_without_branch(self, monkeypatch):
        """Scan works without optional branch."""
        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(secrets_mod, "emit_secrets_events", AsyncMock())

        result = await handle_scan_secrets(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        assert _status(result) == 200
        body = _body(result)
        assert "scan_id" in body

    @pytest.mark.asyncio
    async def test_scan_creates_async_task(self, running_scans, monkeypatch):
        """A background task is created and stored in running_scans."""
        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(secrets_mod, "emit_secrets_events", AsyncMock())

        result = await handle_scan_secrets(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        assert _status(result) == 200
        assert "test-repo" in running_scans
        task = running_scans["test-repo"]
        assert isinstance(task, asyncio.Task)

    @pytest.mark.asyncio
    async def test_scan_internal_error_returns_500(self, monkeypatch):
        """Internal errors during scan setup return 500."""
        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        # Force an error by making get_running_secrets_scans raise
        monkeypatch.setattr(
            secrets_mod,
            "get_running_secrets_scans",
            MagicMock(side_effect=RuntimeError("storage failure")),
        )

        result = await handle_scan_secrets(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        assert _status(result) == 500
        body = _body(result)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_scan_scan_id_format(self, monkeypatch):
        """Scan ID has the expected format (secrets_ prefix + hex)."""
        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(secrets_mod, "emit_secrets_events", AsyncMock())

        result = await handle_scan_secrets(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        body = _body(result)
        scan_id = body["scan_id"]
        assert scan_id.startswith("secrets_")
        # After the prefix, the rest should be hex characters
        hex_part = scan_id[len("secrets_") :]
        assert len(hex_part) == 12
        assert all(c in "0123456789abcdef" for c in hex_part)


# ============================================================================
# handle_get_secrets_scan_status tests
# ============================================================================


class TestHandleGetSecretsScanStatus:
    """Tests for handle_get_secrets_scan_status."""

    @pytest.mark.asyncio
    async def test_get_specific_scan(self, repo_scans):
        """Get a specific scan by ID."""
        scan = _make_secrets_scan(scan_id="secrets_abc")
        repo_scans["secrets_abc"] = scan

        result = await handle_get_secrets_scan_status(repo_id="test-repo", scan_id="secrets_abc")

        assert _status(result) == 200
        body = _body(result)
        assert "scan_result" in body
        assert body["scan_result"]["scan_id"] == "secrets_abc"

    @pytest.mark.asyncio
    async def test_get_specific_scan_not_found(self, repo_scans):
        """Request for non-existent scan_id returns 404."""
        result = await handle_get_secrets_scan_status(repo_id="test-repo", scan_id="nonexistent")

        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_get_latest_scan(self, repo_scans):
        """Without scan_id, returns the latest scan by started_at."""
        older = _make_secrets_scan(
            scan_id="secrets_old",
            started_at=datetime(2024, 1, 10, 10, 0, tzinfo=timezone.utc),
        )
        newer = _make_secrets_scan(
            scan_id="secrets_new",
            started_at=datetime(2024, 1, 20, 10, 0, tzinfo=timezone.utc),
        )
        repo_scans["secrets_old"] = older
        repo_scans["secrets_new"] = newer

        result = await handle_get_secrets_scan_status(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        assert body["scan_result"]["scan_id"] == "secrets_new"

    @pytest.mark.asyncio
    async def test_get_latest_scan_no_scans(self):
        """When no scans exist for the repo, returns 404."""
        result = await handle_get_secrets_scan_status(repo_id="test-repo")

        assert _status(result) == 404
        body = _body(result)
        assert "no secrets scans found" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_get_scan_returns_full_result(self, repo_scans):
        """The scan result includes all expected fields."""
        secret = _make_secret()
        scan = _make_secrets_scan(
            scan_id="secrets_full",
            branch="develop",
            secrets=[secret],
            scanned_history=True,
            history_depth=50,
        )
        repo_scans["secrets_full"] = scan

        result = await handle_get_secrets_scan_status(repo_id="test-repo", scan_id="secrets_full")

        assert _status(result) == 200
        body = _body(result)
        scan_data = body["scan_result"]
        assert scan_data["branch"] == "develop"
        assert scan_data["scanned_history"] is True
        assert scan_data["history_depth"] == 50
        assert len(scan_data["secrets"]) == 1

    @pytest.mark.asyncio
    async def test_get_scan_with_running_status(self, repo_scans):
        """Can retrieve a running scan status."""
        scan = _make_secrets_scan(scan_id="secrets_running", status="running")
        repo_scans["secrets_running"] = scan

        result = await handle_get_secrets_scan_status(
            repo_id="test-repo", scan_id="secrets_running"
        )

        assert _status(result) == 200
        body = _body(result)
        assert body["scan_result"]["status"] == "running"

    @pytest.mark.asyncio
    async def test_get_scan_with_failed_status(self, repo_scans):
        """Can retrieve a failed scan status."""
        scan = _make_secrets_scan(scan_id="secrets_failed", status="failed")
        scan.error = "Secrets scan failed"
        repo_scans["secrets_failed"] = scan

        result = await handle_get_secrets_scan_status(repo_id="test-repo", scan_id="secrets_failed")

        assert _status(result) == 200
        body = _body(result)
        assert body["scan_result"]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_get_scan_for_nonexistent_repo(self):
        """Requesting scans for a repo with no history returns 404."""
        result = await handle_get_secrets_scan_status(repo_id="unknown-repo")

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_latest_among_many(self, repo_scans):
        """Latest scan is correctly identified among many scans."""
        for i in range(5):
            scan = _make_secrets_scan(
                scan_id=f"secrets_{i:03d}",
                started_at=datetime(2024, 1, 10 + i, 10, 0, tzinfo=timezone.utc),
            )
            repo_scans[f"secrets_{i:03d}"] = scan

        result = await handle_get_secrets_scan_status(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        assert body["scan_result"]["scan_id"] == "secrets_004"

    @pytest.mark.asyncio
    async def test_get_status_internal_error(self, monkeypatch):
        """Internal errors return 500."""
        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(
            secrets_mod,
            "get_or_create_secrets_scans",
            MagicMock(side_effect=ValueError("unexpected")),
        )

        result = await handle_get_secrets_scan_status(repo_id="test-repo")

        assert _status(result) == 500


# ============================================================================
# handle_get_secrets tests
# ============================================================================


class TestHandleGetSecrets:
    """Tests for handle_get_secrets."""

    @pytest.mark.asyncio
    async def test_get_secrets_from_completed_scan(self, repo_scans):
        """Returns secrets from the latest completed scan."""
        secret = _make_secret()
        scan = _make_secrets_scan(scan_id="secrets_done", secrets=[secret])
        repo_scans["secrets_done"] = scan

        result = await handle_get_secrets(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        assert len(body["secrets"]) == 1
        assert body["total"] == 1
        assert body["scan_id"] == "secrets_done"

    @pytest.mark.asyncio
    async def test_get_secrets_no_scans(self):
        """Returns 404 when no scans exist."""
        result = await handle_get_secrets(repo_id="test-repo")

        assert _status(result) == 404
        body = _body(result)
        assert "no secrets scans found" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_get_secrets_no_completed_scans(self, repo_scans):
        """Returns 404 when no completed scans exist."""
        scan = _make_secrets_scan(scan_id="secrets_running", status="running")
        repo_scans["secrets_running"] = scan

        result = await handle_get_secrets(repo_id="test-repo")

        assert _status(result) == 404
        body = _body(result)
        assert "no completed" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_filter_by_severity(self, repo_scans):
        """Filtering by severity works correctly."""
        secrets = [
            _make_secret(secret_id="s1", severity=VulnerabilitySeverity.CRITICAL),
            _make_secret(secret_id="s2", severity=VulnerabilitySeverity.HIGH),
            _make_secret(secret_id="s3", severity=VulnerabilitySeverity.MEDIUM),
            _make_secret(secret_id="s4", severity=VulnerabilitySeverity.LOW),
        ]
        scan = _make_secrets_scan(scan_id="secrets_sev", secrets=secrets)
        repo_scans["secrets_sev"] = scan

        result = await handle_get_secrets(repo_id="test-repo", severity="critical")

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["secrets"][0]["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_filter_by_severity_high(self, repo_scans):
        """Filtering by high severity returns only high severity secrets."""
        secrets = [
            _make_secret(secret_id="s1", severity=VulnerabilitySeverity.CRITICAL),
            _make_secret(secret_id="s2", severity=VulnerabilitySeverity.HIGH),
            _make_secret(secret_id="s3", severity=VulnerabilitySeverity.HIGH),
        ]
        scan = _make_secrets_scan(scan_id="secrets_high", secrets=secrets)
        repo_scans["secrets_high"] = scan

        result = await handle_get_secrets(repo_id="test-repo", severity="high")

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert all(s["severity"] == "high" for s in body["secrets"])

    @pytest.mark.asyncio
    async def test_filter_by_secret_type(self, repo_scans):
        """Filtering by secret_type works correctly."""
        secrets = [
            _make_secret(secret_id="s1", secret_type=SecretType.GITHUB_TOKEN),
            _make_secret(secret_id="s2", secret_type=SecretType.AWS_ACCESS_KEY),
            _make_secret(secret_id="s3", secret_type=SecretType.GITHUB_TOKEN),
        ]
        scan = _make_secrets_scan(scan_id="secrets_type", secrets=secrets)
        repo_scans["secrets_type"] = scan

        result = await handle_get_secrets(repo_id="test-repo", secret_type="github_token")

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert all(s["secret_type"] == "github_token" for s in body["secrets"])

    @pytest.mark.asyncio
    async def test_filter_exclude_history(self, repo_scans):
        """Excluding history secrets filters correctly."""
        secrets = [
            _make_secret(secret_id="s1", is_in_history=False),
            _make_secret(secret_id="s2", is_in_history=True),
            _make_secret(secret_id="s3", is_in_history=False),
        ]
        scan = _make_secrets_scan(scan_id="secrets_hist", secrets=secrets)
        repo_scans["secrets_hist"] = scan

        result = await handle_get_secrets(repo_id="test-repo", include_history=False)

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert all(not s["is_in_history"] for s in body["secrets"])

    @pytest.mark.asyncio
    async def test_include_history_default(self, repo_scans):
        """include_history defaults to True, showing all secrets."""
        secrets = [
            _make_secret(secret_id="s1", is_in_history=False),
            _make_secret(secret_id="s2", is_in_history=True),
        ]
        scan = _make_secrets_scan(scan_id="secrets_all", secrets=secrets)
        repo_scans["secrets_all"] = scan

        result = await handle_get_secrets(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2

    @pytest.mark.asyncio
    async def test_combined_filters(self, repo_scans):
        """Multiple filters can be applied simultaneously."""
        secrets = [
            _make_secret(
                secret_id="s1",
                severity=VulnerabilitySeverity.CRITICAL,
                secret_type=SecretType.AWS_ACCESS_KEY,
                is_in_history=False,
            ),
            _make_secret(
                secret_id="s2",
                severity=VulnerabilitySeverity.CRITICAL,
                secret_type=SecretType.GITHUB_TOKEN,
                is_in_history=False,
            ),
            _make_secret(
                secret_id="s3",
                severity=VulnerabilitySeverity.HIGH,
                secret_type=SecretType.AWS_ACCESS_KEY,
                is_in_history=False,
            ),
            _make_secret(
                secret_id="s4",
                severity=VulnerabilitySeverity.CRITICAL,
                secret_type=SecretType.AWS_ACCESS_KEY,
                is_in_history=True,
            ),
        ]
        scan = _make_secrets_scan(scan_id="secrets_combo", secrets=secrets)
        repo_scans["secrets_combo"] = scan

        result = await handle_get_secrets(
            repo_id="test-repo",
            severity="critical",
            secret_type="aws_access_key",
            include_history=False,
        )

        assert _status(result) == 200
        body = _body(result)
        # Only s1 matches: critical + aws_access_key + not in history
        assert body["total"] == 1
        assert body["secrets"][0]["id"] == "s1"

    @pytest.mark.asyncio
    async def test_severity_sorting(self, repo_scans):
        """Secrets are sorted by severity order (critical > high > medium > low)."""
        secrets = [
            _make_secret(secret_id="s_low", severity=VulnerabilitySeverity.LOW),
            _make_secret(secret_id="s_crit", severity=VulnerabilitySeverity.CRITICAL),
            _make_secret(secret_id="s_med", severity=VulnerabilitySeverity.MEDIUM),
            _make_secret(secret_id="s_high", severity=VulnerabilitySeverity.HIGH),
        ]
        scan = _make_secrets_scan(scan_id="secrets_sort", secrets=secrets)
        repo_scans["secrets_sort"] = scan

        result = await handle_get_secrets(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        severities = [s["severity"] for s in body["secrets"]]
        assert severities == ["critical", "high", "medium", "low"]

    @pytest.mark.asyncio
    async def test_pagination_limit(self, repo_scans):
        """Pagination limit is respected."""
        secrets = [
            _make_secret(secret_id=f"s_{i}", severity=VulnerabilitySeverity.HIGH) for i in range(10)
        ]
        scan = _make_secrets_scan(scan_id="secrets_page", secrets=secrets)
        repo_scans["secrets_page"] = scan

        result = await handle_get_secrets(repo_id="test-repo", limit=3)

        assert _status(result) == 200
        body = _body(result)
        assert len(body["secrets"]) == 3
        assert body["total"] == 10
        assert body["limit"] == 3

    @pytest.mark.asyncio
    async def test_pagination_offset(self, repo_scans):
        """Pagination offset is respected."""
        secrets = [
            _make_secret(secret_id=f"s_{i}", severity=VulnerabilitySeverity.HIGH) for i in range(10)
        ]
        scan = _make_secrets_scan(scan_id="secrets_off", secrets=secrets)
        repo_scans["secrets_off"] = scan

        result = await handle_get_secrets(repo_id="test-repo", limit=3, offset=8)

        assert _status(result) == 200
        body = _body(result)
        assert len(body["secrets"]) == 2  # Only 2 left after offset 8
        assert body["total"] == 10
        assert body["offset"] == 8

    @pytest.mark.asyncio
    async def test_pagination_beyond_total(self, repo_scans):
        """Offset beyond total returns empty list."""
        secrets = [_make_secret(secret_id="s_0")]
        scan = _make_secrets_scan(scan_id="secrets_beyond", secrets=secrets)
        repo_scans["secrets_beyond"] = scan

        result = await handle_get_secrets(repo_id="test-repo", offset=100)

        assert _status(result) == 200
        body = _body(result)
        assert len(body["secrets"]) == 0
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_uses_latest_completed_scan(self, repo_scans):
        """Gets secrets from the latest completed scan, ignoring running scans."""
        old_scan = _make_secrets_scan(
            scan_id="secrets_old",
            status="completed",
            secrets=[_make_secret(secret_id="old_secret")],
            started_at=datetime(2024, 1, 10, 10, 0, tzinfo=timezone.utc),
        )
        new_scan = _make_secrets_scan(
            scan_id="secrets_new",
            status="completed",
            secrets=[_make_secret(secret_id="new_secret")],
            started_at=datetime(2024, 1, 20, 10, 0, tzinfo=timezone.utc),
        )
        running_scan = _make_secrets_scan(
            scan_id="secrets_running",
            status="running",
            started_at=datetime(2024, 1, 25, 10, 0, tzinfo=timezone.utc),
        )
        repo_scans["secrets_old"] = old_scan
        repo_scans["secrets_new"] = new_scan
        repo_scans["secrets_running"] = running_scan

        result = await handle_get_secrets(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        assert body["scan_id"] == "secrets_new"

    @pytest.mark.asyncio
    async def test_empty_secrets_list(self, repo_scans):
        """Completed scan with no secrets returns empty list."""
        scan = _make_secrets_scan(scan_id="secrets_empty", secrets=[])
        repo_scans["secrets_empty"] = scan

        result = await handle_get_secrets(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        assert body["secrets"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_filter_no_matching_severity(self, repo_scans):
        """Filtering that produces no results returns empty list."""
        secrets = [_make_secret(severity=VulnerabilitySeverity.HIGH)]
        scan = _make_secrets_scan(scan_id="secrets_nomatch", secrets=secrets)
        repo_scans["secrets_nomatch"] = scan

        result = await handle_get_secrets(repo_id="test-repo", severity="critical")

        assert _status(result) == 200
        body = _body(result)
        assert body["secrets"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_filter_no_matching_type(self, repo_scans):
        """Filtering by non-matching type returns empty list."""
        secrets = [_make_secret(secret_type=SecretType.GITHUB_TOKEN)]
        scan = _make_secrets_scan(scan_id="secrets_notype", secrets=secrets)
        repo_scans["secrets_notype"] = scan

        result = await handle_get_secrets(repo_id="test-repo", secret_type="aws_access_key")

        assert _status(result) == 200
        body = _body(result)
        assert body["secrets"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_response_includes_pagination_metadata(self, repo_scans):
        """Response includes total, limit, offset, and scan_id."""
        scan = _make_secrets_scan(
            scan_id="secrets_meta",
            secrets=[_make_secret()],
        )
        repo_scans["secrets_meta"] = scan

        result = await handle_get_secrets(repo_id="test-repo", limit=50, offset=0)

        assert _status(result) == 200
        body = _body(result)
        assert "total" in body
        assert "limit" in body
        assert "offset" in body
        assert "scan_id" in body
        assert body["limit"] == 50
        assert body["offset"] == 0

    @pytest.mark.asyncio
    async def test_unknown_severity_sort_order(self, repo_scans):
        """Secrets with unknown severity sort after low."""
        secrets = [
            _make_secret(secret_id="s_unk", severity=VulnerabilitySeverity.UNKNOWN),
            _make_secret(secret_id="s_low", severity=VulnerabilitySeverity.LOW),
            _make_secret(secret_id="s_crit", severity=VulnerabilitySeverity.CRITICAL),
        ]
        scan = _make_secrets_scan(scan_id="secrets_unk", secrets=secrets)
        repo_scans["secrets_unk"] = scan

        result = await handle_get_secrets(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        severities = [s["severity"] for s in body["secrets"]]
        assert severities == ["critical", "low", "unknown"]

    @pytest.mark.asyncio
    async def test_get_secrets_internal_error(self, monkeypatch):
        """Internal errors return 500."""
        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(
            secrets_mod,
            "get_or_create_secrets_scans",
            MagicMock(side_effect=TypeError("unexpected")),
        )

        result = await handle_get_secrets(repo_id="test-repo")

        assert _status(result) == 500


# ============================================================================
# handle_list_secrets_scans tests
# ============================================================================


class TestHandleListSecretsScans:
    """Tests for handle_list_secrets_scans."""

    @pytest.mark.asyncio
    async def test_list_scans_empty(self):
        """Returns empty list when no scans exist."""
        result = await handle_list_secrets_scans(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        assert body["scans"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_scans_returns_all(self, repo_scans):
        """Returns all scans for the repository."""
        for i in range(3):
            scan = _make_secrets_scan(
                scan_id=f"secrets_{i:03d}",
                started_at=datetime(2024, 1, 10 + i, 10, 0, tzinfo=timezone.utc),
            )
            repo_scans[f"secrets_{i:03d}"] = scan

        result = await handle_list_secrets_scans(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 3
        assert len(body["scans"]) == 3

    @pytest.mark.asyncio
    async def test_list_scans_sorted_by_time_descending(self, repo_scans):
        """Scans are sorted by started_at descending (newest first)."""
        for i in range(3):
            scan = _make_secrets_scan(
                scan_id=f"secrets_{i:03d}",
                started_at=datetime(2024, 1, 10 + i, 10, 0, tzinfo=timezone.utc),
            )
            repo_scans[f"secrets_{i:03d}"] = scan

        result = await handle_list_secrets_scans(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        scan_ids = [s["scan_id"] for s in body["scans"]]
        assert scan_ids == ["secrets_002", "secrets_001", "secrets_000"]

    @pytest.mark.asyncio
    async def test_filter_by_status_completed(self, repo_scans):
        """Filtering by status returns only matching scans."""
        repo_scans["s1"] = _make_secrets_scan(scan_id="s1", status="completed")
        repo_scans["s2"] = _make_secrets_scan(scan_id="s2", status="running")
        repo_scans["s3"] = _make_secrets_scan(scan_id="s3", status="completed")

        result = await handle_list_secrets_scans(repo_id="test-repo", status="completed")

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert all(s["status"] == "completed" for s in body["scans"])

    @pytest.mark.asyncio
    async def test_filter_by_status_running(self, repo_scans):
        """Filtering by running status works."""
        repo_scans["s1"] = _make_secrets_scan(scan_id="s1", status="completed")
        repo_scans["s2"] = _make_secrets_scan(scan_id="s2", status="running")

        result = await handle_list_secrets_scans(repo_id="test-repo", status="running")

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["scans"][0]["status"] == "running"

    @pytest.mark.asyncio
    async def test_filter_by_status_failed(self, repo_scans):
        """Filtering by failed status works."""
        repo_scans["s1"] = _make_secrets_scan(scan_id="s1", status="failed")

        result = await handle_list_secrets_scans(repo_id="test-repo", status="failed")

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_filter_by_status_no_match(self, repo_scans):
        """Filtering by a status with no matches returns empty list."""
        repo_scans["s1"] = _make_secrets_scan(scan_id="s1", status="completed")

        result = await handle_list_secrets_scans(repo_id="test-repo", status="running")

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 0
        assert body["scans"] == []

    @pytest.mark.asyncio
    async def test_pagination_limit(self, repo_scans):
        """Pagination limit controls the number of returned scans."""
        for i in range(10):
            scan = _make_secrets_scan(
                scan_id=f"secrets_{i:03d}",
                started_at=datetime(2024, 1, 10 + i, 10, 0, tzinfo=timezone.utc),
            )
            repo_scans[f"secrets_{i:03d}"] = scan

        result = await handle_list_secrets_scans(repo_id="test-repo", limit=3)

        assert _status(result) == 200
        body = _body(result)
        assert len(body["scans"]) == 3
        assert body["total"] == 10
        assert body["limit"] == 3

    @pytest.mark.asyncio
    async def test_pagination_offset(self, repo_scans):
        """Pagination offset skips earlier scans."""
        for i in range(5):
            scan = _make_secrets_scan(
                scan_id=f"secrets_{i:03d}",
                started_at=datetime(2024, 1, 10 + i, 10, 0, tzinfo=timezone.utc),
            )
            repo_scans[f"secrets_{i:03d}"] = scan

        result = await handle_list_secrets_scans(repo_id="test-repo", limit=2, offset=3)

        assert _status(result) == 200
        body = _body(result)
        assert len(body["scans"]) == 2
        assert body["total"] == 5
        assert body["offset"] == 3

    @pytest.mark.asyncio
    async def test_pagination_offset_beyond_total(self, repo_scans):
        """Offset beyond total returns empty list."""
        repo_scans["s1"] = _make_secrets_scan(scan_id="s1")

        result = await handle_list_secrets_scans(repo_id="test-repo", offset=100)

        assert _status(result) == 200
        body = _body(result)
        assert len(body["scans"]) == 0
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_scan_entry_fields(self, repo_scans):
        """Each scan entry has the expected fields."""
        secret = _make_secret()
        scan = _make_secrets_scan(
            scan_id="secrets_fields",
            status="completed",
            secrets=[secret],
            files_scanned=100,
            scanned_history=True,
            history_depth=50,
        )
        repo_scans["secrets_fields"] = scan

        result = await handle_list_secrets_scans(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        entry = body["scans"][0]
        assert entry["scan_id"] == "secrets_fields"
        assert entry["status"] == "completed"
        assert "started_at" in entry
        assert "completed_at" in entry
        assert entry["files_scanned"] == 100
        assert entry["scanned_history"] is True
        assert entry["history_depth"] == 50

    @pytest.mark.asyncio
    async def test_completed_scan_has_summary(self, repo_scans):
        """Completed scans include summary with counts."""
        secrets = [
            _make_secret(secret_id="s1", severity=VulnerabilitySeverity.CRITICAL),
            _make_secret(secret_id="s2", severity=VulnerabilitySeverity.HIGH),
            _make_secret(secret_id="s3", severity=VulnerabilitySeverity.MEDIUM),
        ]
        scan = _make_secrets_scan(
            scan_id="secrets_summary",
            status="completed",
            secrets=secrets,
        )
        repo_scans["secrets_summary"] = scan

        result = await handle_list_secrets_scans(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        summary = body["scans"][0]["summary"]
        assert summary is not None
        assert summary["total_secrets"] == 3
        assert summary["critical_count"] == 1
        assert summary["high_count"] == 1
        assert summary["medium_count"] == 1
        assert summary["low_count"] == 0

    @pytest.mark.asyncio
    async def test_running_scan_has_no_summary(self, repo_scans):
        """Running scans have summary set to None."""
        scan = _make_secrets_scan(scan_id="secrets_nosumm", status="running")
        repo_scans["secrets_nosumm"] = scan

        result = await handle_list_secrets_scans(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        assert body["scans"][0]["summary"] is None

    @pytest.mark.asyncio
    async def test_failed_scan_has_no_summary(self, repo_scans):
        """Failed scans have summary set to None."""
        scan = _make_secrets_scan(scan_id="secrets_fail", status="failed")
        repo_scans["secrets_fail"] = scan

        result = await handle_list_secrets_scans(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        assert body["scans"][0]["summary"] is None

    @pytest.mark.asyncio
    async def test_completed_at_none_when_running(self, repo_scans):
        """completed_at is None for running scans."""
        scan = _make_secrets_scan(scan_id="secrets_run", status="running")
        scan.completed_at = None
        repo_scans["secrets_run"] = scan

        result = await handle_list_secrets_scans(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        assert body["scans"][0]["completed_at"] is None

    @pytest.mark.asyncio
    async def test_default_pagination(self, repo_scans):
        """Default pagination is limit=20, offset=0."""
        for i in range(25):
            scan = _make_secrets_scan(
                scan_id=f"secrets_{i:03d}",
                started_at=datetime(2024, 1, 1, i % 24, 0, tzinfo=timezone.utc),
            )
            repo_scans[f"secrets_{i:03d}"] = scan

        result = await handle_list_secrets_scans(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        assert len(body["scans"]) == 20
        assert body["total"] == 25
        assert body["limit"] == 20
        assert body["offset"] == 0

    @pytest.mark.asyncio
    async def test_list_scans_internal_error(self, monkeypatch):
        """Internal errors return 500."""
        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(
            secrets_mod,
            "get_or_create_secrets_scans",
            MagicMock(side_effect=KeyError("unexpected")),
        )

        result = await handle_list_secrets_scans(repo_id="test-repo")

        assert _status(result) == 500


# ============================================================================
# Background scan task behavior tests
# ============================================================================


class TestScanBackgroundTask:
    """Tests for the background secrets scan task behavior."""

    @pytest.mark.asyncio
    async def test_background_scan_updates_result(self, repo_scans, monkeypatch):
        """The background task updates the scan result on success."""
        scan_result = _make_secrets_scan(
            scan_id="bg_scan",
            secrets=[_make_secret()],
        )

        mock_scanner = AsyncMock()
        mock_scanner.scan_repository.return_value = scan_result

        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(secrets_mod, "SecretsScanner", lambda: mock_scanner)
        monkeypatch.setattr(secrets_mod, "emit_secrets_events", AsyncMock())

        result = await handle_scan_secrets(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        scan_id = _body(result)["scan_id"]

        # Let the background task complete
        await asyncio.sleep(0.1)

        # The scan result should be updated
        assert scan_id in repo_scans
        stored = repo_scans[scan_id]
        # The background task replaces the stored result
        assert stored.scan_id == scan_id

    @pytest.mark.asyncio
    async def test_background_scan_with_history(self, repo_scans, monkeypatch):
        """Background task scans git history when include_history is True."""
        current_result = _make_secrets_scan(
            scan_id="bg_hist",
            secrets=[_make_secret(secret_id="current_1")],
        )
        # Create a result for history scan - provide mutable list
        history_result = _make_secrets_scan(
            scan_id="bg_hist_2",
            secrets=[_make_secret(secret_id="hist_1", is_in_history=True)],
        )

        mock_scanner = AsyncMock()
        mock_scanner.scan_repository.return_value = current_result
        mock_scanner.scan_git_history.return_value = history_result

        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(secrets_mod, "SecretsScanner", lambda: mock_scanner)
        monkeypatch.setattr(secrets_mod, "emit_secrets_events", AsyncMock())

        result = await handle_scan_secrets(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
            include_history=True,
            history_depth=50,
        )

        # Let the background task complete
        await asyncio.sleep(0.1)

        mock_scanner.scan_git_history.assert_called_once_with(
            repo_path="/tmp/test-repo",
            depth=50,
            branch=None,
        )

    @pytest.mark.asyncio
    async def test_background_scan_failure_marks_failed(
        self, repo_scans, running_scans, monkeypatch
    ):
        """When the background scan raises, the scan is marked as failed."""
        mock_scanner = AsyncMock()
        mock_scanner.scan_repository.side_effect = RuntimeError("disk error")

        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(secrets_mod, "SecretsScanner", lambda: mock_scanner)

        result = await handle_scan_secrets(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        scan_id = _body(result)["scan_id"]

        # Let the background task fail
        await asyncio.sleep(0.1)

        stored = repo_scans[scan_id]
        assert stored.status == "failed"
        assert stored.error == "Secrets scan failed"
        assert stored.completed_at is not None

    @pytest.mark.asyncio
    async def test_background_scan_cleans_up_running(self, repo_scans, running_scans, monkeypatch):
        """After completion, the task is removed from running_scans."""
        scan_result = _make_secrets_scan(scan_id="bg_clean")

        mock_scanner = AsyncMock()
        mock_scanner.scan_repository.return_value = scan_result

        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(secrets_mod, "SecretsScanner", lambda: mock_scanner)
        monkeypatch.setattr(secrets_mod, "emit_secrets_events", AsyncMock())

        result = await handle_scan_secrets(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        # Initially in running_scans
        assert "test-repo" in running_scans

        # Let background task complete
        await asyncio.sleep(0.1)

        # Cleaned up
        assert "test-repo" not in running_scans

    @pytest.mark.asyncio
    async def test_background_scan_failure_cleans_up_running(self, running_scans, monkeypatch):
        """Even on failure, the task is removed from running_scans."""
        mock_scanner = AsyncMock()
        mock_scanner.scan_repository.side_effect = OSError("no access")

        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(secrets_mod, "SecretsScanner", lambda: mock_scanner)

        await handle_scan_secrets(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
        )

        # Let background task fail
        await asyncio.sleep(0.1)

        assert "test-repo" not in running_scans

    @pytest.mark.asyncio
    async def test_background_scan_emits_events(self, repo_scans, monkeypatch):
        """The background task emits security events on completion."""
        scan_result = _make_secrets_scan(
            scan_id="bg_events",
            secrets=[_make_secret()],
        )

        mock_scanner = AsyncMock()
        mock_scanner.scan_repository.return_value = scan_result
        mock_emit = AsyncMock()

        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(secrets_mod, "SecretsScanner", lambda: mock_scanner)
        monkeypatch.setattr(secrets_mod, "emit_secrets_events", mock_emit)

        result = await handle_scan_secrets(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
            workspace_id="ws_test",
        )

        scan_id = _body(result)["scan_id"]

        # Let background task complete
        await asyncio.sleep(0.1)

        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        assert call_args[0][1] == "test-repo"  # repo_id
        assert call_args[0][2] == scan_id  # scan_id
        assert call_args[0][3] == "ws_test"  # workspace_id

    @pytest.mark.asyncio
    async def test_background_scan_passes_branch_to_scanner(self, repo_scans, monkeypatch):
        """The background task passes branch to the scanner."""
        scan_result = _make_secrets_scan(scan_id="bg_branch")

        mock_scanner = AsyncMock()
        mock_scanner.scan_repository.return_value = scan_result

        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(secrets_mod, "SecretsScanner", lambda: mock_scanner)
        monkeypatch.setattr(secrets_mod, "emit_secrets_events", AsyncMock())

        await handle_scan_secrets(
            repo_path="/tmp/test-repo",
            repo_id="test-repo",
            branch="develop",
        )

        # Let background task complete
        await asyncio.sleep(0.1)

        mock_scanner.scan_repository.assert_called_once_with(
            repo_path="/tmp/test-repo",
            branch="develop",
        )


# ============================================================================
# Security tests
# ============================================================================


class TestSecurityInputValidation:
    """Security-related tests for input validation."""

    @pytest.mark.asyncio
    async def test_repo_id_with_special_characters(self, monkeypatch):
        """Repo IDs with special characters work at handler level."""
        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(secrets_mod, "emit_secrets_events", AsyncMock())

        # Handler does not do repo_id validation itself (that's the router's job).
        # But it should still function correctly.
        result = await handle_scan_secrets(
            repo_path="/tmp/test",
            repo_id="my-repo_123",
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_empty_repo_path(self, monkeypatch):
        """Empty repo_path is accepted (validation is scanner's job)."""
        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(secrets_mod, "emit_secrets_events", AsyncMock())

        result = await handle_scan_secrets(
            repo_path="",
            repo_id="test-repo",
        )
        # Handler accepts it; the scanner will fail in background
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_very_long_repo_path(self, monkeypatch):
        """Very long repo_path does not cause handler crash."""
        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(secrets_mod, "emit_secrets_events", AsyncMock())

        result = await handle_scan_secrets(
            repo_path="/" + "a" * 10000,
            repo_id="test-repo",
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_negative_history_depth(self, monkeypatch):
        """Negative history_depth is accepted (background task handles it)."""
        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(secrets_mod, "emit_secrets_events", AsyncMock())

        result = await handle_scan_secrets(
            repo_path="/tmp/test",
            repo_id="test-repo",
            include_history=True,
            history_depth=-1,
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_zero_limit_pagination(self, repo_scans):
        """Zero limit returns no secrets but with total count."""
        scan = _make_secrets_scan(
            scan_id="sec_zero",
            secrets=[_make_secret()],
        )
        repo_scans["sec_zero"] = scan

        result = await handle_get_secrets(repo_id="test-repo", limit=0)

        assert _status(result) == 200
        body = _body(result)
        assert len(body["secrets"]) == 0
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_negative_offset_pagination(self, repo_scans):
        """Negative offset is treated as slicing from the end."""
        secrets = [_make_secret(secret_id=f"s_{i}") for i in range(5)]
        scan = _make_secrets_scan(scan_id="sec_neg", secrets=secrets)
        repo_scans["sec_neg"] = scan

        # Python handles negative slicing: secrets[-1:-1+100] = secrets[-1:]
        result = await handle_get_secrets(repo_id="test-repo", offset=-1)

        assert _status(result) == 200
        body = _body(result)
        # Negative offset produces a slice from the end
        assert body["total"] == 5

    @pytest.mark.asyncio
    async def test_large_limit_returns_all(self, repo_scans):
        """A limit larger than total returns all secrets."""
        secrets = [_make_secret(secret_id=f"s_{i}") for i in range(3)]
        scan = _make_secrets_scan(scan_id="sec_large", secrets=secrets)
        repo_scans["sec_large"] = scan

        result = await handle_get_secrets(repo_id="test-repo", limit=10000)

        assert _status(result) == 200
        body = _body(result)
        assert len(body["secrets"]) == 3
        assert body["total"] == 3


# ============================================================================
# Edge case tests
# ============================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_multiple_repos_isolated(self):
        """Scans from different repos are isolated."""
        from aragora.server.handlers.codebase.security.storage import (
            get_or_create_secrets_scans,
        )

        repo_a = get_or_create_secrets_scans("repo-a")
        repo_b = get_or_create_secrets_scans("repo-b")

        repo_a["scan_a"] = _make_secrets_scan(scan_id="scan_a", repo_id="repo-a")
        repo_b["scan_b"] = _make_secrets_scan(scan_id="scan_b", repo_id="repo-b")

        result_a = await handle_get_secrets_scan_status(repo_id="repo-a", scan_id="scan_a")
        result_b = await handle_get_secrets_scan_status(repo_id="repo-b", scan_id="scan_b")

        assert _status(result_a) == 200
        assert _status(result_b) == 200
        assert _body(result_a)["scan_result"]["scan_id"] == "scan_a"
        assert _body(result_b)["scan_result"]["scan_id"] == "scan_b"

    @pytest.mark.asyncio
    async def test_cross_repo_scan_not_visible(self):
        """Scan from repo-a is not visible when querying repo-b."""
        from aragora.server.handlers.codebase.security.storage import (
            get_or_create_secrets_scans,
        )

        repo_a = get_or_create_secrets_scans("repo-a")
        repo_a["scan_a"] = _make_secrets_scan(scan_id="scan_a", repo_id="repo-a")

        result = await handle_get_secrets_scan_status(repo_id="repo-b", scan_id="scan_a")

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_scan_with_many_secrets(self, repo_scans):
        """Handler handles a large number of secrets efficiently."""
        secrets = [
            _make_secret(
                secret_id=f"s_{i}",
                severity=VulnerabilitySeverity.HIGH,
            )
            for i in range(500)
        ]
        scan = _make_secrets_scan(scan_id="sec_many", secrets=secrets)
        repo_scans["sec_many"] = scan

        result = await handle_get_secrets(repo_id="test-repo", limit=10)

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 500
        assert len(body["secrets"]) == 10

    @pytest.mark.asyncio
    async def test_list_scans_with_mixed_statuses(self, repo_scans):
        """Listing scans with mixed statuses returns all."""
        repo_scans["s1"] = _make_secrets_scan(
            scan_id="s1",
            status="completed",
            started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        repo_scans["s2"] = _make_secrets_scan(
            scan_id="s2",
            status="running",
            started_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        repo_scans["s3"] = _make_secrets_scan(
            scan_id="s3",
            status="failed",
            started_at=datetime(2024, 1, 3, tzinfo=timezone.utc),
        )

        result = await handle_list_secrets_scans(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 3
        statuses = {s["status"] for s in body["scans"]}
        assert statuses == {"completed", "running", "failed"}

    @pytest.mark.asyncio
    async def test_get_scan_status_scan_to_dict(self, repo_scans):
        """The returned scan_result matches to_dict() output format."""
        secret = _make_secret()
        scan = _make_secrets_scan(
            scan_id="sec_dict",
            secrets=[secret],
            branch="main",
        )
        repo_scans["sec_dict"] = scan

        result = await handle_get_secrets_scan_status(repo_id="test-repo", scan_id="sec_dict")

        body = _body(result)
        scan_data = body["scan_result"]

        # Verify key fields from to_dict()
        assert "scan_id" in scan_data
        assert "repository" in scan_data
        assert "branch" in scan_data
        assert "status" in scan_data
        assert "started_at" in scan_data
        assert "secrets" in scan_data
        assert "summary" in scan_data

    @pytest.mark.asyncio
    async def test_concurrent_scans_different_repos(self, running_scans, monkeypatch):
        """Different repos can have concurrent scans."""
        import aragora.server.handlers.codebase.security.secrets as secrets_mod

        monkeypatch.setattr(secrets_mod, "emit_secrets_events", AsyncMock())

        result_a = await handle_scan_secrets(
            repo_path="/tmp/repo-a",
            repo_id="repo-a",
        )
        result_b = await handle_scan_secrets(
            repo_path="/tmp/repo-b",
            repo_id="repo-b",
        )

        assert _status(result_a) == 200
        assert _status(result_b) == 200
        assert "repo-a" in running_scans
        assert "repo-b" in running_scans

    @pytest.mark.asyncio
    async def test_secrets_data_integrity(self, repo_scans):
        """Secret data is preserved through storage and retrieval."""
        secret = SecretFinding(
            id="integrity_test",
            secret_type=SecretType.AWS_SECRET_KEY,
            file_path="deploy/secrets.yaml",
            line_number=99,
            column_start=5,
            column_end=45,
            matched_text="AKIA****...****wxyz",
            context_line="AWS_SECRET=AKIA****...****wxyz",
            severity=VulnerabilitySeverity.CRITICAL,
            confidence=0.99,
            entropy=4.5,
            is_in_history=True,
            commit_sha="deadbeef",
            commit_author="dev@example.com",
            verified=True,
            remediation="Rotate key immediately",
        )
        scan = _make_secrets_scan(
            scan_id="sec_integrity",
            secrets=[secret],
        )
        repo_scans["sec_integrity"] = scan

        result = await handle_get_secrets(repo_id="test-repo")

        assert _status(result) == 200
        body = _body(result)
        s = body["secrets"][0]
        assert s["id"] == "integrity_test"
        assert s["secret_type"] == "aws_secret_key"
        assert s["file_path"] == "deploy/secrets.yaml"
        assert s["line_number"] == 99
        assert s["severity"] == "critical"
        assert s["confidence"] == 0.99
        assert s["is_in_history"] is True
        assert s["verified"] is True
        assert s["remediation"] == "Rotate key immediately"

    @pytest.mark.asyncio
    async def test_list_scans_scan_entry_completed_at_format(self, repo_scans):
        """completed_at is formatted as ISO string for completed scans."""
        scan = _make_secrets_scan(
            scan_id="sec_iso",
            completed_at=datetime(2024, 3, 15, 14, 30, 0, tzinfo=timezone.utc),
        )
        repo_scans["sec_iso"] = scan

        result = await handle_list_secrets_scans(repo_id="test-repo")

        body = _body(result)
        entry = body["scans"][0]
        assert "2024-03-15" in entry["completed_at"]
        assert "14:30" in entry["completed_at"]

    @pytest.mark.asyncio
    async def test_list_scans_started_at_format(self, repo_scans):
        """started_at is formatted as ISO string."""
        scan = _make_secrets_scan(
            scan_id="sec_start",
            started_at=datetime(2024, 6, 1, 8, 0, 0, tzinfo=timezone.utc),
        )
        repo_scans["sec_start"] = scan

        result = await handle_list_secrets_scans(repo_id="test-repo")

        body = _body(result)
        entry = body["scans"][0]
        assert "2024-06-01" in entry["started_at"]
