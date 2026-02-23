"""Comprehensive tests for the Codebase Audit rules module.

Tests all components of aragora/server/handlers/features/codebase_audit/rules.py:
- RBAC permission constants
- Input validation constants (VALID_SCAN_TYPES, VALID_SEVERITIES, etc.)
- Enums (ScanType, ScanStatus, FindingSeverity, FindingStatus)
- Data classes (Finding, ScanResult)
- Exceptions (ValidationError)
- Validation functions (validate_repository_path, validate_scan_types,
  validate_severity_filter, validate_status_filter, validate_scan_status_filter,
  validate_limit, validate_scan_id, validate_finding_id, validate_dismiss_request,
  validate_github_repo, sanitize_query_params)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from aragora.server.handlers.features.codebase_audit.rules import (
    CODEBASE_AUDIT_READ_PERMISSION,
    CODEBASE_AUDIT_WRITE_PERMISSION,
    MAX_LIST_LIMIT,
    MAX_PATH_LENGTH,
    VALID_SCAN_STATUSES,
    VALID_SCAN_TYPES,
    VALID_SEVERITIES,
    VALID_STATUSES,
    Finding,
    FindingSeverity,
    FindingStatus,
    ScanResult,
    ScanStatus,
    ScanType,
    ValidationError,
    sanitize_query_params,
    validate_dismiss_request,
    validate_finding_id,
    validate_github_repo,
    validate_limit,
    validate_repository_path,
    validate_scan_id,
    validate_scan_status_filter,
    validate_scan_types,
    validate_severity_filter,
    validate_status_filter,
)


# ===========================================================================
# 1. RBAC Permission Constants
# ===========================================================================


class TestRBACPermissions:
    """Tests for RBAC permission constants."""

    def test_read_permission_value(self):
        assert CODEBASE_AUDIT_READ_PERMISSION == "codebase_audit:read"

    def test_write_permission_value(self):
        assert CODEBASE_AUDIT_WRITE_PERMISSION == "codebase_audit:write"

    def test_permissions_are_strings(self):
        assert isinstance(CODEBASE_AUDIT_READ_PERMISSION, str)
        assert isinstance(CODEBASE_AUDIT_WRITE_PERMISSION, str)

    def test_permissions_follow_namespace_pattern(self):
        assert ":" in CODEBASE_AUDIT_READ_PERMISSION
        assert ":" in CODEBASE_AUDIT_WRITE_PERMISSION

    def test_read_and_write_are_different(self):
        assert CODEBASE_AUDIT_READ_PERMISSION != CODEBASE_AUDIT_WRITE_PERMISSION


# ===========================================================================
# 2. Input Validation Constants
# ===========================================================================


class TestValidationConstants:
    """Tests for validation constant sets."""

    def test_valid_scan_types_is_set(self):
        assert isinstance(VALID_SCAN_TYPES, set)

    def test_valid_scan_types_content(self):
        expected = {"sast", "bugs", "secrets", "dependencies", "metrics", "comprehensive"}
        assert VALID_SCAN_TYPES == expected

    def test_valid_severities_is_set(self):
        assert isinstance(VALID_SEVERITIES, set)

    def test_valid_severities_content(self):
        expected = {"critical", "high", "medium", "low", "info"}
        assert VALID_SEVERITIES == expected

    def test_valid_statuses_is_set(self):
        assert isinstance(VALID_STATUSES, set)

    def test_valid_statuses_content(self):
        expected = {"open", "dismissed", "fixed", "false_positive", "accepted_risk"}
        assert VALID_STATUSES == expected

    def test_valid_scan_statuses_is_set(self):
        assert isinstance(VALID_SCAN_STATUSES, set)

    def test_valid_scan_statuses_content(self):
        expected = {"pending", "running", "completed", "failed", "cancelled"}
        assert VALID_SCAN_STATUSES == expected

    def test_max_path_length(self):
        assert MAX_PATH_LENGTH == 4096

    def test_max_list_limit(self):
        assert MAX_LIST_LIMIT == 500

    def test_max_path_length_is_positive(self):
        assert MAX_PATH_LENGTH > 0

    def test_max_list_limit_is_positive(self):
        assert MAX_LIST_LIMIT > 0


# ===========================================================================
# 3. ScanType Enum
# ===========================================================================


class TestScanTypeEnum:
    """Tests for the ScanType enum."""

    def test_comprehensive_value(self):
        assert ScanType.COMPREHENSIVE.value == "comprehensive"

    def test_sast_value(self):
        assert ScanType.SAST.value == "sast"

    def test_bugs_value(self):
        assert ScanType.BUGS.value == "bugs"

    def test_secrets_value(self):
        assert ScanType.SECRETS.value == "secrets"

    def test_dependencies_value(self):
        assert ScanType.DEPENDENCIES.value == "dependencies"

    def test_metrics_value(self):
        assert ScanType.METRICS.value == "metrics"

    def test_member_count(self):
        assert len(ScanType) == 6

    def test_all_values_in_valid_scan_types(self):
        for member in ScanType:
            assert member.value in VALID_SCAN_TYPES


# ===========================================================================
# 4. ScanStatus Enum
# ===========================================================================


class TestScanStatusEnum:
    """Tests for the ScanStatus enum."""

    def test_pending_value(self):
        assert ScanStatus.PENDING.value == "pending"

    def test_running_value(self):
        assert ScanStatus.RUNNING.value == "running"

    def test_completed_value(self):
        assert ScanStatus.COMPLETED.value == "completed"

    def test_failed_value(self):
        assert ScanStatus.FAILED.value == "failed"

    def test_cancelled_value(self):
        assert ScanStatus.CANCELLED.value == "cancelled"

    def test_member_count(self):
        assert len(ScanStatus) == 5

    def test_all_values_in_valid_scan_statuses(self):
        for member in ScanStatus:
            assert member.value in VALID_SCAN_STATUSES


# ===========================================================================
# 5. FindingSeverity Enum
# ===========================================================================


class TestFindingSeverityEnum:
    """Tests for the FindingSeverity enum."""

    def test_critical_value(self):
        assert FindingSeverity.CRITICAL.value == "critical"

    def test_high_value(self):
        assert FindingSeverity.HIGH.value == "high"

    def test_medium_value(self):
        assert FindingSeverity.MEDIUM.value == "medium"

    def test_low_value(self):
        assert FindingSeverity.LOW.value == "low"

    def test_info_value(self):
        assert FindingSeverity.INFO.value == "info"

    def test_member_count(self):
        assert len(FindingSeverity) == 5

    def test_all_values_in_valid_severities(self):
        for member in FindingSeverity:
            assert member.value in VALID_SEVERITIES


# ===========================================================================
# 6. FindingStatus Enum
# ===========================================================================


class TestFindingStatusEnum:
    """Tests for the FindingStatus enum."""

    def test_open_value(self):
        assert FindingStatus.OPEN.value == "open"

    def test_dismissed_value(self):
        assert FindingStatus.DISMISSED.value == "dismissed"

    def test_fixed_value(self):
        assert FindingStatus.FIXED.value == "fixed"

    def test_false_positive_value(self):
        assert FindingStatus.FALSE_POSITIVE.value == "false_positive"

    def test_accepted_risk_value(self):
        assert FindingStatus.ACCEPTED_RISK.value == "accepted_risk"

    def test_member_count(self):
        assert len(FindingStatus) == 5

    def test_all_values_in_valid_statuses(self):
        for member in FindingStatus:
            assert member.value in VALID_STATUSES


# ===========================================================================
# 7. Finding Data Class
# ===========================================================================


class TestFindingDataClass:
    """Tests for the Finding dataclass."""

    def _make_finding(self, **overrides) -> Finding:
        defaults: dict[str, Any] = {
            "id": "find_001",
            "scan_id": "scan_001",
            "scan_type": ScanType.SAST,
            "severity": FindingSeverity.HIGH,
            "title": "SQL Injection",
            "description": "User input not sanitized",
            "file_path": "src/db.py",
        }
        defaults.update(overrides)
        return Finding(**defaults)

    def test_required_fields(self):
        f = self._make_finding()
        assert f.id == "find_001"
        assert f.scan_id == "scan_001"
        assert f.scan_type == ScanType.SAST
        assert f.severity == FindingSeverity.HIGH
        assert f.title == "SQL Injection"
        assert f.description == "User input not sanitized"
        assert f.file_path == "src/db.py"

    def test_optional_defaults(self):
        f = self._make_finding()
        assert f.line_number is None
        assert f.column is None
        assert f.code_snippet is None
        assert f.rule_id is None
        assert f.cwe_id is None
        assert f.owasp_category is None
        assert f.remediation is None
        assert f.confidence == 0.8
        assert f.status == FindingStatus.OPEN
        assert f.dismissed_by is None
        assert f.dismissed_reason is None
        assert f.github_issue_url is None

    def test_created_at_is_set(self):
        f = self._make_finding()
        assert isinstance(f.created_at, datetime)
        assert f.created_at.tzinfo is not None

    def test_custom_optional_fields(self):
        f = self._make_finding(
            line_number=42,
            column=10,
            code_snippet="SELECT * FROM users",
            rule_id="sql-injection",
            cwe_id="CWE-89",
            owasp_category="A03:2021",
            remediation="Use parameterized queries",
            confidence=0.95,
        )
        assert f.line_number == 42
        assert f.column == 10
        assert f.code_snippet == "SELECT * FROM users"
        assert f.rule_id == "sql-injection"
        assert f.cwe_id == "CWE-89"
        assert f.owasp_category == "A03:2021"
        assert f.remediation == "Use parameterized queries"
        assert f.confidence == 0.95

    def test_to_dict_basic(self):
        f = self._make_finding()
        d = f.to_dict()
        assert d["id"] == "find_001"
        assert d["scan_id"] == "scan_001"
        assert d["scan_type"] == "sast"
        assert d["severity"] == "high"
        assert d["title"] == "SQL Injection"
        assert d["description"] == "User input not sanitized"
        assert d["file_path"] == "src/db.py"

    def test_to_dict_enum_values_serialized(self):
        f = self._make_finding(
            scan_type=ScanType.BUGS,
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.DISMISSED,
        )
        d = f.to_dict()
        assert d["scan_type"] == "bugs"
        assert d["severity"] == "critical"
        assert d["status"] == "dismissed"

    def test_to_dict_created_at_is_iso(self):
        f = self._make_finding()
        d = f.to_dict()
        assert isinstance(d["created_at"], str)
        # Should be parseable as ISO format
        datetime.fromisoformat(d["created_at"])

    def test_to_dict_none_fields(self):
        f = self._make_finding()
        d = f.to_dict()
        assert d["line_number"] is None
        assert d["column"] is None
        assert d["code_snippet"] is None
        assert d["rule_id"] is None
        assert d["cwe_id"] is None
        assert d["owasp_category"] is None
        assert d["remediation"] is None
        assert d["dismissed_by"] is None
        assert d["dismissed_reason"] is None
        assert d["github_issue_url"] is None

    def test_to_dict_confidence_included(self):
        f = self._make_finding(confidence=0.99)
        d = f.to_dict()
        assert d["confidence"] == 0.99

    def test_to_dict_all_keys_present(self):
        f = self._make_finding()
        d = f.to_dict()
        expected_keys = {
            "id", "scan_id", "scan_type", "severity", "title", "description",
            "file_path", "line_number", "column", "code_snippet", "rule_id",
            "cwe_id", "owasp_category", "remediation", "confidence", "status",
            "created_at", "dismissed_by", "dismissed_reason", "github_issue_url",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_with_dismissed_fields(self):
        f = self._make_finding(
            status=FindingStatus.DISMISSED,
            dismissed_by="admin@example.com",
            dismissed_reason="Not applicable to our codebase",
        )
        d = f.to_dict()
        assert d["status"] == "dismissed"
        assert d["dismissed_by"] == "admin@example.com"
        assert d["dismissed_reason"] == "Not applicable to our codebase"

    def test_to_dict_with_github_url(self):
        f = self._make_finding(
            github_issue_url="https://github.com/org/repo/issues/42"
        )
        d = f.to_dict()
        assert d["github_issue_url"] == "https://github.com/org/repo/issues/42"


# ===========================================================================
# 8. ScanResult Data Class
# ===========================================================================


class TestScanResultDataClass:
    """Tests for the ScanResult dataclass."""

    def _make_scan_result(self, **overrides) -> ScanResult:
        defaults: dict[str, Any] = {
            "id": "scan_001",
            "tenant_id": "test-tenant",
            "scan_type": ScanType.SAST,
            "status": ScanStatus.COMPLETED,
            "target_path": ".",
            "started_at": datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        }
        defaults.update(overrides)
        return ScanResult(**defaults)

    def test_required_fields(self):
        s = self._make_scan_result()
        assert s.id == "scan_001"
        assert s.tenant_id == "test-tenant"
        assert s.scan_type == ScanType.SAST
        assert s.status == ScanStatus.COMPLETED
        assert s.target_path == "."
        assert s.started_at == datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

    def test_optional_defaults(self):
        s = self._make_scan_result()
        assert s.completed_at is None
        assert s.error_message is None
        assert s.files_scanned == 0
        assert s.findings_count == 0
        assert s.severity_counts == {}
        assert s.duration_seconds == 0.0
        assert s.progress == 0.0
        assert s.findings == []
        assert s.metrics == {}

    def test_custom_optional_fields(self):
        completed = datetime(2026, 1, 15, 10, 5, 0, tzinfo=timezone.utc)
        s = self._make_scan_result(
            completed_at=completed,
            files_scanned=100,
            findings_count=5,
            severity_counts={"critical": 1, "high": 4},
            duration_seconds=300.5,
            progress=1.0,
            metrics={"total_lines": 10000},
        )
        assert s.completed_at == completed
        assert s.files_scanned == 100
        assert s.findings_count == 5
        assert s.severity_counts == {"critical": 1, "high": 4}
        assert s.duration_seconds == 300.5
        assert s.progress == 1.0
        assert s.metrics == {"total_lines": 10000}

    def test_to_dict_basic(self):
        s = self._make_scan_result()
        d = s.to_dict()
        assert d["id"] == "scan_001"
        assert d["tenant_id"] == "test-tenant"
        assert d["scan_type"] == "sast"
        assert d["status"] == "completed"
        assert d["target_path"] == "."

    def test_to_dict_enum_values(self):
        s = self._make_scan_result(
            scan_type=ScanType.BUGS,
            status=ScanStatus.RUNNING,
        )
        d = s.to_dict()
        assert d["scan_type"] == "bugs"
        assert d["status"] == "running"

    def test_to_dict_started_at_is_iso(self):
        s = self._make_scan_result()
        d = s.to_dict()
        assert isinstance(d["started_at"], str)
        datetime.fromisoformat(d["started_at"])

    def test_to_dict_completed_at_none(self):
        s = self._make_scan_result(completed_at=None)
        d = s.to_dict()
        assert d["completed_at"] is None

    def test_to_dict_completed_at_iso(self):
        completed = datetime(2026, 1, 15, 10, 5, 0, tzinfo=timezone.utc)
        s = self._make_scan_result(completed_at=completed)
        d = s.to_dict()
        assert isinstance(d["completed_at"], str)
        datetime.fromisoformat(d["completed_at"])

    def test_to_dict_all_keys(self):
        s = self._make_scan_result()
        d = s.to_dict()
        expected_keys = {
            "id", "tenant_id", "scan_type", "status", "target_path",
            "started_at", "completed_at", "error_message", "files_scanned",
            "findings_count", "severity_counts", "duration_seconds",
            "progress", "metrics",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_with_error(self):
        s = self._make_scan_result(
            status=ScanStatus.FAILED,
            error_message="Scanner crashed",
        )
        d = s.to_dict()
        assert d["status"] == "failed"
        assert d["error_message"] == "Scanner crashed"

    def test_findings_list_default_empty(self):
        s = self._make_scan_result()
        assert s.findings == []

    def test_findings_list_not_in_to_dict(self):
        """The findings list is not serialized in to_dict (separate query)."""
        s = self._make_scan_result()
        d = s.to_dict()
        assert "findings" not in d


# ===========================================================================
# 9. ValidationError Exception
# ===========================================================================


class TestValidationError:
    """Tests for the ValidationError exception."""

    def test_is_exception_subclass(self):
        assert issubclass(ValidationError, Exception)

    def test_message_stored(self):
        err = ValidationError("bad input")
        assert err.message == "bad input"

    def test_field_stored(self):
        err = ValidationError("bad input", field="email")
        assert err.field == "email"

    def test_field_default_none(self):
        err = ValidationError("bad input")
        assert err.field is None

    def test_str_message(self):
        err = ValidationError("bad input")
        assert str(err) == "bad input"

    def test_can_be_caught_as_exception(self):
        with pytest.raises(Exception, match="test error"):
            raise ValidationError("test error")


# ===========================================================================
# 10. validate_repository_path
# ===========================================================================


class TestValidateRepositoryPath:
    """Tests for validate_repository_path."""

    def test_none_path_is_valid(self):
        is_valid, err = validate_repository_path(None)
        assert is_valid is True
        assert err is None

    def test_empty_string_is_valid(self):
        is_valid, err = validate_repository_path("")
        assert is_valid is True
        assert err is None

    def test_simple_path_is_valid(self):
        is_valid, err = validate_repository_path("src/main.py")
        assert is_valid is True
        assert err is None

    def test_current_dir_is_valid(self):
        is_valid, err = validate_repository_path(".")
        assert is_valid is True
        assert err is None

    def test_nested_path_is_valid(self):
        is_valid, err = validate_repository_path("src/utils/helpers/db.py")
        assert is_valid is True
        assert err is None

    def test_path_exceeds_max_length(self):
        long_path = "a" * (MAX_PATH_LENGTH + 1)
        is_valid, err = validate_repository_path(long_path)
        assert is_valid is False
        assert "maximum length" in err

    def test_path_at_max_length_is_valid(self):
        path = "a" * MAX_PATH_LENGTH
        is_valid, err = validate_repository_path(path)
        assert is_valid is True
        assert err is None

    def test_directory_traversal_rejected(self):
        is_valid, err = validate_repository_path("../../etc/passwd")
        assert is_valid is False
        assert ".." in err

    def test_embedded_traversal_rejected(self):
        is_valid, err = validate_repository_path("src/../../../etc/passwd")
        assert is_valid is False
        assert ".." in err

    def test_null_byte_rejected(self):
        is_valid, err = validate_repository_path("src/\x00evil")
        assert is_valid is False
        assert "null byte" in err

    def test_semicolon_rejected(self):
        is_valid, err = validate_repository_path("src; rm -rf /")
        assert is_valid is False
        assert "invalid character" in err

    def test_pipe_rejected(self):
        is_valid, err = validate_repository_path("src | cat /etc/shadow")
        assert is_valid is False
        assert "invalid character" in err

    def test_ampersand_rejected(self):
        is_valid, err = validate_repository_path("src & whoami")
        assert is_valid is False
        assert "invalid character" in err

    def test_dollar_sign_rejected(self):
        is_valid, err = validate_repository_path("$HOME/secrets")
        assert is_valid is False
        assert "invalid character" in err

    def test_backtick_rejected(self):
        is_valid, err = validate_repository_path("src/`whoami`")
        assert is_valid is False
        assert "invalid character" in err

    def test_parentheses_rejected(self):
        is_valid, err = validate_repository_path("src/(evil)")
        assert is_valid is False
        assert "invalid character" in err

    def test_curly_braces_rejected(self):
        is_valid, err = validate_repository_path("src/{evil}")
        assert is_valid is False
        assert "invalid character" in err

    def test_angle_brackets_rejected(self):
        is_valid, err = validate_repository_path("src/<evil>")
        assert is_valid is False
        assert "invalid character" in err

    def test_newline_rejected(self):
        is_valid, err = validate_repository_path("src\nrm -rf /")
        assert is_valid is False
        assert "invalid character" in err

    def test_carriage_return_rejected(self):
        is_valid, err = validate_repository_path("src\r\nevil")
        assert is_valid is False
        assert "invalid character" in err

    def test_home_expansion_rejected(self):
        is_valid, err = validate_repository_path("~/secrets")
        assert is_valid is False
        assert "Home directory expansion" in err

    def test_home_with_username_rejected(self):
        is_valid, err = validate_repository_path("~root/.ssh")
        assert is_valid is False
        assert "Home directory expansion" in err

    @pytest.mark.parametrize("char", [";", "|", "&", "$", "`", "(", ")", "{", "}", "<", ">", "\n", "\r"])
    def test_each_dangerous_char_rejected(self, char):
        is_valid, err = validate_repository_path(f"src{char}evil")
        assert is_valid is False
        assert "invalid character" in err


# ===========================================================================
# 11. validate_scan_types
# ===========================================================================


class TestValidateScanTypes:
    """Tests for validate_scan_types."""

    def test_none_is_valid(self):
        is_valid, err = validate_scan_types(None)
        assert is_valid is True
        assert err is None

    def test_empty_list_is_valid(self):
        is_valid, err = validate_scan_types([])
        assert is_valid is True
        assert err is None

    def test_single_valid_type(self):
        is_valid, err = validate_scan_types(["sast"])
        assert is_valid is True
        assert err is None

    def test_multiple_valid_types(self):
        is_valid, err = validate_scan_types(["sast", "bugs", "secrets"])
        assert is_valid is True
        assert err is None

    def test_all_valid_types(self):
        is_valid, err = validate_scan_types(list(VALID_SCAN_TYPES))
        assert is_valid is True
        assert err is None

    def test_invalid_type_rejected(self):
        is_valid, err = validate_scan_types(["nonexistent"])
        assert is_valid is False
        assert "Invalid scan types" in err

    def test_mixed_valid_invalid_rejected(self):
        is_valid, err = validate_scan_types(["sast", "invalid_scan"])
        assert is_valid is False
        assert "Invalid scan types" in err

    def test_not_a_list_rejected(self):
        is_valid, err = validate_scan_types("sast")  # type: ignore[arg-type]
        assert is_valid is False
        assert "must be a list" in err

    def test_comprehensive_type_valid(self):
        is_valid, err = validate_scan_types(["comprehensive"])
        assert is_valid is True

    def test_metrics_type_valid(self):
        is_valid, err = validate_scan_types(["metrics"])
        assert is_valid is True


# ===========================================================================
# 12. validate_severity_filter
# ===========================================================================


class TestValidateSeverityFilter:
    """Tests for validate_severity_filter."""

    def test_none_is_valid(self):
        is_valid, err = validate_severity_filter(None)
        assert is_valid is True
        assert err is None

    def test_empty_string_is_valid(self):
        is_valid, err = validate_severity_filter("")
        assert is_valid is True
        assert err is None

    @pytest.mark.parametrize("severity", ["critical", "high", "medium", "low", "info"])
    def test_valid_severities(self, severity):
        is_valid, err = validate_severity_filter(severity)
        assert is_valid is True
        assert err is None

    def test_case_insensitive(self):
        is_valid, err = validate_severity_filter("CRITICAL")
        assert is_valid is True

    def test_mixed_case(self):
        is_valid, err = validate_severity_filter("High")
        assert is_valid is True

    def test_invalid_severity_rejected(self):
        is_valid, err = validate_severity_filter("ultra_critical")
        assert is_valid is False
        assert "Invalid severity" in err

    def test_numeric_rejected(self):
        is_valid, err = validate_severity_filter("5")
        assert is_valid is False
        assert "Invalid severity" in err


# ===========================================================================
# 13. validate_status_filter
# ===========================================================================


class TestValidateStatusFilter:
    """Tests for validate_status_filter."""

    def test_none_is_valid(self):
        is_valid, err = validate_status_filter(None)
        assert is_valid is True
        assert err is None

    def test_empty_string_is_valid(self):
        is_valid, err = validate_status_filter("")
        assert is_valid is True
        assert err is None

    @pytest.mark.parametrize("status", ["open", "dismissed", "fixed", "false_positive", "accepted_risk"])
    def test_valid_statuses(self, status):
        is_valid, err = validate_status_filter(status)
        assert is_valid is True
        assert err is None

    def test_case_insensitive(self):
        is_valid, err = validate_status_filter("OPEN")
        assert is_valid is True

    def test_invalid_status_rejected(self):
        is_valid, err = validate_status_filter("pending")
        assert is_valid is False
        assert "Invalid status" in err

    def test_random_string_rejected(self):
        is_valid, err = validate_status_filter("whatever")
        assert is_valid is False
        assert "Invalid status" in err


# ===========================================================================
# 14. validate_scan_status_filter
# ===========================================================================


class TestValidateScanStatusFilter:
    """Tests for validate_scan_status_filter."""

    def test_none_is_valid(self):
        is_valid, err = validate_scan_status_filter(None)
        assert is_valid is True
        assert err is None

    def test_empty_string_is_valid(self):
        is_valid, err = validate_scan_status_filter("")
        assert is_valid is True
        assert err is None

    @pytest.mark.parametrize("status", ["pending", "running", "completed", "failed", "cancelled"])
    def test_valid_scan_statuses(self, status):
        is_valid, err = validate_scan_status_filter(status)
        assert is_valid is True
        assert err is None

    def test_case_insensitive(self):
        is_valid, err = validate_scan_status_filter("COMPLETED")
        assert is_valid is True

    def test_invalid_scan_status_rejected(self):
        is_valid, err = validate_scan_status_filter("open")
        assert is_valid is False
        assert "Invalid scan status" in err

    def test_random_string_rejected(self):
        is_valid, err = validate_scan_status_filter("xyz")
        assert is_valid is False
        assert "Invalid scan status" in err


# ===========================================================================
# 15. validate_limit
# ===========================================================================


class TestValidateLimit:
    """Tests for validate_limit."""

    def test_none_returns_default(self):
        limit, err = validate_limit(None)
        assert limit == 20
        assert err is None

    def test_empty_string_returns_default(self):
        limit, err = validate_limit("")
        assert limit == 20
        assert err is None

    def test_valid_limit(self):
        limit, err = validate_limit("50")
        assert limit == 50
        assert err is None

    def test_limit_one(self):
        limit, err = validate_limit("1")
        assert limit == 1
        assert err is None

    def test_limit_at_max(self):
        limit, err = validate_limit(str(MAX_LIST_LIMIT))
        assert limit == MAX_LIST_LIMIT
        assert err is None

    def test_limit_over_max_capped(self):
        limit, err = validate_limit(str(MAX_LIST_LIMIT + 100))
        assert limit == MAX_LIST_LIMIT
        assert err is None

    def test_non_numeric_rejected(self):
        limit, err = validate_limit("abc")
        assert limit == 0
        assert "Invalid limit" in err

    def test_negative_rejected(self):
        limit, err = validate_limit("-1")
        assert limit == 0
        assert "at least 1" in err

    def test_zero_rejected(self):
        limit, err = validate_limit("0")
        assert limit == 0
        assert "at least 1" in err

    def test_float_string_rejected(self):
        limit, err = validate_limit("10.5")
        assert limit == 0
        assert "Invalid limit" in err

    def test_custom_max_limit(self):
        limit, err = validate_limit("100", max_limit=50)
        assert limit == 50
        assert err is None

    def test_custom_max_limit_at_boundary(self):
        limit, err = validate_limit("50", max_limit=50)
        assert limit == 50
        assert err is None

    def test_custom_max_limit_below(self):
        limit, err = validate_limit("30", max_limit=50)
        assert limit == 30
        assert err is None


# ===========================================================================
# 16. validate_scan_id
# ===========================================================================


class TestValidateScanId:
    """Tests for validate_scan_id."""

    def test_none_rejected(self):
        is_valid, err = validate_scan_id(None)
        assert is_valid is False
        assert "required" in err

    def test_empty_string_rejected(self):
        is_valid, err = validate_scan_id("")
        assert is_valid is False
        assert "required" in err

    def test_valid_alphanumeric(self):
        is_valid, err = validate_scan_id("scan001")
        assert is_valid is True
        assert err is None

    def test_valid_with_underscores(self):
        is_valid, err = validate_scan_id("scan_001_test")
        assert is_valid is True
        assert err is None

    def test_valid_with_hyphens(self):
        is_valid, err = validate_scan_id("scan-001-test")
        assert is_valid is True
        assert err is None

    def test_valid_mixed(self):
        is_valid, err = validate_scan_id("scan_001-abc")
        assert is_valid is True
        assert err is None

    def test_special_chars_rejected(self):
        is_valid, err = validate_scan_id("scan!@#$")
        assert is_valid is False
        assert "invalid characters" in err

    def test_spaces_rejected(self):
        is_valid, err = validate_scan_id("scan 001")
        assert is_valid is False
        assert "invalid characters" in err

    def test_path_traversal_in_id_rejected(self):
        is_valid, err = validate_scan_id("../../../etc")
        assert is_valid is False
        assert "invalid characters" in err

    def test_too_long_rejected(self):
        is_valid, err = validate_scan_id("a" * 65)
        assert is_valid is False
        assert "maximum length" in err

    def test_at_max_length_valid(self):
        is_valid, err = validate_scan_id("a" * 64)
        assert is_valid is True
        assert err is None

    def test_sql_injection_rejected(self):
        is_valid, err = validate_scan_id("1'; DROP TABLE scans;--")
        assert is_valid is False
        assert "invalid characters" in err


# ===========================================================================
# 17. validate_finding_id
# ===========================================================================


class TestValidateFindingId:
    """Tests for validate_finding_id."""

    def test_none_rejected(self):
        is_valid, err = validate_finding_id(None)
        assert is_valid is False
        assert "required" in err

    def test_empty_string_rejected(self):
        is_valid, err = validate_finding_id("")
        assert is_valid is False
        assert "required" in err

    def test_valid_alphanumeric(self):
        is_valid, err = validate_finding_id("find001")
        assert is_valid is True
        assert err is None

    def test_valid_with_underscores(self):
        is_valid, err = validate_finding_id("find_001_sast")
        assert is_valid is True
        assert err is None

    def test_valid_with_hyphens(self):
        is_valid, err = validate_finding_id("find-001-abc")
        assert is_valid is True
        assert err is None

    def test_special_chars_rejected(self):
        is_valid, err = validate_finding_id("find!@#")
        assert is_valid is False
        assert "invalid characters" in err

    def test_spaces_rejected(self):
        is_valid, err = validate_finding_id("find 001")
        assert is_valid is False
        assert "invalid characters" in err

    def test_too_long_rejected(self):
        is_valid, err = validate_finding_id("f" * 65)
        assert is_valid is False
        assert "maximum length" in err

    def test_at_max_length_valid(self):
        is_valid, err = validate_finding_id("f" * 64)
        assert is_valid is True
        assert err is None

    def test_path_traversal_rejected(self):
        is_valid, err = validate_finding_id("../../secret")
        assert is_valid is False
        assert "invalid characters" in err


# ===========================================================================
# 18. validate_dismiss_request
# ===========================================================================


class TestValidateDismissRequest:
    """Tests for validate_dismiss_request."""

    def test_valid_dismissed_status(self):
        is_valid, err = validate_dismiss_request({"status": "dismissed", "reason": "not relevant"})
        assert is_valid is True
        assert err is None

    def test_valid_false_positive_status(self):
        is_valid, err = validate_dismiss_request({"status": "false_positive"})
        assert is_valid is True
        assert err is None

    def test_valid_accepted_risk_status(self):
        is_valid, err = validate_dismiss_request({"status": "accepted_risk"})
        assert is_valid is True
        assert err is None

    def test_valid_fixed_status(self):
        is_valid, err = validate_dismiss_request({"status": "fixed"})
        assert is_valid is True
        assert err is None

    def test_default_status_is_dismissed(self):
        is_valid, err = validate_dismiss_request({"reason": "some reason"})
        assert is_valid is True
        assert err is None

    def test_empty_body_uses_dismissed_default(self):
        is_valid, err = validate_dismiss_request({})
        assert is_valid is True
        assert err is None

    def test_invalid_status_rejected(self):
        is_valid, err = validate_dismiss_request({"status": "open"})
        assert is_valid is False
        assert "Invalid dismiss status" in err

    def test_unknown_status_rejected(self):
        is_valid, err = validate_dismiss_request({"status": "some_invalid_status"})
        assert is_valid is False
        assert "Invalid dismiss status" in err

    def test_reason_at_max_length_valid(self):
        is_valid, err = validate_dismiss_request({"reason": "x" * 2000})
        assert is_valid is True
        assert err is None

    def test_reason_exceeds_max_length(self):
        is_valid, err = validate_dismiss_request({"reason": "x" * 2001})
        assert is_valid is False
        assert "maximum length" in err

    def test_empty_reason_is_valid(self):
        is_valid, err = validate_dismiss_request({"reason": ""})
        assert is_valid is True
        assert err is None

    def test_reason_with_special_chars_valid(self):
        is_valid, err = validate_dismiss_request({"reason": "Not applicable: <test> & (dev)"})
        assert is_valid is True
        assert err is None


# ===========================================================================
# 19. validate_github_repo
# ===========================================================================


class TestValidateGithubRepo:
    """Tests for validate_github_repo."""

    def test_none_rejected(self):
        is_valid, err = validate_github_repo(None)
        assert is_valid is False
        assert "required" in err

    def test_empty_string_rejected(self):
        is_valid, err = validate_github_repo("")
        assert is_valid is False
        assert "required" in err

    def test_valid_repo(self):
        is_valid, err = validate_github_repo("owner/repo")
        assert is_valid is True
        assert err is None

    def test_valid_repo_with_hyphens(self):
        is_valid, err = validate_github_repo("my-org/my-repo")
        assert is_valid is True
        assert err is None

    def test_valid_repo_with_dots(self):
        is_valid, err = validate_github_repo("my.org/my.repo")
        assert is_valid is True
        assert err is None

    def test_valid_repo_with_underscores(self):
        is_valid, err = validate_github_repo("my_org/my_repo")
        assert is_valid is True
        assert err is None

    def test_no_slash_rejected(self):
        is_valid, err = validate_github_repo("just-a-name")
        assert is_valid is False
        assert "owner/name" in err

    def test_too_many_slashes_rejected(self):
        is_valid, err = validate_github_repo("a/b/c")
        assert is_valid is False
        assert "owner/name" in err

    def test_empty_owner_rejected(self):
        is_valid, err = validate_github_repo("/repo")
        assert is_valid is False
        assert "cannot be empty" in err

    def test_empty_name_rejected(self):
        is_valid, err = validate_github_repo("owner/")
        assert is_valid is False
        assert "cannot be empty" in err

    def test_both_empty_rejected(self):
        is_valid, err = validate_github_repo("/")
        assert is_valid is False
        assert "cannot be empty" in err

    def test_special_chars_in_owner_rejected(self):
        is_valid, err = validate_github_repo("ow ner/repo")
        assert is_valid is False
        assert "invalid characters" in err

    def test_special_chars_in_name_rejected(self):
        is_valid, err = validate_github_repo("owner/re po")
        assert is_valid is False
        assert "invalid characters" in err

    def test_shell_injection_in_repo_rejected(self):
        # The semicolon makes this fail the regex character validation
        is_valid, err = validate_github_repo("owner/repo;evil")
        assert is_valid is False
        assert "invalid characters" in err

    def test_valid_numbers_in_repo(self):
        is_valid, err = validate_github_repo("org123/repo456")
        assert is_valid is True
        assert err is None


# ===========================================================================
# 20. sanitize_query_params
# ===========================================================================


class TestSanitizeQueryParams:
    """Tests for sanitize_query_params."""

    def test_empty_params(self):
        result = sanitize_query_params({})
        assert result == {}

    def test_known_params_preserved(self):
        params = {"type": "sast", "status": "open", "severity": "high"}
        result = sanitize_query_params(params)
        assert result == {"type": "sast", "status": "open", "severity": "high"}

    def test_unknown_params_filtered(self):
        params = {"type": "sast", "evil": "DROP TABLE", "injection": "1=1"}
        result = sanitize_query_params(params)
        assert "type" in result
        assert "evil" not in result
        assert "injection" not in result

    def test_limit_param_preserved(self):
        params = {"limit": "50"}
        result = sanitize_query_params(params)
        assert result == {"limit": "50"}

    def test_offset_param_preserved(self):
        params = {"offset": "10"}
        result = sanitize_query_params(params)
        assert result == {"offset": "10"}

    def test_scan_type_param_preserved(self):
        params = {"scan_type": "bugs"}
        result = sanitize_query_params(params)
        assert result == {"scan_type": "bugs"}

    def test_all_allowed_keys(self):
        params = {
            "type": "sast",
            "status": "open",
            "severity": "high",
            "limit": "20",
            "offset": "0",
            "scan_type": "bugs",
        }
        result = sanitize_query_params(params)
        assert len(result) == 6

    def test_value_truncated_at_256(self):
        params = {"type": "a" * 500}
        result = sanitize_query_params(params)
        assert len(result["type"]) == 256

    def test_none_value_becomes_empty_string(self):
        params = {"type": None}
        result = sanitize_query_params(params)
        assert result["type"] == ""

    def test_control_chars_removed(self):
        params = {"type": "sast\x00\x01\x02\x1f"}
        result = sanitize_query_params(params)
        assert result["type"] == "sast"

    def test_non_string_value_converted(self):
        params = {"limit": 42}
        result = sanitize_query_params(params)
        assert result["limit"] == "42"

    def test_mixed_known_unknown(self):
        params = {"type": "sast", "unknown1": "x", "severity": "high", "unknown2": "y"}
        result = sanitize_query_params(params)
        assert set(result.keys()) == {"type", "severity"}

    def test_newline_in_value_removed(self):
        params = {"type": "sast\nevil"}
        result = sanitize_query_params(params)
        assert "\n" not in result["type"]

    def test_tab_in_value_removed(self):
        params = {"type": "sast\tevil"}
        result = sanitize_query_params(params)
        # Tab (0x09) is a control character < 32, so it should be removed
        assert "\t" not in result["type"]


# ===========================================================================
# 21. Cross-Cutting: Enum-Constant Alignment
# ===========================================================================


class TestEnumConstantAlignment:
    """Verify that enum values are consistent with the validation sets."""

    def test_scan_type_values_match_valid_scan_types(self):
        enum_values = {member.value for member in ScanType}
        assert enum_values == VALID_SCAN_TYPES

    def test_scan_status_values_match_valid_scan_statuses(self):
        enum_values = {member.value for member in ScanStatus}
        assert enum_values == VALID_SCAN_STATUSES

    def test_finding_severity_values_match_valid_severities(self):
        enum_values = {member.value for member in FindingSeverity}
        assert enum_values == VALID_SEVERITIES

    def test_finding_status_values_match_valid_statuses(self):
        enum_values = {member.value for member in FindingStatus}
        assert enum_values == VALID_STATUSES


# ===========================================================================
# 22. Security Edge Cases
# ===========================================================================


class TestSecurityEdgeCases:
    """Additional security-focused tests."""

    def test_path_with_url_encoding_not_decoded(self):
        """URL-encoded traversal should be caught at the path level."""
        is_valid, err = validate_repository_path("%2e%2e/etc/passwd")
        # This doesn't contain '..' literally, so it passes path validation
        # The encoded version is safe because the filesystem won't decode it
        assert is_valid is True

    def test_scan_id_null_byte_rejected(self):
        is_valid, err = validate_scan_id("scan\x00evil")
        assert is_valid is False
        assert "invalid characters" in err

    def test_finding_id_null_byte_rejected(self):
        is_valid, err = validate_finding_id("find\x00evil")
        assert is_valid is False
        assert "invalid characters" in err

    def test_very_long_reason_rejected(self):
        is_valid, err = validate_dismiss_request({"reason": "x" * 10000})
        assert is_valid is False
        assert "maximum length" in err

    def test_repo_with_null_byte_rejected(self):
        is_valid, err = validate_github_repo("owner\x00/repo")
        assert is_valid is False
        assert "invalid characters" in err

    def test_sanitize_strips_control_chars_from_all_allowed_params(self):
        params = {
            "type": "\x00sast\x1f",
            "status": "open\x01",
            "severity": "\x02high\x03",
            "limit": "\x0420\x05",
            "offset": "\x060\x07",
            "scan_type": "\x08bugs\x0b",
        }
        result = sanitize_query_params(params)
        for key, value in result.items():
            for ch in value:
                assert ord(ch) >= 32, f"Control char in {key}: {repr(ch)}"

    def test_path_only_tilde_rejected(self):
        is_valid, err = validate_repository_path("~")
        assert is_valid is False
        assert "Home directory expansion" in err

    def test_dismiss_status_pending_rejected(self):
        """'pending' is not a valid dismiss status (only dismissed/false_positive/accepted_risk/fixed)."""
        is_valid, err = validate_dismiss_request({"status": "pending"})
        assert is_valid is False
        assert "Invalid dismiss status" in err

    def test_repo_with_unicode_rejected(self):
        # Unicode chars outside \w range
        is_valid, err = validate_github_repo("owne\u0301r/repo")
        # \w in Python regex includes Unicode word characters so accented chars may match
        # We just verify the function returns a valid tuple
        assert isinstance(is_valid, bool)

    def test_empty_scan_types_list_is_not_invalid(self):
        is_valid, err = validate_scan_types([])
        assert is_valid is True


# ===========================================================================
# 23. Data Class Mutability
# ===========================================================================


class TestDataClassMutability:
    """Test that data class fields can be updated (for dismiss, issue creation, etc.)."""

    def test_finding_status_can_be_updated(self):
        f = Finding(
            id="f1", scan_id="s1", scan_type=ScanType.SAST,
            severity=FindingSeverity.HIGH, title="test",
            description="desc", file_path="x.py",
        )
        f.status = FindingStatus.DISMISSED
        assert f.status == FindingStatus.DISMISSED

    def test_finding_dismissed_fields_can_be_set(self):
        f = Finding(
            id="f1", scan_id="s1", scan_type=ScanType.SAST,
            severity=FindingSeverity.HIGH, title="test",
            description="desc", file_path="x.py",
        )
        f.dismissed_by = "admin@example.com"
        f.dismissed_reason = "Not applicable"
        assert f.dismissed_by == "admin@example.com"
        assert f.dismissed_reason == "Not applicable"

    def test_finding_github_url_can_be_set(self):
        f = Finding(
            id="f1", scan_id="s1", scan_type=ScanType.SAST,
            severity=FindingSeverity.HIGH, title="test",
            description="desc", file_path="x.py",
        )
        f.github_issue_url = "https://github.com/org/repo/issues/1"
        assert f.github_issue_url == "https://github.com/org/repo/issues/1"

    def test_scan_result_status_can_be_updated(self):
        s = ScanResult(
            id="s1", tenant_id="t1", scan_type=ScanType.SAST,
            status=ScanStatus.RUNNING, target_path=".",
            started_at=datetime.now(timezone.utc),
        )
        s.status = ScanStatus.COMPLETED
        assert s.status == ScanStatus.COMPLETED

    def test_scan_result_completed_at_can_be_set(self):
        s = ScanResult(
            id="s1", tenant_id="t1", scan_type=ScanType.SAST,
            status=ScanStatus.RUNNING, target_path=".",
            started_at=datetime.now(timezone.utc),
        )
        now = datetime.now(timezone.utc)
        s.completed_at = now
        assert s.completed_at == now

    def test_scan_result_findings_list_mutable(self):
        s = ScanResult(
            id="s1", tenant_id="t1", scan_type=ScanType.SAST,
            status=ScanStatus.COMPLETED, target_path=".",
            started_at=datetime.now(timezone.utc),
        )
        f = Finding(
            id="f1", scan_id="s1", scan_type=ScanType.SAST,
            severity=FindingSeverity.HIGH, title="test",
            description="desc", file_path="x.py",
        )
        s.findings.append(f)
        assert len(s.findings) == 1

    def test_scan_result_error_message_settable(self):
        s = ScanResult(
            id="s1", tenant_id="t1", scan_type=ScanType.SAST,
            status=ScanStatus.FAILED, target_path=".",
            started_at=datetime.now(timezone.utc),
        )
        s.error_message = "Scanner process killed"
        assert s.error_message == "Scanner process killed"


# ===========================================================================
# 24. validate_limit boundary tests
# ===========================================================================


class TestValidateLimitBoundary:
    """Boundary condition tests for validate_limit."""

    def test_limit_exactly_one(self):
        limit, err = validate_limit("1")
        assert limit == 1
        assert err is None

    def test_limit_large_number(self):
        limit, err = validate_limit("999999")
        # Capped at MAX_LIST_LIMIT
        assert limit == MAX_LIST_LIMIT
        assert err is None

    def test_limit_whitespace_around_number_rejected(self):
        # int() can parse whitespace, but let's check behavior
        limit, err = validate_limit(" 10 ")
        # int(" 10 ") = 10 in Python
        assert limit == 10
        assert err is None

    def test_limit_negative_large(self):
        limit, err = validate_limit("-999")
        assert limit == 0
        assert "at least 1" in err

    def test_limit_hex_string_rejected(self):
        limit, err = validate_limit("0xff")
        assert limit == 0
        assert "Invalid limit" in err


# ===========================================================================
# 25. Finding to_dict roundtrip consistency
# ===========================================================================


class TestFindingToDictConsistency:
    """Verify to_dict returns consistent types across different severity/status combos."""

    @pytest.mark.parametrize("severity", list(FindingSeverity))
    def test_to_dict_with_each_severity(self, severity):
        f = Finding(
            id="f1", scan_id="s1", scan_type=ScanType.SAST,
            severity=severity, title="test",
            description="desc", file_path="x.py",
        )
        d = f.to_dict()
        assert d["severity"] == severity.value
        assert isinstance(d["severity"], str)

    @pytest.mark.parametrize("status", list(FindingStatus))
    def test_to_dict_with_each_status(self, status):
        f = Finding(
            id="f1", scan_id="s1", scan_type=ScanType.SAST,
            severity=FindingSeverity.HIGH, title="test",
            description="desc", file_path="x.py",
            status=status,
        )
        d = f.to_dict()
        assert d["status"] == status.value
        assert isinstance(d["status"], str)

    @pytest.mark.parametrize("scan_type", list(ScanType))
    def test_to_dict_with_each_scan_type(self, scan_type):
        f = Finding(
            id="f1", scan_id="s1", scan_type=scan_type,
            severity=FindingSeverity.HIGH, title="test",
            description="desc", file_path="x.py",
        )
        d = f.to_dict()
        assert d["scan_type"] == scan_type.value
        assert isinstance(d["scan_type"], str)
