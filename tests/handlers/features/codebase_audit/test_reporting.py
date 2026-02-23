"""Comprehensive tests for the Codebase Audit Reporting module.

Covers all public functions in reporting.py:
- calculate_risk_score() — risk scoring with severity weights and confidence
- build_dashboard_data() — tenant dashboard assembly
- build_demo_data() — demo/mock dashboard generation

Tests include:
- Empty inputs, single findings, multiple findings
- Severity weight correctness (critical/high/medium/low/info)
- Confidence scaling
- Risk score capping at 100
- Dashboard structure validation
- Open vs non-open finding filtering
- Severity counts and type counts
- Metrics from latest metrics scan
- Recent scans ordering
- Top findings ordering by severity
- Demo data structure and invariants
- Edge cases (unknown severity, zero confidence, boundary values)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.features.codebase_audit.reporting import (
    build_dashboard_data,
    build_demo_data,
    calculate_risk_score,
)
from aragora.server.handlers.features.codebase_audit.rules import (
    Finding,
    FindingSeverity,
    FindingStatus,
    ScanResult,
    ScanStatus,
    ScanType,
)
from aragora.server.handlers.features.codebase_audit.scanning import (
    _finding_store,
    _get_tenant_findings,
    _get_tenant_scans,
    _scan_store,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_finding(
    finding_id: str = "find_001",
    scan_id: str = "scan_001",
    severity: FindingSeverity = FindingSeverity.HIGH,
    scan_type: ScanType = ScanType.SAST,
    status: FindingStatus = FindingStatus.OPEN,
    confidence: float = 0.8,
    title: str = "Test Finding",
) -> Finding:
    """Create a Finding with sensible defaults."""
    return Finding(
        id=finding_id,
        scan_id=scan_id,
        scan_type=scan_type,
        severity=severity,
        title=title,
        description="A test finding",
        file_path="src/test.py",
        line_number=10,
        confidence=confidence,
        status=status,
    )


def _seed_finding(
    tenant_id: str = "rpt-tenant",
    finding_id: str = "find_abc",
    severity: FindingSeverity = FindingSeverity.HIGH,
    scan_type: ScanType = ScanType.SAST,
    status: FindingStatus = FindingStatus.OPEN,
    confidence: float = 0.8,
    title: str = "Test Finding",
) -> Finding:
    """Create and store a finding in the in-memory store."""
    finding = _make_finding(
        finding_id=finding_id,
        severity=severity,
        scan_type=scan_type,
        status=status,
        confidence=confidence,
        title=title,
    )
    store = _get_tenant_findings(tenant_id)
    store[finding_id] = finding
    return finding


def _seed_scan(
    tenant_id: str = "rpt-tenant",
    scan_id: str = "scan_001",
    scan_type: ScanType = ScanType.SAST,
    status: ScanStatus = ScanStatus.COMPLETED,
    started_at: datetime | None = None,
    metrics: dict[str, Any] | None = None,
) -> ScanResult:
    """Create and store a scan in the in-memory store."""
    scan = ScanResult(
        id=scan_id,
        tenant_id=tenant_id,
        scan_type=scan_type,
        status=status,
        target_path=".",
        started_at=started_at or datetime.now(timezone.utc),
        metrics=metrics or {},
    )
    store = _get_tenant_scans(tenant_id)
    store[scan_id] = scan
    return scan


@pytest.fixture(autouse=True)
def _clean_stores():
    """Reset in-memory stores before each test."""
    _scan_store.clear()
    _finding_store.clear()
    yield
    _scan_store.clear()
    _finding_store.clear()


# ===========================================================================
# calculate_risk_score
# ===========================================================================


class TestCalculateRiskScore:
    """Tests for calculate_risk_score()."""

    def test_empty_findings_returns_zero(self):
        assert calculate_risk_score([]) == 0.0

    def test_single_critical_finding(self):
        # critical weight = 10, confidence = 0.8 => 10 * 0.8 = 8.0
        f = _make_finding(severity=FindingSeverity.CRITICAL, confidence=0.8)
        assert calculate_risk_score([f]) == 8.0

    def test_single_high_finding(self):
        # high weight = 5, confidence = 0.8 => 5 * 0.8 = 4.0
        f = _make_finding(severity=FindingSeverity.HIGH, confidence=0.8)
        assert calculate_risk_score([f]) == 4.0

    def test_single_medium_finding(self):
        # medium weight = 2, confidence = 0.8 => 2 * 0.8 = 1.6
        f = _make_finding(severity=FindingSeverity.MEDIUM, confidence=0.8)
        assert calculate_risk_score([f]) == pytest.approx(1.6)

    def test_single_low_finding(self):
        # low weight = 1, confidence = 0.8 => 1 * 0.8 = 0.8
        f = _make_finding(severity=FindingSeverity.LOW, confidence=0.8)
        assert calculate_risk_score([f]) == pytest.approx(0.8)

    def test_single_info_finding(self):
        # info weight = 0.1, confidence = 0.8 => 0.1 * 0.8 = 0.08
        f = _make_finding(severity=FindingSeverity.INFO, confidence=0.8)
        assert calculate_risk_score([f]) == pytest.approx(0.08)

    def test_confidence_one(self):
        # critical * 1.0 = 10.0
        f = _make_finding(severity=FindingSeverity.CRITICAL, confidence=1.0)
        assert calculate_risk_score([f]) == 10.0

    def test_confidence_zero(self):
        # any weight * 0 = 0
        f = _make_finding(severity=FindingSeverity.CRITICAL, confidence=0.0)
        assert calculate_risk_score([f]) == 0.0

    def test_multiple_findings_accumulate(self):
        # 2 high (5*0.8=4 each) + 1 medium (2*0.8=1.6) = 9.6
        findings = [
            _make_finding(finding_id="a", severity=FindingSeverity.HIGH, confidence=0.8),
            _make_finding(finding_id="b", severity=FindingSeverity.HIGH, confidence=0.8),
            _make_finding(finding_id="c", severity=FindingSeverity.MEDIUM, confidence=0.8),
        ]
        assert calculate_risk_score(findings) == pytest.approx(9.6)

    def test_risk_score_capped_at_100(self):
        # 15 critical findings at confidence 1.0 => 15*10 = 150, capped to 100
        findings = [
            _make_finding(finding_id=f"f{i}", severity=FindingSeverity.CRITICAL, confidence=1.0)
            for i in range(15)
        ]
        assert calculate_risk_score(findings) == 100.0

    def test_risk_score_exactly_100(self):
        # 10 critical at confidence 1.0 => 10*10 = 100
        findings = [
            _make_finding(finding_id=f"f{i}", severity=FindingSeverity.CRITICAL, confidence=1.0)
            for i in range(10)
        ]
        assert calculate_risk_score(findings) == 100.0

    def test_mixed_severities(self):
        # critical(10*0.5=5) + high(5*1.0=5) + medium(2*0.5=1) + low(1*1.0=1) + info(0.1*1.0=0.1) = 12.1
        findings = [
            _make_finding(finding_id="a", severity=FindingSeverity.CRITICAL, confidence=0.5),
            _make_finding(finding_id="b", severity=FindingSeverity.HIGH, confidence=1.0),
            _make_finding(finding_id="c", severity=FindingSeverity.MEDIUM, confidence=0.5),
            _make_finding(finding_id="d", severity=FindingSeverity.LOW, confidence=1.0),
            _make_finding(finding_id="e", severity=FindingSeverity.INFO, confidence=1.0),
        ]
        assert calculate_risk_score(findings) == pytest.approx(12.1)

    def test_returns_float_type(self):
        assert isinstance(calculate_risk_score([]), float)
        f = _make_finding()
        assert isinstance(calculate_risk_score([f]), float)

    def test_all_info_findings_low_score(self):
        # 5 info at 0.5 => 5 * 0.1 * 0.5 = 0.25
        findings = [
            _make_finding(finding_id=f"f{i}", severity=FindingSeverity.INFO, confidence=0.5)
            for i in range(5)
        ]
        assert calculate_risk_score(findings) == pytest.approx(0.25)

    def test_high_confidence_amplifies_score(self):
        low_conf = _make_finding(finding_id="a", severity=FindingSeverity.HIGH, confidence=0.1)
        high_conf = _make_finding(finding_id="b", severity=FindingSeverity.HIGH, confidence=1.0)
        assert calculate_risk_score([high_conf]) > calculate_risk_score([low_conf])


# ===========================================================================
# build_dashboard_data
# ===========================================================================


class TestBuildDashboardData:
    """Tests for build_dashboard_data()."""

    def test_empty_tenant_returns_defaults(self):
        data = build_dashboard_data("empty-tenant")
        assert data["summary"]["total_findings"] == 0
        assert data["summary"]["total_scans"] == 0
        assert data["summary"]["risk_score"] == 0.0
        assert data["metrics"] == {}
        assert data["recent_scans"] == []
        assert data["top_findings"] == []

    def test_severity_counts_zero_when_empty(self):
        data = build_dashboard_data("empty-tenant")
        counts = data["summary"]["severity_counts"]
        for sev in ("critical", "high", "medium", "low", "info"):
            assert counts[sev] == 0

    def test_type_counts_zero_when_empty(self):
        data = build_dashboard_data("empty-tenant")
        counts = data["summary"]["type_counts"]
        for st in ("sast", "bugs", "secrets", "dependencies"):
            assert counts[st] == 0

    def test_counts_only_open_findings(self):
        tid = "rpt-tenant"
        _seed_finding(tenant_id=tid, finding_id="f1", status=FindingStatus.OPEN)
        _seed_finding(tenant_id=tid, finding_id="f2", status=FindingStatus.DISMISSED)
        _seed_finding(tenant_id=tid, finding_id="f3", status=FindingStatus.FIXED)
        _seed_finding(tenant_id=tid, finding_id="f4", status=FindingStatus.OPEN)

        data = build_dashboard_data(tid)
        assert data["summary"]["total_findings"] == 2

    def test_severity_counts_correct(self):
        tid = "rpt-tenant"
        _seed_finding(tenant_id=tid, finding_id="f1", severity=FindingSeverity.CRITICAL)
        _seed_finding(tenant_id=tid, finding_id="f2", severity=FindingSeverity.HIGH)
        _seed_finding(tenant_id=tid, finding_id="f3", severity=FindingSeverity.HIGH)
        _seed_finding(tenant_id=tid, finding_id="f4", severity=FindingSeverity.MEDIUM)
        _seed_finding(tenant_id=tid, finding_id="f5", severity=FindingSeverity.LOW)
        _seed_finding(tenant_id=tid, finding_id="f6", severity=FindingSeverity.INFO)

        data = build_dashboard_data(tid)
        sc = data["summary"]["severity_counts"]
        assert sc["critical"] == 1
        assert sc["high"] == 2
        assert sc["medium"] == 1
        assert sc["low"] == 1
        assert sc["info"] == 1

    def test_type_counts_correct(self):
        tid = "rpt-tenant"
        _seed_finding(tenant_id=tid, finding_id="f1", scan_type=ScanType.SAST)
        _seed_finding(tenant_id=tid, finding_id="f2", scan_type=ScanType.SAST)
        _seed_finding(tenant_id=tid, finding_id="f3", scan_type=ScanType.BUGS)
        _seed_finding(tenant_id=tid, finding_id="f4", scan_type=ScanType.SECRETS)
        _seed_finding(tenant_id=tid, finding_id="f5", scan_type=ScanType.DEPENDENCIES)
        _seed_finding(tenant_id=tid, finding_id="f6", scan_type=ScanType.DEPENDENCIES)

        data = build_dashboard_data(tid)
        tc = data["summary"]["type_counts"]
        assert tc["sast"] == 2
        assert tc["bugs"] == 1
        assert tc["secrets"] == 1
        assert tc["dependencies"] == 2

    def test_metrics_scan_type_not_counted_in_type_counts(self):
        """Metrics scan type findings don't appear in type_counts dict."""
        tid = "rpt-tenant"
        _seed_finding(tenant_id=tid, finding_id="f1", scan_type=ScanType.METRICS)

        data = build_dashboard_data(tid)
        tc = data["summary"]["type_counts"]
        # "metrics" is not a key in type_counts, so the finding isn't counted
        assert tc["sast"] == 0
        assert tc["bugs"] == 0
        assert tc["secrets"] == 0
        assert tc["dependencies"] == 0

    def test_dismissed_findings_excluded_from_severity_counts(self):
        tid = "rpt-tenant"
        _seed_finding(
            tenant_id=tid,
            finding_id="f1",
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.DISMISSED,
        )
        data = build_dashboard_data(tid)
        assert data["summary"]["severity_counts"]["critical"] == 0

    def test_total_scans_count(self):
        tid = "rpt-tenant"
        _seed_scan(tenant_id=tid, scan_id="s1")
        _seed_scan(tenant_id=tid, scan_id="s2")
        _seed_scan(tenant_id=tid, scan_id="s3")

        data = build_dashboard_data(tid)
        assert data["summary"]["total_scans"] == 3

    def test_risk_score_reflects_open_findings(self):
        tid = "rpt-tenant"
        # 1 critical open (10*0.8=8) + 1 high dismissed (excluded) => 8.0
        _seed_finding(
            tenant_id=tid,
            finding_id="f1",
            severity=FindingSeverity.CRITICAL,
            status=FindingStatus.OPEN,
        )
        _seed_finding(
            tenant_id=tid,
            finding_id="f2",
            severity=FindingSeverity.HIGH,
            status=FindingStatus.DISMISSED,
        )

        data = build_dashboard_data(tid)
        assert data["summary"]["risk_score"] == pytest.approx(8.0)

    def test_latest_metrics_from_metrics_scan(self):
        tid = "rpt-tenant"
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        _seed_scan(
            tenant_id=tid,
            scan_id="m1",
            scan_type=ScanType.METRICS,
            started_at=base,
            metrics={"total_lines": 100},
        )
        _seed_scan(
            tenant_id=tid,
            scan_id="m2",
            scan_type=ScanType.METRICS,
            started_at=base + timedelta(hours=1),
            metrics={"total_lines": 200},
        )

        data = build_dashboard_data(tid)
        # Should use the last metrics scan
        assert data["metrics"]["total_lines"] == 200

    def test_metrics_empty_when_no_metrics_scans(self):
        tid = "rpt-tenant"
        _seed_scan(tenant_id=tid, scan_id="s1", scan_type=ScanType.SAST)

        data = build_dashboard_data(tid)
        assert data["metrics"] == {}

    def test_recent_scans_ordered_by_started_at_descending(self):
        tid = "rpt-tenant"
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        _seed_scan(tenant_id=tid, scan_id="s1", started_at=base)
        _seed_scan(tenant_id=tid, scan_id="s3", started_at=base + timedelta(hours=2))
        _seed_scan(tenant_id=tid, scan_id="s2", started_at=base + timedelta(hours=1))

        data = build_dashboard_data(tid)
        ids = [s["id"] for s in data["recent_scans"]]
        assert ids == ["s3", "s2", "s1"]

    def test_recent_scans_limited_to_5(self):
        tid = "rpt-tenant"
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        for i in range(8):
            _seed_scan(
                tenant_id=tid,
                scan_id=f"s{i}",
                started_at=base + timedelta(hours=i),
            )

        data = build_dashboard_data(tid)
        assert len(data["recent_scans"]) == 5
        # Most recent first
        assert data["recent_scans"][0]["id"] == "s7"

    def test_top_findings_ordered_by_severity(self):
        tid = "rpt-tenant"
        _seed_finding(tenant_id=tid, finding_id="f_low", severity=FindingSeverity.LOW, title="Low")
        _seed_finding(
            tenant_id=tid, finding_id="f_crit", severity=FindingSeverity.CRITICAL, title="Critical"
        )
        _seed_finding(
            tenant_id=tid, finding_id="f_high", severity=FindingSeverity.HIGH, title="High"
        )

        data = build_dashboard_data(tid)
        titles = [f["title"] for f in data["top_findings"]]
        assert titles == ["Critical", "High", "Low"]

    def test_top_findings_limited_to_10(self):
        tid = "rpt-tenant"
        for i in range(15):
            _seed_finding(
                tenant_id=tid,
                finding_id=f"f{i}",
                severity=FindingSeverity.MEDIUM,
                title=f"Finding {i}",
            )

        data = build_dashboard_data(tid)
        assert len(data["top_findings"]) == 10

    def test_top_findings_only_open(self):
        tid = "rpt-tenant"
        _seed_finding(tenant_id=tid, finding_id="f1", status=FindingStatus.OPEN)
        _seed_finding(tenant_id=tid, finding_id="f2", status=FindingStatus.FIXED)

        data = build_dashboard_data(tid)
        assert len(data["top_findings"]) == 1
        assert data["top_findings"][0]["id"] == "f1"

    def test_dashboard_data_structure_keys(self):
        data = build_dashboard_data("any-tenant")
        assert "summary" in data
        assert "metrics" in data
        assert "recent_scans" in data
        assert "top_findings" in data

    def test_summary_structure_keys(self):
        data = build_dashboard_data("any-tenant")
        summary = data["summary"]
        assert "total_findings" in summary
        assert "severity_counts" in summary
        assert "type_counts" in summary
        assert "total_scans" in summary
        assert "risk_score" in summary

    def test_recent_scans_are_dicts(self):
        tid = "rpt-tenant"
        _seed_scan(tenant_id=tid, scan_id="s1")
        data = build_dashboard_data(tid)
        assert len(data["recent_scans"]) == 1
        assert isinstance(data["recent_scans"][0], dict)
        assert data["recent_scans"][0]["id"] == "s1"

    def test_top_findings_are_dicts(self):
        tid = "rpt-tenant"
        _seed_finding(tenant_id=tid, finding_id="f1")
        data = build_dashboard_data(tid)
        assert len(data["top_findings"]) == 1
        assert isinstance(data["top_findings"][0], dict)
        assert data["top_findings"][0]["id"] == "f1"

    def test_different_tenants_isolated(self):
        _seed_finding(tenant_id="t1", finding_id="f1")
        _seed_finding(tenant_id="t2", finding_id="f2")
        _seed_scan(tenant_id="t1", scan_id="s1")

        data_t1 = build_dashboard_data("t1")
        data_t2 = build_dashboard_data("t2")

        assert data_t1["summary"]["total_findings"] == 1
        assert data_t2["summary"]["total_findings"] == 1
        assert data_t1["summary"]["total_scans"] == 1
        assert data_t2["summary"]["total_scans"] == 0


# ===========================================================================
# build_demo_data
# ===========================================================================


class TestBuildDemoData:
    """Tests for build_demo_data()."""

    def test_demo_data_has_is_demo_flag(self):
        data = build_demo_data()
        assert data["is_demo"] is True

    def test_demo_data_has_summary(self):
        data = build_demo_data()
        assert "summary" in data
        summary = data["summary"]
        assert "total_findings" in summary
        assert "severity_counts" in summary
        assert "type_counts" in summary
        assert "total_scans" in summary
        assert "risk_score" in summary

    def test_demo_data_has_metrics(self):
        data = build_demo_data()
        assert "metrics" in data
        metrics = data["metrics"]
        assert "total_lines" in metrics
        assert "code_lines" in metrics
        assert "files_analyzed" in metrics
        assert "average_complexity" in metrics

    def test_demo_data_has_findings_list(self):
        data = build_demo_data()
        assert "findings" in data
        assert isinstance(data["findings"], list)
        assert len(data["findings"]) > 0

    def test_demo_data_findings_are_dicts(self):
        data = build_demo_data()
        for f in data["findings"]:
            assert isinstance(f, dict)
            assert "id" in f
            assert "severity" in f
            assert "title" in f

    def test_demo_data_total_findings_matches_list(self):
        data = build_demo_data()
        assert data["summary"]["total_findings"] == len(data["findings"])

    def test_demo_data_type_counts(self):
        data = build_demo_data()
        tc = data["summary"]["type_counts"]
        assert tc["sast"] == 3
        assert tc["bugs"] == 2
        assert tc["secrets"] == 1
        assert tc["dependencies"] == 2

    def test_demo_data_total_scans(self):
        data = build_demo_data()
        assert data["summary"]["total_scans"] == 5

    def test_demo_data_risk_score(self):
        data = build_demo_data()
        assert data["summary"]["risk_score"] == 45.5

    def test_demo_data_severity_counts_sum(self):
        """Sum of severity counts should equal total findings."""
        data = build_demo_data()
        sc = data["summary"]["severity_counts"]
        total = sum(sc.values())
        assert total == data["summary"]["total_findings"]

    def test_demo_data_severity_keys(self):
        data = build_demo_data()
        sc = data["summary"]["severity_counts"]
        for key in ("critical", "high", "medium", "low", "info"):
            assert key in sc

    def test_demo_data_metrics_values_are_numeric(self):
        data = build_demo_data()
        m = data["metrics"]
        assert isinstance(m["total_lines"], (int, float))
        assert isinstance(m["code_lines"], (int, float))
        assert isinstance(m["average_complexity"], (int, float))

    def test_demo_data_findings_have_scan_id(self):
        """All demo findings should reference 'demo_scan'."""
        data = build_demo_data()
        for f in data["findings"]:
            assert f["scan_id"] == "demo_scan"

    def test_demo_data_findings_have_valid_severities(self):
        valid_severities = {"critical", "high", "medium", "low", "info"}
        data = build_demo_data()
        for f in data["findings"]:
            assert f["severity"] in valid_severities

    def test_demo_data_findings_have_valid_scan_types(self):
        valid_types = {"sast", "bugs", "secrets", "dependencies", "metrics", "comprehensive"}
        data = build_demo_data()
        for f in data["findings"]:
            assert f["scan_type"] in valid_types

    def test_demo_data_idempotent_structure(self):
        """Calling build_demo_data twice gives same structure."""
        d1 = build_demo_data()
        d2 = build_demo_data()
        assert d1["summary"]["total_findings"] == d2["summary"]["total_findings"]
        assert d1["summary"]["type_counts"] == d2["summary"]["type_counts"]
        assert d1["summary"]["risk_score"] == d2["summary"]["risk_score"]
        assert d1["metrics"].keys() == d2["metrics"].keys()
        assert len(d1["findings"]) == len(d2["findings"])
